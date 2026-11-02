"""MCP server: generate and edit images through OpenRouter's Images API.

Configuration comes from the environment (the launching project sets it):

- ``OPENROUTER_API_KEY``: the key for this project. Never returned or logged.
- ``IMAGE_GEN_OUTPUT_ROOT``: the only folder images are written to and
  reference images are read from (absolute path).
- ``IMAGE_GEN_DEFAULT_DIR``: where images go when no path is given,
  relative to the root (default ``generated``).
- ``IMAGE_GEN_DEFAULT_MODEL``: the model used when none is given (optional).
- ``IMAGE_GEN_APP_TITLE``: the ``X-Title`` sent to OpenRouter (optional).
"""

import base64
import datetime
import io
import json
import logging
import os
import sys
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

from .paths import default_image_path, media_type_for, numbered, resolve_inside

API = "https://openrouter.ai/api/v1"
GENERATE_TIMEOUT = 300.0
QUERY_TIMEOUT = 30.0
PREVIEW_SIZE = 768
MAX_PRICED_MODELS = 15

log = logging.getLogger("openrouter_image_mcp")

mcp = FastMCP(
    "openrouter-image",
    instructions=(
        "Generates and edits images through OpenRouter. Every generation costs money: "
        "check prices with list_image_models and the key's remaining credit with "
        "get_key_status before larger batches. Images are written only inside the "
        "output root; reference images must be inside it too (or HTTP(S) URLs). Each "
        "image gets a .json sidecar with the prompt, model, parameters and actual cost."
    ),
)


class ConfigError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ConfigError("OPENROUTER_API_KEY is not set for this server; the launching project must provide it.")
    return key


def _output_root() -> str:
    root = os.environ.get("IMAGE_GEN_OUTPUT_ROOT", "").strip()
    if not root or not os.path.isabs(root) or not os.path.isdir(root):
        raise ConfigError("IMAGE_GEN_OUTPUT_ROOT must be an existing absolute folder.")
    return root


def _headers(with_key: bool) -> dict[str, str]:
    headers = {"X-Title": os.environ.get("IMAGE_GEN_APP_TITLE", "openrouter-image-mcp")}
    if with_key:
        headers["Authorization"] = f"Bearer {_api_key()}"
    return headers


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = response.json().get("error", {}).get("message") or response.text
    except ValueError:
        detail = response.text
    raise RuntimeError(f"OpenRouter returned {response.status_code}: {str(detail)[:500]}")


def build_payload(
    prompt: str,
    model: str,
    references: list[dict[str, Any]],
    aspect_ratio: str | None,
    resolution: str | None,
    quality: str | None,
    background: str | None,
    output_format: str,
    seed: int | None,
    n: int,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """The request body for ``POST /images``; unset options are left out."""
    payload: dict[str, Any] = {"model": model, "prompt": prompt, "output_format": output_format, "n": n}
    optional = {
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "quality": quality,
        "background": background,
        "seed": seed,
    }
    payload.update({k: v for k, v in optional.items() if v is not None})
    if references:
        payload["input_references"] = references
    if extra:
        clash = set(extra) & set(payload)
        if clash:
            raise ValueError(f"extra_params repeats named parameters: {', '.join(sorted(clash))}")
        payload.update(extra)
    return payload


def _reference(root: str, ref: str) -> dict[str, Any]:
    if ref.startswith(("http://", "https://")):
        url = ref
    else:
        path = resolve_inside(root, ref)
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        url = f"data:{media_type_for(path)};base64,{data}"
    return {"type": "image_url", "image_url": {"url": url}}


def _preview(data: bytes) -> Image:
    with PILImage.open(io.BytesIO(data)) as img:
        img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def generate_image(
    prompt: str,
    model: str | None = None,
    out_path: str | None = None,
    references: list[str] | None = None,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    quality: str | None = None,
    background: str | None = None,
    output_format: str = "png",
    seed: int | None = None,
    n: int = 1,
    extra_params: dict[str, Any] | None = None,
    preview: bool = True,
) -> list[Any]:
    """Generate (or, with references, edit or restyle) images and save them.

    - model: an OpenRouter image model id; defaults to IMAGE_GEN_DEFAULT_MODEL.
    - out_path: file path inside the output root (relative paths are relative to
      it); default is a timestamped name in the default folder. With n > 1 the
      files are numbered.
    - references: input images, as paths inside the output root or HTTP(S) URLs.
    - aspect_ratio ("16:9"), resolution ("1K", "2K", "4K"), quality, background
      ("transparent", "opaque", "auto"), seed: passed through when set; which
      ones a model accepts is in list_image_models.
    - extra_params: other provider parameters, passed through unchanged.
    - preview: also return a small preview of each image.

    Returns the saved paths, the actual cost and previews. A .json sidecar next
    to each image records the request and the cost.
    """
    model = model or os.environ.get("IMAGE_GEN_DEFAULT_MODEL", "").strip()
    if not model:
        raise ValueError("No model given and IMAGE_GEN_DEFAULT_MODEL is not set.")
    if not 1 <= n <= 10:
        raise ValueError("n must be between 1 and 10.")
    root = _output_root()
    default_dir = os.environ.get("IMAGE_GEN_DEFAULT_DIR", "generated")
    target = resolve_inside(root, out_path or default_image_path(default_dir, prompt, output_format))

    refs = [_reference(root, r) for r in (references or [])]
    payload = build_payload(prompt, model, refs, aspect_ratio, resolution, quality, background,
                            output_format, seed, n, extra_params)

    with httpx.Client(timeout=GENERATE_TIMEOUT) as client:
        response = client.post(f"{API}/images", headers=_headers(True), json=payload)
    _raise_for_status(response)
    body = response.json()
    images = body.get("data") or []
    if not images:
        raise RuntimeError("OpenRouter returned no images.")

    usage = body.get("usage") or {}
    record = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "prompt": prompt,
        "parameters": {k: v for k, v in payload.items() if k not in ("model", "prompt", "input_references")},
        "references": references or [],
        "cost_usd": usage.get("cost"),
        "usage": usage,
        "response_id": body.get("id"),
    }

    os.makedirs(os.path.dirname(target), exist_ok=True)
    saved: list[str] = []
    previews: list[Image] = []
    for i, item in enumerate(images):
        data = base64.b64decode(item["b64_json"])
        path = numbered(target, i, len(images))
        with open(path, "wb") as f:
            f.write(data)
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump({**record, "file": os.path.basename(path), "index": i, "count": len(images),
                       "media_type": item.get("media_type")}, f, indent=2)
        saved.append(path)
        if preview:
            previews.append(_preview(data))

    summary = {"saved": saved, "cost_usd": usage.get("cost"), "model": model}
    return [json.dumps(summary, indent=2), *previews]


def _matches(model: dict[str, Any], text: str | None) -> bool:
    if not text:
        return True
    text = text.lower()
    return text in (model.get("id") or "").lower() or text in (model.get("name") or "").lower()


@mcp.tool()
def list_image_models(filter: str | None = None, with_pricing: bool = False) -> str:
    """List OpenRouter image models, optionally filtered by id or name.

    with_pricing fetches each matching model's endpoints for prices and
    provider parameters (at most 15 models; narrow the filter first). Prices are
    per token for input text, input images and output images; the actual cost of
    a generation is reported after it runs.
    """
    with httpx.Client(timeout=QUERY_TIMEOUT) as client:
        response = client.get(f"{API}/images/models", headers=_headers(False))
        _raise_for_status(response)
        body = response.json()
        models = [m for m in (body.get("data", body) if isinstance(body, dict) else body) if _matches(m, filter)]
        if with_pricing and len(models) > MAX_PRICED_MODELS:
            raise ValueError(f"{len(models)} models match; narrow the filter to {MAX_PRICED_MODELS} or fewer for pricing.")
        result = []
        for m in models:
            entry = {
                "id": m.get("id"),
                "name": m.get("name"),
                "inputs": (m.get("architecture") or {}).get("input_modalities"),
                "parameters": m.get("supported_parameters"),
            }
            if with_pricing and m.get("endpoints"):
                ep = client.get(f"https://openrouter.ai{m['endpoints']}", headers=_headers(False))
                _raise_for_status(ep)
                entry["endpoints"] = [
                    {"provider": e.get("provider_name"), "pricing": e.get("pricing")}
                    for e in ep.json().get("endpoints", [])
                ]
            result.append(entry)
    return json.dumps({"count": len(result), "models": result}, indent=1)


_KEY_FIELDS = (
    "label", "limit", "limit_remaining", "limit_reset", "usage", "usage_daily",
    "usage_weekly", "usage_monthly", "budgets", "effective_budget", "is_free_tier",
)


@mcp.tool()
def get_key_status() -> str:
    """Remaining credit and usage of this project's OpenRouter key.

    Shows the key's limit, what is left, usage today, this week and this month,
    and the budgets OpenRouter enforces. The key itself is never shown.
    """
    with httpx.Client(timeout=QUERY_TIMEOUT) as client:
        response = client.get(f"{API}/key", headers=_headers(True))
    _raise_for_status(response)
    data = response.json().get("data") or {}
    return json.dumps({k: data.get(k) for k in _KEY_FIELDS if k in data}, indent=2)


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    mcp.run()


if __name__ == "__main__":
    main()
