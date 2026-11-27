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

import asyncio
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

from .paths import (
    batch_paths,
    default_image_path,
    free_target,
    is_taken,
    media_type_for,
    numbered,
    resolve_inside,
    sidecar_path,
    with_extension,
)

API = "https://openrouter.ai/api/v1"
GENERATE_TIMEOUT = 300.0
QUERY_TIMEOUT = 30.0
PREVIEW_SIZE = 768
MAX_PRICED_MODELS = 15

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
        body = response.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    detail = (error.get("message") if isinstance(error, dict) else error) or response.text
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
        media_type = media_type_for(path)  # before reading, so a non-image file is never loaded
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        url = f"data:{media_type};base64,{data}"
    return {"type": "image_url", "image_url": {"url": url}}


def _preview(data: bytes) -> Image:
    with PILImage.open(io.BytesIO(data)) as img:
        img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


_PILLOW_FORMATS = {"png": "PNG", "jpeg": "JPEG", "jpg": "JPEG", "webp": "WEBP"}
_SAVE_OPTIONS = {"JPEG": {"quality": 95}, "WEBP": {"lossless": True}}


def _as_format(data: bytes, output_format: str) -> tuple[bytes, str | None, str | None]:
    """*data* as *output_format*, converted with Pillow if the model returned another format.

    Models do not always honour output_format (some return WebP when PNG was
    asked for). Returns the bytes, their media type, and the media type they
    arrived in when they were converted (else None). Bytes already in the right
    format are returned untouched; a format not listed here is left alone.
    """
    target = _PILLOW_FORMATS.get(output_format.lower())
    if target is None:
        return data, None, None
    with PILImage.open(io.BytesIO(data)) as img:
        returned = img.format
        if returned == target:
            return data, PILImage.MIME[target], None
        icc = img.info.get("icc_profile")
        if target == "JPEG" and img.mode not in ("RGB", "L"):
            rgba = img.convert("RGBA")  # JPEG has no transparency: flatten onto white
            img = PILImage.new("RGB", rgba.size, (255, 255, 255))
            img.paste(rgba, mask=rgba.getchannel("A"))
        elif target != "JPEG" and img.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            img = img.convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format=target, icc_profile=icc, **_SAVE_OPTIONS.get(target, {}))
    return buf.getvalue(), PILImage.MIME[target], PILImage.MIME.get(returned or "", returned)


def _write_new(path: str, data: bytes, overwrite: bool) -> str:
    """Write *data* to *path*, or, unless *overwrite*, to the next free ``-vN`` name."""
    if overwrite:
        with open(path, "wb") as f:
            f.write(data)
        return path
    while True:
        path = free_target(path, 1)
        try:
            with open(path, "xb") as f:  # exclusive: a concurrent call cannot slip in between
                f.write(data)
            return path
        except FileExistsError:
            continue


def _save_images(
    images: list[Any], target: str, count: int, output_format: str, record: dict[str, Any], overwrite: bool,
    preview: bool,
) -> tuple[list[str], list[str], list[Image]]:
    """Save every usable image (in *output_format*) and its sidecar, then make previews.

    The images are already paid for, so one bad item never stops the others;
    what went wrong is returned as warnings.
    """
    saved: list[tuple[str, bytes]] = []
    warnings: list[str] = []
    for i, item in enumerate(images):
        try:
            data = base64.b64decode(item["b64_json"])
        except (KeyError, TypeError, ValueError) as e:
            warnings.append(f"Image {i + 1}: no usable image data in the response ({e!r}).")
            continue
        try:
            data, media_type, converted_from = _as_format(data, output_format)
        except Exception as e:  # Pillow raises many types; keep the image as it came
            media_type, converted_from = item.get("media_type"), None
            warnings.append(f"Image {i + 1}: could not be read to check its format ({e}); saved as returned.")
        wanted = numbered(target, i, count)
        try:
            path = _write_new(wanted, data, overwrite)
        except OSError as e:
            warnings.append(f"Image {i + 1}: could not be saved to {wanted}: {e}")
            continue
        saved.append((path, data))
        if path != wanted:
            warnings.append(f"{wanted} was taken; saved as {path}.")
        try:
            with open(sidecar_path(path), "w", encoding="utf-8") as f:
                sidecar = {**record, "file": os.path.basename(path), "index": i, "count": len(images),
                           "media_type": media_type or item.get("media_type")}
                if converted_from:
                    sidecar["converted_from"] = converted_from
                json.dump(sidecar, f, indent=2)
        except OSError as e:
            warnings.append(f"{path}: sidecar not written: {e}")

    previews: list[Image] = []
    if preview:
        for path, data in saved:
            try:
                previews.append(_preview(data))
            except Exception as e:  # Pillow raises many types; the image is saved either way
                warnings.append(f"{path}: no preview ({e}).")
    return [path for path, _ in saved], warnings, previews


# Unstructured: the result mixes text and image previews, which structured output cannot serialize.
@mcp.tool(structured_output=False)
async def generate_image(
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
    overwrite: bool = False,
    preview: bool = True,
) -> list[Any]:
    """Generate (or, with references, edit or restyle) images and save them.

    - model: an OpenRouter image model id; defaults to IMAGE_GEN_DEFAULT_MODEL.
    - out_path: file path inside the output root (relative paths are relative to
      it); default is a timestamped name in the default folder. Its extension
      must match output_format and is added if missing. With n > 1 the files are
      numbered.
    - references: input images, as paths inside the output root or HTTP(S) URLs.
    - aspect_ratio ("16:9"), resolution ("1K", "2K", "4K"), quality, background
      ("transparent", "opaque", "auto"), seed: passed through when set; which
      ones a model accepts is in list_image_models.
    - output_format: "png" (default), "jpeg" or "webp". An image the model
      returns in another format is converted before it is saved.
    - extra_params: other provider parameters, passed through unchanged.
    - overwrite: replace existing files at out_path. Without it an existing
      file is an error, raised before anything is paid for.
    - preview: also return a small preview of each image.

    Returns the saved paths, the actual cost, any warnings and previews. A .json
    sidecar next to each image records the request and the cost.
    """
    model = model or os.environ.get("IMAGE_GEN_DEFAULT_MODEL", "").strip()
    if not model:
        raise ValueError("No model given and IMAGE_GEN_DEFAULT_MODEL is not set.")
    if not 1 <= n <= 10:
        raise ValueError("n must be between 1 and 10.")
    root = _output_root()
    default_dir = os.environ.get("IMAGE_GEN_DEFAULT_DIR", "generated")
    relative = out_path or default_image_path(default_dir, prompt, output_format)
    target = resolve_inside(root, with_extension(relative, output_format))
    if not out_path:
        target = free_target(target, n)
    elif not overwrite:
        taken = [p for p in batch_paths(target, n) if is_taken(p)]
        if taken:
            raise FileExistsError(f"Already exists (pass overwrite=true or another out_path): {', '.join(taken)}")
    for path in batch_paths(target, n):
        if os.path.isdir(path):
            raise IsADirectoryError(f"out_path names a folder: {path}")
    os.makedirs(os.path.dirname(target), exist_ok=True)

    refs = [_reference(root, r) for r in (references or [])]
    payload = build_payload(prompt, model, refs, aspect_ratio, resolution, quality, background,
                            output_format, seed, n, extra_params)

    async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
        response = await client.post(f"{API}/images", headers=_headers(True), json=payload)
    _raise_for_status(response)
    body = response.json()
    images = body.get("data") or []
    usage = body.get("usage") or {}
    if not images:
        raise RuntimeError(f"OpenRouter returned no images (cost {usage.get('cost')} USD, id {body.get('id')}).")

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

    # Numbered by the larger of n and what came back: the names checked above, plus
    # numbered ones if a model returns extra images. Disk and Pillow work runs off the
    # event loop.
    saved, warnings, previews = await asyncio.to_thread(
        _save_images, images, target, max(n, len(images)), output_format, record, overwrite, preview)
    if not saved:
        raise RuntimeError(f"No image could be saved (cost {usage.get('cost')} USD, id {body.get('id')}): "
                           + " ".join(warnings))

    summary: dict[str, Any] = {"saved": saved, "cost_usd": usage.get("cost"), "model": model}
    if warnings:
        summary["warnings"] = warnings
    return [json.dumps(summary, indent=2), *previews]


def _matches(model: dict[str, Any], text: str | None) -> bool:
    if not text:
        return True
    text = text.lower()
    return text in (model.get("id") or "").lower() or text in (model.get("name") or "").lower()


@mcp.tool()
async def list_image_models(filter: str | None = None, with_pricing: bool = False) -> str:
    """List OpenRouter image models, optionally filtered by id or name.

    with_pricing fetches each matching model's endpoints for prices and
    provider parameters (at most 15 models; narrow the filter first). Prices are
    per token for input text, input images and output images; the actual cost of
    a generation is reported after it runs.
    """
    async with httpx.AsyncClient(timeout=QUERY_TIMEOUT) as client:
        response = await client.get(f"{API}/images/models", headers=_headers(False))
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
                ep = await client.get(f"https://openrouter.ai{m['endpoints']}", headers=_headers(False))
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
async def get_key_status() -> str:
    """Remaining credit and usage of this project's OpenRouter key.

    Shows the key's limit, what is left, usage today, this week and this month,
    and the budgets OpenRouter enforces. The key itself is never shown.
    """
    async with httpx.AsyncClient(timeout=QUERY_TIMEOUT) as client:
        response = await client.get(f"{API}/key", headers=_headers(True))
    _raise_for_status(response)
    data = response.json().get("data") or {}
    return json.dumps({k: data.get(k) for k in _KEY_FIELDS if k in data}, indent=2)


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    mcp.run()


if __name__ == "__main__":
    main()
