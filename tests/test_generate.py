import asyncio
import base64
import io
import json

import httpx
from PIL import Image as PILImage

from openrouter_image_mcp import server


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGBA", (64, 64), (255, 0, 0, 128)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeClient:
    """Stands in for httpx.Client: answers POST /images with one PNG."""

    def __init__(self, *args, **kwargs):
        self.sent = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None):
        _FakeClient.last_payload = json
        body = {"id": "gen-1", "data": [{"b64_json": base64.b64encode(_png_bytes()).decode(), "media_type": "image/png"}],
                "usage": {"cost": 0.01}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))


def test_generate_image_through_fastmcp_returns_text_and_preview(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("IMAGE_GEN_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(server.httpx, "Client", _FakeClient)

    result = asyncio.run(server.mcp.call_tool("generate_image", {"prompt": "a dot", "model": "m", "out_path": "out/a.png"}))
    content = result[0] if isinstance(result, tuple) else result

    kinds = [c.type for c in content]
    assert kinds == ["text", "image"]
    summary = json.loads(content[0].text)
    assert summary["cost_usd"] == 0.01
    assert (tmp_path / "out" / "a.png").exists()
    sidecar = json.loads((tmp_path / "out" / "a.json").read_text(encoding="utf-8"))
    assert sidecar["model"] == "m" and sidecar["cost_usd"] == 0.01
    assert _FakeClient.last_payload["prompt"] == "a dot"
