import asyncio
import base64
import io
import json

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError
from PIL import Image as PILImage

from openrouter_image_mcp import server


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGBA", (64, 64), (255, 0, 0, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _item(data: bytes) -> dict:
    return {"b64_json": base64.b64encode(data).decode(), "media_type": "image/png"}


class _Fake:
    """Stands in for OpenRouter: POST /images answers with ``items``, GET /key with a key record.

    With ``gate`` set, POST /images waits until a GET /key has been answered.
    """

    def __init__(self):
        self.items = [_item(_png_bytes())]
        self.posts = []
        self.gate: asyncio.Event | None = None

    def client(self, *args, **kwargs):
        return _FakeClient(self)


class _FakeClient:
    """Stands in for httpx.AsyncClient."""

    def __init__(self, fake):
        self.fake = fake

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        self.fake.posts.append(json)
        if self.fake.gate is not None:
            await self.fake.gate.wait()
        body = {"id": "gen-1", "data": self.fake.items, "usage": {"cost": 0.01}}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    async def get(self, url, headers=None):
        if self.fake.gate is not None:
            self.fake.gate.set()
        body = {"data": {"label": "k", "limit_remaining": 5.0}}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("IMAGE_GEN_OUTPUT_ROOT", str(tmp_path))
    f = _Fake()
    monkeypatch.setattr(server.httpx, "AsyncClient", f.client)
    return f


def _generate(**args):
    """Call generate_image through FastMCP; return the summary and the content types."""
    result = asyncio.run(server.mcp.call_tool("generate_image", {"prompt": "a dot", "model": "m", **args}))
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text), [c.type for c in content]


def test_generate_image_through_fastmcp_returns_text_and_preview(tmp_path, fake):
    summary, kinds = _generate(out_path="out/a.png")

    assert kinds == ["text", "image"]
    assert summary["cost_usd"] == 0.01 and "warnings" not in summary
    assert (tmp_path / "out" / "a.png").exists()
    sidecar = json.loads((tmp_path / "out" / "a.json").read_text(encoding="utf-8"))
    assert sidecar["model"] == "m" and sidecar["cost_usd"] == 0.01
    assert fake.posts[0]["prompt"] == "a dot"


def test_failed_preview_does_not_lose_later_images(tmp_path, fake):
    fake.items = [_item(b"not an image"), _item(_png_bytes())]

    summary, kinds = _generate(out_path="out/a.png", n=2)

    assert kinds == ["text", "image"]
    assert (tmp_path / "out" / "a-1.png").read_bytes() == b"not an image"
    assert (tmp_path / "out" / "a-2.png").exists() and (tmp_path / "out" / "a-2.json").exists()
    assert len(summary["saved"]) == 2
    assert any("no preview" in w for w in summary["warnings"])


def test_item_without_data_does_not_stop_the_batch(tmp_path, fake):
    fake.items = [{"media_type": "image/png"}, _item(_png_bytes())]

    summary, _ = _generate(out_path="out/a.png", n=2)

    assert not (tmp_path / "out" / "a-1.png").exists()
    assert (tmp_path / "out" / "a-2.png").exists()
    assert summary["warnings"][0].startswith("Image 1:")


def test_nothing_usable_is_an_error_that_names_the_cost(fake):
    fake.items = [{"media_type": "image/png"}]

    with pytest.raises(ToolError, match="0.01 USD"):
        _generate(out_path="out/a.png")


def test_a_running_generation_does_not_block_other_calls(tmp_path, fake):
    async def both():
        fake.gate = asyncio.Event()
        generation = server.mcp.call_tool("generate_image", {"prompt": "a dot", "model": "m", "out_path": "out/a.png"})
        status = server.mcp.call_tool("get_key_status", {})
        return await asyncio.wait_for(asyncio.gather(generation, status), timeout=5)

    _, status = asyncio.run(both())

    status_content = status[0] if isinstance(status, tuple) else status
    assert json.loads(status_content[0].text)["limit_remaining"] == 5.0
    assert (tmp_path / "out" / "a.png").exists()
