import httpx
import pytest

from openrouter_image_mcp import server


def _payload(**overrides):
    args = dict(prompt="p", model="m", references=[], aspect_ratio=None, resolution=None, quality=None,
                background=None, output_format="png", seed=None, n=1, extra=None)
    args.update(overrides)
    return server.build_payload(**args)


def test_unset_options_are_left_out():
    assert _payload() == {"model": "m", "prompt": "p", "output_format": "png", "n": 1}


def test_set_options_and_references_are_sent():
    refs = [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]
    p = _payload(aspect_ratio="16:9", background="transparent", seed=7, references=refs)
    assert p["aspect_ratio"] == "16:9" and p["background"] == "transparent" and p["seed"] == 7
    assert p["input_references"] == refs


def test_extra_params_pass_through_but_cannot_override():
    assert _payload(extra={"moderation": "low"})["moderation"] == "low"
    with pytest.raises(ValueError):
        _payload(extra={"model": "other"})


def test_reference_file_becomes_data_url(tmp_path):
    (tmp_path / "r.png").write_bytes(b"\x89PNG")
    ref = server._reference(str(tmp_path), "r.png")
    assert ref["image_url"]["url"].startswith("data:image/png;base64,")


def test_reference_outside_root_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.png").write_bytes(b"x")
    with pytest.raises(ValueError):
        server._reference(str(root), str(tmp_path / "secret.png"))


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(server.ConfigError):
        server._headers(True)


@pytest.mark.parametrize("body, expected", [
    ({"json": {"error": {"message": "Insufficient credits"}}}, "Insufficient credits"),
    ({"json": {"error": "Rate limited"}}, "Rate limited"),
    ({"json": ["unexpected"]}, "unexpected"),
    ({"text": "Bad gateway"}, "Bad gateway"),
])
def test_error_responses_keep_the_status_and_message(body, expected):
    response = httpx.Response(402, request=httpx.Request("POST", "https://openrouter.ai/api/v1/images"), **body)
    with pytest.raises(RuntimeError, match=f"returned 402: .*{expected}"):
        server._raise_for_status(response)


def test_key_status_never_includes_the_key():
    assert "key" not in server._KEY_FIELDS
