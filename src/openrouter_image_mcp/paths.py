"""Output and reference paths, confined to one root folder."""

import datetime
import os
import re

_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def resolve_inside(root: str, path: str) -> str:
    """Return *path* as an absolute path, refusing anything outside *root*.

    Relative paths are taken relative to *root*. Symlinks and ``..`` are
    resolved before the check.
    """
    root_real = os.path.realpath(root)
    candidate = path if os.path.isabs(path) else os.path.join(root_real, path)
    real = os.path.realpath(candidate)
    if os.path.commonpath([os.path.normcase(root_real), os.path.normcase(real)]) != os.path.normcase(root_real):
        raise ValueError(f"Path is outside the output root ({root_real}): {path}")
    return real


def slugify(text: str, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "image"


def default_image_path(default_dir: str, prompt: str, extension: str, now: datetime.datetime | None = None) -> str:
    """A timestamped file name in *default_dir* derived from the prompt."""
    now = now or datetime.datetime.now()
    return os.path.join(default_dir, f"{now:%Y%m%d-%H%M%S}-{slugify(prompt)}.{extension}")


def numbered(path: str, index: int, count: int) -> str:
    """``a.png`` for a single image, ``a-1.png``, ``a-2.png`` ... for several."""
    if count == 1:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}-{index + 1}{ext}"


def media_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _IMAGE_TYPES:
        raise ValueError(f"Unsupported reference image type '{ext}' (use {', '.join(sorted(_IMAGE_TYPES))})")
    return _IMAGE_TYPES[ext]
