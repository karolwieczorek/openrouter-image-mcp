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


_FORMAT_EXTENSIONS = {
    "png": {".png"},
    "jpeg": {".jpg", ".jpeg"},
    "jpg": {".jpg", ".jpeg"},
    "webp": {".webp"},
}


def with_extension(path: str, output_format: str) -> str:
    """*path* with the extension of *output_format*, added if it has none.

    A different extension is refused, so a PNG is never saved as ``.jpg`` and
    an image never shares its ``.json`` sidecar's name.
    """
    fmt = output_format.lower()
    ext = os.path.splitext(path)[1]
    if not ext:
        path, ext = f"{path}.{fmt}", f".{fmt}"
    if ext.lower() == ".json" or ext.lower() not in _FORMAT_EXTENSIONS.get(fmt, {f".{fmt}"}):
        raise ValueError(f"'{path}' does not match output_format '{output_format}'.")
    return path


def sidecar_path(image_path: str) -> str:
    """The ``.json`` record kept next to an image."""
    return os.path.splitext(image_path)[0] + ".json"


def is_taken(image_path: str) -> bool:
    """True if the image or its sidecar already exists."""
    return os.path.lexists(image_path) or os.path.lexists(sidecar_path(image_path))


def batch_paths(target: str, count: int) -> list[str]:
    return [numbered(target, i, count) for i in range(count)]


def free_target(target: str, count: int) -> str:
    """*target*, or ``<name>-v2``, ``-v3`` ... so that no file of the batch exists yet."""
    base, ext = os.path.splitext(target)
    candidate, version = target, 1
    while any(is_taken(p) for p in batch_paths(candidate, count)):
        version += 1
        candidate = f"{base}-v{version}{ext}"
    return candidate


def media_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _IMAGE_TYPES:
        raise ValueError(f"Unsupported reference image type '{ext}' (use {', '.join(sorted(_IMAGE_TYPES))})")
    return _IMAGE_TYPES[ext]
