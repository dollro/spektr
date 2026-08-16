"""Tests for the pre-embedding image downscale.

`IMAGE_EMBED_MAX_PX` was documented in `.env.example` with no `Settings`
field to receive it, so `extra="ignore"` swallowed the key and nothing ever
resized. Pages render at 150 DPI, putting an A4 page on the wire at roughly
1240x1754 — billed area on every provider, and ~28k estimated tokens per page
on Jina's tile meter against a default TPM of 100k.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from ingestion._utils import cap_image_size


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 130, 140)).save(buffer, format="PNG")
    return buffer.getvalue()


def _size(image_bytes: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(image_bytes)).size


def test_downscales_longest_side_to_the_cap() -> None:
    """A 150-DPI A4 page is the case this exists for."""
    out = cap_image_size(_png(1240, 1754), max_px=400)
    assert max(_size(out)) == 400


def test_preserves_aspect_ratio() -> None:
    out = cap_image_size(_png(1000, 500), max_px=400)
    width, height = _size(out)
    assert (width, height) == (400, 200)


def test_landscape_and_portrait_both_capped_on_longest_side() -> None:
    assert max(_size(cap_image_size(_png(900, 300), max_px=200))) == 200
    assert max(_size(cap_image_size(_png(300, 900), max_px=200))) == 200


def test_smaller_image_is_returned_unchanged() -> None:
    """No re-encode when it already fits — byte-identical, not merely equal."""
    original = _png(100, 80)
    assert cap_image_size(original, max_px=400) is original


def test_zero_disables_the_cap() -> None:
    original = _png(1240, 1754)
    assert cap_image_size(original, max_px=0) is original


def test_undecodable_input_fails_open() -> None:
    """A cost optimisation must not become an outage.

    Refusing to embed a page because PIL could not parse it would turn a
    saving into a dropped page, and pages are declared — a page not declared
    is a page CocoIndex deletes.
    """
    garbage = b"not an image at all"
    assert cap_image_size(garbage, max_px=400) is garbage


@pytest.mark.parametrize("max_px", [64, 256, 400, 1024])
def test_output_never_exceeds_the_cap(max_px: int) -> None:
    out = cap_image_size(_png(2000, 1500), max_px=max_px)
    assert max(_size(out)) <= max_px


def test_output_is_meaningfully_smaller() -> None:
    """The point is bytes on the wire, not just pixel count."""
    original = _png(1240, 1754)
    out = cap_image_size(original, max_px=400)
    assert len(out) < len(original)
