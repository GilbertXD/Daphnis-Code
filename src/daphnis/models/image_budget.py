"""Estimate image token reserves for model requests."""

from math import ceil, floor, sqrt


def mimo_image_tokens(width: int, height: int) -> int:
    """Estimate MiMo's image tokens on its 32 px grid before safety margin."""
    patch = 32
    min_pixels, max_pixels = 8_192, 8_388_608
    pixels = width * height
    cols, rows = max(1, round(width / patch)), max(1, round(height / patch))
    aligned_pixels = cols * rows * patch**2
    if aligned_pixels > max_pixels:
        scale = sqrt(max_pixels / pixels)
        cols = max(1, floor(width * scale / patch))
        rows = max(1, floor(height * scale / patch))
    elif aligned_pixels < min_pixels:
        scale = sqrt(min_pixels / pixels)
        cols, rows = ceil(width * scale / patch), ceil(height * scale / patch)
    # A very narrow image can round to one tile on the short side and still exceed the area cap.
    return min(cols * rows, max_pixels // patch**2)


def mimo_image_token_reserve(width: int, height: int) -> int:
    base = mimo_image_tokens(width, height)
    return base + (base + 9) // 10


def deepseek_image_token_reserve(width: int, height: int) -> int:
    return 1024


def conservative_image_token_reserve(width: int, height: int) -> int:
    tiles = ((width + 31) // 32) * ((height + 31) // 32)
    return 2 * max(1024, tiles)
