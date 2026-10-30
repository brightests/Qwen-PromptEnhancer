from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image


def sample_indices_per_second(
    frame_count: int,
    source_fps: float,
    samples_per_second: int,
) -> list[list[int]]:
    if frame_count <= 0 or source_fps <= 0 or samples_per_second <= 0:
        return []
    groups = []
    second_count = math.ceil(frame_count / source_fps)
    for second_index in range(second_count):
        start = round(second_index * source_fps)
        end = min(frame_count, round((second_index + 1) * source_fps))
        window_size = end - start
        sample_count = min(samples_per_second, window_size)
        indices = [
            start + math.floor(sample_index * window_size / sample_count)
            for sample_index in range(sample_count)
        ]
        groups.append(indices)
    return groups


def tensor_frame_data_url(image, frame_index: int = 0, max_edge: int = 1280) -> str:
    frame = image[frame_index, ..., :3].detach().cpu().numpy()
    pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(pixels, mode="RGB")
    pil_image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def tensor_frame_grid_data_url(
    image,
    frame_indices: list[int],
    max_edge: int = 1280,
) -> str:
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    columns = math.ceil(math.sqrt(len(frame_indices)))
    rows = math.ceil(len(frame_indices) / columns)
    cell_size = max(1, max_edge // max(columns, rows))
    canvas = Image.new("RGB", (columns * cell_size, rows * cell_size), (16, 16, 16))
    pasted_rects: list[tuple[int, int, int, int]] = []
    for position, frame_index in enumerate(frame_indices):
        frame = image[frame_index, ..., :3].detach().cpu().numpy()
        pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(pixels, mode="RGB")
        pil_image.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
        column = position % columns
        row = position // columns
        x = column * cell_size + (cell_size - pil_image.width) // 2
        y = row * cell_size + (cell_size - pil_image.height) // 2
        canvas.paste(pil_image, (x, y))
        pasted_rects.append((x, y, x + pil_image.width, y + pil_image.height))
    # Crop away unused canvas so sparse sheets (e.g. a single frame in a
    # 1x1 grid) do not waste vision tokens on black padding. Full grids
    # keep their original dimensions because they cover the whole canvas.
    left = min(rect[0] for rect in pasted_rects)
    top = min(rect[1] for rect in pasted_rects)
    right = max(rect[2] for rect in pasted_rects)
    bottom = max(rect[3] for rect in pasted_rects)
    if (left, top, right, bottom) != (0, 0, canvas.width, canvas.height):
        canvas = canvas.crop((left, top, right, bottom))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_content(image, frame_index: int = 0) -> dict[str, object]:
    return {
        "type": "image_url",
        "image_url": {
            "url": tensor_frame_data_url(image, frame_index),
            "detail": "auto",
        },
    }


def image_grid_content(image, frame_indices: list[int]) -> dict[str, object]:
    return {
        "type": "image_url",
        "image_url": {
            "url": tensor_frame_grid_data_url(image, frame_indices),
            "detail": "auto",
        },
    }


def text_content(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}
