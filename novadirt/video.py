"""Чтение клипов DVR.

Метаданные в этих AVI недостоверны: cv2 сообщает 600 fps / 71340 кадров там,
где реально 834 кадра за 120 с (~7 fps). Поэтому реальный fps выводим как
(число видеопакетов) / (длительность из имени файла), а не из заголовка.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .clips import Clip

_CACHE_NAME = "fps_cache.json"


def packet_count(path: Path) -> int:
    """Число видеопакетов = число кадров. Быстро: контейнер парсится без декодирования."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out.split(",")[0])


class FpsCache:
    def __init__(self, cache_dir: Path):
        self.file = cache_dir / _CACHE_NAME
        self.data: dict[str, dict] = {}
        if self.file.exists():
            self.data = json.loads(self.file.read_text())

    def probe(self, clip: Clip) -> dict:
        key = str(clip.path)
        if key not in self.data:
            n = packet_count(clip.path)
            dur = clip.named_duration
            self.data[key] = {
                "frames": n,
                "named_duration_s": dur,
                "fps": (n / dur) if dur else None,
            }
        return self.data[key]

    def save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.data, indent=1))


@dataclass
class Frame:
    idx: int
    offset_s: float
    ts: datetime | None
    image: np.ndarray


def iter_frames(clip: Clip, fps: float, stride: int = 1, max_frames: int | None = None) -> Iterator[Frame]:
    """Итерация кадров. stride>1 пропускает кадры через grab() без декодирования."""
    cap = cv2.VideoCapture(str(clip.path))
    if not cap.isOpened():
        raise RuntimeError(f"не открывается: {clip.path}")
    try:
        idx = 0
        yielded = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, img = cap.retrieve()
                if not ok:
                    idx += 1
                    continue
                offset = idx / fps if fps else 0.0
                yield Frame(idx, offset, clip.ts_at(offset), img)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    return
            idx += 1
    finally:
        cap.release()
