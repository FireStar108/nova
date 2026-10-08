"""Извлечение кадров под разметку детектора.

Выборка стратифицированная: по каналам, день/ночь и сессиям, равномерно по
таймлайну каждого клипа — чтобы в датасет попали и пустая площадка, и подъезд
машины, и разгрузка, а не 200 почти одинаковых кадров одной сцены.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from novadirt.clips import index_dataset, is_night  # noqa: E402
from novadirt.video import FpsCache, iter_frames  # noqa: E402

# путь к датасету задаётся аргументом или NOVA_DATASET — на другой машине
# он другой, а имена кадров от него не зависят: выборка детерминированная
# (фиксированный шаг по кадрам, сортировка), и разметка сходится с кадрами
DEFAULT_ROOT = Path("/Users/firestar_main/Desktop/nova/папки машин УПП")
OUT = Path(__file__).resolve().parents[1] / "out"
DST = OUT / "trainset" / "images"

# сколько кадров берём с одного клипа канала
PER_CLIP = {"246": 6, "235": 10, "238": 8, "239": 8, "161": 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(os.environ.get("NOVA_DATASET", DEFAULT_ROOT)),
                    help="каталог с папками машин")
    ROOT = ap.parse_args().root
    if not ROOT.is_dir():
        sys.exit(f"нет датасета: {ROOT}\nукажите путь: --root <каталог> "
                 f"или NOVA_DATASET=<каталог>")
    DST.mkdir(parents=True, exist_ok=True)
    cache = FpsCache(OUT)
    sessions = index_dataset(ROOT)
    manifest = []
    stats: dict[tuple[str, str], int] = defaultdict(int)

    for s in sessions:
        tod = "night" if is_night(s) else "day"
        for clip in s.clips:
            ch = clip.channel or "unk"
            want = PER_CLIP.get(ch)
            if not want:
                continue
            info = cache.probe(clip)
            n_frames, fps = info["frames"], info["fps"] or 6.0
            if n_frames < want:
                continue
            stride = max(1, n_frames // want)

            # клип различаем по времени начала: stem[:4] у всех клипов сессии
            # одинаков ("246 "), и кадры разных клипов затирали друг друга
            tag = clip.start.strftime("%H%M%S") if clip.start else clip.path.stem[-6:]

            for fr in iter_frames(clip, fps, stride=stride, max_frames=want):
                name = f"{ch}_{tod}_{_slug(s.vehicle)}_{tag}_{fr.idx:05d}.jpg"
                # cv2.imwrite не умеет в Unicode-пути на Windows (fopen через
                # локальную кодовую страницу) — кодируем в память и пишем сами
                ok, buf = cv2.imencode(".jpg", fr.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (DST / name).write_bytes(buf.tobytes())
                manifest.append({
                    "file": name, "channel": ch, "tod": tod, "day": s.day,
                    "vehicle": s.vehicle, "clip": str(clip.path.relative_to(ROOT)),
                    "frame_idx": fr.idx, "ts": fr.ts.isoformat() if fr.ts else None,
                })
                stats[(ch, tod)] += 1

    cache.save()
    (OUT / "trainset" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"кадров извлечено: {len(manifest)} -> {DST}")
    print(f"\n{'канал':>6} {'день':>7} {'ночь':>7}")
    for ch in sorted({k[0] for k in stats}):
        print(f"{ch:>6} {stats[(ch,'day')]:>7} {stats[(ch,'night')]:>7}")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)[:20]


if __name__ == "__main__":
    main()
