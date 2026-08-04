"""Предразметка грузовиков COCO-моделью в формат YOLO.

Ковш/кузов/экскаватор предразметить нечем — в COCO таких классов нет,
их придётся размечать руками. Здесь мы только снимаем с разметчика
рутину по грузовикам; порог намеренно низкий (recall важнее precision,
лишний бокс удалить дешевле, чем нарисовать пропущенный).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
IMG = OUT / "trainset" / "images"
LBL = OUT / "trainset" / "labels"

CLASSES = ["truck", "truck_body", "grapple", "excavator"]
COCO_TO_TRUCK = {"truck", "bus", "car", "train", "boat"}  # DVR-ракурсы путают эти классы
CONF = 0.15


def main() -> None:
    LBL.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11x.pt")
    images = sorted(IMG.glob("*.jpg"))
    stats = Counter()

    for i in range(0, len(images), 16):
        batch = images[i:i + 16]
        for path, res in zip(batch, model([str(p) for p in batch], verbose=False, conf=CONF)):
            lines = []
            for box in res.boxes:
                if res.names[int(box.cls)] not in COCO_TO_TRUCK:
                    continue
                x, y, w, h = box.xywhn[0].tolist()
                lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            (LBL / f"{path.stem}.txt").write_text("\n".join(lines))
            stats["boxes"] += len(lines)
            stats["empty" if not lines else "labelled"] += 1
        print(f"\r{min(i+16, len(images))}/{len(images)}", end="", flush=True)

    yaml = OUT / "trainset" / "dataset.yaml"
    yaml.write_text(
        "path: .\ntrain: images\nval: images\n\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    print(f"\n\nкадров с боксами: {stats['labelled']}  пустых: {stats['empty']}")
    print(f"боксов всего: {stats['boxes']}")
    print(f"классы: {CLASSES}  ->  {yaml}")


if __name__ == "__main__":
    main()
