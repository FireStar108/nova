"""Обучение детектора на боксовой разметке (каналы 246, 161).

Сплит train/val — **по сессиям, а не по кадрам**. Кадры одной сессии сняты
одной камерой с одной машины с интервалом в секунды: случайный покадровый сплит
положил бы почти одинаковые кадры в train и val, и mAP@val показал бы 0.95 там,
где на новой машине модель не работает вовсе.

Каналы 235/238/239 в детекции не участвуют: кузов там занимает весь кадр,
машина целиком не видна, и боксовые классы на них не определены (см. README).

Запуск:
    .venv/bin/python scripts/05_train_det.py            # yolo11s, 100 эпох
    .venv/bin/python scripts/05_train_det.py --model yolo11m.pt --epochs 200
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out"
TRAINSET = OUT / "trainset"
IMAGES = TRAINSET / "images"
LABELS = TRAINSET / "labels"
SPLIT = OUT / "det_split"

# классы разметки (labels/) и классы обучения — разные множества
LABEL_CLASSES = ["truck", "truck_body", "grapple", "excavator"]
# truck_body размечен на 12 объектах, excavator на 4 — учить по ним нечего,
# а как отдельные классы они только шумят в метриках. Выбрасываем; остальные
# перенумеровываем (0 truck, 2 grapple) -> (0 truck, 1 grapple)
CLASSES = ["truck", "grapple"]
REMAP = {0: 0, 2: 1}

# боксы размечены на всех каналах, а не только на общих планах: детектор ковша
# на 235/238/239 нужен для ROI на инференсе
DET_CHANNELS: set[str] = set()  # пусто = все каналы
VAL_FRACTION = 0.2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="mps / cpu / 0; по умолчанию авто")
    ap.add_argument("--split-only", action="store_true", help="только собрать сплит")
    args = ap.parse_args()

    yaml_path = build_split()
    if args.split_only:
        return

    from ultralytics import YOLO

    YOLO(args.model).train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or _device(),
        project=str(OUT / "runs"),
        name="det",
        exist_ok=True,
        patience=25,  # датасет маленький, плато наступает задолго до 100-й эпохи
        # кадры с DVR: цвет и яркость плавают между камерами и днём/ночью,
        # геометрию камеры трогать нельзя — ракурс фиксированный
        hsv_v=0.5,
        degrees=0.0,
        perspective=0.0,
        fliplr=0.0,  # площадка асимметрична, зеркало ломает геометрию сцены
        mosaic=0.5,
    )
    print(f"\nвеса: {OUT / 'runs' / 'det' / 'weights' / 'best.pt'}")


def build_split() -> Path:
    """Собирает out/det_split с разбиением по сессиям. Возвращает путь к yaml."""
    meta = _manifest()
    frames = []
    for img in sorted(IMAGES.glob("*.jpg")):
        m = meta.get(img.name)
        ch = m["channel"] if m else img.name.split("_")[0]
        if DET_CHANNELS and ch not in DET_CHANNELS:
            continue
        lbl = LABELS / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        session = (m["day"], m["vehicle"]) if m else (img.name, "")
        tod = m["tod"] if m else "day"
        frames.append((img, lbl, session, tod))

    if not frames:
        sys.exit(f"нет кадров с разметкой в {LABELS}")

    _warn_unreviewed()

    val_sessions = _pick_val_sessions(frames)
    if SPLIT.exists():
        shutil.rmtree(SPLIT)
    counts: Counter[str] = Counter()
    per_class: dict[str, Counter[int]] = defaultdict(Counter)

    dropped: Counter[int] = Counter()
    for img, lbl, session, _ in frames:
        part = "val" if session in val_sessions else "train"
        dst_img = SPLIT / "images" / part / img.name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_img.symlink_to(img)

        # метки не симлинкуем, а переписываем: классы перенумерованы под CLASSES
        lines = []
        for line in lbl.read_text().split("\n"):
            if not line.strip():
                continue
            cls, rest = int(line.split()[0]), line.split()[1:]
            if cls not in REMAP:
                dropped[cls] += 1
                continue
            lines.append(" ".join([str(REMAP[cls])] + rest))
            per_class[part][REMAP[cls]] += 1
        dst_lbl = SPLIT / "labels" / part / f"{img.stem}.txt"
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text("\n".join(lines))
        counts[part] += 1

    yaml_path = SPLIT / "dataset.yaml"
    yaml_path.write_text(
        # path обязан быть абсолютным: ultralytics резолвит его от своего
        # datasets_dir, а не от каталога с yaml
        f"path: {SPLIT}\ntrain: images/train\nval: images/val\n\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )

    n_sessions = len({s for _, _, s, _ in frames})
    print(f"сессий: {n_sessions}  из них в val: {len(val_sessions)}")
    print(f"кадров: train {counts['train']}  val {counts['val']}")
    print(f"\n{'класс':>12} {'train':>7} {'val':>7}")
    for i, name in enumerate(CLASSES):
        print(f"{name:>12} {per_class['train'][i]:>7} {per_class['val'][i]:>7}")
        if not per_class["val"][i] and per_class["train"][i]:
            print(f"{'':>12} ! класс отсутствует в val — mAP по нему не измерить")
    if dropped:
        out = ", ".join(f"{LABEL_CLASSES[c]} {n}" for c, n in sorted(dropped.items()))
        print(f"\nисключено из обучения (слишком мало примеров): {out}")
    print(f"\nсплит: {SPLIT}")
    return yaml_path


def _pick_val_sessions(frames: list) -> set:
    """Отбирает сессии в val, сохраняя баланс день/ночь.

    Порядок детерминированный (сортировка), без random — пересборка сплита
    не должна перемешивать train и val между запусками.
    """
    by_tod: dict[str, list] = defaultdict(list)
    seen = set()
    for _, _, session, tod in frames:
        if session not in seen:
            seen.add(session)
            by_tod[tod].append(session)

    val = set()
    for tod in sorted(by_tod):
        sessions = sorted(by_tod[tod])
        n = max(1, round(len(sessions) * VAL_FRACTION))
        # берём с равномерным шагом, а не первые n: соседние по сортировке
        # сессии — часто один день и одна смена
        step = max(1, len(sessions) // n)
        val.update(sessions[::step][:n])
    return val


def _manifest() -> dict[str, dict]:
    path = TRAINSET / "manifest.json"
    if not path.exists():
        return {}
    return {r["file"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def _warn_unreviewed() -> None:
    review = TRAINSET / "review.json"
    checked = 0
    if review.exists():
        checked = sum(1 for v in json.loads(review.read_text(encoding="utf-8")).values() if v)
    if checked == 0:
        print(
            "! разметка не подтверждена руками ни на одном кадре:\n"
            "  labels/ — предразметка COCO-моделью, только класс truck с conf 0.15.\n"
            "  Классы truck_body / grapple / excavator в данных отсутствуют,\n"
            "  обучение по ним ничему не научит. Это прогон пайплайна, не модель.\n"
        )
    else:
        print(f"подтверждено вручную кадров: {checked}\n")


def _device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
