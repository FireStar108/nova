"""Оценка детектора: общий mAP плюс разрез по каналу и день/ночь.

Сводный mAP по val бесполезен для решения «работает или нет»: 246 днём и 246
ночью — разные по сложности задачи (см. 02_probe_yolo: днём COCO-модель не
находила грузовик вообще, ночью conf 0.35). Средняя цифра прячет провал одного
из режимов, поэтому считаем метрику отдельно по каждой группе.

Запуск:
    .venv/bin/python scripts/06_eval_det.py
    .venv/bin/python scripts/06_eval_det.py --weights out/runs/det2/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out"
SPLIT = OUT / "det_split"
TRAINSET = OUT / "trainset"


def _classes() -> list[str]:
    """Имена классов берём из сплита, а не дублируем: 05_train_det выбрасывает
    редкие классы и перенумеровывает остальные, и рассинхрон дал бы метрики,
    подписанные чужими именами."""
    yaml = SPLIT / "dataset.yaml"
    if not yaml.exists():
        sys.exit(f"нет сплита: {SPLIT}\nсначала: python scripts/05_train_det.py --split-only")
    names = []
    for line in yaml.read_text().splitlines():
        if line.startswith("  ") and ":" in line:
            names.append(line.split(":", 1)[1].strip())
    return names


CLASSES = _classes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(OUT / "runs" / "det" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"нет весов: {weights}\nсначала: python scripts/05_train_det.py")
    val_dir = SPLIT / "images" / "val"
    if not val_dir.exists():
        sys.exit(f"нет сплита: {SPLIT}\nсначала: python scripts/05_train_det.py --split-only")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    device = args.device or _device()

    print(f"{'группа':>14} {'кадров':>7} {'mAP50':>7} {'mAP50-95':>9} {'R':>6}")
    print("-" * 48)
    rows = []
    for name, files in _groups(val_dir).items():
        m = _eval_subset(model, name, files, args.imgsz, device)
        if m is None:
            continue
        rows.append((name, len(files), m))
        print(f"{name:>14} {len(files):>7} {m['map50']:>7.3f} {m['map']:>9.3f} {m['recall']:>6.3f}")

    if rows:
        worst = min(rows, key=lambda r: r[2]["map50"])
        print(f"\nхудшая группа: {worst[0]}  mAP50 {worst[2]['map50']:.3f}")
    print("\nпо классам (весь val):")
    m_all = _eval_subset(model, "__all__", sorted(val_dir.glob("*.jpg")), args.imgsz, device)
    if m_all:
        for i, cname in enumerate(CLASSES):
            v = m_all["per_class"].get(i)
            print(f"  {cname:>12}: " + (f"mAP50 {v:.3f}" if v is not None else "нет в val"))


def _groups(val_dir: Path) -> dict[str, list[Path]]:
    meta = _manifest()
    groups: dict[str, list[Path]] = defaultdict(list)
    for img in sorted(val_dir.glob("*.jpg")):
        m = meta.get(img.name)
        ch = m["channel"] if m else img.name.split("_")[0]
        tod = m["tod"] if m else img.name.split("_")[1]
        groups[f"{ch} {tod}"].append(img)
    return dict(sorted(groups.items()))


def _eval_subset(model, name: str, files: list[Path], imgsz: int, device: str) -> dict | None:
    """Прогоняет val на подмножестве кадров через временный сплит-каталог.

    ultralytics умеет валидировать только каталог целиком, поэтому под каждую
    группу собирается каталог симлинков — копировать кадры незачем.
    """
    if not files:
        return None
    tmp = OUT / "eval_tmp" / _slug(name)
    if tmp.exists():
        shutil.rmtree(tmp)
    for img in files:
        lbl = SPLIT / "labels" / "val" / f"{img.stem}.txt"
        (tmp / "images" / "val").mkdir(parents=True, exist_ok=True)
        (tmp / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (tmp / "images" / "val" / img.name).symlink_to(img.resolve())
        if lbl.exists():
            (tmp / "labels" / "val" / lbl.name).symlink_to(lbl.resolve())

    yaml = tmp / "dataset.yaml"
    yaml.write_text(
        f"path: {tmp.resolve()}\ntrain: images/val\nval: images/val\n\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    r = model.val(
        data=str(yaml), imgsz=imgsz, device=device, verbose=False,
        project=str(OUT / "runs"), name=f"val_{_slug(name)}", exist_ok=True, plots=False,
    )
    per_class = {}
    for idx, c in enumerate(r.box.ap_class_index):
        per_class[int(c)] = float(r.box.ap50[idx])
    return {
        "map50": float(r.box.map50), "map": float(r.box.map),
        "recall": float(r.box.mr), "per_class": per_class,
    }


def _manifest() -> dict[str, dict]:
    path = TRAINSET / "manifest.json"
    if not path.exists():
        return {}
    return {r["file"]: r for r in json.loads(path.read_text())}


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


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
