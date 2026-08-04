"""Локальный редактор разметки: детекция боксами + сегментация полигонами.

Запуск:  .venv/bin/python -m labeler.server     ->  http://127.0.0.1:8100

Хранение — прямо в формате YOLO, никакой промежуточной базы: то, что вы
разметили, сразу пригодно для обучения.

Два независимых режима, потому что задачи разные и форматы несовместимы
(YOLO-det и YOLO-seg нельзя держать в одном .txt):

  det  labels/*.txt      "cls x y w h"            каналы 246, 161 — общий план
  seg  labels_seg/*.txt  "cls x1 y1 x2 y2 ..."    каналы 235, 238, 239 — крупный план

Сегментация нужна ради вычитания: процент грязи считается по пикселям лома,
а железо самого грейфера в эту площадь попадать не должно. Боксом ковш из
области не вычесть — в прямоугольнике ковш занимает треть, остальное груз.

Область внимания (ROI вокруг ковша) здесь НЕ размечается: она однозначно
выводится из полигона ковша на инференсе, руками её рисовать значило бы
учить сеть воспроизводить формулу и добавить в данные разброс аннотатора.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
TRAINSET = ROOT / "out" / "trainset"
IMAGES = TRAINSET / "images"
MANIFEST = TRAINSET / "manifest.json"

LABELS = TRAINSET / "labels"
REVIEW = TRAINSET / "review.json"
LABELS_SEG = TRAINSET / "labels_seg"
REVIEW_SEG = TRAINSET / "review_seg.json"

CLASSES = ["truck", "truck_body", "grapple", "excavator"]
SEG_CLASSES = ["grapple", "scrap"]

# на общем плане (246 — площадка с 30 м, 161 — PTZ) полигоны бессмысленны:
# ковш там несколько десятков пикселей, граница лома не читается вообще
SEG_CHANNELS = {"235", "238", "239"}

MODES = {
    "det": (LABELS, REVIEW, CLASSES, None),
    "seg": (LABELS_SEG, REVIEW_SEG, SEG_CLASSES, SEG_CHANNELS),
}

app = FastAPI(title="nova labeler")


def _mode(mode: str):
    if mode not in MODES:
        raise HTTPException(400, f"неизвестный режим: {mode}")
    return MODES[mode]


class Box(BaseModel):
    cls: int
    # нормализованные xywh (центр + размер) — формат YOLO
    x: float
    y: float
    w: float
    h: float


class Poly(BaseModel):
    cls: int
    # плоский список нормализованных координат: x1,y1,x2,y2,... — формат YOLO-seg
    pts: list[float]


class SavePayload(BaseModel):
    boxes: list[Box] = []
    polys: list[Poly] = []
    reviewed: bool = True


def _load_review(path: Path) -> dict[str, bool]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_review(path: Path, data: dict[str, bool]) -> None:
    path.write_text(json.dumps(data, indent=0))


def _meta() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    return {m["file"]: m for m in json.loads(MANIFEST.read_text())}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text()


@app.get("/api/config")
def config() -> dict:
    return {"classes": CLASSES, "seg_classes": SEG_CLASSES,
            "seg_channels": sorted(SEG_CHANNELS)}


@app.get("/api/images")
def images(channel: str = "", tod: str = "", status: str = "",
           mode: str = "det") -> dict:
    _, review_path, _, only_ch = _mode(mode)
    meta = _meta()
    review = _load_review(review_path)
    items = []
    for p in sorted(IMAGES.glob("*.jpg")):
        m = meta.get(p.name, {})
        ch = m.get("channel") or p.name.split("_")[0]
        td = m.get("tod") or ("night" if "_night_" in p.name else "day")
        done = review.get(p.name, False)
        if only_ch and ch not in only_ch:
            continue
        if channel and ch != channel:
            continue
        if tod and td != tod:
            continue
        if status == "todo" and done:
            continue
        if status == "done" and not done:
            continue
        items.append({"file": p.name, "channel": ch, "tod": td,
                      "vehicle": m.get("vehicle", ""), "reviewed": done})
    return {"items": items, "total": len(items)}


@app.get("/api/stats")
def stats(mode: str = "det") -> dict:
    labels, review_path, classes, only_ch = _mode(mode)
    review = _load_review(review_path)
    per_class = [0] * len(classes)
    for txt in labels.glob("*.txt"):
        if not review.get(f"{txt.stem}.jpg"):
            continue  # считаем только проверенное вручную, не предразметку
        for line in txt.read_text().splitlines():
            if line.strip():
                c = int(line.split()[0])
                if 0 <= c < len(classes):
                    per_class[c] += 1
    total = sum(1 for p in IMAGES.glob("*.jpg")
                if not only_ch or p.name.split("_")[0] in only_ch)
    done = sum(1 for v in review.values() if v)
    return {"total": total, "reviewed": done,
            "per_class": dict(zip(classes, per_class))}


@app.get("/api/image/{name}")
def image(name: str) -> FileResponse:
    p = (IMAGES / name).resolve()
    if not p.is_file() or IMAGES.resolve() not in p.parents:
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/label/{name}")
def get_label(name: str, mode: str = "det") -> dict:
    labels, review_path, _, _ = _mode(mode)
    txt = labels / f"{Path(name).stem}.txt"
    boxes: list[dict] = []
    polys: list[dict] = []
    if txt.exists():
        for line in txt.read_text().splitlines():
            parts = line.split()
            if mode == "det" and len(parts) == 5:
                c, x, y, w, h = parts
                boxes.append({"cls": int(c), "x": float(x), "y": float(y),
                              "w": float(w), "h": float(h)})
            elif mode == "seg" and len(parts) >= 7 and len(parts) % 2 == 1:
                polys.append({"cls": int(parts[0]),
                              "pts": [float(v) for v in parts[1:]]})
    return {"boxes": boxes, "polys": polys,
            "reviewed": _load_review(review_path).get(name, False)}


@app.post("/api/label/{name}")
def save_label(name: str, payload: SavePayload, mode: str = "det") -> dict:
    if not (IMAGES / name).is_file():
        raise HTTPException(404)
    labels, review_path, _, _ = _mode(mode)
    labels.mkdir(parents=True, exist_ok=True)
    if mode == "det":
        lines = [
            f"{b.cls} {_clamp(b.x):.6f} {_clamp(b.y):.6f} "
            f"{_clamp(b.w):.6f} {_clamp(b.h):.6f}"
            for b in payload.boxes if b.w > 0.002 and b.h > 0.002
        ]
    else:
        lines = [
            " ".join([str(p.cls)] + [f"{_clamp(v):.6f}" for v in p.pts])
            for p in payload.polys if len(p.pts) >= 6 and len(p.pts) % 2 == 0
        ]
    (labels / f"{Path(name).stem}.txt").write_text("\n".join(lines))
    review = _load_review(review_path)
    review[name] = payload.reviewed
    _save_review(review_path, review)
    return {"saved": len(lines)}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
