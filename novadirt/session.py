"""Конечный автомат сессии: 1 машина = 1 сессия.

Работает в реальном времени по потоку детекций с обзорной камеры (246).
Метрики грязи накапливаются покадрово, итог подтверждается при закрытии
сессии — как и требовалось: real-time + финальная сводка.

Автомат намеренно гистерезисный: на этих DVR-потоках fps плавает 2-25,
детектор моргает, и любое решение по одному кадру даёт рваные сессии.
Поэтому переходы требуют подтверждения N последовательными наблюдениями.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Sequence


class State(str, Enum):
    IDLE = "idle"                # площадка пуста
    TRUCK_PRESENT = "truck"      # машина стоит, разгрузка не началась
    UNLOADING = "unloading"      # ковш работает над кузовом
    LEAVING = "leaving"          # машина пропала из кадра, ждём подтверждения


@dataclass
class Detection:
    """Один бокс в кадре, нормализованные координаты xyxy."""
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def overlaps(grapple: Detection, truck: Detection) -> float:
    """Доля площади ковша, попавшая в бокс машины.

    Не IoU: ковш много меньше кузова, IoU остаётся мизерным даже когда
    ковш полностью внутри кузова, и порог по нему не выставить.
    """
    gx1, gy1, gx2, gy2 = grapple.xyxy
    tx1, ty1, tx2, ty2 = truck.xyxy
    ix1, iy1 = max(gx1, tx1), max(gy1, ty1)
    ix2, iy2 = min(gx2, tx2), min(gy2, ty2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / grapple.area if grapple.area > 0 else 0.0


@dataclass
class SessionRecord:
    """Накопитель одной сессии. Итог подтверждается в finalize()."""
    started: datetime
    ended: datetime | None = None
    unload_started: datetime | None = None
    unload_ended: datetime | None = None
    grab_count: int = 0
    dirt_samples: list[tuple[float, float]] = field(default_factory=list)  # (доля грязи, вес)
    evidence: list[str] = field(default_factory=list)

    def add_dirt(self, ratio: float, weight: float = 1.0) -> None:
        """weight — площадь захвата: маленький грязный ковш не должен весить
        столько же, сколько большой чистый."""
        if 0.0 <= ratio <= 1.0 and weight > 0:
            self.dirt_samples.append((ratio, weight))

    @property
    def dirt_pct(self) -> float | None:
        if not self.dirt_samples:
            return None
        wsum = sum(w for _, w in self.dirt_samples)
        return 100.0 * sum(r * w for r, w in self.dirt_samples) / wsum if wsum else None

    @property
    def duration_s(self) -> float | None:
        if self.ended is None:
            return None
        return (self.ended - self.started).total_seconds()

    def finalize(self, ts: datetime) -> "SessionRecord":
        self.ended = ts
        if self.unload_started and not self.unload_ended:
            self.unload_ended = ts
        return self


class SessionTracker:
    """Гистерезисный автомат.

    confirm_* — сколько подряд кадров должно подтвердить переход.
    Значения в кадрах, а не в секундах, потому что fps по клипам плавает;
    задавать их следует исходя из fps конкретного канала.
    """

    def __init__(
        self,
        confirm_arrive: int = 3,
        confirm_leave: int = 8,
        confirm_unload: int = 2,
        grapple_overlap: float = 0.35,
        min_truck_conf: float = 0.35,
    ):
        self.confirm_arrive = confirm_arrive
        self.confirm_leave = confirm_leave
        self.confirm_unload = confirm_unload
        self.grapple_overlap = grapple_overlap
        self.min_truck_conf = min_truck_conf

        self.state = State.IDLE
        self.current: SessionRecord | None = None
        self.closed: list[SessionRecord] = []
        self._seen = 0        # подряд кадров с машиной
        self._missing = 0     # подряд кадров без машины
        self._unloading = 0   # подряд кадров с ковшом над кузовом
        self._idle_unload = 0

    def update(self, ts: datetime, dets: Sequence[Detection]) -> State:
        trucks = [d for d in dets if d.cls in ("truck", "truck_body") and d.conf >= self.min_truck_conf]
        grapples = [d for d in dets if d.cls == "grapple"]

        # самая крупная машина в кадре = разгружаемая; остальные — фон площадки
        target = max(trucks, key=lambda d: d.area) if trucks else None

        if target is not None:
            self._seen += 1
            self._missing = 0
        else:
            self._missing += 1
            self._seen = 0

        active = bool(target) and any(overlaps(g, target) >= self.grapple_overlap for g in grapples)
        if active:
            self._unloading += 1
            self._idle_unload = 0
        else:
            self._idle_unload += 1
            self._unloading = 0

        if self.state is State.IDLE:
            if self._seen >= self.confirm_arrive:
                self.current = SessionRecord(started=ts)
                self.state = State.TRUCK_PRESENT

        elif self.state is State.TRUCK_PRESENT:
            if self._missing >= self.confirm_leave:
                self._close(ts)
            elif self._unloading >= self.confirm_unload:
                assert self.current is not None
                if self.current.unload_started is None:
                    self.current.unload_started = ts
                self.current.grab_count += 1
                self.state = State.UNLOADING

        elif self.state is State.UNLOADING:
            assert self.current is not None
            if self._missing >= self.confirm_leave:
                self._close(ts)
            elif self._idle_unload >= self.confirm_unload:
                self.current.unload_ended = ts
                self.state = State.TRUCK_PRESENT
            elif active:
                self.current.grab_count += 1

        return self.state

    def flush(self, ts: datetime) -> None:
        """Закрыть открытую сессию — конец потока/клипа."""
        if self.current is not None:
            self._close(ts)

    def _close(self, ts: datetime) -> None:
        if self.current is not None:
            self.closed.append(self.current.finalize(ts))
            self.current = None
        self.state = State.IDLE
        self._seen = self._missing = self._unloading = self._idle_unload = 0
