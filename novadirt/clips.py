"""Индексация датасета: папка = сессия (одна машина), внутри клипы одного канала."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# "246 (2026-07-11 16'54'10 - 2026-07-11 16'56'10).avi"
_NAMED = re.compile(r"^(\d+)\s*\((\d{4}-\d{2}-\d{2} \d{2}'\d{2}'\d{2})\s*-\s*(\d{4}-\d{2}-\d{2} \d{2}'\d{2}'\d{2})\)")
# "283.avi" / "985.avi" — без таймкода в имени
_BARE = re.compile(r"^(\d+)\.avi$", re.I)

_TS = "%Y-%m-%d %H'%M'%S"


@dataclass
class Clip:
    path: Path
    channel: str | None
    start: datetime | None
    end: datetime | None

    @property
    def named_duration(self) -> float | None:
        """Длительность по имени файла. Метаданным AVI доверять нельзя (fps врёт)."""
        if self.start is None or self.end is None:
            return None
        return (self.end - self.start).total_seconds()

    def ts_at(self, offset_s: float) -> datetime | None:
        if self.start is None:
            return None
        return self.start + timedelta(seconds=offset_s)


@dataclass
class Session:
    """Одна машина = одна сессия. Имя папки содержит госномер (ручная разметка оператора)."""
    day: str
    vehicle: str
    clips: list[Clip]

    @property
    def channels(self) -> set[str]:
        return {c.channel for c in self.clips if c.channel}

    @property
    def start(self) -> datetime | None:
        ts = [c.start for c in self.clips if c.start]
        return min(ts) if ts else None

    @property
    def end(self) -> datetime | None:
        ts = [c.end for c in self.clips if c.end]
        return max(ts) if ts else None

    @property
    def span_s(self) -> float | None:
        if self.start and self.end:
            return (self.end - self.start).total_seconds()
        return None

    def clips_on(self, channel: str) -> list[Clip]:
        return sorted(
            (c for c in self.clips if c.channel == channel),
            key=lambda c: c.start or datetime.min,
        )


def parse_clip(path: Path) -> Clip:
    name = path.name
    m = _NAMED.match(name)
    if m:
        return Clip(path, m.group(1), datetime.strptime(m.group(2), _TS), datetime.strptime(m.group(3), _TS))
    m = _BARE.match(name)
    if m:
        return Clip(path, m.group(1), None, None)
    return Clip(path, None, None, None)


def index_dataset(root: Path) -> list[Session]:
    """root = 'папки машин УПП'. Структура: <день>/<машина>/<клипы>.avi"""
    sessions: list[Session] = []
    for day_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for veh_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
            clips = [parse_clip(p) for p in sorted(veh_dir.glob("*.avi"))]
            if clips:
                sessions.append(Session(day_dir.name, veh_dir.name, clips))
    return sessions


def is_night(session: Session) -> bool:
    """Ночь помечена в имени папки-дня; иначе — по часу начала."""
    if "ноч" in session.day.lower():
        return True
    st = session.start
    return bool(st and (st.hour >= 21 or st.hour < 6))
