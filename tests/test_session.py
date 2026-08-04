import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from novadirt.session import Detection, SessionTracker, State, overlaps  # noqa: E402

T0 = datetime(2026, 7, 11, 16, 0, 0)
TRUCK = Detection("truck", 0.9, (0.5, 0.4, 0.9, 0.7))
GRAPPLE_IN = Detection("grapple", 0.8, (0.60, 0.45, 0.68, 0.55))     # внутри кузова
GRAPPLE_OUT = Detection("grapple", 0.8, (0.10, 0.10, 0.18, 0.20))    # над кучей


def run(tr: SessionTracker, frames, start=T0, dt=1.0):
    for i, dets in enumerate(frames):
        tr.update(start + timedelta(seconds=i * dt), dets)
    return tr


def test_overlap_metric_beats_iou_for_small_grapple():
    # ковш целиком внутри кузова -> перекрытие 1.0, тогда как IoU был бы ~0.05
    assert overlaps(GRAPPLE_IN, TRUCK) == 1.0
    assert overlaps(GRAPPLE_OUT, TRUCK) == 0.0


def test_full_session_lifecycle():
    tr = SessionTracker(confirm_arrive=3, confirm_leave=8, confirm_unload=2)
    run(tr, [[TRUCK]] * 5                      # подъезд
           + [[TRUCK, GRAPPLE_IN]] * 10        # разгрузка
           + [[TRUCK]] * 3                     # пауза
           + [[]] * 10)                        # уехала
    assert tr.state is State.IDLE
    assert len(tr.closed) == 1
    s = tr.closed[0]
    assert s.unload_started is not None and s.unload_ended is not None
    assert s.grab_count > 0


def test_flicker_does_not_split_session():
    """Детектор моргает на 1 кадр — сессия не должна разорваться на две."""
    tr = SessionTracker(confirm_arrive=3, confirm_leave=8, confirm_unload=2)
    frames = []
    for _ in range(6):
        frames += [[TRUCK]] * 4 + [[]]   # каждый 5-й кадр пропуск
    run(tr, frames)
    assert len(tr.closed) == 0
    assert tr.state is not State.IDLE


def test_grapple_over_pile_is_not_unloading():
    """Ковш работает с кучей, а не с кузовом — разгрузку начинать нельзя."""
    tr = SessionTracker(confirm_arrive=3, confirm_unload=2)
    run(tr, [[TRUCK, GRAPPLE_OUT]] * 12)
    assert tr.state is State.TRUCK_PRESENT
    assert tr.current is not None and tr.current.unload_started is None


def test_background_truck_ignored_in_favour_of_largest():
    """На площадке всегда есть фоновые машины; целевая — самая крупная."""
    far = Detection("truck", 0.9, (0.05, 0.05, 0.10, 0.09))
    tr = SessionTracker(confirm_arrive=3, confirm_unload=2)
    run(tr, [[far, TRUCK, GRAPPLE_IN]] * 10)
    assert tr.state is State.UNLOADING


def test_weighted_dirt_prefers_large_grabs():
    tr = SessionTracker(confirm_arrive=1)
    run(tr, [[TRUCK]] * 2)
    s = tr.current
    assert s is not None
    s.add_dirt(0.10, weight=10.0)   # большой чистый захват
    s.add_dirt(0.90, weight=1.0)    # маленький грязный
    # простое среднее дало бы 50%, взвешенное ~17%
    assert 15.0 < s.dirt_pct < 20.0


def test_low_confidence_truck_ignored():
    weak = Detection("truck", 0.10, (0.5, 0.4, 0.9, 0.7))
    tr = SessionTracker(confirm_arrive=3, min_truck_conf=0.35)
    run(tr, [[weak]] * 10)
    assert tr.state is State.IDLE


def test_flush_closes_open_session_at_stream_end():
    tr = SessionTracker(confirm_arrive=3)
    run(tr, [[TRUCK]] * 5)
    assert tr.current is not None
    tr.flush(T0 + timedelta(seconds=99))
    assert len(tr.closed) == 1 and tr.closed[0].ended is not None
