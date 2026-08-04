import sys, collections, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from novadirt.clips import index_dataset, is_night
from novadirt.video import FpsCache

ROOT = Path("/Users/firestar_main/Desktop/nova/папки машин УПП")
OUT = Path(__file__).resolve().parents[1] / "out"

sessions = index_dataset(ROOT)
cache = FpsCache(OUT)

per_ch = collections.defaultdict(lambda: {"clips":0,"frames":0,"fps":[],"dur":0.0,"night":0,"day":0})
multi = []
for s in sessions:
    for c in s.clips:
        info = cache.probe(c)
        d = per_ch[c.channel or "?"]
        d["clips"] += 1; d["frames"] += info["frames"]
        if info["fps"]: d["fps"].append(info["fps"])
        if info["named_duration_s"]: d["dur"] += info["named_duration_s"]
        d["night" if is_night(s) else "day"] += 1
    if len(s.channels) > 1: multi.append(s)
cache.save()

print(f"сессий: {len(sessions)}  клипов: {sum(len(s.clips) for s in sessions)}\n")
print(f"{'кан':>5} {'клипов':>7} {'кадров':>9} {'часов':>7} {'fps med':>8} {'fps min-max':>14}  день/ночь")
for ch, d in sorted(per_ch.items(), key=lambda kv: -kv[1]["clips"]):
    f = d["fps"]
    rng = f"{min(f):.1f}-{max(f):.1f}" if f else "-"
    med = f"{st.median(f):.1f}" if f else "-"
    print(f"{ch:>5} {d['clips']:>7} {d['frames']:>9} {d['dur']/3600:>7.2f} {med:>8} {rng:>14}  {d['day']}/{d['night']}")

sp = [s.span_s for s in sessions if s.span_s]
print(f"\nдлительность сессии, с: медиана {st.median(sp):.0f}  мин {min(sp):.0f}  макс {max(sp):.0f}")
print(f"сессий с >1 каналом: {len(multi)}")
for s in multi[:10]: print("   ", s.day, "/", s.vehicle, sorted(s.channels))
