import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from ultralytics import YOLO
from novadirt.clips import index_dataset
from novadirt.video import FpsCache, iter_frames

ROOT = Path("/Users/firestar_main/Desktop/nova/папки машин УПП")
OUT  = Path(__file__).resolve().parents[1] / "out"
(OUT/"probe").mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11x.pt")
cache = FpsCache(OUT)
sessions = index_dataset(ROOT)

# по 1 клипу с каждого интересного канала, день и ночь
picks = []
for ch in ("246","235","238","239"):
    for night in (False, True):
        from novadirt.clips import is_night
        for s in sessions:
            if is_night(s) != night: continue
            cl = s.clips_on(ch)
            if cl: picks.append((ch, "ночь" if night else "день", s, cl[0])); break

hits = collections.Counter()
for ch, tod, s, clip in picks:
    fps = cache.probe(clip)["fps"] or 6.0
    for fr in iter_frames(clip, fps, stride=max(1,int(fps*4)), max_frames=3):
        r = model(fr.image, verbose=False, conf=0.25)[0]
        names = [r.names[int(c)] for c in r.boxes.cls]
        hits.update((ch, n) for n in names)
        tag = f"{ch}_{tod}_{s.vehicle[:12]}_{fr.idx}".replace("/","_").replace(" ","_")
        cv2.imwrite(str(OUT/"probe"/f"{tag}.jpg"), r.plot())
    print(f"{ch} {tod:5s} {s.vehicle[:22]:22s} fps={fps:5.1f}")

print("\n=== что нашла COCO-модель ===")
for (ch,n),c in sorted(hits.items(), key=lambda kv:(kv[0][0],-kv[1])):
    print(f"  {ch}  {n:15s} {c}")
