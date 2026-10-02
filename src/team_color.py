import cv2, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="sample3.mp4")
ap.add_argument("--csv", default="output3/tracks_pitch_clean.csv")
ap.add_argument("--out", default="output3/team_assignments.csv")
ap.add_argument("--samples", type=int, default=8, help="frames sampled per track")
args = ap.parse_args()

df = pd.read_csv(args.csv)

# har track ke liye kuch frames chuno
need = {}   # frame -> [(track_id, x1, y1, x2, y2)]
for tid, g in df.groupby("track_id"):
    idx = np.linspace(0, len(g) - 1, min(args.samples, len(g))).astype(int)
    for _, r in g.iloc[idx].iterrows():
        need.setdefault(int(r["frame"]), []).append(
            (tid, r["x1"], r["y1"], r["x2"], r["y2"]))

cols = {}   # track_id -> list of Lab colours
cap = cv2.VideoCapture(args.video)
fi = -1
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fi += 1
    if fi not in need:
        continue
    H, W = frame.shape[:2]
    for tid, x1, y1, x2, y2 in need[fi]:
        # box ka beech ka hissa (padding hata ke)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * 0.5, (y2 - y1) * 0.5
        crop = frame[int(max(cy - h / 2, 0)):int(min(cy + h / 2, H)),
                     int(max(cx - w / 2, 0)):int(min(cx + w / 2, W))]
        if crop.size == 0:
            continue
        b, g_, r = [crop[..., i].astype(int) for i in range(3)]
        not_grass = ~((g_ > r + 8) & (g_ > b + 8))      # ghaas hata do
        px = crop[not_grass] if not_grass.sum() >= 5 else crop.reshape(-1, 3)
        lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_BGR2LAB).reshape(-1, 3)
        cols.setdefault(tid, []).append(np.median(lab, axis=0))

tids = sorted(cols)
X = np.array([np.median(cols[t], axis=0) for t in tids])

# pure numpy k-means, k=2
rng = np.random.default_rng(0)
best = None
for _ in range(10):
    C = X[rng.choice(len(X), 2, replace=False)]
    for _ in range(50):
        lab_ = np.linalg.norm(X[:, None] - C[None], axis=2).argmin(1)
        newC = np.array([X[lab_ == k].mean(0) if (lab_ == k).any() else C[k] for k in range(2)])
        if np.allclose(newC, C):
            break
        C = newC
    inertia = ((X - C[lab_]) ** 2).sum()
    if best is None or inertia < best[0]:
        best = (inertia, lab_, C)
_, labels, C = best
if C[0][0] > C[1][0]:      # team 0 = gehri jersey (kam L)
    labels = 1 - labels
    C = C[::-1]
out = pd.DataFrame({"track_id": tids, "team_id": labels})
out.to_csv(args.out, index=False)
print("tracks per team:", out["team_id"].value_counts().to_dict())
print("team centre colours (Lab: L, a, b):")
for k in range(2):
    print(f"  team {k}:", C[k].round(0))