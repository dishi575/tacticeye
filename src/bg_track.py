import cv2, json, argparse, csv, os
import numpy as np
from scipy.optimize import linear_sum_assignment

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="sample3.mp4")
ap.add_argument("--out", default="output3")
ap.add_argument("--points", default="homography_points.json")
ap.add_argument("--thr", type=int, default=25)
ap.add_argument("--every_n", type=int, default=1)
ap.add_argument("--max_frames", type=int, default=0, help="0 = whole video")
ap.add_argument("--gate", type=float, default=80, help="max match distance (original px)")
ap.add_argument("--max_lost", type=int, default=30, help="frames to keep a lost track")
args = ap.parse_args()

SCALE = 0.5
os.makedirs(args.out, exist_ok=True)

cap = cv2.VideoCapture(args.video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# median background from 30 spread-out frames
frames = []
for i in np.linspace(0, n - 1, 30).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    if ok:
        frames.append(cv2.resize(f, None, fx=SCALE, fy=SCALE))
bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
del frames

# pitch polygon, shrunk 1.5%
pts = np.array(json.load(open(args.points))["pixel"]) * SCALE
pts = pts.mean(axis=0) + (pts - pts.mean(axis=0)) * 0.985
poly = np.zeros(bg.shape[:2], np.uint8)
cv2.fillPoly(poly, [pts.astype(np.int32)], 255)
kernel = np.ones((5, 5), np.uint8)

def detect(f):
    diff = cv2.absdiff(f, bg).max(axis=2)
    mask = (diff > args.thr).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, poly)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        if 25 < cv2.contourArea(c) < 800:
            x, y, w, h = cv2.boundingRect(c)
            dets.append([(x - 4) / SCALE, (y - 4) / SCALE,
                         (x + w + 4) / SCALE, (y + h + 4) / SCALE])
    return dets

# tracker state: id -> {"c": (cx, cy), "lost": int}
tracks, next_id = {}, 1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fout = open(os.path.join(args.out, "tracks.csv"), "w", newline="")
wr = csv.writer(fout)
wr.writerow(["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2", "cx", "cy"])

fi, rows_total = -1, 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fi += 1
    if fi % args.every_n:
        continue
    if args.max_frames and fi >= args.max_frames:
        break

    dets = detect(cv2.resize(frame, None, fx=SCALE, fy=SCALE))
    centers = np.array([[(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in dets]).reshape(-1, 2)

    ids = list(tracks.keys())
    assigned = {}
    if ids and len(centers):
        prev = np.array([tracks[i]["c"] for i in ids])
        cost = np.linalg.norm(prev[:, None, :] - centers[None, :, :], axis=2)
        r, c = linear_sum_assignment(cost)
        for ri, ci in zip(r, c):
            if cost[ri, ci] <= args.gate:
                assigned[ci] = ids[ri]

    matched_ids = set(assigned.values())
    for i in ids:
        if i not in matched_ids:
            tracks[i]["lost"] += 1
    for i in [i for i in tracks if tracks[i]["lost"] > args.max_lost]:
        del tracks[i]

    for di, d in enumerate(dets):
        if di in assigned:
            tid = assigned[di]
        else:
            tid = next_id
            next_id += 1
        tracks[tid] = {"c": tuple(centers[di]), "lost": 0}
        wr.writerow([fi, tid, 0, 1.0, *[round(v, 1) for v in d],
                     round(centers[di][0], 1), round(centers[di][1], 1)])
        rows_total += 1

    if fi % 100 == 0:
        print(f"frame {fi}/{n}, dets this frame: {len(dets)}")

fout.close()
print("done. rows:", rows_total, "unique ids:", next_id - 1)