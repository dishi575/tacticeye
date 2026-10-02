import argparse
import cv2
import numpy as np
import pandas as pd

def color_for_id(track_id):
    rng = np.random.default_rng(int(track_id))
    return tuple(int(c) for c in rng.integers(60, 255, size=3))

def run(csv_path, out_path, px_per_m=20, margin_m=3, fps=15, trail_len=15):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["pitch_x", "pitch_y"])

    min_x, max_x = df["pitch_x"].min() - margin_m, df["pitch_x"].max() + margin_m
    min_y, max_y = df["pitch_y"].min() - margin_m, df["pitch_y"].max() + margin_m

    width = int((max_x - min_x) * px_per_m)
    height = int((max_y - min_y) * px_per_m)

    def to_px(x, y):
        return int((x - min_x) * px_per_m), int((y - min_y) * px_per_m)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    trails = {}  # track_id->list of recent (px,py)

    for frame_num, group in df.groupby("frame"):
        canvas = np.full((height, width, 3), (60, 140, 60), dtype=np.uint8)  # green pitch

        for _, row in group.iterrows():
            tid = int(row["track_id"])
            cls = int(row["class"])
            px, py = to_px(row["pitch_x"], row["pitch_y"])

            trails.setdefault(tid, [])
            trails[tid].append((px, py))
            trails[tid] = trails[tid][-trail_len:]

            color = (255, 255, 255) if cls == 32 else color_for_id(tid)  # ball = white
            radius = 4 if cls == 32 else 7

            for i in range(1, len(trails[tid])):
                cv2.line(canvas, trails[tid][i - 1], trails[tid][i], color, 1)

            cv2.circle(canvas, (px, py), radius, color, -1)
            if cls != 32:
                cv2.putText(canvas, str(tid), (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        writer.write(canvas)

    writer.release()
    print(f"Saved tactical map video: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="tracks_pitch.csv from apply_homography.py")
    ap.add_argument("--out", default="tactical_map.mp4")
    ap.add_argument("--px_per_m", type=int, default=20, help="pixels per meter, raise for bigger video")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    run(args.csv, args.out, args.px_per_m, fps=args.fps)