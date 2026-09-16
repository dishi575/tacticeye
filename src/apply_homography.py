import cv2
import json
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--points", default="homography_points.json")
ap.add_argument("--tracks", required=True, help="tracks.csv from track.py")
ap.add_argument("--out", default="tracks_pitch.csv")
args = ap.parse_args()

with open(args.points) as f:
    data = json.load(f)

src = np.array(data["pixel"], dtype=np.float32)
dst = np.array(data["world"], dtype=np.float32)

H, _ = cv2.findHomography(src, dst)
np.save("homography_matrix.npy", H)
print("Homography matrix saved to homography_matrix.npy")

df = pd.read_csv(args.tracks)

# use bottom-center of the box (feet position) not the box center, more accurate for pitch position
feet_x = (df["x1"] + df["x2"]) / 2
feet_y = df["y2"]

pts = np.stack([feet_x.values, feet_y.values], axis=1).astype(np.float32).reshape(-1, 1, 2)
world_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

df["pitch_x"] = world_pts[:, 0]
df["pitch_y"] = world_pts[:, 1]

df.to_csv(args.out, index=False)
print(f"Saved {args.out} with pitch_x / pitch_y columns (meters)")
