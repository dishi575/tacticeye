import cv2, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="output3/tracks_pitch_clean.csv")
ap.add_argument("--teams", default="output3/team_assignments.csv")
ap.add_argument("--frame", type=int, default=300)
ap.add_argument("--out", default="output3/team_check.png")
args = ap.parse_args()

S = 10  # px per metre
df = pd.read_csv(args.csv).merge(pd.read_csv(args.teams), on="track_id", how="left")
d = df[df["frame"] == args.frame]

img = np.full((68 * S, 105 * S, 3), (60, 120, 60), np.uint8)
cv2.rectangle(img, (0, 0), (105 * S - 1, 68 * S - 1), (255, 255, 255), 2)
cv2.line(img, (int(52.5 * S), 0), (int(52.5 * S), 68 * S), (255, 255, 255), 2)
cv2.circle(img, (int(52.5 * S), 34 * S), int(9.15 * S), (255, 255, 255), 2)
colors = {0: (255, 120, 0), 1: (255, 255, 255)}   # team 0 neela/gehra, team 1 safed
for _, r in d.iterrows():
    c = colors.get(r["team_id"], (0, 0, 255))
    p = (int(r["pitch_x"] * S), int(r["pitch_y"] * S))
    cv2.circle(img, p, 8, c, -1)
    cv2.circle(img, p, 8, (0, 0, 0), 1)
cv2.imwrite(args.out, img)
print("players in frame:", len(d), "| team counts:", d["team_id"].value_counts().to_dict())