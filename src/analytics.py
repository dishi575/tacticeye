import cv2, json, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="output3/tracks_roles.csv")
ap.add_argument("--out", default="output3")
args = ap.parse_args()

S, L, W = 10, 105, 68
COL = {0: (255, 120, 0), 1: (255, 255, 255)}      # BGR: team 0 neela, team 1 safed
NAME = {0: "dark", 1: "white"}

def draw_lines(img):
    w = (255, 255, 255)
    def R(x1, y1, x2, y2):
        cv2.rectangle(img, (int(x1 * S), int(y1 * S)), (int(x2 * S) - 1, int(y2 * S) - 1), w, 2)
    R(0, 0, L, W)
    R(0, 13.84, 16.5, 54.16); R(L - 16.5, 13.84, L, 54.16)
    R(0, 24.84, 5.5, 43.16);  R(L - 5.5, 24.84, L, 43.16)
    cv2.line(img, (int(L / 2 * S), 0), (int(L / 2 * S), W * S), w, 2)
    cv2.circle(img, (int(L / 2 * S), int(W / 2 * S)), int(9.15 * S), w, 2)
    return img

def grass():
    return np.full((W * S, L * S, 3), (60, 120, 60), np.uint8)

df = pd.read_csv(args.csv).dropna(subset=["team_id"])
df["team_id"] = df["team_id"].astype(int)
stats = {}

for t, d in df.groupby("team_id"):
    # 1) heatmap
    h, _, _ = np.histogram2d(d["pitch_y"], d["pitch_x"], bins=[W, L], range=[[0, W], [0, L]])
    h = cv2.GaussianBlur(h.astype(np.float32), (0, 0), 3)
    h = h / h.max() if h.max() > 0 else h
    h = cv2.resize(h, (L * S, W * S))
    color = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = h[..., None] * 0.75
    img = (grass() * (1 - a) + color * a).astype(np.uint8)
    cv2.imwrite(f"{args.out}/heatmap_team{t}_{NAME[t]}.png", draw_lines(img))

    # 2) team shape per frame
    g = d.groupby("frame")
    width = g["pitch_y"].agg(lambda s: s.max() - s.min())
    length = g["pitch_x"].agg(lambda s: s.max() - s.min())
    stats[NAME[t]] = {
        "tracks": int(d["track_id"].nunique()),
        "avg_players_per_frame": round(float(g.size().mean()), 1),
        "centroid_x_m": round(float(d["pitch_x"].mean()), 1),
        "centroid_y_m": round(float(d["pitch_y"].mean()), 1),
        "avg_width_m": round(float(width.mean()), 1),
        "avg_length_m": round(float(length.mean()), 1),
    }

# 3) average positions map (har track ka average + sabse common role)
img = draw_lines(grass())
for t, d in df.groupby("team_id"):
    for tid, g in d.groupby("track_id"):
        if len(g) < 100:
            continue
        p = (int(g["pitch_x"].mean() * S), int(g["pitch_y"].mean() * S))
        role = g["role"].mode().iat[0]
        cv2.circle(img, p, 13, COL[t], -1)
        cv2.circle(img, p, 13, (0, 0, 0), 1)
        cv2.putText(img, role[0], (p[0] - 6, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.imwrite(f"{args.out}/avg_positions.png", img)

json.dump(stats, open(f"{args.out}/team_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))