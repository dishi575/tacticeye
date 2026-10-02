import numpy as np
import pandas as pd

MIN_FRAMES = 15
MIN_MOVE_M = 4.0     # isse kam hilne wale tracks = static cheezein
src, dst = "output3/tracks_pitch.csv", "output3/tracks_pitch_clean.csv"

df = pd.read_csv(src)
before = df["track_id"].nunique()

# 1) chhote tracks hata do
counts = df.groupby("track_id")["frame"].transform("count")
df = df[counts >= MIN_FRAMES].copy()

# 2) static tracks hata do (net, post, spot marker)
g = df.groupby("track_id")
move = np.hypot(g["pitch_x"].transform("max") - g["pitch_x"].transform("min"),
                g["pitch_y"].transform("max") - g["pitch_y"].transform("min"))
static_ids = sorted(df.loc[move < MIN_MOVE_M, "track_id"].unique())
print("static tracks dropped:", len(static_ids), static_ids)
df = df[move >= MIN_MOVE_M].copy()

# 2b) goal line ke paas wale tracks (staff, net, ball-boys)
mx = df.groupby("track_id")["pitch_x"].transform("mean")
edge_ids = sorted(df.loc[(mx < 8) | (mx > 100), "track_id"].unique())
print("goal-line tracks dropped:", len(edge_ids), edge_ids)
df = df[(mx >= 8) & (mx <= 100)].copy()

# 3) renumber
order = df.groupby("track_id")["frame"].min().sort_values().index
df["track_id"] = df["track_id"].map({old: i + 1 for i, old in enumerate(order)})

df.to_csv(dst, index=False)
print(f"ids: {before} -> {df['track_id'].nunique()}")
print("avg rows per frame:", round(df.groupby('frame').size().mean(), 1))