import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="output3/tracks_pitch_clean.csv")
ap.add_argument("--teams", default="output3/team_assignments.csv")
ap.add_argument("--out", default="output3/tracks_roles.csv")
ap.add_argument("--attack0", choices=["left", "right"], default="right",
                help="team 0 kis taraf attack kar rahi hai; team 1 ulti taraf")
args = ap.parse_args()

PITCH_L = 105.0
GK_ZONE = 16.5   # apne goal line se itni doori ke andar hi GK maana jaye

df = pd.read_csv(args.csv).merge(pd.read_csv(args.teams), on="track_id", how="left")
df["role"] = "UNK"

def depth(x, team):
    """Apne goal se doori (0 = apni goal line, 105 = doosri goal line)."""
    attacks_right = (args.attack0 == "right") == (team == 0)
    return x if attacks_right else PITCH_L - x

for (frame, team), g in df.dropna(subset=["team_id"]).groupby(["frame", "team_id"]):
    d = depth(g["pitch_x"], int(team)).sort_values()
    idx = list(d.index)
    roles = {}
    if d.iloc[0] <= GK_ZONE and len(idx) >= 4:
        roles[idx[0]] = "GK"
        idx = idx[1:]
    n = len(idx)
    for k, i in enumerate(idx):
        roles[i] = ["DEF", "MID", "FWD"][min(k * 3 // max(n, 1), 2)]
    for i, r in roles.items():
        df.at[i, "role"] = r

df.to_csv(args.out, index=False)

print("rows per role:", df["role"].value_counts().to_dict())
# har track ka sabse common role
mode = df[df["role"] != "UNK"].groupby("track_id")["role"].agg(lambda s: s.mode().iat[0])
print("tracks per role:", mode.value_counts().to_dict())
print("saved", args.out)