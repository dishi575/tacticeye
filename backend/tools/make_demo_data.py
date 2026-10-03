"""
Build the demo output files the API serves, from what the CV pipeline already
produced (tracks_pitch.csv) + the source video (for jersey-colour team split).

    python tools/make_demo_data.py --tracks data/tracks_pitch.csv --video data/sample2.mp4 --out data/demo

Writes: tracks_roles.csv, team_stats.json, heatmap_<team>.png, avg_positions.png, match_report.html
Team split = k-means (k=2, pure numpy) on the median torso colour of each track.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEAM_NAME = {0: "red", 1: "green"}  # filled after clustering, see name_teams()
FPS_DEFAULT = 25.0


def kmeans(X, k, seed=0, n_init=10, iters=100):
    rng = np.random.default_rng(seed)
    best, best_in = None, np.inf
    for _ in range(n_init):
        c = X[rng.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            lab = np.linalg.norm(X[:, None] - c[None], axis=2).argmin(1)
            nc = np.array([X[lab == i].mean(0) if (lab == i).any() else c[i] for i in range(k)])
            if np.allclose(nc, c):
                break
            c = nc
        lab = np.linalg.norm(X[:, None] - c[None], axis=2).argmin(1)
        inertia = ((X - c[lab]) ** 2).sum()
        if inertia < best_in:
            best, best_in = (lab, c), inertia
    return best


def torso_colours(df, video):
    """Mean BGR colour of the upper-body region for every detection row."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    by_frame = {f: g for f, g in df.groupby("frame")}
    out = {}
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx in by_frame:
            h, w = img.shape[:2]
            for i, r in by_frame[idx].iterrows():
                x1, x2 = int(max(r.x1, 0)), int(min(r.x2, w))
                y1, y2 = int(max(r.y1, 0)), int(min(r.y2, h))
                bw, bh = x2 - x1, y2 - y1
                if bw < 6 or bh < 12:
                    continue
                # central 50% width, 15%-50% height = shirt, avoids grass/shorts
                crop = img[y1 + int(.15 * bh): y1 + int(.5 * bh), x1 + int(.25 * bw): x2 - int(.25 * bw)]
                if crop.size:
                    out[i] = crop.reshape(-1, 3).mean(0)
        idx += 1
    cap.release()
    return out, fps


def assign_teams(df, colours):
    c = pd.DataFrame.from_dict(colours, orient="index", columns=["b", "g", "r"])
    c["track_id"] = df.loc[c.index, "track_id"].values
    per_track = c.groupby("track_id")[["b", "g", "r"]].median()
    # very dark kit (all channels low) = referee/keeper in black -> not a team
    per_track = per_track[per_track.max(axis=1) > 90]
    X = per_track.values.astype(float)
    lab, cent = kmeans(X, 2)
    # name cluster by dominant channel so labels are human readable
    names = {}
    for k in range(2):
        b, g, r = cent[k]
        names[k] = "red" if r > g else "green"
    if names[0] == names[1]:
        names = {0: "team0", 1: "team1"}
    # stable order: red -> 0, other -> 1
    order = sorted(range(2), key=lambda k: names[k] != "red")
    remap = {k: i for i, k in enumerate(order)}
    team_of = {tid: remap[l] for tid, l in zip(per_track.index, lab)}
    return team_of, {remap[k]: names[k] for k in range(2)}


def pitch_axes(ax, df):
    x0, x1 = df.pitch_x.min() - 4, df.pitch_x.max() + 4
    y0, y1 = df.pitch_y.min() - 4, df.pitch_y.max() + 4
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # origin top-left, like the API contract
    ax.set_aspect("equal")
    ax.set_facecolor("#2e7d32")
    ax.set_xlabel("pitch_x (m)")
    ax.set_ylabel("pitch_y (m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="data/tracks_pitch.csv")
    ap.add_argument("--video", default="data/sample2.mp4")
    ap.add_argument("--out", default="data/demo")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.tracks)
    df = df[df["class"] == 0].reset_index(drop=True)  # players only (ball needs its own model)
    colours, fps = torso_colours(df, a.video)
    team_of, names = assign_teams(df, colours)
    df["team_id"] = df["track_id"].map(team_of)
    df = df.dropna(subset=["team_id"]).copy()
    df["team_id"] = df["team_id"].astype(int)
    df["role"] = "player"  # no role model yet -> single role
    df[["frame", "track_id", "team_id", "role", "pitch_x", "pitch_y"]].to_csv(out / "tracks_roles.csv", index=False)

    # ---- team stats
    stats = {"fps": fps, "team_names": {str(k): v for k, v in names.items()}}
    for t, g in df.groupby("team_id"):
        dist = 0.0
        for _, p in g.sort_values("frame").groupby("track_id"):
            f = p["frame"].to_numpy()
            st = np.hypot(np.diff(p.pitch_x.to_numpy()), np.diff(p.pitch_y.to_numpy()))
            dist += float(st[(np.diff(f) == 1) & (st < 1.0)].sum())
        stats[names[t]] = {
            "team_id": int(t),
            "players_tracked": int(g.track_id.nunique()),
            "total_distance_m": round(dist, 1),
            "avg_x": round(float(g.pitch_x.mean()), 2),
            "avg_y": round(float(g.pitch_y.mean()), 2),
        }
    (out / "team_stats.json").write_text(json.dumps(stats, indent=2))

    # ---- heatmaps
    cmap = {"red": "Reds", "green": "Greens"}
    for t, name in names.items():
        g = df[df.team_id == t]
        fig, ax = plt.subplots(figsize=(7, 6))
        pitch_axes(ax, df)
        ax.hist2d(g.pitch_x, g.pitch_y, bins=[24, 24], cmap=cmap.get(name, "viridis"), alpha=.9,
                  range=[ax.get_xlim(), ax.get_ylim()[::-1]])
        ax.set_title(f"Heatmap - {name} team")
        fig.tight_layout()
        fig.savefig(out / f"heatmap_{name}.png", dpi=110)
        plt.close(fig)

    # ---- average positions
    fig, ax = plt.subplots(figsize=(7, 6))
    pitch_axes(ax, df)
    for t, name in names.items():
        a_ = df[df.team_id == t].groupby("track_id")[["pitch_x", "pitch_y"]].mean()
        ax.scatter(a_.pitch_x, a_.pitch_y, s=140, c={"red": "#d32f2f", "green": "#a5d6a7"}.get(name, "k"),
                   edgecolors="k", label=name, zorder=3)
        for tid, r in a_.iterrows():
            ax.annotate(str(tid), (r.pitch_x, r.pitch_y), ha="center", va="center", fontsize=7, zorder=4)
    ax.legend()
    ax.set_title("Average positions")
    fig.tight_layout()
    fig.savefig(out / "avg_positions.png", dpi=110)
    plt.close(fig)

    # ---- report
    rows = "".join(
        f"<tr><td>{n}</td><td>{stats[n]['players_tracked']}</td><td>{stats[n]['total_distance_m']}</td></tr>"
        for n in names.values()
    )
    (out / "match_report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>TacticEye report</title>"
        "<body style='font-family:sans-serif;max-width:720px;margin:2rem auto'>"
        "<h1>TacticEye match report (demo clip)</h1>"
        f"<p>{df.frame.nunique()} frames @ {fps:.0f} fps, {df.track_id.nunique()} tracks.</p>"
        f"<table border=1 cellpadding=6><tr><th>Team</th><th>Players</th><th>Distance (m)</th></tr>{rows}</table>"
        "<h2>Average positions</h2><img src='/api/avg-positions' width=480>"
        "</body>"
    )
    print("wrote", sorted(p.name for p in out.iterdir()))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
