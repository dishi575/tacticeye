"""
Generate a synthetic tracks.parquet matching SCHEMA.md v1.

Lets people 3/4/6 build the backend, frontend, and analytics layers against
the FINAL shape of the data (team_id and pitch_x/pitch_y fully populated)
without waiting on the real CV pipeline, homography, or clustering to exist.

Usage:
    python generate_synthetic_tracks.py --out tracks.parquet
    python generate_synthetic_tracks.py --out tracks.parquet --frames 1500 --fps 25 --seed 7
"""

import argparse

import numpy as np
import pandas as pd

PITCH_X_MAX = 105.0  # meters
PITCH_Y_MAX = 68.0   # meters

# Pretend camera/homography: a fixed linear map from pitch meters -> pixel
# coords, just so pixel columns aren't left at some meaningless default.
# Real footage won't use this; it's only so synthetic rows are fully
# populated and interchangeable with (eventually) real, post-homography rows.
FRAME_W = 1920
FRAME_H = 1080


def pitch_to_pixel(px, py):
    x = (px / PITCH_X_MAX) * FRAME_W
    y = (py / PITCH_Y_MAX) * FRAME_H
    return x, y


def random_walk(n_frames, start, bounds, step_std, rng):
    """2D random walk clipped to [0, bound] on each axis."""
    pos = np.empty((n_frames, 2), dtype=np.float64)
    pos[0] = start
    for i in range(1, n_frames):
        step = rng.normal(0, step_std, size=2)
        pos[i] = pos[i - 1] + step
        pos[i, 0] = np.clip(pos[i, 0], 0, bounds[0])
        pos[i, 1] = np.clip(pos[i, 1], 0, bounds[1])
    return pos


def generate(n_frames, fps, seed, drop_prob, conf_min, conf_max):
    rng = np.random.default_rng(seed)
    rows = []

    # 11 vs 11 players, track_ids 1..22, teams 0/1. Ball is track_id 0.
    n_players_per_team = 11
    player_specs = []
    tid = 1
    for team in (0, 1):
        # Team 0 starts on the left half, team 1 on the right half —
        # arbitrary, just gives synthetic data plausible starting shape.
        x_lo, x_hi = (5, PITCH_X_MAX / 2 - 5) if team == 0 else (PITCH_X_MAX / 2 + 5, PITCH_X_MAX - 5)
        for _ in range(n_players_per_team):
            start = (rng.uniform(x_lo, x_hi), rng.uniform(5, PITCH_Y_MAX - 5))
            player_specs.append((tid, team, start))
            tid += 1

    ball_start = (PITCH_X_MAX / 2, PITCH_Y_MAX / 2)

    # Precompute a random-walk trajectory per object.
    trajectories = {}
    for pid, team, start in player_specs:
        trajectories[pid] = (
            team,
            random_walk(n_frames, start, (PITCH_X_MAX, PITCH_Y_MAX), step_std=0.4, rng=rng),
        )
    ball_traj = random_walk(n_frames, ball_start, (PITCH_X_MAX, PITCH_Y_MAX), step_std=1.2, rng=rng)

    for frame in range(n_frames):
        # Ball, track_id 0.
        if rng.uniform() > drop_prob:
            px, py = ball_traj[frame]
            x, y = pitch_to_pixel(px, py)
            half_w, half_h = 8, 8
            rows.append(
                dict(
                    frame=frame,
                    track_id=0,
                    class_id=32,
                    class_name="ball",
                    conf=round(float(rng.uniform(conf_min, conf_max)), 3),
                    x1=round(x - half_w, 1), y1=round(y - half_h, 1),
                    x2=round(x + half_w, 1), y2=round(y + half_h, 1),
                    cx=round(x, 1), cy=round(y, 1),
                    team_id=pd.NA,
                    pitch_x=round(float(px), 2), pitch_y=round(float(py), 2),
                )
            )

        # Players.
        for pid, (team, traj) in trajectories.items():
            if rng.uniform() < drop_prob:
                continue  # simulate an occasional missed detection
            px, py = traj[frame]
            x, y = pitch_to_pixel(px, py)
            half_w, half_h = 15, 35
            rows.append(
                dict(
                    frame=frame,
                    track_id=pid,
                    class_id=0,
                    class_name="player",
                    conf=round(float(rng.uniform(conf_min, conf_max)), 3),
                    x1=round(x - half_w, 1), y1=round(y - half_h, 1),
                    x2=round(x + half_w, 1), y2=round(y + half_h, 1),
                    cx=round(x, 1), cy=round(y, 1),
                    team_id=team,
                    pitch_x=round(float(px), 2), pitch_y=round(float(py), 2),
                )
            )

    df = pd.DataFrame(rows)
    df["team_id"] = df["team_id"].astype("Int64")
    df = df[
        [
            "frame", "track_id", "class_id", "class_name", "conf",
            "x1", "y1", "x2", "y2", "cx", "cy",
            "team_id", "pitch_x", "pitch_y",
        ]
    ]
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="tracks.parquet", help="output .parquet path")
    ap.add_argument("--frames", type=int, default=750, help="number of frames (default: 30s @ 25fps)")
    ap.add_argument("--fps", type=int, default=25, help="assumed source fps (metadata only, not stored per-row)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-prob", type=float, default=0.03, help="chance an object is missed in a given frame")
    ap.add_argument("--conf-min", type=float, default=0.55)
    ap.add_argument("--conf-max", type=float, default=0.98)
    args = ap.parse_args()

    df = generate(
        n_frames=args.frames,
        fps=args.fps,
        seed=args.seed,
        drop_prob=args.drop_prob,
        conf_min=args.conf_min,
        conf_max=args.conf_max,
    )
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df)} rows across {args.frames} frames -> {args.out}")
    print(df.head(10).to_string(index=False))
    