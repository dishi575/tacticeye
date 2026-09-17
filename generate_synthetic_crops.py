"""
Generate placeholder "player crop" images to stand in for real cropped
photos (which need Person 1's trained YOLO model + real video, not ready
yet). Each fake crop is a small colored rectangle with jersey-like colors
and some noise/texture, so it's not literally a flat color block.

Team 0 gets a blue-ish palette, team 1 gets a red-ish palette, with enough
random variation that clustering on them is a real (if easy) test of the
pipeline rather than a trivial giveaway.

Once real crops exist (from track.py + YOLO), swap the --out folder for
one full of real cropped player images with the same manifest.csv format
and nothing downstream needs to change.

Usage:
    python generate_synthetic_crops.py --out crops --tracks tracks.parquet
    python generate_synthetic_crops.py --out crops   # no tracks.parquet needed, uses a default 11v11 roster
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

CROP_W, CROP_H = 64, 128

TEAM_BASE_COLORS = {
    0: (40, 70, 200),    # blue-ish
    1: (200, 60, 40),    # red-ish
}


def make_crop(rng, team_id, size=(CROP_W, CROP_H)):
    base = np.array(TEAM_BASE_COLORS.get(int(team_id), (120, 120, 120)), dtype=np.float64)
    noise = rng.normal(0, 25, size=(size[1], size[0], 3))
    img = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
    # A lighter "shorts" band near the bottom third, purely so crops aren't
    # a single flat texture (closer to what a real jersey+shorts crop looks like).
    band_start = int(size[1] * 0.65)
    img[band_start:, :, :] = np.clip(img[band_start:, :, :] + 60, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def default_roster():
    # Matches generate_synthetic_tracks.py's convention: track_id 1..22,
    # first 11 on team 0, next 11 on team 1.
    rows = []
    for tid in range(1, 23):
        team = 0 if tid <= 11 else 1
        rows.append({"track_id": tid, "true_team_id": team})
    return pd.DataFrame(rows)


def roster_from_tracks(tracks_path):
    df = pd.read_parquet(tracks_path)
    df = df[df["class_name"] == "player"]
    roster = (
        df.groupby("track_id")["team_id"]
        .first()
        .reset_index()
        .rename(columns={"team_id": "true_team_id"})
    )
    return roster


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="crops", help="output folder for crop images")
    ap.add_argument("--tracks", default=None, help="optional tracks.parquet to pull real track_id/team_id from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    roster = roster_from_tracks(args.tracks) if args.tracks else default_roster()

    manifest_rows = []
    for _, row in roster.iterrows():
        tid, team = int(row["track_id"]), row["true_team_id"]
        img = make_crop(rng, team)
        fname = f"crop_{tid:03d}.png"
        img.save(os.path.join(args.out, fname))
        manifest_rows.append({"track_id": tid, "crop_path": fname, "true_team_id": team})

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    print(f"Wrote {len(manifest)} crops -> {args.out}/ (manifest.csv included)")