import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

# Pitch coordinates are metres, origin top-left of the pitch (x = length, y = width).
# NOTE: for the demo clip only part of the pitch is calibrated (see homography_points.json),
# so values cover a sub-region, not the full 105 x 68 m.
DATA_DIR = Path(os.getenv("MATCH_DIR", Path(__file__).resolve().parents[1] / "data" / "demo"))

app = FastAPI(title="TacticEye API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


def _stats() -> dict:
    p = DATA_DIR / "team_stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def fps() -> float:
    return float(os.getenv("FPS", _stats().get("fps", 25)))


def team_names() -> dict[int, str]:
    names = _stats().get("team_names") or {"0": "team0", "1": "team1"}
    return {int(k): v for k, v in names.items()}


@lru_cache
def load_tracks() -> pd.DataFrame:
    path = DATA_DIR / "tracks_roles.csv"
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found in {DATA_DIR}")
    df = pd.read_csv(path).dropna(subset=["team_id"])
    df["team"] = df["team_id"].astype(int).map(team_names())
    return df[["frame", "track_id", "team", "role", "pitch_x", "pitch_y"]]


@lru_cache
def load_player_summary() -> list[dict]:
    f_s = fps()
    rows = []
    for tid, g in load_tracks().sort_values("frame").groupby("track_id"):
        x = g["pitch_x"].rolling(15, center=True, min_periods=1).mean().to_numpy()
        y = g["pitch_y"].rolling(15, center=True, min_periods=1).mean().to_numpy()
        f = g["frame"].to_numpy()
        step = np.hypot(np.diff(x), np.diff(y))
        ok = (np.diff(f) == 1) & (step < 1.0)  # >1 m per frame = ID jump, ignore
        secs = ok.sum() / f_s
        dist = float(step[ok].sum())
        rows.append({
            "track_id": int(tid),
            "team": g["team"].mode().iat[0],
            "role": g["role"].mode().iat[0],
            "distance_m": round(dist, 1),
            "avg_speed_ms": round(dist / secs, 2) if secs > 0 else 0.0,
            "seen_s": round(len(g) / f_s, 1),
        })
    return sorted(rows, key=lambda r: -r["distance_m"])


def send_file(name: str, media_type: str | None = None) -> FileResponse:
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found")
    return FileResponse(path, media_type=media_type)


@app.get("/health")
def health():
    return {"status": "ok", "data_dir": str(DATA_DIR)}


@app.get("/api/summary")
def summary():
    df = load_tracks()
    f_s = fps()
    stats = _stats()
    teams = {k: v for k, v in stats.items() if isinstance(v, dict) and "team_id" in v}
    return {
        "fps": f_s,
        "frames": int(df["frame"].nunique()),
        "first_frame": int(df["frame"].min()),
        "last_frame": int(df["frame"].max()),
        "clip_seconds": round((df["frame"].max() - df["frame"].min() + 1) / f_s, 1),
        "tracks": int(df["track_id"].nunique()),
        "team_names": sorted(team_names().values()),
        "pitch": {"length_m": 105, "width_m": 68},
        "bounds": {
            "x_min": round(float(df["pitch_x"].min()), 2), "x_max": round(float(df["pitch_x"].max()), 2),
            "y_min": round(float(df["pitch_y"].min()), 2), "y_max": round(float(df["pitch_y"].max()), 2),
        },
        "teams": teams,
    }


@app.get("/api/frames/{frame}")
def get_frame(frame: int):
    d = load_tracks()
    d = d[d["frame"] == frame]
    if d.empty:
        raise HTTPException(404, f"No players in frame {frame}")
    players = [
        {"track_id": int(r.track_id), "team": r.team, "role": r.role,
         "x": round(float(r.pitch_x), 2), "y": round(float(r.pitch_y), 2)}
        for r in d.itertuples()
    ]
    return {"frame": frame, "time_s": round(frame / fps(), 2), "players": players}


@app.get("/api/tracks")
def get_tracks(
    start: int = Query(0, ge=0),
    end: int = Query(299, ge=0),
    team: str | None = Query(None),
):
    if end < start or end - start > 299:
        raise HTTPException(400, "Use start <= end and a range of at most 300 frames")
    if team is not None and team not in team_names().values():
        raise HTTPException(404, f"team must be one of {sorted(team_names().values())}")
    d = load_tracks()
    d = d[(d["frame"] >= start) & (d["frame"] <= end)]
    if team:
        d = d[d["team"] == team]
    return d.round(2).to_dict(orient="records")


@app.get("/api/players")
def players():
    return load_player_summary()


@app.get("/api/heatmap/{team}")
def heatmap(team: str):
    if team not in team_names().values() or not re.fullmatch(r"[A-Za-z0-9_]+", team):
        raise HTTPException(404, f"team must be one of {sorted(team_names().values())}")
    return send_file(f"heatmap_{team}.png", "image/png")


@app.get("/api/avg-positions")
def avg_positions():
    return send_file("avg_positions.png", "image/png")


@app.get("/api/report", response_class=HTMLResponse)
def report():
    path = DATA_DIR / "match_report.html"
    if not path.exists():
        raise HTTPException(404, "match_report.html not found")
    return path.read_text(encoding="utf-8")
