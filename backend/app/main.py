"""TacticEye read-only API.

Pure Python (csv + math): no pandas/numpy, so it also runs on locked-down machines where
Windows "Application Control" blocks compiled .pyd/.dll files.
"""
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

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
    # also allow other local/LAN dev addresses (e.g. VS Code browser pane, WSL, same-wifi devices)
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)(:\d+)?$",
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
def load_tracks() -> tuple[dict, ...]:
    """Rows of tracks_roles.csv that have a team, sorted by (frame, track_id)."""
    path = DATA_DIR / "tracks_roles.csv"
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found in {DATA_DIR}")
    names = team_names()
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("team_id"):
                continue
            rows.append({
                "frame": int(float(r["frame"])),
                "track_id": int(float(r["track_id"])),
                "team": names.get(int(float(r["team_id"])), f"team{int(float(r['team_id']))}"),
                "role": r.get("role") or "player",
                "pitch_x": float(r["pitch_x"]),
                "pitch_y": float(r["pitch_y"]),
            })
    rows.sort(key=lambda r: (r["frame"], r["track_id"]))
    return tuple(rows)


@lru_cache
def _by_frame() -> dict[int, list[dict]]:
    d: dict[int, list[dict]] = defaultdict(list)
    for r in load_tracks():
        d[r["frame"]].append(r)
    return d


def _rolling_mean(vals: list[float], window: int = 15) -> list[float]:
    """Centered rolling mean with min_periods=1 (same as pandas rolling(15, center=True))."""
    n, half = len(vals), window // 2
    pre = [0.0]
    for v in vals:
        pre.append(pre[-1] + v)
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append((pre[hi] - pre[lo]) / (hi - lo))
    return out


def _mode(values) -> str:
    c = Counter(values)
    top = max(c.values())
    return sorted(k for k, v in c.items() if v == top)[0]


@lru_cache
def load_player_summary() -> list[dict]:
    f_s = fps()
    per_track: dict[int, list[dict]] = defaultdict(list)
    for r in load_tracks():
        per_track[r["track_id"]].append(r)
    out = []
    for tid, g in per_track.items():
        x = _rolling_mean([r["pitch_x"] for r in g])
        y = _rolling_mean([r["pitch_y"] for r in g])
        frames = [r["frame"] for r in g]
        dist, ok_steps = 0.0, 0
        for i in range(len(g) - 1):
            step = math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
            if frames[i + 1] - frames[i] == 1 and step < 1.0:  # >1 m per frame = ID jump, ignore
                dist += step
                ok_steps += 1
        secs = ok_steps / f_s
        out.append({
            "track_id": tid,
            "team": _mode(r["team"] for r in g),
            "role": _mode(r["role"] for r in g),
            "distance_m": round(dist, 1),
            "avg_speed_ms": round(dist / secs, 2) if secs > 0 else 0.0,
            "seen_s": round(len(g) / f_s, 1),
        })
    return sorted(out, key=lambda r: -r["distance_m"])


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
    rows = load_tracks()
    if not rows:
        raise HTTPException(404, "no tracked players in tracks_roles.csv")
    f_s = fps()
    stats = _stats()
    teams = {k: v for k, v in stats.items() if isinstance(v, dict) and "team_id" in v}
    frames = _by_frame()
    first, last = min(frames), max(frames)
    xs = [r["pitch_x"] for r in rows]
    ys = [r["pitch_y"] for r in rows]
    return {
        "fps": f_s,
        "frames": len(frames),
        "first_frame": first,
        "last_frame": last,
        "clip_seconds": round((last - first + 1) / f_s, 1),
        "tracks": len({r["track_id"] for r in rows}),
        "team_names": sorted(team_names().values()),
        "pitch": {"length_m": 105, "width_m": 68},
        "bounds": {
            "x_min": round(min(xs), 2), "x_max": round(max(xs), 2),
            "y_min": round(min(ys), 2), "y_max": round(max(ys), 2),
        },
        "teams": teams,
    }


@app.get("/api/frames/{frame}")
def get_frame(frame: int):
    rows = _by_frame().get(frame)
    if not rows:
        raise HTTPException(404, f"No players in frame {frame}")
    players = [
        {"track_id": r["track_id"], "team": r["team"], "role": r["role"],
         "x": round(r["pitch_x"], 2), "y": round(r["pitch_y"], 2)}
        for r in rows
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
    frames = _by_frame()
    out = []
    for f in range(start, end + 1):
        for r in frames.get(f, ()):
            if team is None or r["team"] == team:
                out.append({**r, "pitch_x": round(r["pitch_x"], 2), "pitch_y": round(r["pitch_y"], 2)})
    return out


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