# TacticEye backend (FastAPI)

Read-only API over the pipeline outputs. Demo data lives in `data/demo/`.
Set `MATCH_DIR` to point at another output folder, `FPS` to override the clip fps (default: from `team_stats.json`).

    pip install -r requirements.txt
    uvicorn app.main:app --reload      # docs at http://127.0.0.1:8000/docs
    pip install -r requirements-dev.txt && python -m pytest

Pitch coordinates: metres, origin top-left (x = length, y = width). The demo clip is only calibrated for a
sub-region of the pitch (x ~ 24-29, y ~ 0-26), so values do not span the full 105 x 68 m.
Teams are named by jersey colour (`red`, `green`); black-kit referees/keepers are excluded.

| Endpoint | Returns |
|---|---|
| GET /health | status |
| GET /api/summary | fps, frames, first/last frame, clip length, track count, team names, per-team stats |
| GET /api/frames/{n} | all players in frame n: track_id, team, role, x, y (404 if none) |
| GET /api/tracks?start&end&team | rows for a frame range (max 300 frames), team = a name from /api/summary |
| GET /api/players | per-track distance and average speed, sorted by distance |
| GET /api/heatmap/{team} | PNG heatmap |
| GET /api/avg-positions | PNG average positions |
| GET /api/report | HTML match report |

## Regenerate demo data
    pip install -r requirements-dev.txt
    python tools/make_demo_data.py --tracks data/tracks_pitch.csv --video data/sample2.mp4 --out data/demo

## Docker
    docker build -t tacticeye-api . && docker run -p 8000:8000 tacticeye-api

## Next tasks
- POST /api/pipeline/run (video upload, background job)
- Ball detection, passing lanes, xT
