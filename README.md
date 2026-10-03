# TacticEye

Football match video in, tactical analytics out. TacticEye detects and tracks every player in a broadcast-style clip, maps them onto the pitch, splits them into teams, and serves the result through an API to a web dashboard (tactical map replay, player stats, heatmaps, match report).

```
 video ──► YOLO + ByteTrack ──► pixel tracks ──► homography ──► pitch coordinates (m)
                                     │                                   │
                                     └──► jersey colour / SigLIP ──► team split
                                                                         │
                                  stats · heatmaps · average positions ◄─┘
                                                   │
                                          FastAPI backend  ──►  Next.js dashboard
```

## What is in this repo

| Path | What it does |
|---|---|
| `track.py`, `src/track.py` | YOLOv8 + ByteTrack on a video, writes `tracks.csv` (frame, track_id, class, conf, bbox, centre) |
| `homography_points.json`, `homography_matrix.npy`, `src/calibrate.py`, `src/apply_homography.py` | Pixel to pitch-metre mapping; produces `tracks_pitch.csv` |
| `cluster_teams.py`, `siglip_embeddings_colab.ipynb`, `generate_synthetic_crops.py` | Team split from SigLIP embeddings of player crops (numpy k-means, no scikit-learn needed) |
| `generate_synthetic_tracks.py` | Synthetic `tracks.parquet` for building the backend/frontend without the CV pipeline |
| `src/analytics.py`, `src/roles.py`, `src/tactical_map.py`, `src/report.py`, `docs/` | Analytics, role guess, tactical map, heatmaps and the HTML match report |
| `backend/` | **FastAPI** read-only API over the pipeline outputs, with tests, Dockerfile and demo data |
| `frontend/` | **Next.js** dashboard that talks to the API |
| `PROJECT_README.md` | Two-terminal quick start |

## Quick start (dashboard on the demo clip)

Requirements: Python 3.10+, Node 18.18+.

```bash
# Terminal 1 - API  (http://127.0.0.1:8000/docs)
cd backend
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 2 - dashboard  (http://localhost:3000)
cd frontend
cp .env.local.example .env.local  # Windows: copy .env.local.example .env.local
npm install
npm run dev
```

Open <http://localhost:3000>. Press **Play** on the tactical map to replay the clip.

The API serves the files in `backend/data/demo/`. Point it at another match with `MATCH_DIR=/path/to/output` (and `FPS=25` if the clip is not 30 fps; by default fps is read from `team_stats.json`).

## Dashboard

- **Stats cards:** clip length, frames, tracks, distance per team.
- **Tactical map:** bird's-eye replay of player positions with play/pause, scrubber, speed, team filter and trails.
- **Players table:** distance, average speed and time seen per track.
- **Heatmaps and average positions** per team.
- **Match report** (HTML) embedded in the page.

## API

Pitch coordinates are metres, origin top-left (x = length, y = width). Interactive docs at `/docs`.

| Endpoint | Returns |
|---|---|
| `GET /health` | status |
| `GET /api/summary` | fps, frame range, clip length, track count, team names, coordinate bounds, per-team stats |
| `GET /api/frames/{n}` | all players in frame `n` (track_id, team, role, x, y) |
| `GET /api/tracks?start&end&team` | rows for a frame range (max 300 frames) |
| `GET /api/players` | per-track distance and average speed |
| `GET /api/heatmap/{team}` | PNG heatmap |
| `GET /api/avg-positions` | PNG average positions |
| `GET /api/report` | HTML match report |

Tests: `cd backend && pip install -r requirements-dev.txt && python -m pytest`.
Docker: `cd backend && docker build -t tacticeye-api . && docker run -p 8000:8000 tacticeye-api`.

## Running the CV pipeline on your own clip

```bash
pip install -r requirements.txt                       # ultralytics, opencv, numpy, pandas, ...
python track.py --video sample2.mp4 --model yolov8m.pt --imgsz 1280 --conf 0.15 --out output4
# convert pixels to pitch metres (see src/apply_homography.py) -> tracks_pitch.csv
cd backend
python tools/make_demo_data.py --tracks ../tracks_pitch.csv --video ../sample2.mp4 --out data/demo
```

`tools/make_demo_data.py` splits teams by median shirt colour (k-means, k=2), drops very dark kits (referee/keeper) from the team split, and writes `tracks_roles.csv`, `team_stats.json`, heatmaps, average positions and the report into `data/demo/`.

Tips: `yolov8n.pt` is the fastest but misses many players in a wide shot; `yolov8m.pt` at `--imgsz 1280` detects far more (about 22 people plus the ball in a test frame) at roughly 2 s/frame on a 2-core CPU. Use `--every_n 6` for a quick trial run.

## Known limitations

- **Calibration covers only part of the pitch.** The homography is fitted from four points in one penalty box, so it extrapolates poorly to the rest of the frame (e.g. the centre circle maps to ~28 m instead of ~52 m). Recalibrate with 6+ spread-out landmarks (centre circle, halfway line, both penalty boxes) for full-pitch accuracy. Until then the map shows only the calibrated region and absolute distances are approximate.
- **Detection density depends on the model.** With `yolov8n` the demo data has only 1-5 players per frame; use a larger model for realistic numbers.
- **Roles** are a single value (`player`) in the demo data; role detection is not wired into the API yet.
- **Ball, passing lanes and xT** are not implemented yet.
- The API is read-only: there is no video upload endpoint yet.

## Roadmap

- [ ] Re-run tracking with a larger YOLO model and regenerate the demo data
- [ ] Recalibrate the homography with spread-out pitch landmarks
- [ ] `POST /api/pipeline/run`: upload a video and run the pipeline as a background job
- [ ] Ball tracking, passing lanes, expected threat (xT)
- [ ] Wire role detection (`src/roles.py`) into the API
