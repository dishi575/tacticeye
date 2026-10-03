# TacticEye

Football match video in, tactical analytics out. TacticEye detects and tracks every player in a broadcast-style clip, maps them onto the pitch, splits them into teams, and serves the result through an API to a web dashboard (tactical map replay, player stats, heatmaps, match report).



## System Architecture

```mermaid
flowchart TD

    A["Match Video"] --> B["YOLOv8<br/>Player Detection"]
    B --> C["ByteTrack<br/>Multi-Object Tracking"]
    C --> D["Player Tracks<br/>Track ID + Bounding Box"]

    D --> E["Pixel Coordinates<br/>(u, v)"]

    E --> F["Homography<br/>Perspective Transformation"]
    F --> G["Pitch Coordinates<br/>(X, Y) in meters"]

    D --> H["Player Crop"]

    H --> I["Jersey Colour<br/>Features"]
    H --> J["SigLIP<br/>Visual Embeddings"]

    I --> K["Team Classification"]
    J --> K
    K --> L["Team Assignment"]

    G --> M["Player-Level Dataset"]
    L --> M
    D --> M

    M --> N["Analytics Engine"]

    N --> O["Player Statistics"]
    N --> P["Heatmaps"]
    N --> Q["Average Positions"]
    N --> R["Team Shape"]
    N --> S["Movement & Distance"]

    O --> T["FastAPI"]
    P --> T
    Q --> T
    R --> T
    S --> T

    T --> U["Next.js Dashboard"]

    U --> V["Tactical Map"]
    U --> W["Player Statistics"]
    U --> X["Heatmaps"]
    U --> Y["Average Positions"]
    U --> Z["Match Report"]

    %% Styling

    classDef input fill:#E8F1FF,stroke:#2563EB,color:#172554,stroke-width:2px;
    classDef vision fill:#F1E8FF,stroke:#7C3AED,color:#3B0764,stroke-width:2px;
    classDef geometry fill:#E8F8EF,stroke:#16A34A,color:#14532D,stroke-width:2px;
    classDef team fill:#FFF3E0,stroke:#EA580C,color:#7C2D12,stroke-width:2px;
    classDef analytics fill:#E6F7F7,stroke:#0F766E,color:#134E4A,stroke-width:2px;
    classDef backend fill:#FFF7D6,stroke:#CA8A04,color:#713F12,stroke-width:2px;
    classDef frontend fill:#EEF2FF,stroke:#4F46E5,color:#312E81,stroke-width:2px;

    class A input;
    class B,C,D,H vision;
    class E,F,G geometry;
    class I,J,K,L team;
    class M,N,O,P,Q,R,S analytics;
    class T backend;
    class U,V,W,X,Y,Z frontend;



## Requirements: Python 3.10+, Node 18.18+.

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
