"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Summary, TrackRow, teamColor } from "@/lib/api";

const HOLD = 8; // keep a player on screen this many frames after a missed detection
const TRAIL = 25;

type Props = { summary: Summary; rows: TrackRow[] };

export default function TacticalMap({ summary, rows }: Props) {
  const { first_frame, last_frame, fps, bounds } = summary;
  // start where the most players are visible, so the map is not empty on first load
  const [frame, setFrame] = useState(() => {
    const counts = new Map<number, number>();
    for (const r of rows) counts.set(r.frame, (counts.get(r.frame) ?? 0) + 1);
    let best = first_frame, bestN = 0;
    counts.forEach((n, f) => { if (n > bestN) { bestN = n; best = f; } });
    return Math.max(first_frame, best - 20);
  });
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [trails, setTrails] = useState(true);
  const [teamFilter, setTeamFilter] = useState<string>("all");
  const frameRef = useRef(frame);
  frameRef.current = frame;

  // frame -> track_id -> row
  const index = useMemo(() => {
    const m = new Map<number, Map<number, TrackRow>>();
    for (const r of rows) {
      let f = m.get(r.frame);
      if (!f) m.set(r.frame, (f = new Map()));
      f.set(r.track_id, r);
    }
    return m;
  }, [rows]);

  const trackIds = useMemo(() => Array.from(new Set(rows.map((r) => r.track_id))), [rows]);

  function posAt(tid: number, f: number): TrackRow | undefined {
    for (let k = 0; k <= HOLD; k++) {
      const r = index.get(f - k)?.get(tid);
      if (r) return r;
    }
  }

  // playback loop
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      const next = frameRef.current + 1;
      if (next > last_frame) {
        setPlaying(false);
        return;
      }
      setFrame(next);
    }, 1000 / (fps * speed));
    return () => clearInterval(id);
  }, [playing, speed, fps, last_frame]);

  // view box fits the calibrated region with padding
  const pad = 4;
  const x0 = bounds.x_min - pad, x1 = bounds.x_max + pad;
  const y0 = bounds.y_min - pad, y1 = bounds.y_max + pad;
  const W = x1 - x0, H = y1 - y0;

  const gridX: number[] = [];
  for (let x = Math.ceil(x0 / 5) * 5; x <= x1; x += 5) gridX.push(x);
  const gridY: number[] = [];
  for (let y = Math.ceil(y0 / 5) * 5; y <= y1; y += 5) gridY.push(y);

  const visible = trackIds
    .map((tid) => ({ tid, cur: posAt(tid, frame) }))
    .filter((p): p is { tid: number; cur: TrackRow } => !!p.cur && (teamFilter === "all" || p.cur.team === teamFilter));

  const trailFor = (tid: number) => {
    const pts: string[] = [];
    for (let f = Math.max(first_frame, frame - TRAIL); f <= frame; f++) {
      const r = index.get(f)?.get(tid);
      if (r) pts.push(`${r.pitch_x},${r.pitch_y}`);
    }
    return pts.join(" ");
  };

  const time = ((frame - first_frame) / fps).toFixed(2);

  return (
    <div className="card">
      <div className="card-head">
        <h2>Tactical map</h2>
        <span className="muted">
          frame {frame} · t = {time}s · {visible.length} players visible
        </span>
      </div>

      <svg viewBox={`${x0} ${y0} ${W} ${H}`} className="pitch" preserveAspectRatio="xMidYMid meet">
        <rect x={x0} y={y0} width={W} height={H} fill="#14532d" />
        {gridX.map((x) => (
          <g key={`gx${x}`}>
            <line x1={x} x2={x} y1={y0} y2={y1} stroke="#ffffff22" strokeWidth={0.08} />
            <text x={x + 0.3} y={y0 + 1.2} fontSize={0.9} fill="#ffffff88">{x}m</text>
          </g>
        ))}
        {gridY.map((y) => (
          <g key={`gy${y}`}>
            <line x1={x0} x2={x1} y1={y} y2={y} stroke="#ffffff22" strokeWidth={0.08} />
            <text x={x0 + 0.3} y={y - 0.3} fontSize={0.9} fill="#ffffff88">{y}m</text>
          </g>
        ))}
        {trails &&
          visible.map(({ tid, cur }) => (
            <polyline
              key={`t${tid}`}
              points={trailFor(tid)}
              fill="none"
              stroke={teamColor(cur.team)}
              strokeOpacity={0.45}
              strokeWidth={0.25}
            />
          ))}
        {visible.map(({ tid, cur }) => (
          <g key={tid} transform={`translate(${cur.pitch_x} ${cur.pitch_y})`}>
            <circle r={1.1} fill={teamColor(cur.team)} stroke="#000" strokeWidth={0.18} />
            <text textAnchor="middle" dy={0.45} fontSize={1.2} fontWeight={700} fill="#000">{tid}</text>
          </g>
        ))}
      </svg>

      <div className="controls">
        <button onClick={() => (frame >= last_frame ? (setFrame(first_frame), setPlaying(true)) : setPlaying(!playing))}>
          {playing ? "Pause" : frame >= last_frame ? "Replay" : "Play"}
        </button>
        <input
          type="range"
          min={first_frame}
          max={last_frame}
          value={frame}
          onChange={(e) => setFrame(Number(e.target.value))}
          aria-label="frame"
        />
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="speed">
          {[0.25, 0.5, 1, 2].map((s) => (
            <option key={s} value={s}>{s}x</option>
          ))}
        </select>
        <select value={teamFilter} onChange={(e) => setTeamFilter(e.target.value)} aria-label="team">
          <option value="all">all teams</option>
          {summary.team_names.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <label className="check">
          <input type="checkbox" checked={trails} onChange={(e) => setTrails(e.target.checked)} /> trails
        </label>
      </div>
      <p className="muted small">
        Only part of the pitch is calibrated for this clip, so the map shows that region (y axis points down).
      </p>
    </div>
  );
}
