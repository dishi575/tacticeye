"use client";

import { useEffect, useMemo, useState } from "react";
import TacticalMap from "@/components/TacticalMap";
import {
  API, PlayerStat, Summary, TrackRow, avgPositionsUrl, getAllTracks, getPlayers, getSummary,
  heatmapUrl, reportUrl, teamColor,
} from "@/lib/api";

type SortKey = "distance_m" | "avg_speed_ms" | "seen_s" | "track_id";

export default function Home() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [rows, setRows] = useState<TrackRow[]>([]);
  const [players, setPlayers] = useState<PlayerStat[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<SortKey>("distance_m");
  const [teamFilter, setTeamFilter] = useState("all");

  useEffect(() => {
    (async () => {
      try {
        const s = await getSummary();
        setSummary(s);
        const [r, p] = await Promise.all([getAllTracks(s.first_frame, s.last_frame), getPlayers()]);
        setRows(r);
        setPlayers(p);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const shown = useMemo(() => {
    const list = players.filter((p) => teamFilter === "all" || p.team === teamFilter);
    return [...list].sort((a, b) => (sort === "track_id" ? a.track_id - b.track_id : b[sort] - a[sort]));
  }, [players, sort, teamFilter]);

  if (error)
    return (
      <main className="wrap">
        <div className="card error">
          <h2>Backend se connect nahi ho paaya</h2>
          <p>{error}</p>
          <p>
            Backend chalao: <code>cd backend &amp;&amp; uvicorn app.main:app --reload</code>. API URL abhi:{" "}
            <code>{API}</code> (badalne ke liye <code>frontend/.env.local</code> me <code>NEXT_PUBLIC_API_URL</code>).
          </p>
        </div>
      </main>
    );

  if (!summary) return <main className="wrap"><p className="muted">Loading…</p></main>;

  return (
    <main className="wrap">
      <header>
        <h1>TacticEye</h1>
        <p className="muted">Football tracking dashboard · demo clip</p>
      </header>

      <section className="stats">
        <Stat label="Clip length" value={`${summary.clip_seconds}s`} />
        <Stat label="Frames" value={`${summary.frames}`} sub={`@ ${summary.fps.toFixed(0)} fps`} />
        <Stat label="Tracks" value={`${summary.tracks}`} />
        {Object.entries(summary.teams).map(([name, t]) => (
          <Stat
            key={name}
            label={`${name} team`}
            value={`${t.total_distance_m} m`}
            sub={`${t.players_tracked} players tracked`}
            color={teamColor(name)}
          />
        ))}
      </section>

      {rows.length ? <TacticalMap summary={summary} rows={rows} /> : <p className="muted">Loading tracks…</p>}

      <section className="card">
        <div className="card-head">
          <h2>Players</h2>
          <div className="controls inline">
            <select value={teamFilter} onChange={(e) => setTeamFilter(e.target.value)} aria-label="team filter">
              <option value="all">all teams</option>
              {summary.team_names.map((t) => <option key={t}>{t}</option>)}
            </select>
            <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} aria-label="sort by">
              <option value="distance_m">sort: distance</option>
              <option value="avg_speed_ms">sort: speed</option>
              <option value="seen_s">sort: time seen</option>
              <option value="track_id">sort: id</option>
            </select>
          </div>
        </div>
        <table>
          <thead>
            <tr><th>ID</th><th>Team</th><th>Role</th><th>Distance (m)</th><th>Avg speed (m/s)</th><th>Seen (s)</th></tr>
          </thead>
          <tbody>
            {shown.map((p) => (
              <tr key={p.track_id}>
                <td>#{p.track_id}</td>
                <td><span className="dot" style={{ background: teamColor(p.team) }} /> {p.team}</td>
                <td>{p.role}</td>
                <td>{p.distance_m}</td>
                <td>{p.avg_speed_ms}</td>
                <td>{p.seen_s}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card">
        <h2>Heatmaps &amp; positions</h2>
        <div className="images">
          {summary.team_names.map((t) => (
            <figure key={t}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={heatmapUrl(t)} alt={`${t} heatmap`} />
              <figcaption>{t} heatmap</figcaption>
            </figure>
          ))}
          <figure>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={avgPositionsUrl} alt="average positions" />
            <figcaption>average positions</figcaption>
          </figure>
        </div>
      </section>

      <section className="card">
        <div className="card-head">
          <h2>Match report</h2>
          <a href={reportUrl} target="_blank" rel="noreferrer">open in new tab</a>
        </div>
        <iframe src={reportUrl} title="match report" className="report" />
      </section>
    </main>
  );
}

function Stat({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="stat" style={color ? { borderTopColor: color } : undefined}>
      <div className="muted small">{label}</div>
      <div className="big">{value}</div>
      {sub && <div className="muted small">{sub}</div>}
    </div>
  );
}
