import base64, json, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="output3")
ap.add_argument("--fps", type=float, default=30)
args = ap.parse_args()
D = args.dir
NAME = {0: "Dark", 1: "White"}

df = pd.read_csv(f"{D}/tracks_roles.csv").dropna(subset=["team_id"])
df["team_id"] = df["team_id"].astype(int)
stats = json.load(open(f"{D}/team_stats.json"))

def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()

# per-track distance covered (smoothed, ID-jumps ignored)
rows = []
for tid, g in df.sort_values("frame").groupby("track_id"):
    x = g["pitch_x"].rolling(15, center=True, min_periods=1).mean().to_numpy()
    y = g["pitch_y"].rolling(15, center=True, min_periods=1).mean().to_numpy()
    f = g["frame"].to_numpy()
    step = np.hypot(np.diff(x), np.diff(y))
    ok = (np.diff(f) == 1) & (step < 1.0)          # >1 m/frame = ID jump
    dist = step[ok].sum()
    secs = ok.sum() / args.fps
    if secs < 5:
        continue
    rows.append({
        "Track": int(tid),
        "Team": NAME[int(g["team_id"].mode().iat[0])],
        "Role": g["role"].mode().iat[0],
        "Distance (m)": round(dist, 1),
        "Avg speed (m/s)": round(dist / secs, 2),
        "Seen (s)": round(len(g) / args.fps, 1),
    })
top = pd.DataFrame(rows).sort_values("Distance (m)", ascending=False).head(10)

total_s = df["frame"].nunique() / args.fps
stat_rows = "".join(
    f"<tr><td>{k.replace('_', ' ')}</td><td>{stats['dark'][k]}</td><td>{stats['white'][k]}</td></tr>"
    for k in stats["dark"])

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>TacticEye Match Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 16px;color:#1b1b1b}}
h1{{margin-bottom:0}} .sub{{color:#666;margin-top:4px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}} th{{background:#f2f2f2}}
.row{{display:flex;gap:16px;flex-wrap:wrap}} .row div{{flex:1;min-width:300px}}
img{{width:100%;border:1px solid #ccc}} .note{{background:#fff7e0;padding:10px;border-left:4px solid #e0a800}}
</style></head><body>
<h1>TacticEye: Match Report</h1>
<p class="sub">Clip length: {total_s:.0f} s | Source: fixed top-view camera | Pitch assumed 105 x 68 m</p>

<h2>Team shape</h2>
<table><tr><th>Metric</th><th>Dark</th><th>White</th></tr>{stat_rows}</table>

<h2>Average positions</h2>
<img src="data:image/png;base64,{b64(D + '/avg_positions.png')}">

<h2>Heatmaps</h2>
<div class="row">
<div><h3>Dark</h3><img src="data:image/png;base64,{b64(D + '/heatmap_team0_dark.png')}"></div>
<div><h3>White</h3><img src="data:image/png;base64,{b64(D + '/heatmap_team1_white.png')}"></div>
</div>

<h2>Most distance covered (top 10 tracks)</h2>
{top.to_html(index=False)}

<p class="note"><b>Method notes:</b> players detected via background subtraction (static camera),
tracked with Hungarian matching, mapped to pitch metres using a 4-corner homography.
Teams from jersey colour (Lab k-means). Roles are depth-based (relative to own goal), not true formation detection.
Distances are approximate: a player may appear under more than one track ID.</p>
</body></html>"""

open(f"{D}/match_report.html", "w", encoding="utf-8").write(html)
print("saved", f"{D}/match_report.html")