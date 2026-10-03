from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_summary():
    r = client.get("/api/summary").json()
    assert r["tracks"] > 0 and r["frames"] > 0
    assert set(r["team_names"]) == {"red", "green"}
    assert r["bounds"]["x_min"] < r["bounds"]["x_max"]


def test_frame_and_404():
    first = client.get("/api/summary").json()["first_frame"]
    r = client.get(f"/api/frames/{first}")
    assert r.status_code == 200 and r.json()["players"]
    assert client.get("/api/frames/99999").status_code == 404


def test_tracks_range_limit_and_team_filter():
    assert client.get("/api/tracks?start=0&end=500").status_code == 400
    rows = client.get("/api/tracks?start=0&end=299&team=red").json()
    assert rows and all(x["team"] == "red" for x in rows)
    assert client.get("/api/tracks?team=blue").status_code == 404


def test_players():
    p = client.get("/api/players").json()
    assert p and {"track_id", "team", "distance_m", "avg_speed_ms"} <= set(p[0])


def test_images_and_report():
    assert client.get("/api/heatmap/red").headers["content-type"] == "image/png"
    assert client.get("/api/heatmap/blue").status_code == 404
    assert client.get("/api/avg-positions").status_code == 200
    assert "TacticEye" in client.get("/api/report").text
