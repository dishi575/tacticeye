import cv2
import json
import argparse
import numpy as np

# Standard pitch 105 x 68 m. Click order: TL, TR, BR, BL
CORNERS_WORLD = [[0, 0], [105, 0], [105, 68], [0, 68]]

points_px = []   # ORIGINAL frame coordinates
disp = None
scale = 1.0

def redraw():
    global disp
    disp = base.copy()
    for i, (ox, oy) in enumerate(points_px):
        x, y = int(ox * scale), int(oy * scale)
        cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(disp, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_px) < N_POINTS:
        points_px.append([x / scale, y / scale])   # back to original coords
        print(f"Point {len(points_px)}: original pixel ({x/scale:.0f},{y/scale:.0f})")
        redraw()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="frame.jpg")
    ap.add_argument("--out", default="homography_points.json")
    ap.add_argument("--mode", choices=["corners", "custom"], default="corners",
                    help="corners = click 4 pitch corners (TL,TR,BR,BL); "
                         "custom = click N landmarks and type their metres")
    ap.add_argument("--n", type=int, default=4, help="points for custom mode")
    args = ap.parse_args()

    N_POINTS = 4 if args.mode == "corners" else args.n

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not open {args.image}")

    h, w = img.shape[:2]
    scale = min(1600 / w, 900 / h, 1.0)
    base = cv2.resize(img, None, fx=scale, fy=scale)
    disp = base.copy()
    print(f"Frame size: {w}x{h}, display scale: {scale:.2f}")

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("frame", click_event)

    if args.mode == "corners":
        print("Click pitch corners in order: 1=Top-Left, 2=Top-Right, "
              "3=Bottom-Right, 4=Bottom-Left")
    print("'z' = undo last point, Enter = done")

    while True:
        cv2.imshow("frame", disp)
        k = cv2.waitKey(20) & 0xFF
        if k == 13:
            break
        if k == ord("z") and points_px:
            points_px.pop()
            redraw()
    cv2.destroyAllWindows()

    if len(points_px) != N_POINTS:
        raise SystemExit(f"Need {N_POINTS} points, got {len(points_px)}. Run again.")

    if args.mode == "corners":
        points_world = CORNERS_WORLD
    else:
        points_world = []
        for i in range(N_POINTS):
            raw = input(f"Point {i+1} real-world x,y in metres: ")
            x_str, y_str = raw.split(",")
            points_world.append([float(x_str), float(y_str)])

    with open(args.out, "w") as f:
        json.dump({"pixel": points_px, "world": points_world}, f, indent=2)
    print(f"Saved {args.out}")

    # Sanity check: warp frame to top-down view, 10 px = 1 m
    src = np.array(points_px, dtype=np.float32)
    dst = np.array(points_world, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    S = np.diag([10, 10, 1]).astype(np.float64)
    out = cv2.warpPerspective(img, S @ H, (1050, 680))
    cv2.imwrite("check.jpg", out)
    print("Saved check.jpg: pitch seedha rectangle dikhna chahiye")