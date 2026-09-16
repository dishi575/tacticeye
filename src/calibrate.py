"""
Standard pitch reference points (meters), origin = one corner of the full pitch,
x = along the length (0-105), y = along the width (0-68):

  Corner flags:            (0,0)  (105,0)  (0,68)  (105,68)
  Penalty box corners:     left box: (0,13.84) (16.5,13.84) (0,54.16) (16.5,54.16)
                            right box: (88.5,13.84) (105,13.84) (88.5,54.16) (105,54.16)
  Penalty spot:             left: (11,34)   right: (94,34)
  Center spot:               (52.5,34)
  Halfway line touching sidelines: (52.5,0)  (52.5,68)

Pick 4 points you can clearly see and click them in this order, top-left to
bottom-right doesn't matter as long as click order matches the order you type
their real-world coordinates in below.
"""
import cv2
import json
import argparse

points_px = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_px) < 4:
        points_px.append([x, y])
        print(f"Point {len(points_px)}: pixel ({x},{y})")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, str(len(points_px)), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("frame", img)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="frame.jpg")
    ap.add_argument("--out", default="homography_points.json")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not open {args.image}")

    cv2.imshow("frame", img)
    cv2.setMouseCallback("frame", click_event)
    print("Click 4 points on the pitch markings in the window. Press any key when done.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points_px) != 4:
        raise SystemExit(f"Need exactly 4 points, got {len(points_px)}. Run again.")

    points_world = []
    print("\nNow type the real-world (x,y) in meters for each point you just clicked, in the SAME order.")
    print("See the comment at the top of this file for standard pitch reference coordinates.\n")
    for i in range(4):
        raw = input(f"Point {i+1} real-world x,y (e.g. 0,13.84): ")
        x_str, y_str = raw.split(",")
        points_world.append([float(x_str), float(y_str)])

    with open(args.out, "w") as f:
        json.dump({"pixel": points_px, "world": points_world}, f, indent=2)

    print(f"Saved {args.out}")
