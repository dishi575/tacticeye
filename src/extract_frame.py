import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--out", default="frame.jpg")
ap.add_argument("--frame_num", type=int, default=0)
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_num)
ok, frame = cap.read()
if not ok:
    print("Could not read that frame, try a different --frame_num")
else:
    cv2.imwrite(args.out, frame)
    print(f"Saved {args.out} — open it and pick 4 clear points on the pitch markings.")
cap.release()
