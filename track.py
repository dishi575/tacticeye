import argparse
import csv
import os
from ultralytics import YOLO

PERSON_CLASS = 0
BALL_CLASS = 32  # COCO "sports ball"

def run(video_path, out_dir, model_name="yolov8n.pt", conf=0.25, every_n=1, imgsz=640):
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(model_name)

    csv_path = os.path.join(out_dir, "tracks.csv")
    out_video_path = os.path.join(out_dir, "annotated.mp4")

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2", "cx", "cy"])

    results_gen = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        classes=[PERSON_CLASS, BALL_CLASS],
        conf=conf,
        imgsz=imgsz,
        vid_stride=every_n,
        persist=True,
        stream=True,
        save=True,
        project=out_dir,
        name="predict",
        exist_ok=True,
        device="cpu",
    )

    frame_idx = 0
    for r in results_gen:
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            xyxy = r.boxes.xyxy.cpu().tolist()
            for tid, cls, cf, (x1, y1, x2, y2) in zip(ids, clss, confs, xyxy):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                writer.writerow([frame_idx, tid, cls, round(cf, 3),
                                  round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                                  round(cx, 1), round(cy, 1)])
        frame_idx += 1

    csv_file.close()
    print(f"Tracking data: {csv_path}")
    print(f"Annotated output saved under: {os.path.join(out_dir, 'predict')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to input .mp4")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt is fastest on CPU")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every_n", type=int, default=1, help="process every Nth frame (raise this on CPU)")
    ap.add_argument("--imgsz", type=int, default=640, help="lower to 480/416 for more speed")
    args = ap.parse_args()

    run(args.video, args.out, args.model, args.conf, args.every_n, args.imgsz)
