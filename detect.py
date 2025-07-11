# detect.py
import os
import cv2
from ultralytics import YOLO
from pathlib import Path

def extract_objects(video_path, output_dir='data/objects/', conf=0.4, frame_skip=30):
    model = YOLO("yolov8n.pt")  # hoặc yolov8s.pt
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_frame_id = 0
    basename = Path(video_path).stem

    results_meta = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip != 0:
            frame_id += 1
            continue

        results = model(frame)[0]
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            if conf_score < conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj = frame[y1:y2, x1:x2]
            label = model.names[cls_id]
            obj_path = f"{output_dir}/{basename}_{frame_id}_{i}_{label}.jpg"
            cv2.imwrite(obj_path, obj)

            results_meta.append({
                "path": obj_path,
                "label": label,
                "video": video_path,
                "frame": frame_id,
                "bbox": [x1, y1, x2, y2]
            })

        frame_id += 1

    cap.release()
    return results_meta

