import os
import cv2
import torch
import clip
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from types import SimpleNamespace
from PIL import Image
import sys

sys.path.append("D:/MCMOT/ByteTrack")
from yolox.tracker.byte_tracker import BYTETracker

if not hasattr(np, 'float'):
    np.float = np.float64

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def extract_and_track(video_path, output_dir="data/tracking/", conf=0.2, frame_skip=30, visualize=False):
    model = YOLO("yolov8s.pt")  # Khuyến khích dùng yolov8s.pt cho kết quả tốt hơn
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được video {video_path}")
        return []

    frame_id = 0
    basename = Path(video_path).stem

    args = SimpleNamespace(
        track_thresh=conf,
        track_buffer=30,
        match_thresh=0.7,
        mot20=False,
        min_box_area=0  # Fix: không bỏ lọc box nhỏ
    )

    tracker = BYTETracker(args)

    results_meta = []

    # Video writer
    if visualize:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip != 0:
            frame_id += 1
            continue

        results = model(frame)[0]
        dets = []

        # Filter only person class (YOLOv8 person usually has id 0)
        person_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]

        print(f"[Frame {frame_id}] Person detections: {len(person_boxes)}")

        for box in person_boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            score = float(box.conf[0])
            dets.append([x1, y1, x2, y2, score])  # Bỏ cls_id để tránh lỗi ByteTrack

            if visualize:
                label = model.names[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Convert to tensor
        if dets:
            dets = torch.tensor(dets, dtype=torch.float32).to(device)
        else:
            dets = torch.zeros((0, 5), dtype=torch.float32).to(device)

        img_info = frame.shape[:2]
        img_size = (frame.shape[0], frame.shape[1])

        online_targets = tracker.update(dets, img_info, img_size)

        print(f"[Frame {frame_id}] Tracked objects: {len(online_targets)}")

        for t in online_targets:
            tlwh = t.tlwh
            track_id = t.track_id
            x1, y1, w, h = map(int, tlwh)
            x2, y2 = x1 + w, y1 + h

            # Avoid empty crop
            if x1 >= x2 or y1 >= y2:
                continue

            obj_crop = frame[y1:y2, x1:x2]
            obj_path = f"{output_dir}/{basename}_{frame_id}_{track_id}.jpg"
            cv2.imwrite(obj_path, obj_crop)

            # CLIP feature extraction
            pil_img = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB))
            img_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = clip_model.encode_image(img_tensor)
                feat /= feat.norm(dim=-1, keepdim=True)
                feat = feat.cpu().numpy()[0]

            results_meta.append({
                "track_id": track_id,
                "path": obj_path,
                "video": video_path,
                "frame": frame_id,
                "bbox": [x1, y1, x2, y2],
                "clip_feat": feat.tolist()
            })

            if visualize:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        if visualize:
            if out is None:
                out = cv2.VideoWriter(f"{output_dir}/{basename}_tracked.mp4", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            out.write(frame)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1

    cap.release()
    if visualize and out is not None:
        out.release()
    cv2.destroyAllWindows()

    return results_meta
