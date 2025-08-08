# core/person_tracker.py
import numpy as np
from ultralytics import YOLO

# 👉 import đúng cho BoxMOT 15.x
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

class PersonTracker:
    def __init__(self, config):
        self.model = YOLO(config['yolo_model_path'])
        self.conf_thres = float(config['yolo_confidence_threshold'])
        self.iou_thres  = float(config.get('yolo_iou_threshold', 0.45))
        self.imgsz      = int(config.get('img_size', 640))
        self.device     = config.get('device', 'cpu')

        # Khởi tạo ByteTrack (API mới)
        self.tracker = ByteTrack(
            min_conf   = float(config.get('min_conf', 0.1)),
            track_thresh = float(config.get('track_thresh', 0.45)),
            match_thresh = float(config.get('match_thresh', 0.8)),
            track_buffer = int(config.get('track_buffer', 25)),
            frame_rate   = int(config.get('fps', 30)),
            per_class    = False,
        )

    def _make_dets(self, results):
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy()                 # (N,4)
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)  # (N,1)
        cls  = boxes.cls.cpu().numpy().reshape(-1, 1)   # (N,1)
        # chỉ lấy person (COCO=0). Nếu muốn theo tất cả class thì bỏ filter này
        mask = (cls.flatten() == 0)
        xyxy, conf, cls = xyxy[mask], conf[mask], cls[mask]

        if xyxy.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.hstack([xyxy, conf, cls]).astype(np.float32)

    def track(self, frame):
        results = self.model.predict(
            source=frame, imgsz=self.imgsz,
            conf=self.conf_thres, iou=self.iou_thres,
            verbose=False, device=self.device
        )
        dets = self._make_dets(results)

        # ByteTrack update: trả về ndarray [x1,y1,x2,y2,id,conf,cls,det_ind]
        tracks = self.tracker.update(dets, frame)

        out = []
        if tracks is None or len(tracks) == 0:
            return out
        # tracks là np.ndarray
        for row in tracks:
            x1, y1, x2, y2, tid = map(int, row[:5])
            out.append([x1, y1, x2, y2, tid])
        return out
