# -*- coding: utf-8 -*-
from typing import Dict, Any, List
import numpy as np
import yaml

# Giả sử các class đã được import
from src.backends.onnx_backend import ONNXBackend
from src.backends.rknn_backend import RKNNBackend
from src.models.yolo.yolov8_person import YOLOv8Person
# ... import các models và trackers khác
from src.tracker.bytetrack import ByteTrackWrapper
from src.models.arcface import ArcFace

class FaceIDPipeline:
    """
    Mục đích:
        Đây là lớp "nhạc trưởng", điều phối toàn bộ pipeline từ đầu đến cuối.
        Nó chịu trách nhiệm load cấu hình, khởi tạo các thành phần (backend, model, tracker)
        và xử lý luồng dữ liệu qua từng frame.
    """

    def __init__(self, detector_config_path: str, tracker_config_path: str, use_rknn: bool = False):
        """
        Mục đích:
            Khởi tạo toàn bộ pipeline dựa trên các file cấu hình.

        Tham số:
            detector_config_path (str): Đường dẫn đến file `detector.yaml`.
            tracker_config_path (str): Đường dẫn đến file `tracker.yaml`.
            use_rknn (bool): Cờ để xác định dùng backend RKNN hay ONNX.
        """
        # 1. Load configs
        with open(detector_config_path) as f:
            self.detector_config = yaml.safe_load(f)
        with open(tracker_config_path) as f:
            self.tracker_config = yaml.safe_load(f)

        # 2. Load backend
        backend_class = RKNNBackend if use_rknn else ONNXBackend
        model_ext = ".rknn" if use_rknn else ".onnx"
        
        # 3. Load models (detector và recognizer)
        detector_model_path = self.detector_config['model_path'] + model_ext
        self.detector_backend = backend_class(detector_model_path)
        
        # Dựa vào config để chọn class model phù hợp
        detector_type = self.detector_config.get('type', 'yolov8_person')
        if detector_type == 'yolov8_person':
            self.detector = YOLOv8Person(**self.detector_config['params'])
        # ... else if cho retinaface, yolov8_face ...

        # Tương tự, load model ArcFace
        # self.arcface_backend = backend_class(...)
        # self.arcface = ArcFace()
        
        # 4. Load tracker
        self.tracker = ByteTrackWrapper(**self.tracker_config['bytetrack_params'])
        pass

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Mục đích:
            Xử lý một frame ảnh duy nhất để thực hiện tracking và nhận diện.

        Tham số:
            frame (np.ndarray): Frame ảnh đầu vào, định dạng BGR (từ OpenCV).

        Trả về:
            List[Dict[str, Any]]: Danh sách các đối tượng được track trong frame.
                                  Mỗi dictionary chứa thông tin:
                                  {
                                      "track_id": int,
                                      "bbox": [x1, y1, x2, y2],
                                      "face_embedding": np.ndarray (or None)
                                  }
        Logic chính:
            1. **Tiền xử lý ảnh**: Chuyển BGR -> RGB, resize ảnh về kích thước input của detector,
               chuẩn hóa (normalize).
            2. **Phát hiện (Detect)**:
               - `raw_outputs = self.detector_backend.run(preprocessed_frame)`
               - `detections = self.detector.postprocess(raw_outputs, frame.shape)`
            3. **Theo vết (Track)**:
               - `tracked_objects = self.tracker.update(detections, frame.shape)`
            4. **Nhận diện (Recognize) cho từng đối tượng được track**:
               - Lặp qua `tracked_objects`.
               - Với mỗi đối tượng, crop vùng ảnh `bbox` từ `frame` gốc.
               - Crop tiếp vùng mặt từ ảnh người (hoặc dùng 1 model face detector nhỏ).
               - Tiền xử lý ảnh mặt (resize 112x112, align, normalize).
               - Chạy model ArcFace để lấy embedding:
                 `embedding = self.arcface.get_embedding(...)`
            5. **Tập hợp kết quả**: Tạo danh sách dictionary output.
        """
        pass