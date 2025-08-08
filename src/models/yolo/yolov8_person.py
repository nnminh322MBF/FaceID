# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np

# Giả sử các hàm này đã được implement và import
from src.postprocess.nms import non_max_suppression
# from src.postprocess.yolo_decode import decode_yolo_output

class YOLOv8Person:
    """
    Mục đích:
        Đóng gói logic xử lý cho model YOLOv8 phát hiện người.
        Lớp này nhận output thô từ backend, thực hiện hậu xử lý (post-processing)
        để trả về kết quả detection cuối cùng.
    """

    def __init__(self, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Mục đích:
            Khởi tạo detector với các ngưỡng cần thiết.

        Tham số:
            conf_threshold (float): Ngưỡng tin cậy. Chỉ những box có score cao hơn
                                    ngưỡng này mới được xem xét.
            iou_threshold (float): Ngưỡng IoU cho NMS.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # Các tham số khác của YOLO: strides, anchors (nếu cần)
        pass

    def postprocess(self, raw_outputs: List[np.ndarray], original_image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Mục đích:
            Chuyển đổi output thô từ model thành danh sách các bounding box đã được lọc.

        Tham số:
            raw_outputs (List[np.ndarray]): Output trực tiếp từ backend (ONNX/RKNN).
                                            Thường là một tensor có shape (1, num_classes + 4, num_proposals).
            original_image_shape (Tuple[int, int]): Kích thước (height, width) của ảnh gốc
                                                    để scale tọa độ box về đúng vị trí.

        Trả về:
            np.ndarray: Mảng các detection cuối cùng. Mỗi hàng là một detection.
                        Shape: (num_detections, 6).
                        Format: [x1, y1, x2, y2, score, class_id].

        Logic chính:
            1. Decode output: Dùng `decode_yolo_output` để chuyển đổi tensor output thành
               danh sách các box, scores, và class_ids.
            2. Lọc theo confidence: Loại bỏ các box có score thấp hơn `self.conf_threshold`.
            3. Áp dụng NMS: Dùng `non_max_suppression` để loại bỏ các box trùng lặp.
            4. Scale tọa độ: Chuyển tọa độ box từ kích thước của model input (ví dụ 640x640)
               về kích thước của ảnh gốc.
        """
        pass