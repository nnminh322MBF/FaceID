# -*- coding: utf-8 -*-
import numpy as np

def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Mục đích:
        Thực hiện thuật toán Non-Maximum Suppression (NMS) để lọc các bounding box bị trùng lặp.
        Đây là một hàm tiện ích chung, có thể được sử dụng bởi bất kỳ model detector nào.

    Tham số:
        boxes (np.ndarray): Mảng chứa tọa độ các bounding box, shape (N, 4).
                            Format có thể là (x1, y1, x2, y2) hoặc (cx, cy, w, h).
                            Cần có xử lý nội bộ để đồng nhất.
        scores (np.ndarray): Mảng chứa điểm tin cậy (confidence score) tương ứng với mỗi box, shape (N,).
        iou_threshold (float): Ngưỡng IoU (Intersection over Union). Các box có IoU lớn hơn
                               ngưỡng này với box có score cao hơn sẽ bị loại bỏ.

    Trả về:
        np.ndarray: Mảng chứa chỉ số (indices) của các box được giữ lại sau khi lọc.
    """
    pass