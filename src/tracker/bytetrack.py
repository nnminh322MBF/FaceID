# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np

class ByteTrackWrapper:
    """
    Mục đích:
        Một lớp "wrapper" bao bọc thư viện ByteTrack gốc.
        Mục đích là để cung cấp một giao diện (interface) nhất quán cho pipeline,
        giúp việc thay đổi tracker (ví dụ từ ByteTrack sang Sort) trở nên dễ dàng hơn.
    """

    def __init__(self, track_thresh: float, track_buffer: int, match_thresh: float, frame_rate: int):
        """
        Mục đích:
            Khởi tạo tracker với các tham số của ByteTrack.

        Tham số:
            track_thresh (float): Ngưỡng tin cậy để khởi tạo một track mới.
            track_buffer (int): Số frame mà một track "mất dấu" sẽ được giữ lại trước khi bị xóa.
            match_thresh (float): Ngưỡng IoU để khớp detection với track đã có.
            frame_rate (int): Tốc độ khung hình của video.
        """
        # from yolox.tracker.byte_tracker import STrack, BYTETracker # Import thật
        # self.tracker = BYTETracker(...)
        pass

    def update(self, detections: np.ndarray, image_info: Tuple[int, int]) -> np.ndarray:
        """
        Mục đích:
            Cập nhật trạng thái của tracker với các detection mới từ frame hiện tại.

        Tham số:
            detections (np.ndarray): Mảng các detection từ model detector.
                                     Shape: (num_detections, 5+).
                                     Format: [x1, y1, x2, y2, score, ...].
            image_info (Tuple[int, int]): Kích thước (height, width) của ảnh.

        Trả về:
            np.ndarray: Mảng chứa thông tin các track đang hoạt động.
                        Shape: (num_active_tracks, 5).
                        Format: [x1, y1, x2, y2, track_id].
        """
        pass