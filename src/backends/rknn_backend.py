# -*- coding: utf-8 -*-
from typing import List, Any
import numpy as np
# from rknnlite.api import RKNNLite # Sẽ được import thật

class RKNNBackend:
    """
    Mục đích:
        Đây là lớp trừu tượng để làm việc với các model đã được convert sang định dạng RKNN.
        Chịu trách nhiệm load model .rknn và chạy inference trên NPU của Rockchip.
        Lớp này giúp che giấu sự phức tạp của thư viện rknnlite.
    """

    def __init__(self, model_path: str):
        """
        Mục đích:
            Khởi tạo backend bằng cách load một model .rknn.

        Tham số:
            model_path (str): Đường dẫn đến file model .rknn.
        """
        self.model_path = model_path
        # self.rknn = RKNNLite()
        # self.rknn.load_rknn(model_path)
        # self.rknn.init_runtime()
        pass

    def run(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        Mục đích:
            Thực hiện inference trên NPU.

        Tham số:
            input_data (np.ndarray): Một tensor numpy đã được tiền xử lý.
                                     Lưu ý: RKNN thường yêu cầu input ở định dạng
                                     NHWC và kiểu dữ liệu int8 nếu đã lượng tử hóa.

        Trả về:
            List[np.ndarray]: Danh sách các tensor output thô từ model.
        """
        pass