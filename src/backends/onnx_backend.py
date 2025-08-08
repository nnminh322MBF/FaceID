# -*- coding: utf-8 -*-
from typing import List, Any
import numpy as np
# import onnxruntime # Sẽ được import thật

class ONNXBackend:
    """
    Mục đích:
        Đây là lớp trừu tượng (abstraction layer) để làm việc với các model ONNX.
        Nó chịu trách nhiệm load model từ file .onnx và thực hiện inference.
        Pipeline chính sẽ gọi backend này mà không cần biết chi tiết về ONNX Runtime.
    """

    def __init__(self, model_path: str):
        """
        Mục đích:
            Khởi tạo backend bằng cách load một model ONNX.

        Tham số:
            model_path (str): Đường dẫn đến file model .onnx.
        """
        self.model_path = model_path
        # self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        # self.input_name = self.session.get_inputs()[0].name
        # self.output_names = [output.name for output in self.session.get_outputs()]
        pass

    def run(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        Mục đích:
            Thực hiện inference trên dữ liệu đầu vào.

        Tham số:
            input_data (np.ndarray): Một tensor numpy đã được tiền xử lý (resize, normalize)
                                     phù hợp với yêu cầu của model.
                                     Thường có shape (1, 3, height, width).

        Trả về:
            List[np.ndarray]: Một danh sách các tensor output thô từ model.
                              Số lượng và shape của các tensor này phụ thuộc vào model cụ thể.
        """
        pass