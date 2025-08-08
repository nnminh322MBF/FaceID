# -*- coding: utf-8 -*-
import numpy as np

class ArcFace:
    """
    Mục đích:
        Đóng gói logic cho model ArcFace. Lớp này nhận vào một ảnh khuôn mặt
        đã được crop và trả về vector đặc trưng (embedding).
    """

    def __init__(self):
        """
        Mục đích:
            Khởi tạo model ArcFace.
        """
        pass

    def get_embedding(self, cropped_face_image: np.ndarray) -> np.ndarray:
        """
        Mục đích:
            Trích xuất vector đặc trưng 512 chiều từ một ảnh khuôn mặt.

        Tham số:
            cropped_face_image (np.ndarray): Ảnh khuôn mặt đã được crop và align.
                                             Ảnh này cần được tiền xử lý (resize về 112x112,
                                             chuẩn hóa) trước khi đưa vào hàm này.

        Trả về:
            np.ndarray: Vector đặc trưng (embedding) của khuôn mặt, shape (512,).
                        Vector này đã được chuẩn hóa (L2 normalized).
        
        Logic chính:
            1. (Bên ngoài lớp này) Load model ArcFace qua backend.
            2. (Trong lớp này) Hàm này sẽ nhận ảnh đã được tiền xử lý.
            3. (Bên ngoài lớp này) Chạy inference qua backend để lấy output thô.
            4. (Trong lớp này) Output thô chính là embedding, thực hiện L2 normalization
               trên vector này.
        """
        pass