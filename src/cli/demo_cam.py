# -*- coding: utf-8 -*-
import cv2
import argparse

from src.app.pipeline import FaceIDPipeline
# from some_utils.drawing import draw_results # Hàm tiện ích để vẽ bbox và id

def main():
    """
    Mục đích:
        Chạy một bản demo pipeline với input từ webcam hoặc file video.
        Đây là điểm bắt đầu (entrypoint) để người dùng cuối chạy ứng dụng.
    """
    parser = argparse.ArgumentParser(description="Face ID Tracking and Recognition Demo")
    parser.add_argument("--detector_config", type=str, required=True, help="Path to detector config yaml.")
    parser.add_argument("--tracker_config", type=str, required=True, help="Path to tracker config yaml.")
    parser.add_argument("--source", type=str, default="0", help="Video source (webcam id or video file path).")
    parser.add_argument("--use_rknn", action="store_true", help="Use RKNN backend instead of ONNX.")
    args = parser.parse_args()

    # Khởi tạo pipeline
    pipeline = FaceIDPipeline(
        detector_config_path=args.detector_config,
        tracker_config_path=args.tracker_config,
        use_rknn=args.use_rknn
    )

    # Mở nguồn video
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Xử lý frame
        results = pipeline.process_frame(frame)

        # Vẽ kết quả lên frame
        # output_frame = draw_results(frame, results)

        # Hiển thị
        # cv2.imshow("FaceID Demo", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()