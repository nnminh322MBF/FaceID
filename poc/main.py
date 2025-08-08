import cv2
import yaml
from core.person_tracker import PersonTracker
from core.face_recognizer import FaceRecognizer
from utils.visuals import draw_tracked_person
import os

def _open_video(src):
    # Cho phép "0" (string) -> 0 (int)
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    return cap

def main():
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    tracker = PersonTracker(config)
    recognizer = FaceRecognizer(config)

    known_faces_dir = 'data/known_faces'
    if not os.path.exists(config['face_db_path']) and os.path.exists(known_faces_dir):
        print("Face database not found. Creating a new one.")
        recognizer.create_database(known_faces_dir)
    
    src = config['video_source']
    if isinstance(src, str) and not src.isdigit() and not os.path.exists(src):
        print(f"Error: Video file not found at {src}")
        return
        
    cap = _open_video(src)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{src}'")
        return

    writer = None
    out_path = config.get('output_video_path')
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Lấy 1 frame để xác định size/fps nếu cần
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read first frame.")
            return
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # quay lại đầu video

    tracked_identities = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracked_boxes = tracker.track(frame)
        current_tracked_ids = set()

        for bbox_with_id in tracked_boxes:
            *bbox, track_id = bbox_with_id
            current_tracked_ids.add(track_id)

            if track_id not in tracked_identities or tracked_identities[track_id][0] == "Unknown":
                name, confidence = recognizer.recognize(frame, bbox)
                if name:
                    tracked_identities[track_id] = (name, confidence)
            
            display_name, display_conf = tracked_identities.get(track_id, ("Unknown", None))
            draw_tracked_person(frame, bbox, track_id, display_name, display_conf)

        # dọn ID đã rời khung
        for obs_id in list(tracked_identities.keys()):
            if obs_id not in current_tracked_ids:
                tracked_identities.pop(obs_id, None)

        # Lưu ảnh ra file, không dùng cv2.imshow
        # cv2.imwrite('output_frame.png', frame)
        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()