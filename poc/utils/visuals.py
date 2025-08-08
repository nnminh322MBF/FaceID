import cv2

def draw_tracked_person(frame, bbox, track_id, name=None, confidence=None):
    """
    Vẽ bounding box và thông tin lên khung hình.
    """
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0) # Green
    
    # Vẽ bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Chuẩn bị text
    label = f"ID: {track_id}"
    if name:
        conf_str = f"{confidence:.2f}" if confidence else ""
        label = f"{name} ({conf_str}) - ID: {track_id}"
        
    # Vẽ nền cho text
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
    
    # Vẽ text
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)