import insightface
import numpy as np
import os
import cv2
from sklearn.preprocessing import normalize
from core.face_vectorstore import FaceVectorStore


class FaceRecognizer:
    def __init__(self, config):
        self.model_name = config["arcface_model"]
        self.db_path = config["face_db_path"]
        self.recognition_threshold = float(config["face_recognition_threshold"])

        insight_root = os.path.abspath(
            config.get("insightface_root", "./models/insightface")
        )
        os.makedirs(insight_root, exist_ok=True)

        self.model = insightface.app.FaceAnalysis(
            name=self.model_name,
            root=insight_root,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.model.prepare(ctx_id=-1, det_size=(640, 640))

        self.vector_store = FaceVectorStore(db_path=self.db_path, embedding_dim=512)

    def create_database(self, known_faces_dir):
        print("Creating face database...")
        all_embeddings = []
        all_labels = []
        for person_name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    img = cv2.imread(image_path)
                    if img is None:
                        continue

                    faces = self.model.get(img)
                    if faces and len(faces) == 1:
                        emb = (
                            normalize(faces[0].normed_embedding.reshape(1, -1))
                            .astype("float32")
                            .flatten()
                        )
                        all_embeddings.append(emb)
                        all_labels.append(person_name)
                    else:
                        print(
                            f"Warning: No face or multiple faces detected in {image_path}"
                        )
        if all_embeddings:
            self.vector_store.add(embeddings=all_embeddings, labels=all_labels)
            self.vector_store.save()

    def recognize(self, frame, person_bbox):
        x1, y1, x2, y2 = person_bbox
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None, None

        faces = self.model.get(person_crop)
        if not faces:
            return None, None

        main_face = max(
            faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )
        embedding = normalize(main_face.normed_embedding.reshape(1, -1)).astype(
            "float32"
        )

        if self.vector_store.get_ntotal() > 0:
            cosine_similarity, label = self.vector_store.search(
                embedding_query=embedding
            )

            if cosine_similarity > self.recognition_threshold:
                return label, cosine_similarity

        return "Unknown", None
    
    def add_new_employee(self, employee_name, employee_image):
        if isinstance(employee_image, str):
            if os.path.exists(employee_image):
                img = cv2.imread(employee_image)
            else:
                print(f"Lỗi: Đường dẫn file không tồn tại: {employee_image}")
                return 

        img = employee_image

        face = self.model.get(img=img)
        if not face or len(face) != 1:
            print("Cần chính xác một khuôn mặt trong ảnh để thêm nhân viên.")
            return
        
        main_face = face[0]
        embedding = main_face.normed_embedding.reshape(1,-1).astype("float32")

        self.vector_store.add(embeddings=embedding, labels=employee_name)
        self.vector_store.save()
        print(f"Đã thêm nhân viên {employee_name} thành công.")