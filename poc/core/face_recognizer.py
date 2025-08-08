import insightface
import numpy as np
import faiss
import os
import cv2
from sklearn.preprocessing import normalize

class FaceRecognizer:
    def __init__(self, config):
        self.model_name = config['arcface_model']
        self.db_path = config['face_db_path']
        self.labels_path = config.get('face_labels_path', os.path.splitext(self.db_path)[0] + ".labels.txt")
        self.recognition_threshold = float(config['face_recognition_threshold'])

        insight_root = os.path.abspath(config.get('insightface_root', './models/insightface'))
        os.makedirs(insight_root, exist_ok=True)

        self.model = insightface.app.FaceAnalysis(
            name=self.model_name,
            root=insight_root,                     # <<< quan trọng
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection','recognition']
        )
        self.model.prepare(ctx_id=-1, det_size=(640, 640))

        # Initialize Faiss index
        self.embedding_dim = 512
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.labels = []
        
        # Load existing database if it exists
        if os.path.exists(self.db_path) and os.path.exists(self.labels_path):
            self.load_database()

    def create_database(self, known_faces_dir):
        print("Creating face database...")
        all_embeddings = []
        
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
                        emb = normalize(faces[0].normed_embedding.reshape(1, -1)).astype('float32').flatten()
                        all_embeddings.append(emb)
                        self.labels.append(person_name)
                    else:
                        print(f"Warning: No face or multiple faces detected in {image_path}")

        if all_embeddings:
            embeddings_matrix = np.array(all_embeddings, dtype='float32')
            self.index.add(embeddings_matrix)
            self.save_database()
            print(f"Database created with {self.index.ntotal} faces.")

    def save_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        faiss.write_index(self.index, self.db_path)
        with open(self.labels_path, 'w', encoding='utf-8') as f:
            for label in self.labels:
                f.write(f"{label}\n")

    def load_database(self):
        self.index = faiss.read_index(self.db_path)
        with open(self.labels_path, 'r', encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines()]
        print(f"Loaded database with {self.index.ntotal} faces.")

    def recognize(self, frame, person_bbox):
        x1, y1, x2, y2 = person_bbox
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None, None

        faces = self.model.get(person_crop)
        if not faces:
            return None, None
            
        # Lấy khuôn mặt lớn nhất
        main_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        embedding = normalize(main_face.normed_embedding.reshape(1, -1)).astype('float32')

        if self.index.ntotal > 0:
            distances, indices = self.index.search(embedding, 1)
            l2_sq = float(distances[0][0])
            cosine_similarity = 1.0 - (l2_sq / 2.0)  # <-- FIX Ở ĐÂY

            if cosine_similarity > self.recognition_threshold:
                label_index = int(indices[0][0])
                return self.labels[label_index], cosine_similarity
        
        return "Unknown", None
