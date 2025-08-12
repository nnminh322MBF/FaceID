import faiss
import os
import numpy as np
from typing import List


class FaceVectorStore:
    def __init__(self, db_path: str, embedding_dim: int):
        self.db_path = db_path
        self.labels_path = os.path.splitext(self.db_path)[0] + "label.txt"
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.labels = []

        if os.path.exists(self.db_path) and os.path.exists(self.labels_path):
            self.__load()

    def __load(self):
        self.index = faiss.read_index(self.db_path)
        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f.readlines()]

    def add(self, embeddings: np.ndarray, labels: List[str]):
        if embeddings.shape[0] != len(labels):
            raise ValueError("embedding number must be equal to label length")

        self.index.add(embeddings)
        self.labels.append(labels)

    def search(self, embedding_query: np.ndarray, top_k: int = 1):
        if self.index.ntotal == 0:
            return None, 0.0

        distances, indices = self.index.search(embedding_query, top_k)
        L2_dis = distances[0][0]
        cosine_similarity = 1 - (L2_dis / 2.0)
        label_index = self.labels[int(indices.item())]

        return cosine_similarity, label_index
    
    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        faiss.write_index(self.index, self.db_path)
        with open(self.labels_path, 'w', encoding='utf-8') as f:
            for label in self.labels:
                f.write(f"{label}\n")
    
    def get_ntotal(self):
        return self.index.ntotal
    

