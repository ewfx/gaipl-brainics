# telemetry_recommender.py
import json
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TelemetryRecommender:
    def __init__(self, telemetry_file):
        self.telemetry_data = json.load(open(telemetry_file))
        self.entries = [entry['description'] for entry in self.telemetry_data]

        self.vectorizer = TfidfVectorizer()
        self.embeddings = self.vectorizer.fit_transform(self.entries).toarray()
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings).astype('float32'))

    def recommend(self, query):
        query_vec = self.vectorizer.transform([query]).toarray().astype('float32')
        distances, indices = self.index.search(query_vec, k=3)
        return [self.telemetry_data[i] for i in indices[0]]
