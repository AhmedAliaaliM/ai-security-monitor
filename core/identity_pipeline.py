"""
Pipeline B: Identity - face detection + recognition.
Uses insightface's 'buffalo_l' model pack (RetinaFace detector + ArcFace
recognition, both pretrained - no training needed).
"""

import json
import os
import numpy as np

MATCH_THRESHOLD = 0.45  # cosine similarity - tune based on testing


class FaceDatabase:
    def __init__(self, path: str):
        self.path = path
        self.entries = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                raw = json.load(f)
            self.entries = {name: np.array(vec, dtype=np.float32) for name, vec in raw.items()}

    def save(self):
        raw = {name: vec.tolist() for name, vec in self.entries.items()}
        with open(self.path, "w") as f:
            json.dump(raw, f)

    def add(self, name: str, embedding: np.ndarray):
        self.entries[name] = embedding.astype(np.float32)
        self.save()

    def remove(self, name: str):
        if name in self.entries:
            del self.entries[name]
            self.save()

    def match(self, embedding: np.ndarray, threshold: float = MATCH_THRESHOLD):
        if not self.entries:
            return None, 0.0
        best_name = None
        best_sim = -1.0
        query = embedding / (np.linalg.norm(embedding) + 1e-8)
        for name, vec in self.entries.items():
            ref = vec / (np.linalg.norm(vec) + 1e-8)
            sim = float(np.dot(query, ref))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim


class IdentityPipeline:
    def __init__(self, db_path: str = "known_faces.json", det_size=(640, 640)):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=det_size)
        self.db = FaceDatabase(db_path)

    def process_frame(self, frame_bgr):
        faces = self.app.get(frame_bgr)
        results = []
        for face in faces:
            name, sim = self.db.match(face.embedding)
            results.append({
                "bbox": [float(x) for x in face.bbox],
                "matched_name": name,
                "similarity": round(sim, 3),
            })
        return results

    def enroll_from_image(self, name: str, frame_bgr):
        faces = self.app.get(frame_bgr)
        if not faces:
            return False
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        self.db.add(name, largest.embedding)
        return True

    def decide(self, frame_results, unauthorized_label="UNAUTHORIZED"):
        if not frame_results:
            return {"status": "no_face", "detail": None}
        matched = [r for r in frame_results if r["matched_name"] is not None]
        if matched:
            best = max(matched, key=lambda r: r["similarity"])
            return {"status": "authorized", "detail": best}
        best_unknown = max(frame_results, key=lambda r: r["similarity"])
        return {"status": unauthorized_label, "detail": best_unknown}