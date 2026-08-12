"""
Pipeline B: Identity — face detection + recognition.

Uses insightface's 'buffalo_l' model pack, which bundles:
  - a RetinaFace-based detector (det_10g.onnx)
  - an ArcFace recognition model (w600k_r50.onnx) producing 512-dim embeddings

Both are pretrained — no training/fine-tuning needed for this pipeline.
The only "training" here is enrollment: adding known faces to a small
local database so new faces can be matched against them.

Face database is stored as a JSON file: {name: [512 floats], ...}
This is fine for a handful of authorized people; swap for a proper
vector DB if this ever needs to scale to hundreds of identities.
"""

import json
import os
import numpy as np

MATCH_THRESHOLD = 0.45  # cosine similarity — tune based on false-accept/false-reject testing


class FaceDatabase:
    def __init__(self, path: str):
        self.path = path
        self.entries = {}  # name -> np.array(512,)
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
        """
        Returns (name, similarity) for the best match above threshold,
        or (None, best_similarity) if nothing matches closely enough.
        """
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
    def __init__(self, db_path: str = "known_faces.json", det_size=(640, 640), match_threshold: float = MATCH_THRESHOLD):
        # Imported here (not at module top) so environments that only use
        # Pipeline A aren't forced to have insightface installed.
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=det_size)  # ctx_id=-1 -> CPU
        self.match_threshold = match_threshold
        self.db = FaceDatabase(db_path)

    def process_frame(self, frame_bgr):
        """
        Runs detection + recognition on one frame.
        Returns a list of dicts, one per detected face:
            {bbox, name_or_None, similarity}
        """
        faces = self.app.get(frame_bgr)
        results = []
        for face in faces:
            name, sim = self.db.match(face.embedding, threshold=self.match_threshold)
            results.append({
                "bbox": [float(x) for x in face.bbox],
                "matched_name": name,
                "similarity": round(sim, 3),
            })
        return results

    def enroll_from_image(self, name: str, frame_bgr):
        """
        Enrolls a new authorized person from a single clear photo.
        Returns True if a face was found and enrolled, False otherwise.
        For best results use a well-lit, front-facing, single-face photo.
        """
        faces = self.app.get(frame_bgr)
        if not faces:
            return False
        # If multiple faces are in the enrollment photo, use the largest
        # (most likely the intended subject, not someone in the background).
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        self.db.add(name, largest.embedding)
        return True

    def decide(self, frame_results, unauthorized_label="UNAUTHORIZED"):
        """
        Turns raw per-face results into the decision categories used by
        the rest of the pipeline (matches the DECISION & LOGIC block from
        the original architecture diagram).
        """
        if not frame_results:
            return {"status": "no_face", "detail": None}

        matched = [r for r in frame_results if r["matched_name"] is not None]
        if matched:
            best = max(matched, key=lambda r: r["similarity"])
            return {"status": "authorized", "detail": best}

        # faces present but none matched
        best_unknown = max(frame_results, key=lambda r: r["similarity"])
        return {"status": unauthorized_label, "detail": best_unknown}