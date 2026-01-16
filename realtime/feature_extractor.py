import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python import core
import numpy as np
import os
from itertools import combinations


def _resolve_model_path(filename):
    """Return an existing path for model file by checking common locations.
    Checks project root, mediapipe/ under project root, current working dir, and absolute path.
    Falls back to the provided filename so MediaPipe can attempt to resolve it.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(repo_root, filename),
        os.path.join(repo_root, "mediapipe", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "mediapipe", filename),
        os.path.abspath(filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return filename

NUM_HAND_LANDMARKS = 21
FEATURES_PER_LM = 3
NUM_FACE_LANDMARKS = 468
FACE_KEY_POINTS = [33, 133, 160, 158, 144, 145, 153, 154, 155, 173, 362, 263, 387, 385, 373, 374, 380, 381, 382, 398, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 1, 2, 98, 327, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 152, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]

USE_DISTANCES = True
USE_ANGLES = True
USE_VELOCITY = True
USE_FACE = True
USE_HEAD_POSE = True

class StreamingFeatureExtractor:
    def __init__(self, num_hands=2):
        self.num_hands = num_hands
        self.prev_features = None
        self.hand_landmarker = None
        self.face_landmarker = None
        self.base_feature_dim = self._compute_base_dim()
        self.total_feature_dim = self.base_feature_dim * 2 if USE_VELOCITY else self.base_feature_dim

    def _compute_base_dim(self):
        dim = self.num_hands * NUM_HAND_LANDMARKS * FEATURES_PER_LM
        if USE_DISTANCES:
            dim += self.num_hands * (NUM_HAND_LANDMARKS * (NUM_HAND_LANDMARKS - 1)) // 2
        if USE_ANGLES:
            dim += self.num_hands * 4
        if USE_FACE:
            dim += len(FACE_KEY_POINTS) * FEATURES_PER_LM + 15
            if USE_HEAD_POSE:
                dim += 3
        return dim

    def normalize_landmarks(self, lm):
        lm = lm.reshape(-1, 3)
        wrist = lm[0].copy()
        lm -= wrist
        scale = np.linalg.norm(lm, axis=1).max()
        if scale > 0:
            lm /= scale
        return lm.flatten()

    def compute_distances(self, lm, num_landmarks):
        lm = lm.reshape(-1, 3)
        return np.array([np.linalg.norm(lm[i] - lm[j]) for i, j in combinations(range(num_landmarks), 2)])

    def compute_angles(self, lm):
        lm = lm.reshape(-1, 3)
        angles = []
        for i in [5, 9, 13, 17]:
            v1 = lm[i] - lm[0]
            v2 = lm[i + 1] - lm[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 * n2 == 0:
                angles.append(0.0)
            else:
                cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angles.append(np.arccos(cos))
        return np.array(angles)

    def compute_face_features(self, face_landmarks):
        if not face_landmarks or len(face_landmarks) == 0:
            return np.zeros(15)
        lm = face_landmarks[0]
        features = []
        left_eye_left = np.array([lm[33].x, lm[33].y, lm[33].z])
        left_eye_right = np.array([lm[133].x, lm[133].y, lm[133].z])
        right_eye_left = np.array([lm[362].x, lm[362].y, lm[362].z])
        right_eye_right = np.array([lm[263].x, lm[263].y, lm[263].z])
        features.extend([np.linalg.norm(left_eye_left - left_eye_right), np.linalg.norm(right_eye_left - right_eye_right), np.linalg.norm(left_eye_right - right_eye_left)])
        mouth_left = np.array([lm[61].x, lm[61].y, lm[61].z])
        mouth_right = np.array([lm[291].x, lm[291].y, lm[291].z])
        mouth_top = np.array([lm[13].x, lm[13].y, lm[13].z])
        mouth_bottom = np.array([lm[14].x, lm[14].y, lm[14].z])
        features.extend([np.linalg.norm(mouth_left - mouth_right), np.linalg.norm(mouth_top - mouth_bottom)])
        forehead = np.array([lm[10].x, lm[10].y, lm[10].z])
        chin = np.array([lm[152].x, lm[152].y, lm[152].z])
        left_cheek = np.array([lm[234].x, lm[234].y, lm[234].z])
        right_cheek = np.array([lm[454].x, lm[454].y, lm[454].z])
        features.extend([np.linalg.norm(forehead - chin), np.linalg.norm(left_cheek - right_cheek)])
        eye_center = (left_eye_left + right_eye_right) / 2
        mouth_center = (mouth_left + mouth_right) / 2
        features.extend([np.linalg.norm(eye_center - mouth_center), np.linalg.norm(np.array([lm[1].x, lm[1].y, lm[1].z]) - chin)])
        left_symmetry = np.linalg.norm(left_eye_left - mouth_left)
        right_symmetry = np.linalg.norm(right_eye_right - mouth_right)
        features.extend([left_symmetry, right_symmetry, abs(left_symmetry - right_symmetry)])
        jaw_left = np.array([lm[93].x, lm[93].y, lm[93].z])
        jaw_right = np.array([lm[323].x, lm[323].y, lm[323].z])
        features.append(np.linalg.norm(jaw_left - jaw_right))
        return np.array(features)
    
    def compute_head_pose(self, face_landmarks):
        if not face_landmarks or len(face_landmarks) == 0:
            return np.zeros(3)
        lm = face_landmarks[0]
        nose_tip = np.array([lm[1].x, lm[1].y, lm[1].z])
        chin = np.array([lm[152].x, lm[152].y, lm[152].z])
        left_eye = np.array([lm[33].x, lm[33].y, lm[33].z])
        right_eye = np.array([lm[263].x, lm[263].y, lm[263].z])
        return np.array([nose_tip[1] - chin[1], (left_eye[0] + right_eye[0]) / 2 - nose_tip[0], left_eye[1] - right_eye[1]])

    def extract(self, rgb_frame):
        from mediapipe.tasks.python.core import base_options
        if self.hand_landmarker is None:
            hand_model = _resolve_model_path("hand_landmarker.task")
            options = HandLandmarkerOptions(base_options=base_options.BaseOptions(model_asset_path=hand_model), num_hands=self.num_hands, min_hand_detection_confidence=0.5)
            self.hand_landmarker = HandLandmarker.create_from_options(options)
        if USE_FACE and self.face_landmarker is None:
            try:
                face_model = _resolve_model_path("face_landmarker.task")
                face_options = FaceLandmarkerOptions(base_options=base_options.BaseOptions(model_asset_path=face_model), num_faces=1, min_face_detection_confidence=0.5)
                self.face_landmarker = FaceLandmarker.create_from_options(face_options)
            except:
                pass
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_result = self.hand_landmarker.detect(mp_image)
        face_result = None
        if USE_FACE and self.face_landmarker:
            try:
                face_result = self.face_landmarker.detect(mp_image)
            except:
                pass
        frame_features = []
        if hand_result.hand_landmarks:
            hand_data = []
            for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                try:
                    handed = hand_result.handedness[idx][0].category_name
                except:
                    handed = "Left" if idx == 0 else "Right"
                lm = []
                for p in hand_lms:
                    lm.extend([p.x, p.y, p.z])
                lm = self.normalize_landmarks(np.array(lm))
                hand_data.append((handed, lm))
            hand_data = sorted(hand_data, key=lambda x: x[0])
            for _, lm in hand_data:
                frame_features.extend(lm)
                if USE_DISTANCES:
                    frame_features.extend(self.compute_distances(lm, NUM_HAND_LANDMARKS))
                if USE_ANGLES:
                    frame_features.extend(self.compute_angles(lm))
        expected_hand_features = self.num_hands * NUM_HAND_LANDMARKS * FEATURES_PER_LM
        if USE_DISTANCES:
            expected_hand_features += self.num_hands * (NUM_HAND_LANDMARKS * (NUM_HAND_LANDMARKS - 1)) // 2
        if USE_ANGLES:
            expected_hand_features += self.num_hands * 4
        if len(frame_features) < expected_hand_features:
            frame_features.extend([0.0] * (expected_hand_features - len(frame_features)))
        if USE_FACE:
            if face_result and face_result.face_landmarks:
                face_lm = face_result.face_landmarks[0]
                for idx in FACE_KEY_POINTS:
                    if idx < len(face_lm):
                        frame_features.extend([face_lm[idx].x, face_lm[idx].y, face_lm[idx].z])
                    else:
                        frame_features.extend([0.0, 0.0, 0.0])
                frame_features.extend(self.compute_face_features(face_result.face_landmarks))
                if USE_HEAD_POSE:
                    frame_features.extend(self.compute_head_pose(face_result.face_landmarks))
            else:
                face_feature_size = len(FACE_KEY_POINTS) * FEATURES_PER_LM + 15
                if USE_HEAD_POSE:
                    face_feature_size += 3
                frame_features.extend([0.0] * face_feature_size)
        frame_features = np.array(frame_features, dtype=np.float32)
        
        if USE_VELOCITY:
            if self.prev_features is None:
                velocity = np.zeros_like(frame_features)
            else:
                if len(frame_features) != len(self.prev_features):
                    velocity = np.zeros_like(frame_features)
                else:
                    velocity = frame_features - self.prev_features
            self.prev_features = frame_features.copy()
            frame_features = np.concatenate([frame_features, velocity])
        
        return frame_features
