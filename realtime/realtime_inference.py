import cv2
import time
import numpy as np
import os
import torch

from realtime.feature_extractor import StreamingFeatureExtractor
from realtime.buffer import FeatureBuffer
from realtime.ctc_decoder import greedy_ctc_decode
from realtime.model import SignRecognitionModel
from sign_vocab import idx_to_sign, BLANK_IDX

# ---------------- CONFIG ---------------- #
# GPU Detection and Configuration
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "sign_model.pth"))
# If running from a subpackage folder, the models file might be in a parent workspace directory.
if not os.path.exists(MODEL_PATH):
    for i in range(1, 5):
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * i), "models", "sign_model.pth"))
        if os.path.exists(candidate):
            MODEL_PATH = candidate
            break
SEQUENCE_LENGTH = 45
DISPLAY_COOLDOWN = 2.5  # seconds between predictions (increased for stability)
MIN_CONFIDENCE = 0.70   # minimum confidence to show prediction (increased threshold)
STABILITY_THRESHOLD = 0.02  # maximum hand movement to consider "stable"
STABILITY_FRAMES = 30   # number of stable frames required before prediction
# --------------------------------------- #

# ---------------- MODEL ---------------- #
def load_model(feature_dim):
    model = SignRecognitionModel(
        input_dim=feature_dim,
        hidden_dim=256,
        output_dim=len(idx_to_sign)
    ).to(DEVICE)

    if not torch.cuda.is_available():
        state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    else:
        state = torch.load(MODEL_PATH)

    model.load_state_dict(state)
    model.eval()
    return model

def get_hand_bounding_box(hand_landmarks, image_width, image_height):
    """Calculate bounding box for hand landmarks"""
    x_coords = [lm.x * image_width for lm in hand_landmarks]
    y_coords = [lm.y * image_height for lm in hand_landmarks]
    
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_width, x_max + padding)
    y_max = min(image_height, y_max + padding)
    
    return (x_min, y_min, x_max, y_max)

def calculate_hand_movement(current_landmarks, previous_landmarks):
    """Calculate average movement of hand landmarks between frames"""
    if previous_landmarks is None:
        return float('inf')
    
    movements = []
    for curr, prev in zip(current_landmarks, previous_landmarks):
        dx = curr.x - prev.x
        dy = curr.y - prev.y
        dz = curr.z - prev.z
        movement = np.sqrt(dx**2 + dy**2 + dz**2)
        movements.append(movement)
    
    return np.mean(movements)

# ---------------- MAIN ---------------- #
def main():
    print("[INFO] Starting real-time sign recognition")
    print("[INFO] Waiting for stable hand position before prediction...")
    extractor = StreamingFeatureExtractor()

    # Infer feature dimension
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feat = extractor.extract(dummy_frame)
    FEATURE_DIM = len(feat)
    print(f"[INFO] Feature dimension: {FEATURE_DIM}")


    buffer = FeatureBuffer(max_len=SEQUENCE_LENGTH, feature_dim=FEATURE_DIM)

    try:
        model = load_model(FEATURE_DIM)
        print("[INFO] Model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[ERROR] Make sure models/sign_model.pth exists and was trained properly.")
        return

    cap = None
    last_displayed = ""
    last_display_time = 0
    recognized_word = ""
    confidence = 0.0
    previous_hand_landmarks = None
    stable_frames_count = 0
    is_hand_stable = False
    hand_detected = False

    try:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract features (this also detects hands)
            features = extractor.extract(rgb)
            
            # Get hand detection results for visualization from extractor
            # We need to detect again for visualization since extractor doesn't return landmarks
            from mediapipe.tasks.python.core import base_options
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
            import mediapipe as mp
            
            # Use the extractor's hand landmarker if available
            if extractor.hand_landmarker is not None:
                hand_landmarker = extractor.hand_landmarker
            else:
                # Create a temporary one (this shouldn't happen as extract() initializes it)
                from realtime.feature_extractor import _resolve_model_path
                hand_model_path = _resolve_model_path("hand_landmarker.task")
                options = HandLandmarkerOptions(
                    base_options=base_options.BaseOptions(model_asset_path=hand_model_path),
                    num_hands=2,
                    min_hand_detection_confidence=0.5
                )
                hand_landmarker = HandLandmarker.create_from_options(options)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = hand_landmarker.detect(mp_image)
            
            hand_detected = False
            current_hand_landmarks = None
            
            # Draw hand bounding boxes and landmarks
            if hand_result.hand_landmarks:
                hand_detected = True
                h, w, _ = frame.shape
                
                for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    # Get bounding box
                    bbox = get_hand_bounding_box(hand_landmarks, w, h)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                                (0, 255, 0), 2)
                    
                    # Draw hand landmarks
                    for landmark in hand_landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    
                    # Get handedness
                    try:
                        handedness = hand_result.handedness[idx][0].category_name
                        cv2.putText(frame, handedness, (bbox[0], bbox[1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except:
                        pass
                    
                    # Store first hand for stability check
                    if idx == 0:
                        current_hand_landmarks = hand_landmarks
                
                # Check hand stability
                if current_hand_landmarks is not None and previous_hand_landmarks is not None:
                    movement = calculate_hand_movement(current_hand_landmarks, previous_hand_landmarks)
                    
                    if movement < STABILITY_THRESHOLD:
                        stable_frames_count += 1
                    else:
                        stable_frames_count = 0
                    
                    is_hand_stable = stable_frames_count >= STABILITY_FRAMES
                else:
                    stable_frames_count = 0
                    is_hand_stable = False
                
                previous_hand_landmarks = current_hand_landmarks
            else:
                # No hand detected - reset stability
                stable_frames_count = 0
                is_hand_stable = False
                previous_hand_landmarks = None

            # Add features to buffer if hand detected
            if features is not None and hand_detected:
                buffer.add(features)

            # Run inference when buffer is full AND hand is stable
            if buffer.is_full() and is_hand_stable:
                seq = buffer.get()  # (T, F)
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(x)        # (1, T, C)
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1).squeeze(0).cpu().numpy()  # (T,)
                    
                    # Calculate confidence (average max probability)
                    max_probs = torch.max(probs, dim=-1)[0].squeeze(0).cpu().numpy()
                    confidence = float(np.mean(max_probs))
                
                # Use majority voting: find most common prediction (excluding blank)
                # Filter out blank predictions
                non_blank_preds = preds[preds != BLANK_IDX]
                
                if len(non_blank_preds) > 0:
                    # Get the most common prediction
                    from collections import Counter
                    vote_counts = Counter(non_blank_preds)
                    most_common_idx = vote_counts.most_common(1)[0][0]
                    text = idx_to_sign[most_common_idx]
                    
                    # Calculate confidence for this specific prediction
                    # Get probability of the most common prediction across all frames
                    pred_probs = probs.squeeze(0).cpu().numpy()  # (T, C)
                    confidence = float(np.mean(pred_probs[:, most_common_idx]))
                else:
                    text = None

                now = time.time()
                # Only show predictions with high confidence
                if text and confidence >= MIN_CONFIDENCE and now - last_display_time > DISPLAY_COOLDOWN:
                    print(f"[SIGN]: {text.upper()} (Confidence: {confidence*100:.1f}%)")
                    last_displayed = text
                    last_display_time = now
                    recognized_word = text
                    
                    # Clear buffer after successful prediction to wait for next sign
                    buffer = FeatureBuffer(max_len=SEQUENCE_LENGTH, feature_dim=FEATURE_DIM)
                    stable_frames_count = 0

            # Display status information
            status_y = 20

            # Display status information
            status_y = 20
            
            # Hand detection status
            if hand_detected:
                status_text = "Hand Detected" if is_hand_stable else "Hand Moving..."
                status_color = (0, 255, 0) if is_hand_stable else (0, 165, 255)
                cv2.putText(frame, status_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            else:
                cv2.putText(frame, "No Hand Detected", (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            status_y += 30
            
            # Buffer status
            buffer_fill = len(buffer.buffer)
            buffer_text = f"Buffer: {buffer_fill}/{SEQUENCE_LENGTH}"
            cv2.putText(frame, buffer_text, (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw buffer progress bar
            bar_x, bar_y = 10, status_y + 10
            bar_width, bar_height = 200, 15
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (100, 100, 100), 2)
            fill_width = int((buffer_fill / SEQUENCE_LENGTH) * bar_width)
            if fill_width > 0:
                color = (0, 255, 0) if is_hand_stable else (0, 165, 255)
                cv2.rectangle(frame, (bar_x + 2, bar_y + 2), 
                             (bar_x + fill_width - 2, bar_y + bar_height - 2), 
                             color, -1)
            
            status_y += 45
            
            # Stability indicator
            if hand_detected:
                stability_pct = (stable_frames_count / STABILITY_FRAMES) * 100
                stability_text = f"Stability: {stability_pct:.0f}%"
                cv2.putText(frame, stability_text, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                status_y += 25

            # Display recognized word and confidence on frame
            if recognized_word:
                # Create semi-transparent overlay for better text visibility
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, frame.shape[0] - 100), (500, frame.shape[0] - 10), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Display word and confidence
                cv2.putText(frame, f"DETECTED: {recognized_word.upper()}", 
                           (20, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", 
                           (20, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Hold sign steady for detection | ESC to quit", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Sign Recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()
