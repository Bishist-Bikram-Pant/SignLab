import cv2
import time
import numpy as np
import os
import torch
import pyttsx3

from realtime.feature_extractor import StreamingFeatureExtractor
from realtime.buffer import FeatureBuffer
from realtime.ctc_decoder import greedy_ctc_decode
from realtime.model import SignRecognitionModel
from sign_vocab import idx_to_sign, BLANK_IDX

# ---------------- CONFIG ---------------- #
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "sign_model.pth"))
# If running from a subpackage folder, the models file might be in a parent workspace directory.
if not os.path.exists(MODEL_PATH):
    for i in range(1, 5):
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * i), "models", "sign_model.pth"))
        if os.path.exists(candidate):
            MODEL_PATH = candidate
            break
SEQUENCE_LENGTH = 45
SPEAK_COOLDOWN = 1.0  # seconds
# --------------------------------------- #

# ---------------- TTS ---------------- #
# Initialize lazily to avoid pyttsx3 driver destructor issues on interpreter shutdown.
tts = None
TTS_AVAILABLE = False

def speak(text):
    global tts, TTS_AVAILABLE
    if not TTS_AVAILABLE:
        try:
            tts = pyttsx3.init()
            tts.setProperty("rate", 160)
            TTS_AVAILABLE = True
        except Exception:
            tts = None
            TTS_AVAILABLE = False
            return

    if tts is None:
        return

    try:
        tts.say(text)
        tts.runAndWait()
    except Exception:
        # Don't let TTS errors crash the app
        pass

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

# ---------------- MAIN ---------------- #
def main():
    print("[INFO] Starting real-time sign recognition")
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
    last_spoken = ""
    last_speak_time = 0

    try:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            features = extractor.extract(rgb)
            if features is not None:
                buffer.add(features)

            # Run inference when buffer is full
            if buffer.is_full():
                seq = buffer.get()  # (T, F)
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(x)        # (1, T, C)
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1).squeeze(0).cpu().numpy()

                decoded = greedy_ctc_decode(preds, blank=BLANK_IDX)
                text = " ".join(idx_to_sign[i] for i in decoded)

                now = time.time()
                if text and text != last_spoken and now - last_speak_time > SPEAK_COOLDOWN:
                    print("[SIGN]:", text)
                    speak(text)
                    last_spoken = text
                    last_speak_time = now

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
        try:
            tts.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
