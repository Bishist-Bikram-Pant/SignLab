#!/usr/bin/env python3
"""
Download MediaPipe Face Landmarker model
"""
import os
import urllib.request

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_PATH = "mediapipe/face_landmarker.task"

os.makedirs("mediapipe", exist_ok=True)

if os.path.exists(FACE_MODEL_PATH):
    print(f"✓ Face landmarker model already exists: {FACE_MODEL_PATH}")
else:
    print(f"📥 Downloading face landmarker model...")
    print(f"   URL: {FACE_MODEL_URL}")
    print(f"   Destination: {FACE_MODEL_PATH}")
    
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        size_mb = os.path.getsize(FACE_MODEL_PATH) / (1024 * 1024)
        print(f"✓ Downloaded successfully ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"   Please download manually from: {FACE_MODEL_URL}")
        print(f"   Save it to: {FACE_MODEL_PATH}")

# Check hand model
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
# Prefer to place the hand model inside the mediapipe/ directory so other modules can find it.
HAND_MODEL_DEST = os.path.join("mediapipe", "hand_landmarker.task")
HAND_MODEL_LOCAL = os.path.abspath("hand_landmarker.task")

if os.path.exists(HAND_MODEL_DEST):
    print(f"✓ Hand landmarker model exists: {HAND_MODEL_DEST}")
else:
    # If user already downloaded the model to the project root, copy it into mediapipe/
    if os.path.exists(HAND_MODEL_LOCAL):
        print(f"Copying existing hand model from {HAND_MODEL_LOCAL} to {HAND_MODEL_DEST}")
        try:
            import shutil
            shutil.copy(HAND_MODEL_LOCAL, HAND_MODEL_DEST)
            size_mb = os.path.getsize(HAND_MODEL_DEST) / (1024 * 1024)
            print(f"✓ Copied successfully ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"❌ Copy failed: {e}")
            print(f"   Please move the model manually to: {HAND_MODEL_DEST}")
    else:
        print(f"📥 Downloading hand landmarker model...")
        print(f"   URL: {HAND_MODEL_URL}")
        print(f"   Destination: {HAND_MODEL_DEST}")
        try:
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_DEST)
            size_mb = os.path.getsize(HAND_MODEL_DEST) / (1024 * 1024)
            print(f"✓ Downloaded successfully ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print(f"   Please download manually from: {HAND_MODEL_URL}")
            print(f"   Save it to: {HAND_MODEL_DEST}")

print("\n" + "="*60)
print("Setup complete! You can now:")
print("  1. python collect_data.py  (collect training data)")
print("  2. python train.py         (train the model)")
print("  3. python -m realtime.realtime_inference  (run inference)")
print("="*60)
