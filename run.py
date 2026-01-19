#!/usr/bin/env python3
"""
Quick start script for real-time sign language recognition
"""
import subprocess
import sys

def main():
    from sign_vocab import SIGNS
    
    print("=" * 70)
    print("  Real-Time Sign Language Recognition - WLASL Dataset")
    print("=" * 70)
    print()
    print("Controls:")
    print("  - ESC: Quit")
    print()
    print(f"Signs recognized ({len(SIGNS)} words):")
    
    # Display signs in rows of 5
    for i in range(0, len(SIGNS), 5):
        row = SIGNS[i:i+5]
        print("  " + ", ".join(row))
    
    print()
    print("=" * 70)
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "realtime.realtime_inference"], check=True)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
