# sign_vocab.py
import json

# ASL words for this project - expanded to 10 words
SIGNS = [
    "hello", "goodbye", "thanks", "please", "yes", 
    "no", "sorry", "help", "love", "friend"
]

BLANK_IDX = 0

# Updated vocabulary for word recognition (10 common ASL words)
idx_to_sign = {
    0: "<blank>",
    1: "hello",
    2: "goodbye",
    3: "thanks",
    4: "please",
    5: "yes",
    6: "no",
    7: "sorry",
    8: "help",
    9: "love",
    10: "friend"
}

sign_to_idx = {v: k for k, v in idx_to_sign.items()}

# Save for reference
with open("sign_vocab.json", "w") as f:
    json.dump({"sign_to_idx": sign_to_idx, "idx_to_sign": idx_to_sign}, f)
