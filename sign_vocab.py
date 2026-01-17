# sign_vocab.py
import json

# Example signs in your dataset
SIGNS = [
    "hello", "thank_you", "yes", "no", "i_love_you", "please"
]

# Map signs to integers
sign_to_idx = {s: i+1 for i, s in enumerate(SIGNS)}  # start at 1
sign_to_idx["<blank>"] = 0  # CTC blank token

# sign_lab/sign_vocab.py

BLANK_IDX = 0

idx_to_sign = {
    0: "<blank>",
    1: "hello",
    2: "I",
    3: "yes",
    4: "no",
    5: "please",
    6: "sorry",
    7: "thank",
    8: "you",
    9: "help",
    10: "want",
    11: "food",
    12: "drink",
    13: "asl_a",
    14: "asl_b",
    15: "asl_c",
    16: "asl_d"
}

sign_to_idx = {v: k for k, v in idx_to_sign.items()}

# Save for reference
with open("sign_vocab.json", "w") as f:
    json.dump({"sign_to_idx": sign_to_idx, "idx_to_sign": idx_to_sign}, f)
