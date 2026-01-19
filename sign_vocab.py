"""
Sign vocabulary for ASL recognition
Auto-generated from ASL Alphabet dataset - ALL LETTERS
"""

# All signs in the vocabulary (excluding blank token)
SIGNS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Blank token index
BLANK_IDX = 0

# Mapping from sign name to index
sign_to_idx = {"<blank>": 0}
for idx, sign in enumerate(SIGNS, start=1):
    sign_to_idx[sign] = idx

# Reverse mapping from index to sign name
idx_to_sign = {idx: sign for sign, idx in sign_to_idx.items()}

# Number of classes (including blank)
NUM_CLASSES = len(sign_to_idx)

def get_sign_name(idx):
    """Get sign name from index"""
    return idx_to_sign.get(idx, "<unknown>")

def get_sign_idx(name):
    """Get index from sign name"""
    return sign_to_idx.get(name, 0)
