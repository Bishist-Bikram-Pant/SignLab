"""
Sign vocabulary for ASL recognition
Auto-generated from WLASL dataset
"""

# All signs in the vocabulary (excluding blank token)
SIGNS = ['book', 'drink', 'computer', 'before', 'chair', 'go', 'clothes', 'who', 'candy', 'cousin', 'deaf', 'walk', 'yes', 'thin', 'help', 'year', 'fine', 'no', 'what', 'now', 'many', 'hot', 'woman', 'cool', 'thanksgiving', 'like', 'black', 'finish', 'table', 'mother']

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
