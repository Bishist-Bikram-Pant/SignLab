"""
Train the sign language recognition model using PyTorch.

TRAINING WORKFLOW:
1. Loads all .npy files from dataset/{sign_name}/ folders
2. Splits data into 80% training, 20% testing
3. Runs bidirectional GRU through sequences
4. Uses CrossEntropyLoss on frame predictions
5. Keeps best model (lowest test loss)
6. Saves to models/sign_model.pth

HOW TO USE:
  python train.py
  
WHAT YOU SEE:
  - Epoch number and training progress
  - Train Loss: how well model fits training data
  - Test Loss: how well model generalizes
  - Accuracy: % of sequences correctly classified
  - ✓ Best model saved = better checkpoint found

IF YOU GET "NO DATA FOUND":
  → Run: python collect_data.py
  → Collect at least a few samples first

CONFIGS TO TWEAK:
  - BATCH_SIZE: higher = faster but needs more memory
  - EPOCHS: more epochs = longer training but potentially better
  - LEARNING_RATE: lower = slower but more stable training
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from realtime.model import SignRecognitionModel
from sign_vocab import sign_to_idx, idx_to_sign

# Config
DATASET_DIR = "dataset"
MODEL_PATH = "models/sign_model.pth"
SEQUENCE_LENGTH = 45
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"\n📱 Using device: {DEVICE}\n")

class SignDataset(torch.utils.data.Dataset):
    """Loads all .npy files from dataset/ folder and pads/truncates to fixed length"""
    def __init__(self, dataset_dir, target_length=SEQUENCE_LENGTH):
        self.sequences = []
        self.labels = []
        self.sign_names = []
        self.target_length = target_length
        
        for sign, idx in sign_to_idx.items():
            if sign == "<blank>":
                continue
            
            sign_dir = os.path.join(dataset_dir, sign)
            if not os.path.exists(sign_dir):
                print(f"⚠️  Warning: {sign_dir} not found")
                continue
            
            # Load all .npy files for this sign
            files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
            for file in files:
                path = os.path.join(sign_dir, file)
                seq = np.load(path)
                
                # Ensure consistent feature dimension
                from realtime.feature_extractor import StreamingFeatureExtractor
                expected_feature_dim = StreamingFeatureExtractor().total_feature_dim
                
                if seq.shape[1] < expected_feature_dim:
                    padding = np.zeros((seq.shape[0], expected_feature_dim - seq.shape[1]))
                    seq = np.hstack([seq, padding])
                elif seq.shape[1] > expected_feature_dim:
                    seq = seq[:, :expected_feature_dim]
                
                # Pad or truncate to target length
                if len(seq) < target_length:
                    # Pad with zeros
                    padding = np.zeros((target_length - len(seq), seq.shape[1]))
                    seq = np.vstack([seq, padding])
                elif len(seq) > target_length:
                    # Truncate
                    seq = seq[:target_length]
                
                self.sequences.append(seq)
                self.labels.append(idx)
                self.sign_names.append(sign)
        
        print(f"✓ Loaded {len(self.sequences)} sequences from {dataset_dir}/")
        
        # Show breakdown
        if len(self.sequences) > 0:
            for sign, idx in sign_to_idx.items():
                if sign == "<blank>":
                    continue
                count = sum(1 for l in self.labels if l == idx)
                if count > 0:
                    print(f"  - {sign:12s}: {count:3d} samples")
        
        if len(self.sequences) == 0:
            raise RuntimeError(
                f"❌ No data found in {dataset_dir}\n"
                f"   Run: python collect_data.py"
            )
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label

def train():
    print("="*70)
    print("  TRAINING SIGN RECOGNITION MODEL")
    print("="*70)
    print()
    
    # Load dataset
    print("📂 Loading dataset...")
    dataset = SignDataset(DATASET_DIR)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Get feature dimension from first sample
    sample, _ = dataset[0]
    feature_dim = sample.shape[1]
    output_dim = len(idx_to_sign)
    
    print(f"\n📊 Training setup:")
    print(f"  - Feature dimension: {feature_dim}")
    print(f"  - Output classes: {output_dim}")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")
    print()
    
    # Build model
    print("🧠 Building model...")
    model = SignRecognitionModel(
        input_dim=feature_dim,
        hidden_dim=256,
        output_dim=output_dim,
        num_layers=2
    ).to(DEVICE)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Ensure the parent directory for MODEL_PATH exists so torch.save won't fail
    model_dir = os.path.dirname(MODEL_PATH) or "."
    os.makedirs(model_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_loss = float('inf')
    
    print()
    print("🎯 Training...")
    print("-" * 70)
    
    # Training loop
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        
        for batch_idx, (seqs, labels) in enumerate(train_loader):
            seqs = seqs.to(DEVICE)  # (B, T, F)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(seqs)  # (B, T, C)
            
            # Compute loss: classify sign from sequence
            # For each frame, predict the sign
            logits_flat = logits.reshape(-1, logits.shape[-1])  # (B*T, C)
            labels_repeated = labels.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)  # (B*T,)
            
            loss = nn.CrossEntropyLoss()(logits_flat, labels_repeated)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Test
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for seqs, labels in test_loader:
                seqs = seqs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                logits = model(seqs)
                
                # Get prediction: most common class across sequence
                preds = logits.argmax(dim=-1)  # (B, T)
                # Use CPU for mode operation (MPS doesn't support it)
                preds_cpu = preds.cpu()
                pred_labels = preds_cpu.mode(dim=1)[0].to(DEVICE)
                
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)
                
                logits_flat = logits.reshape(-1, logits.shape[-1])
                labels_repeated = labels.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
                loss = nn.CrossEntropyLoss()(logits_flat, labels_repeated)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        
        # Print progress
        status = " ✓ Best!" if test_loss < best_loss else ""
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Val: {test_loss:.4f} | "
              f"Acc: {accuracy:5.1f}%{status}")
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), MODEL_PATH)
        
        scheduler.step()
    
    print("-" * 70)
    print()
    print("="*70)
    print("✓ Training complete!")
    print(f"✓ Model saved to {MODEL_PATH}")
    print()
    print("NEXT STEP:")
    print("  python -m realtime.realtime_inference")
    print("="*70 + "\n")

if __name__ == "__main__":
    train()
