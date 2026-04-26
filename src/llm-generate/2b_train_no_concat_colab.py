from google.colab import drive
drive.mount('/content/drive')

print("✓ Google Drive mounted successfully!")

# ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import os
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ---

# ==================== IMPORTANT: SET THIS PATH ====================
# Path to your embeddings directory (from 2a_create_all_embeddings_colab.ipynb)
# Example: "/content/drive/MyDrive/RNN_All_Embeddings/20231203_153045"
EMBEDDINGS_DIR = "/content/drive/MyDrive/RNN_All_Embeddings/20251204_045020"

# ==================================================================

# Output directory for training results
TRAINING_OUTPUT_DIR = os.path.join(
    "/content/drive/MyDrive/RNN_Training_Results",
    datetime.now().strftime("%Y%m%d_%H%M%S")
)
os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)

print(f"Embeddings directory: {EMBEDDINGS_DIR}")
print(f"Training results will be saved to: {TRAINING_OUTPUT_DIR}")

# Batch Size
TRAIN_BATCH_SIZE = 16

# Model Configuration
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.4

# Training Configuration
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10

print(f"\nConfiguration:")
print(f"  Model: Hidden={HIDDEN_DIM}, Layers={NUM_LAYERS}, Dropout={DROPOUT}")
print(f"  Training: Epochs={EPOCHS}, LR={LEARNING_RATE}, Batch={TRAIN_BATCH_SIZE}")

# ---

# Load embeddings
embeddings_file = os.path.join(EMBEDDINGS_DIR, "embeddings.pkl")
metadata_file = os.path.join(EMBEDDINGS_DIR, "embeddings_metadata.json")

print(f"Loading embeddings from: {embeddings_file}")

if not os.path.exists(embeddings_file):
    raise FileNotFoundError(
        f"Embeddings not found at: {embeddings_file}\n"
        f"Please run '2a_create_all_embeddings_colab.ipynb' first and update EMBEDDINGS_DIR."
    )

with open(embeddings_file, 'rb') as f:
    embeddings_data = pickle.load(f)

# Use only original embeddings (not concatenated with injected)
train_sequences = embeddings_data['train']['original_sequences']
train_labels = embeddings_data['train']['labels']

val_sequences = embeddings_data['val']['original_sequences']
val_labels = embeddings_data['val']['labels']

test_sequences = embeddings_data['test']['original_sequences']
test_labels = embeddings_data['test']['labels']

print(f"✓ Loaded embeddings:")
print(f"  Train: {len(train_sequences)} documents")
print(f"  Val:   {len(val_sequences)} documents")
print(f"  Test:  {len(test_sequences)} documents")
print(f"\n✓ Using only original sentence embeddings (no concatenation)")
print(f"  Embedding dim: {train_sequences[0].shape[1]}")

# Load metadata
if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"\n✓ Metadata loaded")
    print(f"  Creation date: {metadata['creation_timestamp']}")
    print(f"  Model: {metadata['model_id']}")
    print(f"  Original embedding dim: {metadata.get('original_embedding_dim', 'N/A')}")
    print(f"  Injected embedding dim: {metadata.get('injected_embedding_dim', 'N/A')}")

# ---

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✓ Using device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ---

class SequenceArtifactDetector(nn.Module):
    """
    Bidirectional LSTM for detecting AI-generated text based on sequential patterns.
    Processes variable-length document sequences.
    """

    def __init__(self, embedding_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super(SequenceArtifactDetector, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = hidden_dim * 2  # *2 for bidirectional

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1),
            nn.Softmax(dim=1)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, lengths):
        """
        Args:
            x: (batch_size, max_seq_len, embedding_dim)
            lengths: (batch_size,) actual lengths of sequences
        """
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        packed_out, (hidden, cell) = self.lstm(packed)

        # Unpack
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attended = torch.sum(lstm_out * attention_weights, dim=1)

        # Classification
        output = self.classifier(attended)
        return output

print("✓ Model architecture defined")

# ---

class SequenceDataset(Dataset):
    """Dataset that returns variable-length sequences."""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    """
    Collate function to pad variable-length sequences in a batch.

    Args:
        batch: List of (sequence, label) tuples

    Returns:
        padded_sequences: (batch_size, max_len, embedding_dim)
        labels: (batch_size, 1)
        lengths: (batch_size,) actual sequence lengths
    """
    sequences, labels = zip(*batch)

    # Get lengths
    lengths = torch.tensor([len(seq) for seq in sequences])

    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # Stack labels
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return padded_sequences, labels, lengths

print("✓ Dataset class and collate function defined")

# ---

train_dataset = SequenceDataset(train_sequences, train_labels)
val_dataset = SequenceDataset(val_sequences, val_labels)
test_dataset = SequenceDataset(test_sequences, test_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print(f"\n✓ DataLoaders created:")
print(f"  Training batches: {len(train_loader)}")
print(f"  Validation batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")
print(f"\n  Training samples: {len(train_dataset)}")
print(f"  Validation samples: {len(val_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# ---

def train_model(train_loader, val_loader, embedding_dim, epochs=50, lr=0.001,
                hidden_dim=256, num_layers=2, dropout=0.3):
    print("\n" + "="*70)
    print("Training RNN Sequence Detector")
    print("="*70)

    model = SequenceArtifactDetector(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": []
    }

    print(f"\nModel Configuration:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print("="*70 + "\n")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, lengths)
            loss = criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y, lengths in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                lengths = lengths.to(device)

                outputs = model(batch_X, lengths)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout
            }
            torch.save(checkpoint, os.path.join(TRAINING_OUTPUT_DIR, "best_model.pth"))
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}\n")

    # Load best model
    checkpoint = torch.load(os.path.join(TRAINING_OUTPUT_DIR, "best_model.pth"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history

print("✓ Training function defined")

# ---

# Get embedding dimension from concatenated sequences
embedding_dim = train_sequences[0].shape[1]  # This is original_dim + injected_dim

print(f"Using embedding dimension: {embedding_dim}")

model, history = train_model(
    train_loader, val_loader, embedding_dim,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

# ---

# Save training history
with open(os.path.join(TRAINING_OUTPUT_DIR, "training_history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"✓ Training history saved to Google Drive")

# ---

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Training and Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(history['learning_rate'], marker='o', color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True)

axes[1, 1].axis('off')
summary_text = f"""
Training Summary
{'='*40}
Best Val Accuracy: {max(history['val_acc']):.2f}%
Final Train Acc: {history['train_acc'][-1]:.2f}%
Final Val Acc: {history['val_acc'][-1]:.2f}%
Total Epochs: {len(history['train_loss'])}

Model Configuration
{'='*40}
Hidden Dim: {HIDDEN_DIM}
Num Layers: {NUM_LAYERS}
Dropout: {DROPOUT}
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center')

plt.tight_layout()
plt.savefig(os.path.join(TRAINING_OUTPUT_DIR, "training_curves.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Training curves saved to Google Drive")

# ---

print("\n" + "="*70)
print("Evaluating on Test Set")
print("="*70)

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_X, batch_y, lengths in tqdm(test_loader, desc="Testing"):
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        lengths = lengths.to(device)

        outputs = model(batch_X, lengths)
        predictions = (outputs > 0.5).float()

        all_preds.extend(predictions.cpu().numpy())
        all_true.extend(batch_y.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_true, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_true, all_preds, average='binary', zero_division=0
)
cm = confusion_matrix(all_true, all_preds)

# Print results
print("\n" + "="*70)
print("FINAL TEST RESULTS")
print("="*70)
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Human    AI")
print(f"Actual Human    {cm[0][0]:3d}    {cm[0][1]:3d}")
print(f"       AI       {cm[1][0]:3d}    {cm[1][1]:3d}")
print("="*70)

# ---

# Save test metrics
test_results = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "num_test_samples": len(test_dataset),
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(TRAINING_OUTPUT_DIR, "test_results.json"), "w") as f:
    json.dump(test_results, f, indent=2)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Human', 'AI'],
            yticklabels=['Human', 'AI'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Test Set')
plt.savefig(os.path.join(TRAINING_OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ Test results saved to Google Drive")

# ---

config = {
    "embeddings_path": EMBEDDINGS_DIR,
    "training_output_path": TRAINING_OUTPUT_DIR,
    "dataset": {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset)
    },
    "model": {
        "architecture": "Bidirectional LSTM with Attention",
        "embedding_dim": embedding_dim,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "uses_sliding_windows": False
    },
    "training": {
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": TRAIN_BATCH_SIZE,
        "patience": PATIENCE,
        "actual_epochs_trained": len(history['train_loss'])
    },
    "results": {
        "best_val_accuracy": max(history['val_acc']),
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1)
    },
    "device": device,
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(TRAINING_OUTPUT_DIR, "full_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"✓ Full configuration saved to Google Drive")

# ---

print("\n" + "="*70)
print("TRAINING COMPLETE - SUMMARY")
print("="*70)
print(f"\n✓ All results saved to Google Drive:")
print(f"  {TRAINING_OUTPUT_DIR}\n")
print(f"Saved files:")
print(f"  1. best_model.pth - Best trained model")
print(f"  2. training_history.json - Training history")
print(f"  3. training_curves.png - Training visualization")
print(f"  4. test_results.json - Test metrics")
print(f"  5. confusion_matrix.png - Confusion matrix")
print(f"  6. full_config.json - Complete configuration\n")
print(f"Final Results:")
print(f"  Best Val Accuracy: {max(history['val_acc']):.2f}%")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Test F1 Score: {f1*100:.2f}%\n")
print(f"Model Architecture:")
print(f"  Proper RNN sequence processing (no sliding windows)")
print(f"  Each document processed as a single variable-length sequence\n")
print(f"Training completed using embeddings from:")
print(f"  {EMBEDDINGS_DIR}")
print("="*70)