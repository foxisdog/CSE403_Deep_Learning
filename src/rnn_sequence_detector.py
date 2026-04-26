import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import nltk
import logging
import sys
import os
from tqdm import tqdm
import random
import math

# ========================================
# Configuration
# ========================================
NUM_ROWS_TO_USE = 10  # Number of documents to process from dataset
MAX_SENTENCES_PER_DOC = 40  # Maximum sentences per document
WINDOW_SIZE = 10  # Number of sentences in each sequence window
WINDOW_STEP = 5  # Sliding window step size
BATCH_SIZE = 8  # Batch size for LLM inference
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding extraction

# ========================================
# Setup & Initialization
# ========================================

# HuggingFace Login
hf_token = ""  # Add your token here if needed
if hf_token:
    try:
        login(token=hf_token)
    except Exception as e:
        print(f"Warning: Login failed. {e}")

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

# Model Loading
model_id = "meta-llama/Llama-3.2-1B-Instruct"
print(f"Loading {model_id}...")

try:
    tokenizer_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {"token": hf_token} if hf_token else {}
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
        **model_kwargs
    )
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")
    sys.exit(1)

# NLTK Setup
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ========================================
# RNN/LSTM Model Architecture
# ========================================

class SequenceArtifactDetector(nn.Module):
    """
    Bidirectional LSTM for detecting AI-generated text based on sequential patterns.

    Architecture:
    1. Input Layer: Sequence of [Original ⊕ Transformed] embeddings
    2. Bi-LSTM: Captures forward and backward sequential dependencies
    3. Global Average Pooling: Aggregates patterns across all time steps
    4. Classifier: Maps to binary prediction (AI vs Human)
    """

    def __init__(self, embedding_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super(SequenceArtifactDetector, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM
        # Input: (batch, seq_len, embedding_dim * 2)  # *2 for [Orig ⊕ Transformed]
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # After Bi-LSTM, hidden dim becomes hidden_dim * 2 (forward + backward)
        lstm_output_dim = hidden_dim * 2

        # Attention mechanism (optional but helps focus on important time steps)
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
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embedding_dim * 2)

        Returns:
            output: (batch_size, 1) - probability of being AI-generated
        """
        # LSTM forward pass
        # lstm_out: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention to get weighted representation
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_dim * 2)

        # Alternative: Global Average Pooling (uncomment to use instead of attention)
        # averaged = torch.mean(lstm_out, dim=1)  # (batch_size, hidden_dim * 2)

        # Classification
        output = self.classifier(attended)  # (batch_size, 1)

        return output

# ========================================
# Dataset Class with Sliding Window
# ========================================

class SequenceDataset(Dataset):
    """
    Dataset that creates sliding window sequences from documents.

    Each sample is a sequence of (original, transformed) embedding pairs.
    This allows the RNN to learn sequential patterns across sentences.
    """

    def __init__(self, sequences, labels):
        """
        Args:
            sequences: List of tensors, each of shape (window_size, embedding_dim * 2)
            labels: List of labels (0 for human, 1 for AI)
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

# ========================================
# Helper Functions
# ========================================

def get_sentences(text):
    """Split text into sentences"""
    if not text or not isinstance(text, str):
        return []
    return nltk.tokenize.sent_tokenize(text)

def get_sentence_embedding(texts):
    """
    Get embeddings for a batch of sentences using mean pooling.

    Args:
        texts: List of strings or single string

    Returns:
        embeddings: (batch_size, hidden_size)
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts or all(not t for t in texts):
        return torch.zeros((len(texts) if texts else 1, llm_model.config.hidden_size), device=device)

    tokenizer.padding_side = "right"

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = llm_model(**inputs)

    last_hidden = outputs.hidden_states[-1]
    mask = inputs['attention_mask'].unsqueeze(-1)

    sum_emb = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_emb = sum_emb / counts

    return mean_emb

def transform_sentences(texts, mode, description=None):
    """
    Transform sentences using LLM (Reduce or Inject mode).

    Args:
        texts: List of sentences to transform
        mode: "reduce" or "inject"
        description: Progress bar description

    Returns:
        List of transformed sentences
    """
    tokenizer.padding_side = "left"

    if mode == "reduce":
        task_instruction = "Simplify the sentence. Keep only the main Subject, Verb, and Object."
        rule = "Output ONLY ONE simplified sentence. Do not add extra sentences or explanations."
    elif mode == "inject":
        task_instruction = "Rewrite the sentence to be more descriptive and vivid."
        rule = "Output ONLY ONE rewritten sentence. Do not add extra sentences or explanations."
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Prepare prompts
    prompts = []
    for sentence in texts:
        prompt = (
            f"Task: {task_instruction}\n"
            f"Rule: {rule}\n"
            f"Input: {sentence}\n"
            f"Output:"
        )
        prompts.append(prompt)

    # Batch processing
    results = []

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=description or f"Transforming ({mode})"):
        batch_prompts = prompts[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode outputs
        for j, output_ids in enumerate(outputs):
            # Remove input prompt tokens
            input_length = inputs['input_ids'][j].shape[0]
            generated_ids = output_ids[input_length:]

            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Validation: Take only first sentence
            generated_sentences = get_sentences(generated_text)
            if generated_sentences:
                results.append(generated_sentences[0])
            else:
                # Fallback to original
                results.append(texts[i+j] if i+j < len(texts) else "")

    return results

def create_sliding_windows(sequences, labels, window_size=WINDOW_SIZE, step=WINDOW_STEP):
    """
    Create sliding window samples from sequences.

    This is the KEY to solving the data scarcity problem:
    - A document with 40 sentences can generate ~6-7 training samples
    - This provides sufficient data variability for RNN training

    Args:
        sequences: List of (seq_len, embedding_dim) tensors
        labels: List of labels
        window_size: Size of each window
        step: Step size for sliding

    Returns:
        windows: List of (window_size, embedding_dim) tensors
        window_labels: List of labels for each window
    """
    windows = []
    window_labels = []

    for seq, label in zip(sequences, labels):
        seq_len = seq.shape[0]

        if seq_len < window_size:
            # If sequence is too short, pad it
            padding = torch.zeros((window_size - seq_len, seq.shape[1]), device=seq.device)
            padded_seq = torch.cat([seq, padding], dim=0)
            windows.append(padded_seq)
            window_labels.append(label)
        else:
            # Create sliding windows
            for start_idx in range(0, seq_len - window_size + 1, step):
                end_idx = start_idx + window_size
                window = seq[start_idx:end_idx]
                windows.append(window)
                window_labels.append(label)

    return windows, window_labels

# ========================================
# Data Processing Pipeline
# ========================================

def collect_raw_documents():
    """
    Collect raw documents from dataset.

    Returns:
        List of (text, label) tuples
    """
    print(f"\nLoading dataset...")
    try:
        ds = load_dataset("artnitolog/llm-generated-texts", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    raw_samples = []
    excluded_cols = {'id', 'prompt', 'dataset_name', 'classes'}
    rows_processed = 0

    print(f"Collecting documents (Target: {NUM_ROWS_TO_USE} rows)...")

    for row in tqdm(ds, desc="Scanning Dataset"):
        if rows_processed >= NUM_ROWS_TO_USE:
            break

        all_cols = row.keys()
        human_col = next((c for c in all_cols if 'human' in c.lower()), None)
        if not human_col:
            continue

        # Find AI columns
        ai_candidates = []
        for col in all_cols:
            if col in excluded_cols or col == human_col:
                continue
            if row[col] and isinstance(row[col], str) and len(row[col]) > 10:
                ai_candidates.append(col)

        if not ai_candidates or not row[human_col]:
            continue

        # Add human sample
        raw_samples.append((row[human_col], 0))

        # Add AI sample
        selected_ai_col = random.choice(ai_candidates)
        raw_samples.append((row[selected_ai_col], 1))

        rows_processed += 1

    print(f"Total documents collected: {len(raw_samples)}")
    return raw_samples

def process_documents_to_sequences(documents, desc_prefix=""):
    """
    Process documents into sequences of [Original ⊕ Transformed] embeddings.

    This is the CORE of the sequence-aware approach:
    1. Split each document into sentences
    2. Transform sentences (Reduce → Inject)
    3. Create embedding pairs [Original, Transformed]
    4. Return sequences (one per document)

    Args:
        documents: List of (text, label) tuples
        desc_prefix: Prefix for progress bar descriptions

    Returns:
        sequences: List of (seq_len, embedding_dim * 2) tensors
        labels: List of labels
    """
    print(f"\n[{desc_prefix}] Processing {len(documents)} documents into sequences...")

    all_sequences = []
    all_labels = []

    for doc_text, label in tqdm(documents, desc=f"[{desc_prefix}] Processing docs"):
        # 1. Split into sentences
        sentences = get_sentences(doc_text)
        if not sentences:
            continue

        # Limit sentences per document
        sentences = sentences[:MAX_SENTENCES_PER_DOC]

        if len(sentences) < 3:  # Skip documents that are too short
            continue

        # 2. Transform sentences: Reduce → Inject
        reduced_sentences = transform_sentences(sentences, "reduce", description=None)
        injected_sentences = transform_sentences(reduced_sentences, "inject", description=None)

        # 3. Extract embeddings
        # Process in batches to avoid OOM
        orig_embeddings_list = []
        inj_embeddings_list = []

        for i in range(0, len(sentences), EMBEDDING_BATCH_SIZE):
            batch_orig = sentences[i:i+EMBEDDING_BATCH_SIZE]
            batch_inj = injected_sentences[i:i+EMBEDDING_BATCH_SIZE]

            emb_orig = get_sentence_embedding(batch_orig).cpu()
            emb_inj = get_sentence_embedding(batch_inj).cpu()

            orig_embeddings_list.append(emb_orig)
            inj_embeddings_list.append(emb_inj)

        # Concatenate all batches
        orig_embeddings = torch.cat(orig_embeddings_list, dim=0)
        inj_embeddings = torch.cat(inj_embeddings_list, dim=0)

        # 4. Create sequence: [Orig ⊕ Transformed] for each sentence
        # Shape: (num_sentences, embedding_dim * 2)
        sequence = torch.cat([orig_embeddings, inj_embeddings], dim=1)

        all_sequences.append(sequence)
        all_labels.append(label)

    print(f"[{desc_prefix}] Created {len(all_sequences)} sequences")
    return all_sequences, all_labels

# ========================================
# Training Functions
# ========================================

def train_model(train_loader, val_loader, embedding_dim, epochs=50, lr=0.001, hidden_dim=256, num_layers=2, dropout=0.3):
    """
    Train the RNN-based sequence detector.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        embedding_dim: Dimension of embeddings
        epochs: Number of training epochs
        lr: Learning rate
        hidden_dim: Hidden dimension for LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate for LSTM and classifier

    Returns:
        Trained model
    """
    print("\n=== Training RNN Sequence Detector ===")

    # Initialize model
    model = SequenceArtifactDetector(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("=" * 50)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)
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
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, "best_rnn_model.pth")
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}% precious")
    print(f"{'-'*50}\n")

    # Load best model
    checkpoint = torch.load("best_rnn_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

# ========================================
# Inference Functions
# ========================================

def predict_document(doc_text, model):
    """
    Predict whether a document is AI-generated or human-written.

    This function processes the entire document as a sequence, capturing
    the "rhythm" and "flow" of sentence transformations.

    Args:
        doc_text: Text to classify
        model: Trained RNN model

    Returns:
        score: Probability of being AI-generated (0-1)
        label: "AI Generated" or "Human Written"
    """
    # 1. Split into sentences
    sentences = get_sentences(doc_text)
    if not sentences or len(sentences) < 3:
        return 0.5, "Uncertain"

    sentences = sentences[:MAX_SENTENCES_PER_DOC]

    # 2. Transform
    reduced = transform_sentences(sentences, "reduce", description=None)
    injected = transform_sentences(reduced, "inject", description=None)

    # 3. Get embeddings
    emb_orig = get_sentence_embedding(sentences)
    emb_inj = get_sentence_embedding(injected)

    # 4. Create sequence
    sequence = torch.cat([emb_orig, emb_inj], dim=1).cpu()

    # 5. Create sliding windows
    windows, _ = create_sliding_windows([sequence], [0], window_size=WINDOW_SIZE, step=WINDOW_SIZE)

    # 6. Predict on each window and average
    model.eval()
    predictions = []

    with torch.no_grad():
        for window in windows:
            window = window.unsqueeze(0).to(device)  # Add batch dimension
            output = model(window)
            predictions.append(output.item())

    # Average predictions across windows
    avg_score = np.mean(predictions) if predictions else 0.5
    label = "AI Generated" if avg_score > 0.5 else "Human Written"

    return avg_score, label

# ========================================
# Main Execution
# ========================================

def main():
    print("\n" + "="*70)
    print("Sequence-Aware Artifact Detector (RNN/LSTM)")
    print("="*70)

    # 1. Collect documents
    raw_docs = collect_raw_documents()
    if not raw_docs:
        print("No documents collected. Exiting.")
        return

    # 2. Split into train/val/test
    train_docs, temp_docs = train_test_split(raw_docs, test_size=0.4, random_state=42)
    val_docs, test_docs = train_test_split(temp_docs, test_size=0.5, random_state=42)

    print(f"\nDocument split: Train={len(train_docs)}, Val={len(val_docs)}, Test={len(test_docs)}")

    # 3. Process documents into sequences
    print("\n--- Processing Training Data ---")
    train_sequences, train_labels = process_documents_to_sequences(train_docs, "Train")

    print("\n--- Processing Validation Data ---")
    val_sequences, val_labels = process_documents_to_sequences(val_docs, "Val")

    if not train_sequences or not val_sequences:
        print("Error: No sequences created. Exiting.")
        return

    # 4. Create sliding windows
    print("\n--- Creating Sliding Windows ---")
    train_windows, train_window_labels = create_sliding_windows(
        train_sequences, train_labels,
        window_size=WINDOW_SIZE,
        step=WINDOW_STEP
    )
    val_windows, val_window_labels = create_sliding_windows(
        val_sequences, val_labels,
        window_size=WINDOW_SIZE,
        step=WINDOW_STEP
    )

    print(f"Training windows: {len(train_windows)}")
    print(f"Validation windows: {len(val_windows)}")

    # 5. Create datasets and dataloaders
    train_dataset = SequenceDataset(train_windows, train_window_labels)
    val_dataset = SequenceDataset(val_windows, val_window_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 6. Train model
    embedding_dim = llm_model.config.hidden_size
    # Recommended model sizes
    recommended_hidden_dim = 128 # Smaller RNN
    recommended_num_layers = 1 # Shallower RNN
    recommended_dropout = 0.4

    model = train_model(
        train_loader, val_loader, embedding_dim,
        epochs=50, lr=0.001,
        hidden_dim=recommended_hidden_dim,
        num_layers=recommended_num_layers,
        dropout=recommended_dropout
    )

    # 7. Evaluate on test set (document level)
    print("\n--- Evaluating on Test Documents ---")

    all_scores = []
    all_preds = []
    all_true = []

    for doc_text, true_label in tqdm(test_docs, desc="Testing"):
        score, pred_label = predict_document(doc_text, model)
        pred = 1 if score > 0.5 else 0

        all_scores.append(score)
        all_preds.append(pred)
        all_true.append(true_label)

    # Calculate metrics
    accuracy = accuracy_score(all_true, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(all_true, all_preds)

    # Print results
    print("\n" + "="*70)
    print("FINAL TEST RESULTS (Document-Level)")
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

if __name__ == "__main__":
    main()
