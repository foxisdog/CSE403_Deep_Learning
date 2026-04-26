import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
import logging
import sys
import os
from tqdm import tqdm
import random
import math

# --- Setup & Configuration ---
NUM_ROWS_TO_USE = 10  # Strict limit: Process exactly this many rows (e.g., 100 rows -> 200 samples)
MAX_SENTENCES_PER_DOC = 20 # Limit sentences per document to reduce memory usage during LLM inference

# 1. Login
hf_token = ""

if hf_token:
    try:
        login(token=hf_token)
    except Exception as e:
        print(f"Warning: Login failed. {e}")
else:
    print("No HF token provided. Running anonymously.")

# 2. Model Configuration
model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

print(f"Loading {model_id} on {device}...")

try:
    # Load Tokenizer
    tokenizer_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load Model
    model_kwargs = {"token": hf_token} if hf_token else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
        **model_kwargs
    )
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")
    sys.exit(1)

# Initialize Generation Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,
    batch_size=8 # Changed batch size back to 8 as requested
)

# NLTK Setup
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Advanced MLP Classifier with Residual Connections ---

class ResidualBlock(nn.Module):
    """Residual block with skip connection for better gradient flow"""
    def __init__(self, dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += identity  # Skip connection
        out = self.relu(out)

        return out

class AdvancedArtifactDetectorMLP(nn.Module):
    """
    Advanced MLP with:
    - Batch Normalization for stable training
    - Residual connections for better gradient flow
    - Layer Normalization
    - Attention mechanism for feature weighting
    - Multiple dropout layers for regularization
    """
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128], dropout=0.4):
        super(AdvancedArtifactDetectorMLP, self).__init__()

        # Input projection with Batch Normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.BatchNorm1d(hidden_dims[3]),
            nn.ReLU(),
            nn.Dropout(dropout * 0.4)
        )

        # Self-attention layer for feature weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0] // 4),
            nn.Tanh(),
            nn.Linear(hidden_dims[0] // 4, hidden_dims[0]),
            nn.Sigmoid()
        )

        # Final classification layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[3], 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Initialize weights using He initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)

        # Apply attention weighting
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Deep layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Output
        x = self.output(x)

        return x

# --- Helper Functions ---

def get_sentences(text):
    """Returns ALL sentences from the text."""
    if not text or not isinstance(text, str):
        return []
    return nltk.tokenize.sent_tokenize(text)

def get_sentence_embedding(text):
    """
    Calculates embedding for a single sentence (or batch of sentences).
    Uses standard Mean Pooling (no chunking needed for sentences).
    """
    if not text:
        return torch.zeros((1, model.config.hidden_size), device=device)

    # Ensure padding side is right for embedding extraction
    tokenizer.padding_side = "right"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.hidden_states[-1]
    mask = inputs['attention_mask'].unsqueeze(-1)

    sum_emb = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_emb = sum_emb / counts

    return mean_emb

def run_transformation_pipeline(texts, mode, description="Transforming", doc_contexts=None):
    """
    Runs the pipeline using a GENERATOR to maximize GPU throughput.

    Args:
        texts: List of sentences to transform
        mode: "reduce" or "inject"
        description: Progress bar description
        doc_contexts: Optional list of (doc_text, sent_start_idx, sent_end_idx) for context
    """
    # Switch padding to left for generation
    tokenizer.padding_side = "left"

    if mode == "reduce":
        task_instruction = "Simplify the TARGET sentence. Keep the main Subject, Verb, and Object."
        rule = "Output ONLY ONE simplified sentence. Do not add extra sentences. Do not explain."
    elif mode == "inject":
        task_instruction = "Rewrite the TARGET sentence to be more descriptive and vivid."
        rule = "Output ONLY ONE rewritten sentence. Do not add extra sentences. Do not explain."

    chat_inputs = []

    for i, sentence in enumerate(texts):
        # Build prompt with or without context
        if doc_contexts and i < len(doc_contexts):
            doc_text, sent_idx_in_doc = doc_contexts[i]
            # Provide full document context
            prompt = (
                f"Task: {task_instruction}\n"
                f"Rule: {rule}\n\n"
                f"Full document for context:\n{doc_text}\n\n"
                f"TARGET sentence to transform: {sentence}\n"
                f"Transformed sentence:"
            )
        else:
            # Fallback: No context (original behavior)
            prompt = (
                f"Task: {task_instruction}\n"
                f"Rule: {rule}\n"
                f"Input Sentence: {sentence}"
            )

        chat_inputs.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])


    def input_generator():
        for item in chat_inputs:
            yield item

    results = []

    pipeline_iterator = pipe(
        input_generator(),
        batch_size=8, # Changed batch size to 8 as requested
        max_new_tokens=64,
        do_sample=False
    )

    # Check if we should show progress bar (description provided)
    iterator = tqdm(pipeline_iterator, total=len(texts), desc=description) if description else pipeline_iterator

    for idx, out in enumerate(iterator):
        try:
            generated_conv = out[0]['generated_text']
            content = generated_conv[-1]['content']

            if "Input Sentence:" in content:
                content = content.split("Input Sentence:")[-1].strip()

            # Clean up: Remove any explanations or extra text
            content = content.strip()

            # Validation: Check if output is a single sentence
            output_sentences = get_sentences(content)

            if len(output_sentences) == 0:
                # Fallback: Use original sentence
                results.append(texts[idx] if idx < len(texts) else "")
            elif len(output_sentences) == 1:
                # Perfect: Single sentence output
                results.append(output_sentences[0])
            else:
                # Multiple sentences: Take only the first one
                results.append(output_sentences[0])

        except Exception as e:
            # Fallback: Use original sentence on error
            results.append(texts[idx] if idx < len(texts) else "")

    return results

# --- Core Logic: Flattening & Processing ---

def collect_raw_documents():
    """
    Collects raw documents (Human/AI pairs) up to TOTAL_DATASET_LIMIT.
    Returns: List of tuples (text, label)
    """
    print(f"\n Loading FULL Dataset...")
    try:
        ds = load_dataset("artnitolog/llm-generated-texts", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    raw_samples = []
    excluded_cols = {'id', 'prompt', 'dataset_name', 'classes'}
    rows_processed_count = 0

    print(f"Collecting valid samples (Target: {NUM_ROWS_TO_USE} Rows)...")

    for row in tqdm(ds, desc="Scanning Dataset"):
        if NUM_ROWS_TO_USE is not None and rows_processed_count >= NUM_ROWS_TO_USE:
            break

        all_cols = row.keys()
        human_col = next((c for c in all_cols if 'human' in c.lower()), None)
        if not human_col: continue

        ai_candidates = []
        for col in all_cols:
            if col in excluded_cols: continue
            if col == human_col: continue
            if row[col] and isinstance(row[col], str) and len(row[col]) > 10:
                ai_candidates.append(col)

        if not ai_candidates or not row[human_col]: continue

        # 1. Human Sample
        raw_samples.append((row[human_col], 0))

        # 2. AI Sample (Random)
        selected_ai_col = random.choice(ai_candidates)
        raw_samples.append((row[selected_ai_col], 1))

        rows_processed_count += 1

    print(f"Total Documents Collected: {len(raw_samples)}")
    return raw_samples

def create_sentence_dataset(documents, desc_prefix=""):
    """
    FLATTENING STRATEGY WITH CONTEXT:
    1. Split each doc into sentences.
    2. Inherit label (0/1).
    3. Run Reduce -> Inject on ALL sentences WITH document context.
    4. Embed Original & Injected sentences.
    5. Concat and return Tensor X, y.
    """
    print(f"\n[{desc_prefix}] Flattening {len(documents)} documents into sentences...")

    all_sentences = []
    all_labels = []
    doc_contexts = []  # Store (full_doc_text, sent_start_idx, sent_end_idx) for each sentence

    # 1. Extract Sentences & Inherit Labels + Build Context Info
    for text, label in documents:
        sents = get_sentences(text)
        if not sents: continue

        # Limit sentences per document for memory management
        sents = sents[:MAX_SENTENCES_PER_DOC]

        # For each sentence, store the full document text for context
        for sent_idx, sent in enumerate(sents):
            all_sentences.append(sent)
            all_labels.append(label)
            doc_contexts.append((text, sent_idx))  # (full_doc, sentence_position)

    print(f"[{desc_prefix}] Total Sentences: {len(all_sentences)}")
    print(f"[{desc_prefix}] Document contexts prepared: {len(doc_contexts)}")

    # 2. Batch Transformation WITH CONTEXT (Reduce -> Inject)
    reduced_sentences = run_transformation_pipeline(
        all_sentences,
        "reduce",
        description=f"[{desc_prefix}] Reducing (with context)",
        doc_contexts=doc_contexts
    )

    # CRITICAL: Verify sentence count is preserved
    assert len(reduced_sentences) == len(all_sentences), \
        f"[ERROR] Sentence count mismatch after reduce: {len(all_sentences)} \u2192 {len(reduced_sentences)}"

    # For inject, we can reuse the same contexts (reduced sentences still belong to same docs)
    injected_sentences = run_transformation_pipeline(
        reduced_sentences,
        "inject",
        description=f"[{desc_prefix}] Injecting (with context)",
        doc_contexts=doc_contexts
    )

    # CRITICAL: Verify sentence count is preserved
    assert len(injected_sentences) == len(reduced_sentences), \
        f"[ERROR] Sentence count mismatch after inject: {len(reduced_sentences)} \u2192 {len(injected_sentences)}"

    print(f"[{desc_prefix}] \u2713 Sentence count preserved: {len(all_sentences)} \u2192 {len(reduced_sentences)} \u2192 {len(injected_sentences)}")

    # 3. Feature Extraction (Embeddings)
    print(f"[{desc_prefix}] Calculating Sentence Embeddings...")

    # Process embeddings in batches to prevent OOM
    batch_size = 32
    features_list = []

    for i in tqdm(range(0, len(all_sentences), batch_size), desc=f"[{desc_prefix}] Embedding"):
        batch_orig = all_sentences[i : i+batch_size]
        batch_inj = injected_sentences[i : i+batch_size]

        # Get batch embeddings
        emb_orig = get_sentence_embedding(batch_orig)
        emb_inj = get_sentence_embedding(batch_inj)

        # Concat: [Orig, Inj]
        feats = torch.cat((emb_orig, emb_inj), dim=1)
        features_list.append(feats.cpu()) # Store on CPU

    if not features_list:
        return None, None

    X = torch.cat(features_list, dim=0)
    y = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)

    return X, y

# --- Advanced Training with Optimization Techniques ---

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup and cosine annealing
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]

def train_advanced_model(X_train, y_train, X_val, y_val):
    """
    Advanced training with:
    - AdamW optimizer with weight decay
    - Cosine annealing with warmup
    - Gradient clipping
    - Focal loss for class imbalance
    - Early stopping
    - Model checkpointing
    """
    print("\n=== Advanced Training Pipeline ===")

    input_dim = X_train.shape[1]

    # Initialize model
    model = AdvancedArtifactDetectorMLP(
        input_dim=input_dim,
        hidden_dims=[1024, 512, 256, 128],
        dropout=0.4
    ).to(device)

    # Loss function (Focal Loss for hard examples)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # Will be adjusted by scheduler
        betas=(0.9, 0.999),
        weight_decay=0.01,  # L2 regularization
        eps=1e-8
    )

    # Learning rate scheduler with warmup
    epochs = 50
    warmup_epochs = 5
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        min_lr=1e-6
    )

    # Data loader with shuffle
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True,
        drop_last=True  # For batch normalization
    )

    # Training configuration
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    gradient_clip_value = 1.0

    train_losses = []
    val_losses = []
    val_accs = []

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Epochs: {epochs}, Warmup: {warmup_epochs}, Patience: {patience}")
    print(f"Optimizer: AdamW (lr=0.001, weight_decay=0.01)")
    print(f"Scheduler: Cosine Annealing with Warmup")
    print(f"Loss: Focal Loss (alpha=0.25, gamma=2.0)")
    print("=" * 50)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_X, batch_y in progress_bar:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

            optimizer.step()

            train_loss += loss.item()

            # Calculate training accuracy
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()
            val_pred = (val_out > 0.5).float()
            val_acc = (val_pred == y_val).sum().item() / len(y_val) * 100

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Model checkpointing and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, "best_model.pth")
            print(f"  \u2713 New best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'-'*50}\n")

    # Load best model
    checkpoint = torch.load("best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

# --- Inference: Sentence-Level Voting with Monte Carlo Dropout ---

def predict_essay(essay_text, mlp_model, num_mc_samples=5):
    """
    Voting System with Monte Carlo Dropout and Document Context:
    1. Split essay into N sentences.
    2. Process each sentence with full document context -> Probability (with MC dropout).
    3. Take Mean of Probabilities.
    4. Threshold > 0.5 -> AI.

    Monte Carlo Dropout provides better uncertainty estimation.
    """
    # 1. Split
    sentences = get_sentences(essay_text)
    if not sentences:
        return 0.5, "Uncertain"

    # Limit sentences per document for memory management
    sentences = sentences[:MAX_SENTENCES_PER_DOC]

    # 2. Prepare document context for each sentence
    doc_contexts = [(essay_text, i) for i in range(len(sentences))]

    # 3. Transform (with document context)
    red = run_transformation_pipeline(
        sentences,
        "reduce",
        description=None,
        doc_contexts=doc_contexts
    )
    inj = run_transformation_pipeline(
        red,
        "inject",
        description=None,
        doc_contexts=doc_contexts
    )

    # 3. Embed
    with torch.no_grad():
        emb_o = get_sentence_embedding(sentences)
        emb_i = get_sentence_embedding(inj)
        features = torch.cat((emb_o, emb_i), dim=1)

    # 4. Monte Carlo Dropout inference for better predictions
    mlp_model.eval() # Set model to evaluation mode to handle BatchNorm1d correctly with batch_size=1
    for module in mlp_model.modules():
        if isinstance(module, nn.Dropout):
            module.train() # Re-enable dropout for Monte Carlo sampling

    mc_predictions = []
    with torch.no_grad():
        for _ in range(num_mc_samples):
            probs = mlp_model(features)
            mc_predictions.append(probs)

    # Average predictions across MC samples
    mc_predictions = torch.stack(mc_predictions, dim=0)
    mean_probs = mc_predictions.mean(dim=0)

    # 5. Vote (Mean across sentences)
    mean_score = mean_probs.mean().item()

    label = "AI Generated" if mean_score > 0.5 else "Human Written"
    return mean_score, label

# --- Main Execution Flow ---

def main():
    # 1. Collect Raw Documents
    raw_docs = collect_raw_documents()
    if not raw_docs: return

    # 2. Split Documents (Train/Val/Test)
    train_docs, temp_docs = train_test_split(raw_docs, test_size=0.4, random_state=42)
    val_docs, test_docs = train_test_split(temp_docs, test_size=0.5, random_state=42)

    print(f"\nDocument Split: Train={len(train_docs)}, Val={len(val_docs)}, Test={len(test_docs)}")

    # 3. Prepare Sentence-Level Training Data
    print("\n--- Processing Training Data (Flattening) ---")
    X_train, y_train = create_sentence_dataset(train_docs, desc_prefix="Train")

    print("\n--- Processing Validation Data (Flattening) ---")
    X_val, y_val = create_sentence_dataset(val_docs, desc_prefix="Val")

    if X_train is None or X_val is None:
        print("Error creating datasets.")
        return

    # Move to GPU
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    # 4. Train Advanced MLP
    mlp = train_advanced_model(X_train, y_train, X_val, y_val)

    # 5. Final Evaluation (Document Level Voting with MC Dropout)
    print("\n--- Final Evaluation on Test Documents (Voting System with MC Dropout) ---")

    correct_docs = 0
    total_docs = len(test_docs)

    for text, label in tqdm(test_docs, desc="Voting on Docs"):
        score, pred_label = predict_essay(text, mlp, num_mc_samples=5)

        # Check correctness
        is_ai = 1 if score > 0.5 else 0
        if is_ai == label:
            correct_docs += 1

    doc_acc = (correct_docs / total_docs) * 100
    print(f"\n{'-'*40}")
    print(f"FINAL DOCUMENT ACCURACY (Voting + MC Dropout): {doc_acc:.2f}%")
    print(f"Correct: {correct_docs} / {total_docs}")
    print(f"{'-'*40}")

if __name__ == "__main__":
    main()
