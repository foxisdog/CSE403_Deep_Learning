import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
import sys
import re
from tqdm import tqdm
import random
import math

# --- Setup & Configuration ---
NUM_ROWS_TO_USE = 10
MAX_SENTENCES_PER_DOC = 20  # Reduced to minimize memory even further

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
    tokenizer_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,
    batch_size=1  # Reduced batch size to 1 to reduce memory footprint
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

# --- Advanced MLP (Same as optimized version) ---

class AdvancedArtifactDetectorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128], dropout=0.4):
        super(AdvancedArtifactDetectorMLP, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0] // 4),
            nn.Tanh(),
            nn.Linear(hidden_dims[0] // 4, hidden_dims[0]),
            nn.Sigmoid()
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

        self.output = nn.Sequential(
            nn.Linear(hidden_dims[3], 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

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
        x = self.input_proj(x)
        attention_weights = self.attention(x)
        x = x * attention_weights
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

# --- Helper Functions ---

def get_sentences(text):
    if not text or not isinstance(text, str):
        return []
    return nltk.tokenize.sent_tokenize(text)

def get_sentence_embedding(text):
    if not text:
        return torch.zeros((1, model.config.hidden_size), device=device)

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

# --- DOCUMENT-LEVEL TRANSFORMATION (NEW!) ---

def parse_numbered_output(output_text, expected_count):
    """
    Parse numbered output format:
    1. Sentence one.
    2. Sentence two.
    etc.
    """
    lines = output_text.strip().split('\n')
    sentences = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try to match pattern: "1. Sentence here"
        match = re.match(r'^\d+\.\s*(.+)$', line)
        if match:
            sentences.append(match.group(1).strip())
        elif len(sentences) < expected_count and line:
            # Fallback: treat as sentence if it doesn't start with number
            sentences.append(line)

    return sentences

# Statistics tracking
mismatch_stats = {
    'total_docs': 0,
    'mismatches': 0,
    'too_few': 0,
    'too_many': 0,
    'examples': []
}

def print_mismatch_statistics():
    """Print accumulated statistics about sentence count mismatches"""
    stats = mismatch_stats
    print(f"\n{'='*60}")
    print(f"SENTENCE COUNT MISMATCH STATISTICS")
    print(f"{'='*60}")
    print(f"Total Documents Processed: {stats['total_docs']}")
    print(f"Mismatches: {stats['mismatches']} ({stats['mismatches']/max(stats['total_docs'],1)*100:.1f}%%)")
    print(f"  - Too Few Sentences: {stats['too_few']}")
    print(f"  - Too Many Sentences: {stats['too_many']}")
    print(f"{'='*60}")

    if stats['examples']:
        print(f"\nSample Mismatches (showing first 3):")
        for i, ex in enumerate(stats['examples'][:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Expected: {ex['expected']} sentences")
            print(f"Got: {ex['got']} sentences")
            print(f"Original sentences:")
            for j, s in enumerate(ex['original'][:3]):
                print(f"  {j+1}. {s[:80]}...")
            print(f"Transformed output:")
            for j, s in enumerate(ex['transformed'][:3]):
                print(f"  {j+1}. {s[:80]}...")
            if len(ex['transformed']) > 3:
                print(f"  ... and {len(ex['transformed']) - 3} more")
        print(f"{'='*60}\n")

def run_transformation_document_level(doc_groups, mode, description="Transforming"):
    """
    Transform documents in ONE shot per document.

    Args:
        doc_groups: List of (doc_text, [sentences], label) tuples
        mode: "reduce" or "inject"

    Returns:
        List of transformed sentence lists
    """
    tokenizer.padding_side = "left"

    if mode == "reduce":
        task = "Simplify each sentence. Keep the main Subject, Verb, and Object."
        example = (
            "Example:\n"
            "Input: The incredibly talented musician played a beautiful symphony.\n"
            "Output: The musician played a symphony."
        )
    elif mode == "inject":
        task = "Rewrite each sentence to be more descriptive and vivid. Add adjectives and adverbs."
        example = (
            "Example:\n"
            "Input: The dog ran.\n"
            "Output: The energetic dog ran quickly."
        )

    results = []

    for doc_text, sentences, label in tqdm(doc_groups, desc=description):
        n_sents = len(sentences)
        mismatch_stats['total_docs'] += 1

        if n_sents == 0:
            results.append([])
            continue

        # Truncate if too long
        if n_sents > MAX_SENTENCES_PER_DOC:
            print(f"Warning: Document has {n_sents} sentences, truncating to {MAX_SENTENCES_PER_DOC}")
            sentences = sentences[:MAX_SENTENCES_PER_DOC]
            n_sents = MAX_SENTENCES_PER_DOC

        # Build numbered input
        numbered_input = '\n'.join([f"{i+1}. {s}" for i, s in enumerate(sentences)])

        prompt = f"""Task: {task}

{example}

CRITICAL INSTRUCTIONS:
- You will receive exactly {n_sents} sentences
- You MUST output exactly {n_sents} transformed sentences
- Use numbered format: 1. ... 2. ... 3. ...
- Do NOT merge or split sentences
- Do NOT add explanations

Input document ({n_sents} sentences):
{numbered_input}

Output ({n_sents} transformed sentences):"""

        chat_input = [
            {"role": "system", "content": "You are a precise text transformation assistant."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Generate
            output = pipe(
                [chat_input],
                max_new_tokens=min(len(doc_text) * 2, 1024),
                do_sample=False
            )

            generated_text = output[0][0]['generated_text']
            content = generated_text[-1]['content']

            # Parse numbered output
            transformed_sents = parse_numbered_output(content, n_sents)

            # Check for mismatch
            if len(transformed_sents) != n_sents:
                mismatch_stats['mismatches'] += 1

                if len(transformed_sents) < n_sents:
                    mismatch_stats['too_few'] += 1
                else:
                    mismatch_stats['too_many'] += 1

                # Store example (only first 5)
                if len(mismatch_stats['examples']) < 5:
                    mismatch_stats['examples'].append({
                        'expected': n_sents,
                        'got': len(transformed_sents),
                        'original': sentences,
                        'transformed': transformed_sents,
                        'mode': mode
                    })

                print(f"\n⚠️  MISMATCH DETECTED:")
                print(f"   Expected: {n_sents} sentences")
                print(f"   Got: {len(transformed_sents)} sentences")
                print(f"   Mode: {mode}")
                print(f"   Original (first 2):")
                for i, s in enumerate(sentences[:2]):
                    print(f"     {i+1}. {s[:100]}...")
                print(f"   Transformed (first 2):")
                for i, s in enumerate(transformed_sents[:2]):
                    print(f"     {i+1}. {s[:100]}...")
                print(f"   → Skipping this document\n")

                # Skip this document - use None as marker
                results.append(None)
            else:
                # Success!
                results.append(transformed_sents)

        except Exception as e:
            print(f"  ❌ Error transforming document: {e}. Skipping.")
            results.append(None)

    return results

# --- Data Processing ---

def collect_raw_documents():
    print(f"\n Loading Dataset...")
    try:
        ds = load_dataset("artnitolog/llm-generated-texts", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    raw_samples = []
    excluded_cols = {'id', 'prompt', 'dataset_name', 'classes'}
    rows_processed_count = 0

    print(f"Collecting samples (Target: {NUM_ROWS_TO_USE} Rows)...")

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

        raw_samples.append((row[human_col], 0))
        selected_ai_col = random.choice(ai_candidates)
        raw_samples.append((row[selected_ai_col], 1))

        rows_processed_count += 1

    print(f"Total Documents Collected: {len(raw_samples)}")
    return raw_samples

def create_sentence_dataset_document_level(documents, desc_prefix=""):
    """
    DOCUMENT-LEVEL STRATEGY:
    1. Group sentences by document
    2. Transform entire documents (1 LLM call per doc)
    3. Flatten back to sentence list
    4. Embed and return features
    """
    print(f"\n[{desc_prefix}] Processing {len(documents)} documents (DOCUMENT-LEVEL)...")

    # Reset statistics for this phase
    mismatch_stats['total_docs'] = 0
    mismatch_stats['mismatches'] = 0
    mismatch_stats['too_few'] = 0
    mismatch_stats['too_many'] = 0
    mismatch_stats['examples'] = []

    # 1. Prepare document groups
    doc_groups = []
    for text, label in documents:
        sents = get_sentences(text)
        if not sents: continue
        doc_groups.append((text, sents, label))

    total_sents = sum(len(sents) for _, sents, _ in doc_groups)
    print(f"[{desc_prefix}] Total: {len(doc_groups)} docs, {total_sents} sentences")

    # 2. Document-level transformation (FAST!)
    reduced_doc_groups = run_transformation_document_level(
        doc_groups,
        "reduce",
        description=f"[{desc_prefix}] Reducing (doc-level)"
    )

    # Filter out None results before inject
    valid_for_inject = []
    for (text, orig_sents, label), reduced_sents in zip(doc_groups, reduced_doc_groups):
        if reduced_sents is not None:
            valid_for_inject.append((text, reduced_sents, label))

    print(f"[{desc_prefix}] After reduce: {len(valid_for_inject)}/{len(doc_groups)} valid docs")

    injected_doc_groups = run_transformation_document_level(
        valid_for_inject,
        "inject",
        description=f"[{desc_prefix}] Injecting (doc-level)"
    )

    # Print statistics
    print_mismatch_statistics()

    # 3. Flatten back to sentence list (only valid documents)
    all_sentences = []
    all_labels = []
    reduced_sentences = []
    injected_sentences = []

    skipped_count = 0
    valid_idx = 0

    for i, (doc_text, orig_sents, label) in enumerate(doc_groups):
        red_sents = reduced_doc_groups[i]

        if red_sents is None:
            skipped_count += 1
            continue

        inj_sents = injected_doc_groups[valid_idx]
        valid_idx += 1

        if inj_sents is None:
            skipped_count += 1
            continue

        # Both valid - add to dataset
        all_sentences.extend(orig_sents)
        reduced_sentences.extend(red_sents)
        injected_sentences.extend(inj_sents)
        all_labels.extend([label] * len(orig_sents))

    print(f"[{desc_prefix}] ✓ Successfully processed: {len(doc_groups) - skipped_count}/{len(doc_groups)} docs")
    print(f"[{desc_prefix}] ✓ Flattened: {len(all_sentences)} sentences")

    # 4. Feature extraction
    print(f"[{desc_prefix}] Calculating Embeddings...")
    batch_size = 32
    features_list = []

    for i in tqdm(range(0, len(all_sentences), batch_size), desc=f"[{desc_prefix}] Embedding"):
        batch_orig = all_sentences[i : i+batch_size]
        batch_inj = injected_sentences[i : i+batch_size]

        emb_orig = get_sentence_embedding(batch_orig)
        emb_inj = get_sentence_embedding(batch_inj)

        feats = torch.cat((emb_orig, emb_inj), dim=1)
        features_list.append(feats.cpu())

    if not features_list:
        return None, None

    X = torch.cat(features_list, dim=0)
    y = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)

    return X, y

# --- Training (Same as optimized) ---

class FocalLoss(nn.Module):
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
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]

def train_advanced_model(X_train, y_train, X_val, y_val):
    print("\n=== Advanced Training Pipeline ===")

    input_dim = X_train.shape[1]
    model = AdvancedArtifactDetectorMLP(
        input_dim=input_dim,
        hidden_dims=[1024, 512, 256, 128],
        dropout=0.4
    ).to(device)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)

    epochs = 50
    warmup_epochs = 5
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs, min_lr=1e-6)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True, drop_last=True)

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    gradient_clip_value = 1.0

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    print(f"Epochs: {epochs}, Warmup: {warmup_epochs}, Patience: {patience}")
    print("=" * 50)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_X, batch_y in progress_bar:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()
            val_pred = (val_out > 0.5).float()
            val_acc = (val_pred == y_val).sum().item() / len(y_val) * 100

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%% | "
              f"LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, "best_model_document_level.pth")
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    print(f"\n{'='*50}")
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%%")
    print(f"{'='*50}\n")

    checkpoint = torch.load("best_model_document_level.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

# --- Inference ---

def predict_essay(essay_text, mlp_model, num_mc_samples=5):
    """Document-level prediction with MC Dropout"""
    sentences = get_sentences(essay_text)
    if not sentences:
        return 0.5, "Uncertain"

    # Document-level transformation (1 call!)
    doc_groups = [(essay_text, sentences, 0)]

    red_groups = run_transformation_document_level(doc_groups, "reduce", description=None)

    # Check if reduce succeeded
    if red_groups[0] is None:
        print(f"Warning: Reduce transformation failed for essay. Returning uncertain.")
        return 0.5, "Uncertain (transformation failed)"

    inj_groups = run_transformation_document_level(
        [(essay_text, red_groups[0], 0)],
        "inject",
        description=None
    )

    # Check if inject succeeded
    if inj_groups[0] is None:
        print(f"Warning: Inject transformation failed for essay. Returning uncertain.")
        return 0.5, "Uncertain (transformation failed)"

    red_sents = red_groups[0]
    inj_sents = inj_groups[0]

    # Embed
    with torch.no_grad():
        emb_o = get_sentence_embedding(sentences)
        emb_i = get_sentence_embedding(inj_sents)
        features = torch.cat((emb_o, emb_i), dim=1)

    # MC Dropout
    mlp_model.train()
    mc_predictions = []
    with torch.no_grad():
        for _ in range(num_mc_samples):
            probs = mlp_model(features)
            mc_predictions.append(probs)

    mc_predictions = torch.stack(mc_predictions, dim=0)
    mean_probs = mc_predictions.mean(dim=0)
    mean_score = mean_probs.mean().item()

    label = "AI Generated" if mean_score > 0.5 else "Human Written"
    return mean_score, label

# --- Main ---

def main():
    raw_docs = collect_raw_documents()
    if not raw_docs: return

    train_docs, temp_docs = train_test_split(raw_docs, test_size=0.4, random_state=42)
    val_docs, test_docs = train_test_split(temp_docs, test_size=0.5, random_state=42)

    print(f"\nDocument Split: Train={len(train_docs)}, Val={len(val_docs)}, Test={len(test_docs)}")

    print("\n--- Processing Training Data (DOCUMENT-LEVEL) ---")
    X_train, y_train = create_sentence_dataset_document_level(train_docs, desc_prefix="Train")

    print("\n--- Processing Validation Data (DOCUMENT-LEVEL) ---")
    X_val, y_val = create_sentence_dataset_document_level(val_docs, desc_prefix="Val")

    if X_train is None or X_val is None:
        print("Error creating datasets.")
        return

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    mlp = train_advanced_model(X_train, y_train, X_val, y_val)

    print("\n--- Final Evaluation (Document-Level Voting + MC Dropout) ---")

    correct_docs = 0
    total_docs = len(test_docs)

    for text, label in tqdm(test_docs, desc="Voting on Docs"):
        score, pred_label = predict_essay(text, mlp, num_mc_samples=5)
        is_ai = 1 if score > 0.5 else 0
        if is_ai == label:
            correct_docs += 1

    doc_acc = (correct_docs / total_docs) * 100
    print(f"\n{'-'*40}")
    print(f"FINAL DOCUMENT ACCURACY (Document-Level + MC Dropout): {doc_acc:.2f}%%")
    print(f"Correct: {correct_docs} / {total_docs}")
    print(f"{'-'*40}")

if __name__ == "__main__":
    main()
