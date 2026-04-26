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
import logging
import sys
import os
from tqdm import tqdm
import random

# --- Setup & Configuration ---
NUM_ROWS_TO_USE = None  # Strict limit: Process exactly this many rows (e.g., 100 rows -> 200 samples)

# 1. Login
hf_token = "YOUR_HF_TOKEN_HERE"

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
    # Note: Using 'torch_dtype' as it is the correct argument for Hugging Face AutoModel.
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
    batch_size=32
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

# --- MLP Classifier Definition ---

class ArtifactDetectorMLP(nn.Module):
    def __init__(self, input_dim):
        super(ArtifactDetectorMLP, self).__init__()
        # Input dim will be 2 * Hidden_Size (Original + Injected)
        self.layer1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# --- Helper Functions ---

def get_sentences(text):
    """Returns ALL sentences from the text (No slicing)."""
    if not text or not isinstance(text, str):
        return []
    return nltk.tokenize.sent_tokenize(text)

def run_transformation_pipeline(texts, mode, description="Transforming"):
    """
    Runs the pipeline using a GENERATOR to maximize GPU throughput.
    """
    tokenizer.padding_side = "left"

    if mode == "reduce":
        system_prompt = (
            "Task: Simplify the sentence. Keep the main Subject, Verb, and Object.\n"
            "Rule: Do not change the meaning. Do not loop. Remove extra details.\n"
            "Input Sentence: "
        )
    elif mode == "inject":
        system_prompt = (
            "Task: Rewrite the sentence to be more descriptive and vivid.\n"
            "Rule: Add adjectives and adverbs. Keep the original meaning.\n"
            "Input Sentence: "
        )

    # 1. Prepare Inputs
    chat_inputs = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": system_prompt + t}
        ] for t in texts
    ]

    # 2. Define Generator
    def input_generator():
        for item in chat_inputs:
            yield item

    results = []

    # 3. Iterate over the pipeline output
    pipeline_iterator = pipe(
        input_generator(),
        batch_size=32,
        max_new_tokens=64,
        do_sample=False
    )

    # Use tqdm on the iterator for visual feedback
    for out in tqdm(pipeline_iterator, total=len(texts), desc=description):
        try:
            generated_conv = out[0]['generated_text']
            content = generated_conv[-1]['content']

            if "Input Sentence:" in content:
                content = content.split("Input Sentence:")[-1].strip()
            results.append(content.strip())
        except Exception:
            results.append("")

    return results

def get_embedding_smart_chunking(text, max_tokens=512):
    """
    Calculates document embedding using Sentence-Aware Smart Chunking.
    """
    if not text:
        return torch.zeros((1, model.config.hidden_size), device=device)

    sentences = get_sentences(text)
    if not sentences:
        return torch.zeros((1, model.config.hidden_size), device=device)

    # 1. Create Chunks
    chunks = []
    current_chunk = []
    current_length = 0

    tokenizer.padding_side = "right"

    for sent in sentences:
        token_len = len(tokenizer.encode(sent, add_special_tokens=False))

        if current_length + token_len > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sent)
        current_length += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # 2. Embed Chunks
    chunk_embeddings = []

    batch_size = 4
    for i in range(0, len(chunks), batch_size):
        batch_texts = chunks[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.hidden_states[-1]
        mask = inputs['attention_mask'].unsqueeze(-1)

        sum_emb = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_emb = sum_emb / counts

        chunk_embeddings.append(mean_emb)

    # 3. Average across chunks
    if not chunk_embeddings:
        return torch.zeros((1, model.config.hidden_size), device=device)

    all_chunks_tensor = torch.cat(chunk_embeddings, dim=0)
    doc_embedding = torch.mean(all_chunks_tensor, dim=0, keepdim=True)

    return doc_embedding

# --- Dataset Processing (Strict & Balanced) ---

def process_full_dataset():
    print(f"\n Loading FULL Dataset...")
    try:
        ds = load_dataset("artnitolog/llm-generated-texts", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    # 1. Collect Raw Data (Strict 1:1 Logic)
    raw_samples = []

    # Exclude metadata columns
    excluded_cols = {'id', 'prompt', 'dataset_name', 'classes'}

    # Counter for Processed Source Rows
    rows_processed_count = 0

    print(f"Collecting valid samples (Target: {NUM_ROWS_TO_USE} Rows)...")

    for row in tqdm(ds, desc="Scanning Dataset"):
        # Check Stop Condition
        if NUM_ROWS_TO_USE is not None and rows_processed_count >= NUM_ROWS_TO_USE:
            break

        all_cols = row.keys()

        # Identify Human Column
        human_col = next((c for c in all_cols if 'human' in c.lower()), None)
        if not human_col: continue # Skip if no human text found

        # Identify AI Candidate Columns
        ai_candidates = []
        for col in all_cols:
            if col in excluded_cols: continue
            if col == human_col: continue

            # Ensure the column has content
            if row[col] and isinstance(row[col], str) and len(row[col]) > 10:
                ai_candidates.append(col)

        if not ai_candidates: continue # Skip if no AI text found
        if not row[human_col]: continue

        # --- Balanced Sampling Logic ---

        # 1. Add Human Sample (Label 0)
        human_text = row[human_col]
        raw_samples.append((human_text, 0))

        # 2. Add Random AI Sample (Label 1)
        selected_ai_col = random.choice(ai_candidates)
        ai_text = row[selected_ai_col]
        raw_samples.append((ai_text, 1))

        # Increment Row Counter (1 Row = 1 Human + 1 AI)
        rows_processed_count += 1

    print(f"\nValidation:")
    print(f"Target Rows: {NUM_ROWS_TO_USE}")
    print(f"Processed Rows: {rows_processed_count}")
    print(f"Total Samples Collected: {len(raw_samples)} (Should be {2 * rows_processed_count})")
    print(f"Human Samples: {sum(1 for _, l in raw_samples if l == 0)}")
    print(f"AI Samples: {sum(1 for _, l in raw_samples if l == 1)}")

    # 2. Flatten into Sentences
    all_sentences = []
    doc_boundaries = []
    current_idx = 0
    valid_samples = []

    print("Tokenizing sentences (FULL TEXT)...")
    for text, label in raw_samples:
        sents = get_sentences(text) # ALL sentences
        if not sents: continue

        all_sentences.extend(sents)
        doc_boundaries.append((current_idx, current_idx + len(sents)))
        valid_samples.append((text, label))
        current_idx += len(sents)

    # 3. Batch Transformation
    reduced_sentences = run_transformation_pipeline(
        all_sentences,
        "reduce",
        description="Step 1/2: Reducing"
    )

    injected_sentences = run_transformation_pipeline(
        reduced_sentences,
        "inject",
        description="Step 2/2: Injecting"
    )

    # 4. Feature Extraction
    print("\nCalculating Smart Chunked Embeddings...")
    features_list = []
    labels_list = []

    for i, (start, end) in enumerate(tqdm(doc_boundaries, desc="Embedding Docs")):
        inj_segment = " ".join(injected_sentences[start:end])
        orig_text = valid_samples[i][0]

        emb_orig = get_embedding_smart_chunking(orig_text)
        emb_inj = get_embedding_smart_chunking(inj_segment)

        feat = torch.cat((emb_orig, emb_inj), dim=1)

        features_list.append(feat.cpu())
        labels_list.append(valid_samples[i][1])

    if not features_list:
        return None, None

    X = torch.cat(features_list, dim=0)
    y = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)

    return X, y

# --- Training & Evaluation ---

def train_and_evaluate(X, y):
    print("\n Splitting Data (60/20/20)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    input_dim = X.shape[1]
    mlp = ArtifactDetectorMLP(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.0005)

    epochs = 20
    best_val_acc = 0.0

    print("\n Starting Training...")

    for epoch in range(epochs):
        mlp.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = mlp(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        mlp.eval()
        with torch.no_grad():
            val_out = mlp(X_val)
            val_loss = criterion(val_out, y_val).item()
            val_pred = (val_out > 0.5).float()
            val_acc = (val_pred == y_val).sum().item() / len(y_val) * 100

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(mlp.state_dict(), "best_model.pth")

    print(f"\n Training Complete. Best Val Acc: {best_val_acc:.2f}%")

    print("\n Evaluating on Test Set...")
    mlp.load_state_dict(torch.load("best_model.pth", weights_only=True))
    mlp.eval()

    with torch.no_grad():
        test_out = mlp(X_test)
        test_pred = (test_out > 0.5).float()
        test_acc = (test_pred == y_test).sum().item() / len(y_test) * 100

    print(f"{'='*30}")
    print(f"FINAL TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'='*30}")

    return mlp

def main():
    # 1. Process Data
    X, y = process_full_dataset()

    if X is None:
        print("Data preparation failed.")
        return

    # 2. Train & Eval
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()

# ---

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
import logging
import sys
import os
from tqdm import tqdm
import random

# --- Setup & Configuration ---
NUM_ROWS_TO_USE = 100  # Strict limit: Process exactly this many rows (e.g., 100 rows -> 200 samples)

# 1. Login
hf_token = "YOUR_HF_TOKEN_HERE"

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
    # Note: Using 'torch_dtype' as it is the correct argument for Hugging Face AutoModel.
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
    batch_size=32
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

# --- MLP Classifier Definition ---

class ArtifactDetectorMLP(nn.Module):
    def __init__(self, input_dim):
        super(ArtifactDetectorMLP, self).__init__()
        # Input dim will be 2 * Hidden_Size (Original + Injected)
        self.layer1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# --- Helper Functions ---

def get_sentences(text):
    """Returns ALL sentences from the text (No slicing)."""
    if not text or not isinstance(text, str):
        return []
    return nltk.tokenize.sent_tokenize(text)

def run_transformation_pipeline(texts, mode, description="Transforming"):
    """
    Runs the pipeline using a GENERATOR to maximize GPU throughput.
    """
    tokenizer.padding_side = "left"

    if mode == "reduce":
        system_prompt = (
            "Task: Simplify the sentence. Keep the main Subject, Verb, and Object.\n"
            "Rule: Do not change the meaning. Do not loop. Remove extra details.\n"
            "Input Sentence: "
        )
    elif mode == "inject":
        system_prompt = (
            "Task: Rewrite the sentence to be more descriptive and vivid.\n"
            "Rule: Add adjectives and adverbs. Keep the original meaning.\n"
            "Input Sentence: "
        )

    # 1. Prepare Inputs
    chat_inputs = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": system_prompt + t}
        ] for t in texts
    ]

    # 2. Define Generator
    def input_generator():
        for item in chat_inputs:
            yield item

    results = []

    # 3. Iterate over the pipeline output
    pipeline_iterator = pipe(
        input_generator(),
        batch_size=32,
        max_new_tokens=64,
        do_sample=False
    )

    # Use tqdm on the iterator for visual feedback
    for out in tqdm(pipeline_iterator, total=len(texts), desc=description):
        try:
            generated_conv = out[0]['generated_text']
            content = generated_conv[-1]['content']

            if "Input Sentence:" in content:
                content = content.split("Input Sentence:")[-1].strip()
            results.append(content.strip())
        except Exception:
            results.append("")

    return results

def get_embedding_smart_chunking(text, max_tokens=512):
    """
    Calculates document embedding using Sentence-Aware Smart Chunking.
    """
    if not text:
        return torch.zeros((1, model.config.hidden_size), device=device)

    sentences = get_sentences(text)
    if not sentences:
        return torch.zeros((1, model.config.hidden_size), device=device)

    # 1. Create Chunks
    chunks = []
    current_chunk = []
    current_length = 0

    tokenizer.padding_side = "right"

    for sent in sentences:
        token_len = len(tokenizer.encode(sent, add_special_tokens=False))

        if current_length + token_len > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sent)
        current_length += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # 2. Embed Chunks
    chunk_embeddings = []

    batch_size = 4
    for i in range(0, len(chunks), batch_size):
        batch_texts = chunks[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.hidden_states[-1]
        mask = inputs['attention_mask'].unsqueeze(-1)

        sum_emb = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_emb = sum_emb / counts

        chunk_embeddings.append(mean_emb)

    # 3. Average across chunks
    if not chunk_embeddings:
        return torch.zeros((1, model.config.hidden_size), device=device)

    all_chunks_tensor = torch.cat(chunk_embeddings, dim=0)
    doc_embedding = torch.mean(all_chunks_tensor, dim=0, keepdim=True)

    return doc_embedding

# --- Dataset Processing (Strict & Balanced) ---

def process_full_dataset():
    print(f"\n Loading FULL Dataset...")
    try:
        ds = load_dataset("artnitolog/llm-generated-texts", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    # 1. Collect Raw Data (Strict 1:1 Logic)
    raw_samples = []

    # Exclude metadata columns
    excluded_cols = {'id', 'prompt', 'dataset_name', 'classes'}

    # Counter for Processed Source Rows
    rows_processed_count = 0

    print(f"Collecting valid samples (Target: {NUM_ROWS_TO_USE} Rows)...")

    for row in tqdm(ds, desc="Scanning Dataset"):
        # Check Stop Condition
        if NUM_ROWS_TO_USE is not None and rows_processed_count >= NUM_ROWS_TO_USE:
            break

        all_cols = row.keys()

        # Identify Human Column
        human_col = next((c for c in all_cols if 'human' in c.lower()), None)
        if not human_col: continue # Skip if no human text found

        # Identify AI Candidate Columns
        ai_candidates = []
        for col in all_cols:
            if col in excluded_cols: continue
            if col == human_col: continue

            # Ensure the column has content
            if row[col] and isinstance(row[col], str) and len(row[col]) > 10:
                ai_candidates.append(col)

        if not ai_candidates: continue # Skip if no AI text found
        if not row[human_col]: continue

        # --- Balanced Sampling Logic ---

        # 1. Add Human Sample (Label 0)
        human_text = row[human_col]
        raw_samples.append((human_text, 0))

        # 2. Add Random AI Sample (Label 1)
        selected_ai_col = random.choice(ai_candidates)
        ai_text = row[selected_ai_col]
        raw_samples.append((ai_text, 1))

        # Increment Row Counter (1 Row = 1 Human + 1 AI)
        rows_processed_count += 1

    print(f"\nValidation:")
    print(f"Target Rows: {NUM_ROWS_TO_USE}")
    print(f"Processed Rows: {rows_processed_count}")
    print(f"Total Samples Collected: {len(raw_samples)} (Should be {2 * rows_processed_count})")
    print(f"Human Samples: {sum(1 for _, l in raw_samples if l == 0)}")
    print(f"AI Samples: {sum(1 for _, l in raw_samples if l == 1)}")

    # 2. Flatten into Sentences
    all_sentences = []
    doc_boundaries = []
    current_idx = 0
    valid_samples = []

    print("Tokenizing sentences (FULL TEXT)...")
    for text, label in raw_samples:
        sents = get_sentences(text) # ALL sentences
        if not sents: continue

        all_sentences.extend(sents)
        doc_boundaries.append((current_idx, current_idx + len(sents)))
        valid_samples.append((text, label))
        current_idx += len(sents)

    # 3. Batch Transformation
    reduced_sentences = run_transformation_pipeline(
        all_sentences,
        "reduce",
        description="Step 1/2: Reducing"
    )

    injected_sentences = run_transformation_pipeline(
        reduced_sentences,
        "inject",
        description="Step 2/2: Injecting"
    )

    # 4. Feature Extraction
    print("\nCalculating Smart Chunked Embeddings...")
    features_list = []
    labels_list = []

    for i, (start, end) in enumerate(tqdm(doc_boundaries, desc="Embedding Docs")):
        inj_segment = " ".join(injected_sentences[start:end])
        orig_text = valid_samples[i][0]

        emb_orig = get_embedding_smart_chunking(orig_text)
        emb_inj = get_embedding_smart_chunking(inj_segment)

        feat = torch.cat((emb_orig, emb_inj), dim=1)

        features_list.append(feat.cpu())
        labels_list.append(valid_samples[i][1])

    if not features_list:
        return None, None

    X = torch.cat(features_list, dim=0)
    y = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)

    return X, y

# --- Training & Evaluation ---

def train_and_evaluate(X, y):
    print("\n Splitting Data (60/20/20)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    input_dim = X.shape[1]
    mlp = ArtifactDetectorMLP(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.0005)

    epochs = 20
    best_val_acc = 0.0

    print("\n Starting Training...")

    for epoch in range(epochs):
        mlp.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = mlp(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        mlp.eval()
        with torch.no_grad():
            val_out = mlp(X_val)
            val_loss = criterion(val_out, y_val).item()
            val_pred = (val_out > 0.5).float()
            val_acc = (val_pred == y_val).sum().item() / len(y_val) * 100

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(mlp.state_dict(), "best_model.pth")

    print(f"\n Training Complete. Best Val Acc: {best_val_acc:.2f}%")

    print("\n Evaluating on Test Set...")
    mlp.load_state_dict(torch.load("best_model.pth", weights_only=True))
    mlp.eval()

    with torch.no_grad():
        test_out = mlp(X_test)
        test_pred = (test_out > 0.5).float()
        test_acc = (test_pred == y_test).sum().item() / len(y_test) * 100

    print(f"{'='*30}")
    print(f"FINAL TEST ACCURACY: {test_acc:.2f}%")
    print(f"{'='*30}")

    return mlp

def main():
    # 1. Process Data
    X, y = process_full_dataset()

    if X is None:
        print("Data preparation failed.")
        return

    # 2. Train & Eval
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()

# ---

import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset
import nltk
import sys
import os
import random
import time
from IPython.display import clear_output  # 화면 갱신용

# --- Configuration ---
MODEL_PATH = "best_model.pth"
hf_token = "YOUR_HF_TOKEN_HERE"

# 1. Login
if hf_token:
    try:
        login(token=hf_token)
    except Exception as e:
        print(f"Warning: Login failed. {e}")

# 2. Model Setup (GLOBAL SCOPE - 무조건 실행됨)
model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_id} on {device}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", output_hidden_states=True, token=hf_token
    )
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# 파이프라인도 전역으로 설정
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", pad_token_id=tokenizer.eos_token_id, batch_size=32)

# --- MLP Classifier Structure ---
class ArtifactDetectorMLP(nn.Module):
    def __init__(self, input_dim):
        super(ArtifactDetectorMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# --- Helper Functions ---
def get_sentences(text):
    if not text or not isinstance(text, str): return []
    return nltk.tokenize.sent_tokenize(text)

def run_transformation_pipeline(texts, mode):
    tokenizer.padding_side = "left"
    if mode == "reduce":
        sys_prompt = "Task: Simplify. Keep Subject, Verb, Object. Rule: No meaning change. Input: "
    else:
        sys_prompt = "Task: Rewrite to be descriptive. Rule: Add adjectives. Input: "

    chat_inputs = [[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": sys_prompt + t}] for t in texts]
    results = []

    batch_size = 32
    for i in range(0, len(chat_inputs), batch_size):
        batch = chat_inputs[i : i + batch_size]
        outputs = pipe(batch, max_new_tokens=64, do_sample=False)
        for out in outputs:
            try:
                c = out[0]['generated_text'][-1]['content']
                if "Input:" in c: c = c.split("Input:")[-1]
                elif "Input Sentence:" in c: c = c.split("Input Sentence:")[-1]
                results.append(c.strip())
            except: results.append("")
    return results

def get_embedding_smart_chunking(text, max_tokens=512):
    if not text: return torch.zeros((1, model.config.hidden_size), device=device)
    sentences = get_sentences(text)
    if not sentences: return torch.zeros((1, model.config.hidden_size), device=device)

    chunks, cur_chunk, cur_len = [], [], 0
    tokenizer.padding_side = "right"
    for s in sentences:
        l = len(tokenizer.encode(s, add_special_tokens=False))
        if cur_len + l > max_tokens and cur_chunk:
            chunks.append(" ".join(cur_chunk)); cur_chunk = []; cur_len = 0
        cur_chunk.append(s); cur_len += l
    if cur_chunk: chunks.append(" ".join(cur_chunk))

    embs = []
    for i in range(0, len(chunks), 4):
        inps = tokenizer(chunks[i:i+4], return_tensors="pt", padding=True, truncation=True, max_length=max_tokens).to(device)
        with torch.no_grad(): out = model(**inps)
        mask = inps['attention_mask'].unsqueeze(-1)
        embs.append((out.hidden_states[-1] * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9))

    return torch.mean(torch.cat(embs, 0), dim=0, keepdim=True)

# --- Statistics & Dashboard Class ---
class Stats:
    def __init__(self):
        self.h_total = 0
        self.h_hit = 0
        self.a_total = 0
        self.a_hit = 0

        # 기록: Human인데 AI 점수가 가장 높게 나온 경우 (Worst Human)
        self.max_human_prob = 0.0
        # 기록: AI인데 AI 점수가 가장 낮게 나온 경우 (Worst AI)
        self.min_ai_prob = 1.0

    def update(self, is_human, is_correct, prob):
        if is_human:
            self.h_total += 1
            if is_correct: self.h_hit += 1
            if prob > self.max_human_prob: self.max_human_prob = prob
        else:
            self.a_total += 1
            if is_correct: self.a_hit += 1
            if prob < self.min_ai_prob: self.min_ai_prob = prob

    def print_dashboard(self, current_row, current_type, current_prob, sent_len):
        clear_output(wait=True) # 화면 갱신

        h_miss = self.h_total - self.h_hit
        a_miss = self.a_total - self.a_hit

        # 0 나누기 방지
        h_acc = (self.h_hit / self.h_total * 100) if self.h_total > 0 else 0
        a_acc = (self.a_hit / self.a_total * 100) if self.a_total > 0 else 0

        print(f"Processing Row: {current_row} | Type: [{current_type}] ({sent_len} sents) | Prob: {current_prob:.4f}")
        print("="*60)
        print(f" HUMAN HIT  : {self.h_hit} / {self.h_total} ({h_acc:.1f}%)")
        print(f" HUMAN MISS : {h_miss} / {self.h_total}")
        print(f" >> Highest Prob (Worst Human): {self.max_human_prob:.4f}")
        print("-" * 60)
        print(f" AI HIT     : {self.a_hit} / {self.a_total} ({a_acc:.1f}%)")
        print(f" AI MISS    : {a_miss} / {self.a_total}")
        print(f" >> Lowest Prob (Worst AI)    : {self.min_ai_prob:.4f}")
        print("="*60)
        print("Press Stop Button (or Ctrl+C) to finish.")

# --- Main Logic ---
def main():
    # 1. Load MLP
    print(f"\nLoading trained model from {MODEL_PATH}...")
    try:
        input_dim = model.config.hidden_size * 2
        mlp = ArtifactDetectorMLP(input_dim).to(device)
        mlp.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        mlp.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: best_model.pth not found. Please train first.")
        return

    # 2. Load Dataset
    print("Loading Dataset...")
    ds = load_dataset("artnitolog/llm-generated-texts", split="train")

    stats = Stats()

    print("\nStarting Real-time Inference...")

    try:
        for i, row in enumerate(ds):

            # 1. Identify Human Column
            col_human = next((c for c in row.keys() if 'human' in c.lower()), None)
            if not col_human or not row[col_human]: continue

            # 2. Identify ALL AI Columns (Exclude metadata)
            exclude_cols = ['id', 'prompt', 'dataset_name', 'classes', col_human]
            ai_cols = [c for c in row.keys() if c not in exclude_cols and row[c]]

            # 3. Targets: (Text, Label, TypeStr)
            # Human 1개
            targets = [(row[col_human], 0, "Human")]
            # AI 모두 (Random Choice 아님 -> All)
            for ac in ai_cols:
                targets.append((row[ac], 1, f"AI-{ac}"))

            # 4. Process
            for text, true_label, type_str in targets:
                sents = get_sentences(text)
                if not sents: continue

                # --- Pipeline Execution ---
                red = run_transformation_pipeline(sents, "reduce")
                inj = run_transformation_pipeline(red, "inject")
                inj_text = " ".join(inj)

                # --- Prediction ---
                with torch.no_grad():
                    emb_orig = get_embedding_smart_chunking(text)
                    emb_inj = get_embedding_smart_chunking(inj_text)
                    feat = torch.cat((emb_orig, emb_inj), dim=1)
                    prob = mlp(feat).item()
                    pred_label = 1 if prob > 0.5 else 0

                # --- Update Dashboard ---
                is_correct = (pred_label == true_label)
                stats.update(is_human=(true_label==0), is_correct=is_correct, prob=prob)

                stats.print_dashboard(
                    current_row=i,
                    current_type=type_str,
                    current_prob=prob,
                    sent_len=len(sents)
                )

    except KeyboardInterrupt:
        print("\n\nStopped by User.")
        stats.print_dashboard(
            current_row="STOPPED", current_type="N/A", current_prob=0.0, sent_len=0
        )

if __name__ == "__main__":
    main()