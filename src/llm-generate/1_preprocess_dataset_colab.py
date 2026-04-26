from google.colab import drive
drive.mount('/content/drive')

print("✓ Google Drive mounted successfully!")

# ---

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset
import nltk
import os
from tqdm import tqdm
import random
import json
import pickle
from datetime import datetime

# ---

# Google Drive Output Path
GDRIVE_BASE = "/content/drive/MyDrive/RNN_Preprocessed_Data"
OUTPUT_DIR = os.path.join(GDRIVE_BASE, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Preprocessed data will be saved to:")
print(f"  {OUTPUT_DIR}")

# Dataset Configuration
NUM_ROWS_TO_USE = None  # None = use ALL data
MAX_SENTENCES_PER_DOC = None  # None = process ALL sentences in each document

# Batch Size for LLM inference
BATCH_SIZE = 8

# HuggingFace Token
hf_token = "YOUR_HF_TOKEN_HERE"  # Add your token here if needed

# Resume Configuration
RESUME_FROM_CHECKPOINT = False  # Set to True to resume from checkpoint
CHECKPOINT_DIR = ""  # Path to checkpoint directory if resuming (e.g., "/content/drive/MyDrive/RNN_Preprocessed_Data/20231203_143022")
SAVE_CHECKPOINT_EVERY = 10  # Save checkpoint every N documents

print(f"\nConfiguration:")
print(f"  Dataset: {'FULL (all rows)' if NUM_ROWS_TO_USE is None else f'{NUM_ROWS_TO_USE} rows'}")
print(f"  Max sentences per doc: {'ALL (no limit)' if MAX_SENTENCES_PER_DOC is None else MAX_SENTENCES_PER_DOC}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Save checkpoint every: {SAVE_CHECKPOINT_EVERY} documents")
print(f"  Resume from checkpoint: {RESUME_FROM_CHECKPOINT}")
if RESUME_FROM_CHECKPOINT:
    print(f"  Checkpoint directory: {CHECKPOINT_DIR}")

# ---

# HuggingFace Login
if hf_token:
    try:
        login(token=hf_token)
        print("✓ Logged in to HuggingFace")
    except Exception as e:
        print(f"Warning: Login failed. {e}")

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✓ Using device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ---

# Load LLM Model
model_id = "meta-llama/Llama-3.2-1B-Instruct"
print(f"\nLoading {model_id}...")

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
        **model_kwargs
    )
    print(f"✓ Model loaded successfully")
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")
    raise

# ---

# NLTK Setup
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
print("✓ NLTK resources ready")

# ---

def get_sentences(text):
    """Split text into sentences"""
    if not text or not isinstance(text, str):
        return []
    return nltk.tokenize.sent_tokenize(text)

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

        for j, output_ids in enumerate(outputs):
            input_length = inputs['input_ids'][j].shape[0]
            generated_ids = output_ids[input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generated_sentences = get_sentences(generated_text)
            if generated_sentences:
                results.append(generated_sentences[0])
            else:
                results.append(texts[i+j] if i+j < len(texts) else "")

    return results

print("✓ Helper functions defined")

# ---

def collect_raw_documents(num_rows=None):
    """
    Collect raw documents from dataset.

    Args:
        num_rows: Number of rows to use (None = all data)

    Returns:
        List of (text, label) tuples
    """
    print(f"\nLoading dataset...")
    try:
        ds = load_dataset("artnitolog/llm-generated-texts", split="train")
        print(f"✓ Dataset loaded: {len(ds)} total rows")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    raw_samples = []
    excluded_cols = {'id', 'prompt', 'dataset_name', 'classes'}
    rows_processed = 0

    target_rows = num_rows if num_rows else len(ds)
    print(f"Collecting documents (Target: {target_rows if num_rows else 'ALL'} rows)...")

    for row in tqdm(ds, desc="Scanning Dataset"):
        if num_rows and rows_processed >= num_rows:
            break

        all_cols = row.keys()
        human_col = next((c for c in all_cols if 'human' in c.lower()), None)
        if not human_col:
            continue

        ai_candidates = []
        for col in all_cols:
            if col in excluded_cols or col == human_col:
                continue
            if row[col] and isinstance(row[col], str) and len(row[col]) > 10:
                ai_candidates.append(col)

        if not ai_candidates or not row[human_col]:
            continue

        raw_samples.append((row[human_col], 0))  # Human
        selected_ai_col = random.choice(ai_candidates)
        raw_samples.append((row[selected_ai_col], 1))  # AI

        rows_processed += 1

    print(f"✓ Total documents collected: {len(raw_samples)}")
    return raw_samples

# Collect documents
raw_docs = collect_raw_documents(num_rows=NUM_ROWS_TO_USE)

# ---

def preprocess_all_documents(documents):
    """
    Preprocess all documents: split into sentences, apply Reduce → Inject transformations.

    Supports resuming from checkpoint if interrupted.

    Args:
        documents: List of (text, label) tuples

    Returns:
        List of preprocessed document dictionaries with:
        - original_sentences: List of original sentences
        - reduced_sentences: List of reduced sentences
        - injected_sentences: List of injected sentences
        - label: Document label (0=human, 1=AI)
    """
    global OUTPUT_DIR # Declare OUTPUT_DIR as global here

    print(f"\n{'='*70}")
    print(f"Preprocessing {len(documents)} documents")
    print(f"{'='*70}\n")

    # Checkpoint file paths
    checkpoint_file = os.path.join(OUTPUT_DIR, "checkpoint.pkl")
    progress_file = os.path.join(OUTPUT_DIR, "progress.json")

    # Try to load checkpoint if resuming
    preprocessed_docs = []
    start_idx = 0

    if RESUME_FROM_CHECKPOINT and CHECKPOINT_DIR:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pkl")
        progress_path = os.path.join(CHECKPOINT_DIR, "progress.json")

        if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
            print(f"📂 Loading checkpoint from: {CHECKPOINT_DIR}")
            with open(checkpoint_path, 'rb') as f:
                preprocessed_docs = pickle.load(f)
            with open(progress_path, 'r') as f:
                progress = json.load(f)
            start_idx = progress['last_processed_idx'] + 1
            print(f"✓ Loaded {len(preprocessed_docs)} preprocessed documents")
            print(f"✓ Resuming from document {start_idx + 1}/{len(documents)}\n")

            # Update OUTPUT_DIR to use the checkpoint directory
            OUTPUT_DIR = CHECKPOINT_DIR
        else:
            print(f"⚠️  Checkpoint not found at {CHECKPOINT_DIR}")
            print(f"Starting from beginning...\n")

    skipped_docs = 0

    try:
        for idx in range(start_idx, len(documents)):
            doc_text, label = documents[idx]

            # Split into sentences
            sentences = get_sentences(doc_text)
            if not sentences or len(sentences) < 3:
                skipped_docs += 1
                continue

            # Process ALL sentences (no limit)
            if MAX_SENTENCES_PER_DOC is not None:
                sentences = sentences[:MAX_SENTENCES_PER_DOC]

            # Apply transformations
            print(f"\n[Document {idx+1}/{len(documents)}] Processing {len(sentences)} sentences...")

            reduced_sentences = transform_sentences(
                sentences,
                "reduce",
                description=f"[Doc {idx+1}] Reducing"
            )

            injected_sentences = transform_sentences(
                reduced_sentences,
                "inject",
                description=f"[Doc {idx+1}] Injecting"
            )

            # Store preprocessed data
            preprocessed_docs.append({
                'doc_id': idx,
                'original_sentences': sentences,
                'reduced_sentences': reduced_sentences,
                'injected_sentences': injected_sentences,
                'label': label,
                'num_sentences': len(sentences)
            })

            # Save checkpoint periodically
            if (idx + 1) % SAVE_CHECKPOINT_EVERY == 0:
                print(f"\n💾 Saving checkpoint at document {idx+1}...")
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(preprocessed_docs, f)
                with open(progress_file, 'w') as f:
                    json.dump({
                        'last_processed_idx': idx,
                        'total_processed': len(preprocessed_docs),
                        'total_skipped': skipped_docs,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                print(f"✓ Checkpoint saved ({len(preprocessed_docs)} documents)")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Processing interrupted by user!")
        print(f"💾 Saving emergency checkpoint...")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(preprocessed_docs, f)
        with open(progress_file, 'w') as f:
            json.dump({
                'last_processed_idx': idx - 1,
                'total_processed': len(preprocessed_docs),
                'total_skipped': skipped_docs,
                'timestamp': datetime.now().isoformat(),
                'status': 'interrupted'
            }, f, indent=2)
        print(f"✓ Emergency checkpoint saved!")
        print(f"\nTo resume, set:")
        print(f"  RESUME_FROM_CHECKPOINT = True")
        print(f"  CHECKPOINT_DIR = \"{OUTPUT_DIR}\"")
        raise

    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print(f"💾 Saving emergency checkpoint...")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(preprocessed_docs, f)
        with open(progress_file, 'w') as f:
            json.dump({
                'last_processed_idx': idx - 1 if idx > 0 else 0,
                'total_processed': len(preprocessed_docs),
                'total_skipped': skipped_docs,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }, f, indent=2)
        print(f"✓ Emergency checkpoint saved!")
        print(f"\nTo resume, set:")
        print(f"  RESUME_FROM_CHECKPOINT = True")
        print(f"  CHECKPOINT_DIR = \"{OUTPUT_DIR}\"")
        raise

    print(f"\n{'='*70}")
    print(f"Preprocessing complete!")
    print(f"  Total documents: {len(documents)}")
    print(f"  Preprocessed: {len(preprocessed_docs)}")
    print(f"  Skipped (too short): {skipped_docs}")
    print(f"{'='*70}\n")

    return preprocessed_docs

# Preprocess all documents
preprocessed_data = preprocess_all_documents(raw_docs)

# ---

# Save preprocessed data as pickle file
preprocessed_file = os.path.join(OUTPUT_DIR, "preprocessed_data.pkl")

print(f"Saving preprocessed data to Google Drive...")
with open(preprocessed_file, 'wb') as f:
    pickle.dump(preprocessed_data, f)

print(f"✓ Preprocessed data saved: {preprocessed_file}")
print(f"  File size: {os.path.getsize(preprocessed_file) / 1e6:.2f} MB")

# ---

# Calculate statistics
total_sentences = sum(doc['num_sentences'] for doc in preprocessed_data)
human_docs = sum(1 for doc in preprocessed_data if doc['label'] == 0)
ai_docs = sum(1 for doc in preprocessed_data if doc['label'] == 1)

metadata = {
    "preprocessing_timestamp": datetime.now().isoformat(),
    "dataset_name": "artnitolog/llm-generated-texts",
    "model_used": model_id,
    "total_documents": len(preprocessed_data),
    "human_documents": human_docs,
    "ai_documents": ai_docs,
    "total_sentences": total_sentences,
    "max_sentences_per_doc": "ALL (no limit)" if MAX_SENTENCES_PER_DOC is None else MAX_SENTENCES_PER_DOC,
    "batch_size": BATCH_SIZE,
    "device": device,
    "preprocessing_steps": [
        "1. Split text into sentences (NLTK)",
        "2. Reduce: Simplify to Subject-Verb-Object",
        "3. Inject: Rewrite to be more descriptive"
    ],
    "data_structure": {
        "doc_id": "Document index",
        "original_sentences": "List of original sentences",
        "reduced_sentences": "List of reduced sentences",
        "injected_sentences": "List of injected sentences",
        "label": "0 = human, 1 = AI",
        "num_sentences": "Number of sentences in document"
    }
}

# Save metadata
metadata_file = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Metadata saved: {metadata_file}")

# ---

# Show a sample document
if preprocessed_data:
    sample_doc = preprocessed_data[0]

    print(f"\n{'='*70}")
    print("SAMPLE PREPROCESSED DOCUMENT")
    print(f"{'='*70}\n")
    print(f"Document ID: {sample_doc['doc_id']}")
    print(f"Label: {'AI' if sample_doc['label'] == 1 else 'Human'}")
    print(f"Number of sentences: {sample_doc['num_sentences']}")
    print(f"\nFirst 3 sentences:")
    print(f"\n[Original]")
    for i in range(min(3, len(sample_doc['original_sentences']))):
        print(f"{i+1}. {sample_doc['original_sentences'][i]}")
    print(f"\n[Reduced]")
    for i in range(min(3, len(sample_doc['reduced_sentences']))):
        print(f"{i+1}. {sample_doc['reduced_sentences'][i]}")
    print(f"\n[Injected]")
    for i in range(min(3, len(sample_doc['injected_sentences']))):
        print(f"{i+1}. {sample_doc['injected_sentences'][i]}")
    print(f"\n{'='*70}")

# ---

print(f"\n{'='*70}")
print("PREPROCESSING COMPLETE - SUMMARY")
print(f"{'='*70}\n")
print(f"✓ All preprocessed data saved to Google Drive:")
print(f"  {OUTPUT_DIR}\n")
print(f"Files saved:")
print(f"  1. preprocessed_data.pkl - All preprocessed documents")
print(f"  2. metadata.json - Dataset information\n")
print(f"Statistics:")
print(f"  Total documents: {len(preprocessed_data)}")
print(f"  Human documents: {human_docs}")
print(f"  AI documents: {ai_docs}")
print(f"  Total sentences: {total_sentences}")
print(f"  Avg sentences/doc: {total_sentences/len(preprocessed_data):.1f}\n")
print(f"Next step:")
print(f"  Use '2_train_with_preprocessed_data_colab.ipynb' to train models")
print(f"  with this preprocessed data.\n")
print(f"{'='*70}")