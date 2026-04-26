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

# Google Drive Output Path (HC3-specific directory)
GDRIVE_BASE = "/content/drive/MyDrive/hc3_preprocessed_data"
OUTPUT_DIR = os.path.join(GDRIVE_BASE, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"HC3 Preprocessed data will be saved to:")
print(f"  {OUTPUT_DIR}")

# HC3 Dataset Configuration
HC3_SPLITS = ['open_qa', 'finance', 'medicine', 'wiki_csai', 'reddit_eli5']
SAMPLES_PER_SPLIT = 200  # 200 samples per split = 1000 total
RANDOM_SEED = 42

# Processing Configuration
MAX_SENTENCES_PER_DOC = None  # None = process ALL sentences
BATCH_SIZE = 8

# HuggingFace Token
hf_token = "YOUR_HF_TOKEN_HERE"

# Checkpoint Configuration
RESUME_FROM_CHECKPOINT = False
CHECKPOINT_DIR = ""  # Set this if resuming
SAVE_CHECKPOINT_EVERY = 10

print(f"\nConfiguration:")
print(f"  Dataset: Hello-SimpleAI/HC3")
print(f"  Splits: {', '.join(HC3_SPLITS)}")
print(f"  Samples per split: {SAMPLES_PER_SPLIT}")
print(f"  Total samples: {len(HC3_SPLITS) * SAMPLES_PER_SPLIT}")
print(f"  Random seed: {RANDOM_SEED}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Resume from checkpoint: {RESUME_FROM_CHECKPOINT}")

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

def collect_hc3_documents(splits, samples_per_split, random_seed=42):
    """
    Collect random samples from HC3 dataset.

    HC3 structure:
    - Each row has 'question', 'human_answers', 'chatgpt_answers'
    - human_answers and chatgpt_answers are lists of strings

    Returns:
        List of (text, label, split_name) tuples
    """
    random.seed(random_seed)
    raw_samples = []

    print(f"\n{'='*70}")
    print("Loading HC3 Dataset")
    print(f"{'='*70}\n")

    for split_name in splits:
        print(f"\nLoading split: {split_name}")
        try:
            # Load directly from JSONL files via URL (bypass legacy script)
            # This avoids the HC3.py script entirely
            url = f"https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{split_name}.jsonl"

            ds = load_dataset(
                "json",
                data_files=url,
                split="train"
            )
            print(f"  Total rows in {split_name}: {len(ds)}")

            # Shuffle and iterate until we get exactly samples_per_split valid samples
            collected_count = 0
            sampled_indices = random.sample(range(len(ds)), len(ds))  # Shuffle all indices

            for idx in tqdm(sampled_indices, desc=f"  Collecting {split_name}"):
                if collected_count >= samples_per_split:
                    break

                row = ds[idx]

                # Get human and ChatGPT answers
                human_answers = row.get('human_answers', [])
                chatgpt_answers = row.get('chatgpt_answers', [])

                # Skip if no valid answers
                if not human_answers or not chatgpt_answers:
                    continue

                # Pick first answer from each (or random if multiple)
                human_text = human_answers[0] if len(human_answers) == 1 else random.choice(human_answers)
                chatgpt_text = chatgpt_answers[0] if len(chatgpt_answers) == 1 else random.choice(chatgpt_answers)

                # Skip if too short
                if len(human_text.strip()) < 50 or len(chatgpt_text.strip()) < 50:
                    continue

                # Add both human and AI samples
                raw_samples.append((human_text, 0, split_name))  # Human
                raw_samples.append((chatgpt_text, 1, split_name))  # AI
                collected_count += 1  # Count pairs, not individual samples

            print(f"  ✓ Collected {collected_count} pairs ({collected_count * 2} samples) from {split_name}")

        except Exception as e:
            print(f"  ❌ Error loading {split_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print(f"✓ Total documents collected: {len(raw_samples)}")
    print(f"  Human: {sum(1 for _, label, _ in raw_samples if label == 0)}")
    print(f"  AI: {sum(1 for _, label, _ in raw_samples if label == 1)}")

    # Per-split breakdown
    print(f"\nPer-split breakdown:")
    for split_name in splits:
        split_samples = [s for s in raw_samples if s[2] == split_name]
        split_human = sum(1 for _, label, _ in split_samples if label == 0)
        split_ai = sum(1 for _, label, _ in split_samples if label == 1)
        print(f"  {split_name}: {len(split_samples)} total ({split_human} human, {split_ai} AI)")

    print(f"{'='*70}\n")

    return raw_samples

# Collect HC3 documents
raw_docs = collect_hc3_documents(HC3_SPLITS, SAMPLES_PER_SPLIT, RANDOM_SEED)

# ---

def preprocess_all_documents(documents):
    """
    Preprocess all documents: split into sentences, apply Reduce → Inject transformations.

    Args:
        documents: List of (text, label, split_name) tuples

    Returns:
        List of preprocessed document dictionaries
    """
    global OUTPUT_DIR

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
            OUTPUT_DIR = CHECKPOINT_DIR
        else:
            print(f"⚠️  Checkpoint not found at {CHECKPOINT_DIR}")
            print(f"Starting from beginning...\n")

    skipped_docs = 0

    try:
        for idx in range(start_idx, len(documents)):
            doc_text, label, split_name = documents[idx]

            # Split into sentences
            sentences = get_sentences(doc_text)
            if not sentences or len(sentences) < 3:
                skipped_docs += 1
                continue

            if MAX_SENTENCES_PER_DOC is not None:
                sentences = sentences[:MAX_SENTENCES_PER_DOC]

            # Apply transformations
            print(f"\n[Document {idx+1}/{len(documents)}] [{split_name}] Processing {len(sentences)} sentences...")

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
                'split_name': split_name,
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

# Per-split statistics
split_stats = {}
for split_name in HC3_SPLITS:
    split_docs = [doc for doc in preprocessed_data if doc['split_name'] == split_name]
    split_stats[split_name] = {
        'total_docs': len(split_docs),
        'human_docs': sum(1 for doc in split_docs if doc['label'] == 0),
        'ai_docs': sum(1 for doc in split_docs if doc['label'] == 1),
        'total_sentences': sum(doc['num_sentences'] for doc in split_docs)
    }

metadata = {
    "preprocessing_timestamp": datetime.now().isoformat(),
    "dataset_name": "Hello-SimpleAI/HC3",
    "model_used": model_id,
    "splits_used": HC3_SPLITS,
    "samples_per_split": SAMPLES_PER_SPLIT,
    "random_seed": RANDOM_SEED,
    "total_documents": len(preprocessed_data),
    "human_documents": human_docs,
    "ai_documents": ai_docs,
    "total_sentences": total_sentences,
    "split_statistics": split_stats,
    "max_sentences_per_doc": "ALL (no limit)" if MAX_SENTENCES_PER_DOC is None else MAX_SENTENCES_PER_DOC,
    "batch_size": BATCH_SIZE,
    "device": device,
    "preprocessing_steps": [
        "1. Randomly sample documents from 5 HC3 splits",
        "2. Split text into sentences (NLTK)",
        "3. Reduce: Simplify to Subject-Verb-Object",
        "4. Inject: Rewrite to be more descriptive"
    ],
    "data_structure": {
        "doc_id": "Document index",
        "split_name": "HC3 split name",
        "original_sentences": "List of original sentences",
        "reduced_sentences": "List of reduced sentences",
        "injected_sentences": "List of injected sentences",
        "label": "0 = human, 1 = AI (ChatGPT)",
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
    print(f"Split: {sample_doc['split_name']}")
    print(f"Label: {'AI (ChatGPT)' if sample_doc['label'] == 1 else 'Human'}")
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
print("HC3 PREPROCESSING COMPLETE - SUMMARY")
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

if len(preprocessed_data) > 0:
    print(f"  Avg sentences/doc: {total_sentences/len(preprocessed_data):.1f}\n")
else:
    print(f"  Avg sentences/doc: N/A (No documents preprocessed)\n")

print(f"Per-split statistics:")
for split_name, stats in split_stats.items():
    print(f"  {split_name}:")
    print(f"    Total: {stats['total_docs']}, Human: {stats['human_docs']}, AI: {stats['ai_docs']}, Sentences: {stats['total_sentences']}")
print(f"\nNext step:")
print(f"  Use '2a_create_hc3_embeddings_colab.ipynb' to create embeddings")
print(f"{'='*70}")