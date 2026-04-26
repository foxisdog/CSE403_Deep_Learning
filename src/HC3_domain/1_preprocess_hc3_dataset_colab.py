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
import json
import pickle
from datetime import datetime

# ---

# Google Drive Output Path
GDRIVE_BASE = "/content/drive/MyDrive/hc3_preprocessed_data"
OUTPUT_DIR = os.path.join(GDRIVE_BASE, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"HC3 Preprocessed data will be saved to:")
print(f"  {OUTPUT_DIR}")

# Batch Size for LLM inference
BATCH_SIZE = 8

# HuggingFace Token
hf_token = "YOUR_HF_TOKEN_HERE"

# HC3 domains to process
HC3_DOMAINS = ['finance', 'medicine', 'open_qa', 'reddit_eli5', 'wiki_csai']

print(f"\nConfiguration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  HC3 Domains: {', '.join(HC3_DOMAINS)}")

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

print("\nLoading HC3 dataset...")
print("Method: Direct download from HuggingFace Hub")

from huggingface_hub import hf_hub_download
import json

hc3_data = {}

# HC3 dataset is available as JSON files in the repository
# Let's download them directly
for domain in HC3_DOMAINS:
    print(f"\nLoading {domain}...")

    try:
        # Download the JSON file for this domain
        file_path = hf_hub_download(
            repo_id="Hello-SimpleAI/HC3",
            filename=f"{domain}.jsonl",
            repo_type="dataset",
            token=hf_token
        )

        # Load the JSONL file
        data_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))

        # Create a dataset-like structure
        hc3_data[domain] = {
            'train': data_list
        }

        print(f"  ✓ {domain}: {len(data_list)} samples loaded from JSONL")

    except Exception as e1:
        print(f"  JSONL attempt failed: {e1}")

        # Try .json instead of .jsonl
        try:
            file_path = hf_hub_download(
                repo_id="Hello-SimpleAI/HC3",
                filename=f"{domain}.json",
                repo_type="dataset",
                token=hf_token
            )

            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)

            hc3_data[domain] = {
                'train': data_list
            }

            print(f"  ✓ {domain}: {len(data_list)} samples loaded from JSON")

        except Exception as e2:
            print(f"  JSON attempt failed: {e2}")

            # Try loading the data directory structure
            try:
                from huggingface_hub import list_repo_files

                print(f"  Listing available files for {domain}...")
                all_files = list_repo_files("Hello-SimpleAI/HC3", repo_type="dataset")
                domain_files = [f for f in all_files if domain in f.lower()]

                print(f"  Found {len(domain_files)} files containing '{domain}':")
                for f in domain_files[:10]:  # Show first 10
                    print(f"    - {f}")

                if domain_files:
                    print(f"  Attempting to download first matching file...")
                    file_path = hf_hub_download(
                        repo_id="Hello-SimpleAI/HC3",
                        filename=domain_files[0],
                        repo_type="dataset",
                        token=hf_token
                    )

                    # Try to parse it
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.endswith('.jsonl'):
                            data_list = [json.loads(line) for line in f]
                        else:
                            data_list = json.load(f)

                    hc3_data[domain] = {
                        'train': data_list
                    }
                    print(f"  ✓ {domain}: {len(data_list)} samples")

            except Exception as e3:
                print(f"  ❌ All attempts failed: {e3}")

print(f"\n{'='*70}")
if len(hc3_data) > 0:
    print(f"✓ Successfully loaded {len(hc3_data)} HC3 domains")
    for domain, data in hc3_data.items():
        print(f"  {domain}: {len(data['train'])} samples")
else:
    print("❌ WARNING: No datasets loaded!")
    print("\nPlease check if the dataset is available at:")
    print("https://huggingface.co/datasets/Hello-SimpleAI/HC3")
print(f"{'='*70}")

# ---

import random

def preprocess_domain(domain_name, dataset, target_pairs=200):
    """
    Preprocess a single HC3 domain.

    Args:
        domain_name: Name of the domain
        dataset: HuggingFace dataset for this domain
        target_pairs: Number of pairs to randomly select and process (default: 200)

    Returns:
        List of preprocessed documents with domain label
    """
    print(f"\n{'='*70}")
    print(f"Processing {domain_name} (Target: {target_pairs} pairs)")
    print(f"{'='*70}\n")

    preprocessed_docs = []
    skipped_pairs = 0

    # Create a copy and shuffle to ensure random selection
    data_list = list(dataset['train'])
    random.shuffle(data_list)

    pairs_collected = 0

    # Use tqdm to show progress towards the target count
    with tqdm(total=target_pairs, desc=f"[{domain_name}] Sampling") as pbar:
        for row in data_list:
            if pairs_collected >= target_pairs:
                break

            # Get human answer
            human_answer = row.get('human_answers', [''])[0] if row.get('human_answers') else ""

            # Get chatgpt answer
            chatgpt_answer = row.get('chatgpt_answers', [''])[0] if row.get('chatgpt_answers') else ""

            # IMPORTANT: Skip this pair if EITHER human or AI is invalid
            if not human_answer or len(human_answer) < 10:
                skipped_pairs += 1
                continue
            if not chatgpt_answer or len(chatgpt_answer) < 10:
                skipped_pairs += 1
                continue

            # Split into sentences for both
            human_sentences = get_sentences(human_answer)
            ai_sentences = get_sentences(chatgpt_answer)

            # Skip if either has too few sentences
            if not human_sentences or len(human_sentences) < 3:
                skipped_pairs += 1
                continue
            if not ai_sentences or len(ai_sentences) < 3:
                skipped_pairs += 1
                continue

            # Process Human document
            human_reduced = transform_sentences(
                human_sentences,
                "reduce",
                description=f"[{domain_name}] Pair {pairs_collected+1} (Human) Reducing"
            )

            human_injected = transform_sentences(
                human_reduced,
                "inject",
                description=f"[{domain_name}] Pair {pairs_collected+1} (Human) Injecting"
            )

            # Assign doc_id based on current list length
            doc_id_human = len(preprocessed_docs)

            preprocessed_docs.append({
                'doc_id': doc_id_human,
                'split': domain_name,
                'original_sentences': human_sentences,
                'reduced_sentences': human_reduced,
                'injected_sentences': human_injected,
                'label': 0,  # Human
                'num_sentences': len(human_sentences)
            })

            # Process AI document
            ai_reduced = transform_sentences(
                ai_sentences,
                "reduce",
                description=f"[{domain_name}] Pair {pairs_collected+1} (AI) Reducing"
            )

            ai_injected = transform_sentences(
                ai_reduced,
                "inject",
                description=f"[{domain_name}] Pair {pairs_collected+1} (AI) Injecting"
            )

            preprocessed_docs.append({
                'doc_id': doc_id_human + 1,
                'split': domain_name,
                'original_sentences': ai_sentences,
                'reduced_sentences': ai_reduced,
                'injected_sentences': ai_injected,
                'label': 1,  # AI
                'num_sentences': len(ai_sentences)
            })

            pairs_collected += 1
            pbar.update(1)

    # Verify equal counts
    human_count = sum(1 for d in preprocessed_docs if d['label'] == 0)
    ai_count = sum(1 for d in preprocessed_docs if d['label'] == 1)

    print(f"\n{'='*70}")
    print(f"{domain_name} complete!")
    print(f"  Preprocessed: {len(preprocessed_docs)} ({human_count} Human, {ai_count} AI)")
    print(f"  Pairs collected: {pairs_collected}")
    print(f"  Skipped pairs (during search): {skipped_pairs}")
    if human_count != ai_count:
        print(f"  ⚠️  WARNING: Human ({human_count}) != AI ({ai_count})")
    else:
        print(f"  ✓ Balanced: Human and AI counts are equal")
    print(f"{'='*70}\n")

    return preprocessed_docs

print("✓ Preprocessing function defined (Random sampling enabled)")


# ---

# Process each domain
all_preprocessed_data = []

for domain_name, dataset in hc3_data.items():
    domain_docs = preprocess_domain(domain_name, dataset)
    all_preprocessed_data.extend(domain_docs)

print(f"\n{'='*70}")
print(f"ALL DOMAINS PROCESSED")
print(f"{'='*70}")
print(f"Total preprocessed documents: {len(all_preprocessed_data)}")

# Count by domain
domain_counts = {}
for doc in all_preprocessed_data:
    domain = doc['split']
    domain_counts[domain] = domain_counts.get(domain, 0) + 1

print(f"\nDocuments by domain:")
for domain, count in sorted(domain_counts.items()):
    print(f"  {domain}: {count}")

# ---

# Save preprocessed data as pickle file
preprocessed_file = os.path.join(OUTPUT_DIR, "preprocessed_data.pkl")

print(f"\nSaving preprocessed data to Google Drive...")
with open(preprocessed_file, 'wb') as f:
    pickle.dump(all_preprocessed_data, f)

print(f"✓ Preprocessed data saved: {preprocessed_file}")
print(f"  File size: {os.path.getsize(preprocessed_file) / 1e6:.2f} MB")

# ---

# Calculate statistics
total_sentences = sum(doc['num_sentences'] for doc in all_preprocessed_data)
human_docs = sum(1 for doc in all_preprocessed_data if doc['label'] == 0)
ai_docs = sum(1 for doc in all_preprocessed_data if doc['label'] == 1)

# Count by domain
domain_stats = {}
for domain in HC3_DOMAINS:
    domain_docs = [d for d in all_preprocessed_data if d['split'] == domain]
    domain_stats[domain] = {
        'total_docs': len(domain_docs),
        'human_docs': sum(1 for d in domain_docs if d['label'] == 0),
        'ai_docs': sum(1 for d in domain_docs if d['label'] == 1),
        'total_sentences': sum(d['num_sentences'] for d in domain_docs)
    }

metadata = {
    "preprocessing_timestamp": datetime.now().isoformat(),
    "dataset_name": "Hello-SimpleAI/HC3",
    "model_used": model_id,
    "total_documents": len(all_preprocessed_data),
    "human_documents": human_docs,
    "ai_documents": ai_docs,
    "total_sentences": total_sentences,
    "splits_used": HC3_DOMAINS,
    "domain_statistics": domain_stats,
    "batch_size": BATCH_SIZE,
    "device": device,
    "preprocessing_steps": [
        "1. Split text into sentences (NLTK)",
        "2. Reduce: Simplify to Subject-Verb-Object",
        "3. Inject: Rewrite to be more descriptive"
    ],
    "data_structure": {
        "doc_id": "Document index",
        "split": "HC3 domain (finance/medicine/open_qa/reddit_eli5/wiki_csai)",
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

print(f"\n{'='*70}")
print("HC3 PREPROCESSING COMPLETE - SUMMARY")
print(f"{'='*70}\n")
print(f"✓ All preprocessed data saved to Google Drive:")
print(f"  {OUTPUT_DIR}\n")
print(f"Files saved:")
print(f"  1. preprocessed_data.pkl - All preprocessed documents")
print(f"  2. metadata.json - Dataset information\n")
print(f"Statistics:")
print(f"  Total documents: {len(all_preprocessed_data)}")
print(f"  Human documents: {human_docs}")
print(f"  AI documents: {ai_docs}")
print(f"  Total sentences: {total_sentences}")

# Avoid division by zero
if len(all_preprocessed_data) > 0:
    print(f"  Avg sentences/doc: {total_sentences/len(all_preprocessed_data):.1f}\n")
else:
    print(f"  Avg sentences/doc: N/A (no documents processed)\n")

# Verify balance
if human_docs == ai_docs:
    print(f"✓ DATA BALANCE VERIFIED: Human ({human_docs}) == AI ({ai_docs})")
else:
    print(f"⚠️  WARNING: IMBALANCED DATA - Human ({human_docs}) != AI ({ai_docs})")

print(f"\nDomain breakdown:")
for domain, stats in sorted(domain_stats.items()):
    h_count = stats['human_docs']
    a_count = stats['ai_docs']
    balance_status = "✓ Balanced" if h_count == a_count else "⚠️  Imbalanced"
    print(f"  {domain}: {stats['total_docs']} docs ({h_count} Human, {a_count} AI) - {balance_status}")

print(f"\nNext step:")
print(f"  Use '2a_create_hc3_embeddings_colab.ipynb' to create embeddings")
print(f"  from this preprocessed data.\n")
print(f"{'='*70}")