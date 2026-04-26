from google.colab import drive
drive.mount('/content/drive')

print("✓ Google Drive mounted successfully!")

# ---

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import json
import pickle
from datetime import datetime

# ---

# ==================== IMPORTANT: SET THIS PATH ====================
# Path to your HC3 preprocessed data directory
# Example: "/content/drive/MyDrive/hc3_preprocessed_data/20231203_143022"
PREPROCESSED_DATA_DIR = "/content/drive/MyDrive/hc3_preprocessed_data/20251205_063825"

# ==================================================================

# Output directory for embeddings
EMBEDDINGS_OUTPUT_DIR = os.path.join(
    "/content/drive/MyDrive/hc3_embeddings",
    datetime.now().strftime("%Y%m%d_%H%M%S")
)
os.makedirs(EMBEDDINGS_OUTPUT_DIR, exist_ok=True)

print(f"HC3 Preprocessed data directory: {PREPROCESSED_DATA_DIR}")
print(f"HC3 Embeddings will be saved to: {EMBEDDINGS_OUTPUT_DIR}")

# Batch Size for embedding extraction
EMBEDDING_BATCH_SIZE = 32

# HuggingFace Token
hf_token = "YOUR_HF_TOKEN_HERE"

print(f"\nConfiguration:")
print(f"  Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")

# ---

# Load preprocessed data
preprocessed_file = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_data.pkl")
metadata_file = os.path.join(PREPROCESSED_DATA_DIR, "metadata.json")

print(f"Loading HC3 preprocessed data from: {preprocessed_file}")

if not os.path.exists(preprocessed_file):
    raise FileNotFoundError(
        f"Preprocessed data not found at: {preprocessed_file}\n"
        f"Please run '1_preprocess_hc3_dataset_colab.ipynb' first and update PREPROCESSED_DATA_DIR."
    )

with open(preprocessed_file, 'rb') as f:
    preprocessed_data = pickle.load(f)

print(f"✓ Loaded {len(preprocessed_data)} preprocessed documents")

# Load metadata
if os.path.exists(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Metadata loaded")
    print(f"  Preprocessing date: {metadata['preprocessing_timestamp']}")
    print(f"  Total documents: {metadata['total_documents']}")
    print(f"  Total sentences: {metadata['total_sentences']}")
    print(f"  HC3 Splits: {', '.join(metadata['splits_used'])}")

# Group documents by domain (split)
# Each document should have a 'split' field indicating the domain
domain_data = {}
for doc in preprocessed_data:
    split = doc.get('split', 'unknown')
    if split not in domain_data:
        domain_data[split] = []
    domain_data[split].append(doc)

print(f"\n✓ Grouped documents by domain:")
for domain, docs in sorted(domain_data.items()):
    human_count = sum(1 for d in docs if d['label'] == 0)
    ai_count = sum(1 for d in docs if d['label'] == 1)
    print(f"  {domain}: {len(docs)} docs ({human_count} Human, {ai_count} AI)")

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

# Load LLM Model (for embeddings)
model_id = "meta-llama/Llama-3.2-1B-Instruct"
print(f"\nLoading {model_id} for embeddings...")

try:
    tokenizer_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {"token": hf_token} if hf_token else {}
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
        **model_kwargs
    )
    print(f"✓ Model loaded successfully")
    print(f"  Hidden size: {llm_model.config.hidden_size}")
except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}")
    raise

# ---

def get_sentence_embedding(texts):
    """
    Get embeddings for a batch of sentences using mean pooling.

    Args:
        texts: List of strings

    Returns:
        embeddings: (batch_size, hidden_size)
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts or all(not t for t in texts):
        return torch.zeros((len(texts) if texts else 1, llm_model.config.hidden_size), device=device)

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

print("✓ Embedding extraction function defined")

# ---

# Data is already grouped by domain in domain_data dict
# No train/val/test splitting - keep each domain separate

print(f"\n✓ Domain organization complete:")
print(f"  Total domains: {len(domain_data)}")
print(f"  Total documents: {sum(len(docs) for docs in domain_data.values())}")

# Save domain organization info
domain_info = {
    "total_domains": len(domain_data),
    "domains": {},
    "timestamp": datetime.now().isoformat()
}

for domain, docs in sorted(domain_data.items()):
    human_count = sum(1 for d in docs if d['label'] == 0)
    ai_count = sum(1 for d in docs if d['label'] == 1)
    total_sentences = sum(len(d['original_sentences']) for d in docs)

    domain_info["domains"][domain] = {
        "total_docs": len(docs),
        "human_docs": human_count,
        "ai_docs": ai_count,
        "total_sentences": total_sentences,
        "avg_sentences_per_doc": total_sentences / len(docs) if docs else 0
    }

    print(f"\n  {domain}:")
    print(f"    Docs: {len(docs)} ({human_count} Human, {ai_count} AI)")
    print(f"    Sentences: {total_sentences} (avg {total_sentences/len(docs):.1f} per doc)")

with open(os.path.join(EMBEDDINGS_OUTPUT_DIR, "domain_organization.json"), "w") as f:
    json.dump(domain_info, f, indent=2)

print(f"\n✓ Domain organization info saved")

# ---

def convert_to_embeddings(data_split, split_name):
    """
    Convert both original and injected sentences to embedding sequences.

    This function generates TWO separate embedding sequences per document:
    1. Original sentence embeddings
    2. Injected sentence embeddings

    Args:
        data_split: List of preprocessed documents
        split_name: Name of split (for progress bar)

    Returns:
        original_sequences: List of (seq_len, embedding_dim) tensors
        injected_sequences: List of (seq_len, embedding_dim) tensors
        labels: List of labels
    """
    print(f"\n{'='*70}")
    print(f"Converting {split_name} documents to embeddings")
    print(f"{'='*70}\n")

    all_original_sequences = []
    all_injected_sequences = []
    all_labels = []

    for doc in tqdm(data_split, desc=f"[{split_name}] Embedding"):
        original_sentences = doc['original_sentences']
        injected_sentences = doc['injected_sentences']
        label = doc['label']

        # Extract embeddings in batches
        orig_embeddings_list = []
        inj_embeddings_list = []

        for i in range(0, len(original_sentences), EMBEDDING_BATCH_SIZE):
            batch_orig = original_sentences[i:i+EMBEDDING_BATCH_SIZE]
            batch_inj = injected_sentences[i:i+EMBEDDING_BATCH_SIZE]

            emb_orig = get_sentence_embedding(batch_orig).cpu()
            emb_inj = get_sentence_embedding(batch_inj).cpu()

            orig_embeddings_list.append(emb_orig)
            inj_embeddings_list.append(emb_inj)

        # Concatenate all batches
        orig_embeddings = torch.cat(orig_embeddings_list, dim=0)
        inj_embeddings = torch.cat(inj_embeddings_list, dim=0)

        # Store separately (NOT concatenated)
        all_original_sequences.append(orig_embeddings)
        all_injected_sequences.append(inj_embeddings)
        all_labels.append(label)

    print(f"[{split_name}] Created {len(all_original_sequences)} original sequences")
    print(f"[{split_name}] Created {len(all_injected_sequences)} injected sequences")
    return all_original_sequences, all_injected_sequences, all_labels

print("✓ Conversion function defined")

# ---

# Convert each domain to embeddings separately
domain_embeddings = {}

for domain_name in sorted(domain_data.keys()):
    domain_docs = domain_data[domain_name]

    orig_seqs, inj_seqs, labels = convert_to_embeddings(domain_docs, domain_name)

    domain_embeddings[domain_name] = {
        'original_sequences': orig_seqs,
        'injected_sequences': inj_seqs,
        'labels': labels
    }

print(f"\n✓ Generated embeddings for {len(domain_embeddings)} domains")

# ---

print(f"\n{'='*70}")
print("Saving HC3 domain-wise embeddings to Google Drive")
print(f"{'='*70}\n")

# Save embeddings organized by domain
embeddings_data = domain_embeddings

embeddings_file = os.path.join(EMBEDDINGS_OUTPUT_DIR, "embeddings.pkl")
with open(embeddings_file, 'wb') as f:
    pickle.dump(embeddings_data, f)

print(f"✓ HC3 embeddings saved to: {embeddings_file}")

# Calculate file size
file_size_mb = os.path.getsize(embeddings_file) / (1024 * 1024)
print(f"  File size: {file_size_mb:.2f} MB")

# Print domain summary
print(f"\n✓ Saved embeddings for {len(embeddings_data)} domains:")
for domain in sorted(embeddings_data.keys()):
    n_samples = len(embeddings_data[domain]['labels'])
    print(f"  {domain}: {n_samples} documents")

# ---

# Save metadata
first_domain = list(domain_embeddings.keys())[0]
first_orig = domain_embeddings[first_domain]['original_sequences'][0]
first_inj = domain_embeddings[first_domain]['injected_sequences'][0]

embeddings_metadata = {
    "creation_timestamp": datetime.now().isoformat(),
    "dataset_name": "Hello-SimpleAI/HC3",
    "preprocessed_data_dir": PREPROCESSED_DATA_DIR,
    "model_id": model_id,
    "embedding_dim": llm_model.config.hidden_size,
    "embedding_batch_size": EMBEDDING_BATCH_SIZE,
    "organization": "domain_wise",
    "domains": sorted(list(domain_embeddings.keys())),
    "domain_samples": {
        domain: len(domain_embeddings[domain]['labels'])
        for domain in sorted(domain_embeddings.keys())
    },
    "original_embedding_dim": first_orig.shape[1],
    "injected_embedding_dim": first_inj.shape[1],
    "embedding_types": {
        "original_sequences": "Original sentence embeddings (seq_len, embedding_dim)",
        "injected_sequences": "Injected sentence embeddings (seq_len, embedding_dim)"
    },
    "device": device,
    "note": "HC3 dataset embeddings organized by domain - Contains BOTH original and injected embeddings separately for each domain."
}

metadata_file = os.path.join(EMBEDDINGS_OUTPUT_DIR, "embeddings_metadata.json")
with open(metadata_file, 'w') as f:
    json.dump(embeddings_metadata, f, indent=2)

print(f"\n✓ Metadata saved to: {metadata_file}")

# ---

# Verify shapes
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

# Show first domain
first_domain = sorted(domain_embeddings.keys())[0]
first_orig = domain_embeddings[first_domain]['original_sequences'][0]
first_inj = domain_embeddings[first_domain]['injected_sequences'][0]
first_label = domain_embeddings[first_domain]['labels'][0]

print(f"\nFirst domain ({first_domain}) - first document:")
print(f"  Original embedding shape: {first_orig.shape}")
print(f"  Injected embedding shape: {first_inj.shape}")
print(f"  Label: {first_label}")

print(f"\nDomains and sample counts:")
for domain in sorted(domain_embeddings.keys()):
    n_samples = len(domain_embeddings[domain]['labels'])
    human_count = sum(1 for l in domain_embeddings[domain]['labels'] if l == 0)
    ai_count = sum(1 for l in domain_embeddings[domain]['labels'] if l == 1)
    print(f"  {domain}: {n_samples} samples ({human_count} Human, {ai_count} AI)")

print(f"\nUsage Examples:")
print(f"  1. Access domain: data['{first_domain}']")
print(f"  2. Use original only: data['{first_domain}']['original_sequences']")
print(f"  3. Use injected only: data['{first_domain}']['injected_sequences']")
print(f"  4. Concatenate both: torch.cat([orig, inj], dim=1) -> (seq_len, {first_orig.shape[1]*2})")

# ---

print("\n" + "="*70)
print("HC3 EMBEDDING GENERATION COMPLETE - SUMMARY")
print("="*70)
print(f"\n✓ HC3 embeddings saved to Google Drive:")
print(f"  {EMBEDDINGS_OUTPUT_DIR}\n")
print(f"Saved files:")
print(f"  1. embeddings.pkl - All embedding tensors (domain-wise)")
print(f"  2. embeddings_metadata.json - Embedding metadata")
print(f"  3. domain_organization.json - Domain organization info\n")
print(f"Embedding Statistics:")
print(f"  Total domains: {len(domain_embeddings)}")
print(f"  Domains: {', '.join(sorted(domain_embeddings.keys()))}")
print(f"  Total documents: {sum(len(d['labels']) for d in domain_embeddings.values())}")

first_domain = sorted(domain_embeddings.keys())[0]
first_orig = domain_embeddings[first_domain]['original_sequences'][0]
first_inj = domain_embeddings[first_domain]['injected_sequences'][0]

print(f"  Original embedding dim: {first_orig.shape[1]}")
print(f"  Injected embedding dim: {first_inj.shape[1]}\n")

print(f"Domain breakdown:")
for domain in sorted(domain_embeddings.keys()):
    n_samples = len(domain_embeddings[domain]['labels'])
    human_count = sum(1 for l in domain_embeddings[domain]['labels'] if l == 0)
    ai_count = sum(1 for l in domain_embeddings[domain]['labels'] if l == 1)
    print(f"  {domain}: {n_samples} docs ({human_count} Human, {ai_count} AI)")

print(f"\nData Structure:")
print(f"  embeddings_data['{first_domain}']['original_sequences'] - List of original embeddings")
print(f"  embeddings_data['{first_domain}']['injected_sequences'] - List of injected embeddings")
print(f"  embeddings_data['{first_domain}']['labels'] - List of labels\n")
print(f"Next step:")
print(f"  Use '3_evaluate_model_performance.ipynb' to evaluate model on each domain")
print(f"  Each domain can be evaluated separately for domain-specific performance analysis")
print("="*70)