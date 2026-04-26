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
# Path to your preprocessed data directory
# Example: "/content/drive/MyDrive/RNN_Preprocessed_Data/20231203_143022"
PREPROCESSED_DATA_DIR = "/content/drive/MyDrive/RNN_Preprocessed_Data/20251203_133243"

# ==================================================================

# Output directory for embeddings
EMBEDDINGS_OUTPUT_DIR = os.path.join(
    "/content/drive/MyDrive/RNN_All_Embeddings",
    datetime.now().strftime("%Y%m%d_%H%M%S")
)
os.makedirs(EMBEDDINGS_OUTPUT_DIR, exist_ok=True)

print(f"Preprocessed data directory: {PREPROCESSED_DATA_DIR}")
print(f"Embeddings will be saved to: {EMBEDDINGS_OUTPUT_DIR}")

# Batch Size for embedding extraction
EMBEDDING_BATCH_SIZE = 32

# HuggingFace Token
hf_token = ""  # Add your token here if needed

print(f"\nConfiguration:")
print(f"  Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")

# ---

# Load preprocessed data
preprocessed_file = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_data.pkl")
metadata_file = os.path.join(PREPROCESSED_DATA_DIR, "metadata.json")

print(f"Loading preprocessed data from: {preprocessed_file}")

if not os.path.exists(preprocessed_file):
    raise FileNotFoundError(
        f"Preprocessed data not found at: {preprocessed_file}\n"
        f"Please run '1_preprocess_dataset_colab.ipynb' first and update PREPROCESSED_DATA_DIR."
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

# Split into train/val/test (60/20/20)
train_data, temp_data = train_test_split(preprocessed_data, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"\nData split:")
print(f"  Train: {len(train_data)} documents")
print(f"  Val:   {len(val_data)} documents")
print(f"  Test:  {len(test_data)} documents")

# Save split info with doc_ids for reproducibility
split_info = {
    "total_docs": len(preprocessed_data),
    "train_docs": len(train_data),
    "val_docs": len(val_data),
    "test_docs": len(test_data),
    "split_ratios": "60/20/20",
    "random_seed": 42,
    "train_doc_ids": [doc['doc_id'] for doc in train_data],
    "val_doc_ids": [doc['doc_id'] for doc in val_data],
    "test_doc_ids": [doc['doc_id'] for doc in test_data],
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(EMBEDDINGS_OUTPUT_DIR, "data_split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)

print(f"✓ Split info saved to Google Drive (with doc_ids for reproducibility)")

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

# Convert all splits to embeddings
train_orig, train_inj, train_labels = convert_to_embeddings(train_data, "Train")
val_orig, val_inj, val_labels = convert_to_embeddings(val_data, "Val")
test_orig, test_inj, test_labels = convert_to_embeddings(test_data, "Test")

# ---

print(f"\n{'='*70}")
print("Saving all embeddings to Google Drive")
print(f"{'='*70}\n")

# Save embeddings with BOTH original and injected sequences
embeddings_data = {
    'train': {
        'original_sequences': train_orig,
        'injected_sequences': train_inj,
        'labels': train_labels
    },
    'val': {
        'original_sequences': val_orig,
        'injected_sequences': val_inj,
        'labels': val_labels
    },
    'test': {
        'original_sequences': test_orig,
        'injected_sequences': test_inj,
        'labels': test_labels
    }
}

embeddings_file = os.path.join(EMBEDDINGS_OUTPUT_DIR, "embeddings.pkl")
with open(embeddings_file, 'wb') as f:
    pickle.dump(embeddings_data, f)

print(f"✓ All embeddings saved to: {embeddings_file}")

# Calculate file size
file_size_mb = os.path.getsize(embeddings_file) / (1024 * 1024)
print(f"  File size: {file_size_mb:.2f} MB")

# ---

# Save metadata
embeddings_metadata = {
    "creation_timestamp": datetime.now().isoformat(),
    "preprocessed_data_dir": PREPROCESSED_DATA_DIR,
    "model_id": model_id,
    "embedding_dim": llm_model.config.hidden_size,
    "embedding_batch_size": EMBEDDING_BATCH_SIZE,
    "train_samples": len(train_orig),
    "val_samples": len(val_orig),
    "test_samples": len(test_orig),
    "original_embedding_dim": train_orig[0].shape[1] if train_orig else 0,
    "injected_embedding_dim": train_inj[0].shape[1] if train_inj else 0,
    "embedding_types": {
        "original_sequences": "Original sentence embeddings (seq_len, embedding_dim)",
        "injected_sequences": "Injected sentence embeddings (seq_len, embedding_dim)"
    },
    "device": device,
    "note": "Contains BOTH original and injected embeddings separately. Use either independently or combine as needed."
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

print(f"\nTrain set first document:")
print(f"  Original embedding shape: {train_orig[0].shape}")
print(f"  Injected embedding shape: {train_inj[0].shape}")
print(f"  Label: {train_labels[0]}")

print(f"\nUsage Examples:")
print(f"  1. Use original only: data['train']['original_sequences']")
print(f"  2. Use injected only: data['train']['injected_sequences']")
print(f"  3. Concatenate both: torch.cat([orig, inj], dim=1) -> (seq_len, {train_orig[0].shape[1]*2})")
print(f"  4. Use as separate inputs in multi-input model")

# ---

print("\n" + "="*70)
print("EMBEDDING GENERATION COMPLETE - SUMMARY")
print("="*70)
print(f"\n✓ All embeddings saved to Google Drive:")
print(f"  {EMBEDDINGS_OUTPUT_DIR}\n")
print(f"Saved files:")
print(f"  1. embeddings.pkl - All embedding tensors")
print(f"  2. embeddings_metadata.json - Embedding metadata")
print(f"  3. data_split_info.json - Split information with doc_ids\n")
print(f"Embedding Statistics:")
print(f"  Train samples: {len(train_orig)}")
print(f"  Val samples:   {len(val_orig)}")
print(f"  Test samples:  {len(test_orig)}")
print(f"  Original embedding dim: {train_orig[0].shape[1] if train_orig else 0}")
print(f"  Injected embedding dim: {train_inj[0].shape[1] if train_inj else 0}\n")
print(f"Data Structure:")
print(f"  embeddings_data['train']['original_sequences'] - List of original embeddings")
print(f"  embeddings_data['train']['injected_sequences'] - List of injected embeddings")
print(f"  embeddings_data['train']['labels'] - List of labels\n")
print(f"Next step:")
print(f"  Use this data for training models")
print(f"  Can use original, injected, or both depending on your model architecture")
print("="*70)