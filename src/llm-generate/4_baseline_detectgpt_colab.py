from google.colab import drive
drive.mount('/content/drive')

print("✓ Google Drive mounted successfully!")

# ---

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import numpy as np
from tqdm import tqdm
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---

# Output directory for results
OUTPUT_DIR = os.path.join(
    "/content/drive/MyDrive/DetectGPT_Baseline_Results",
    datetime.now().strftime("%Y%m%d_%H%M%S")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models
SCORING_MODEL = "gpt2"  # Model to compute log probabilities
MASKING_MODEL = "google/flan-t5-base"  # Model for perturbations (masking & filling)

# DetectGPT Parameters
NUM_PERTURBATIONS = 50  # Number of perturbations per text (paper uses 100, we use 50 for speed)
MASK_RATIO = 0.15  # Ratio of tokens to mask for perturbation
CHUNK_SIZE = 512  # Maximum tokens per chunk (for long texts)

# Sampling
NUM_SAMPLES = 500  # Number of samples to evaluate (use smaller for speed, None for all)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Scoring model: {SCORING_MODEL}")
print(f"Masking model: {MASKING_MODEL}")
print(f"Perturbations per text: {NUM_PERTURBATIONS}")
print(f"Mask ratio: {MASK_RATIO}")
print(f"Samples to evaluate: {NUM_SAMPLES if NUM_SAMPLES else 'ALL'}")

# ---

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✓ Using device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ---

print(f"\nLoading scoring model: {SCORING_MODEL}...")
scoring_tokenizer = AutoTokenizer.from_pretrained(SCORING_MODEL)
scoring_model = AutoModelForCausalLM.from_pretrained(SCORING_MODEL)
scoring_model = scoring_model.to(device)
scoring_model.eval()
print(f"✓ Scoring model loaded")

print(f"\nLoading masking model: {MASKING_MODEL}...")
masking_tokenizer = AutoTokenizer.from_pretrained(MASKING_MODEL)
masking_model = AutoModelForSeq2SeqLM.from_pretrained(MASKING_MODEL)
masking_model = masking_model.to(device)
masking_model.eval()
print(f"✓ Masking model loaded")

print(f"\n✓ All models loaded successfully")

# ---

def load_test_data(num_samples=None):
    """
    Load dataset and prepare test samples.
    Uses the same dataset as other experiments.

    Args:
        num_samples: Number of samples to collect (None = all)

    Returns:
        List of (text, label) tuples
    """
    print(f"\nLoading dataset...")
    ds = load_dataset("artnitolog/llm-generated-texts", split="train")
    print(f"✓ Dataset loaded: {len(ds)} rows")

    # Collect samples
    samples = []
    excluded_cols = {'id', 'prompt', 'dataset_name', 'classes'}

    print(f"Collecting samples...")
    for row in tqdm(ds, desc="Processing"):
        if num_samples and len(samples) >= num_samples:
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

        # Add samples (alternate to maintain balance)
        if num_samples is None or len(samples) < num_samples:
            samples.append((row[human_col], 0))  # Human = 0
        if num_samples is None or len(samples) < num_samples:
            selected_ai_col = random.choice(ai_candidates)
            samples.append((row[selected_ai_col], 1))  # AI = 1

    print(f"✓ Total samples: {len(samples)}")
    print(f"  Human: {sum(1 for _, label in samples if label == 0)}")
    print(f"  AI: {sum(1 for _, label in samples if label == 1)}")

    return samples

# Set random seed for reproducibility
random.seed(42)
test_data = load_test_data(num_samples=NUM_SAMPLES)

# ---

def get_log_likelihood(text, model, tokenizer, max_length=512):
    """
    Compute average log-likelihood of text.

    Args:
        text: Input text string
        model: Language model
        tokenizer: Tokenizer
        max_length: Maximum sequence length

    Returns:
        Average log-likelihood per token
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    input_ids = inputs['input_ids']

    # Compute log-likelihood
    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        # Negative log-likelihood (loss)
        neg_log_likelihood = outputs.loss.item()

    # Return negative to get log-likelihood
    return -neg_log_likelihood


def perturb_text(text, model, tokenizer, mask_ratio=0.15):
    """
    Generate a perturbed version of text by masking and filling.

    Args:
        text: Input text string
        model: Seq2Seq model for filling
        tokenizer: Tokenizer
        mask_ratio: Ratio of tokens to mask

    Returns:
        Perturbed text string
    """
    # Tokenize
    tokens = text.split()
    if len(tokens) == 0:
        return text

    # Randomly select positions to mask
    num_mask = max(1, int(len(tokens) * mask_ratio))
    mask_positions = random.sample(range(len(tokens)), min(num_mask, len(tokens)))

    # Create masked text (replace with <extra_id_0>, <extra_id_1>, etc.)
    masked_tokens = tokens.copy()
    for i, pos in enumerate(sorted(mask_positions)):
        masked_tokens[pos] = f"<extra_id_{i}>"

    masked_text = " ".join(masked_tokens)

    # Use T5 to fill in the masks
    inputs = tokenizer(
        masked_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=1.0
        )

    filled_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Simple reconstruction: just use the filled text
    # (In practice, DetectGPT does more sophisticated span replacement)
    return filled_text if filled_text else text


def compute_detectgpt_score(text, num_perturbations=50):
    """
    Compute DetectGPT score (probability curvature).

    Higher score = more likely to be AI-generated

    Args:
        text: Input text string
        num_perturbations: Number of perturbations to generate

    Returns:
        DetectGPT score (original_ll - mean_perturbed_ll)
    """
    # Get log-likelihood of original text
    original_ll = get_log_likelihood(text, scoring_model, scoring_tokenizer)

    # Generate perturbations and compute their log-likelihoods
    perturbed_lls = []
    for _ in range(num_perturbations):
        perturbed_text = perturb_text(text, masking_model, masking_tokenizer, MASK_RATIO)
        perturbed_ll = get_log_likelihood(perturbed_text, scoring_model, scoring_tokenizer)
        perturbed_lls.append(perturbed_ll)

    # Compute mean perturbed log-likelihood
    mean_perturbed_ll = np.mean(perturbed_lls)

    # DetectGPT score: original - mean_perturbed (curvature)
    # Higher score = more likely AI-generated
    score = original_ll - mean_perturbed_ll

    return score

print("✓ DetectGPT functions defined")

# ---

print("\n" + "="*70)
print("Running DetectGPT Evaluation")
print("="*70)
print(f"\nNote: This will take a while (~{NUM_PERTURBATIONS} perturbations per sample)\n")

scores = []
true_labels = []

for text, label in tqdm(test_data, desc="Computing DetectGPT scores"):
    try:
        score = compute_detectgpt_score(text, num_perturbations=NUM_PERTURBATIONS)
        scores.append(score)
        true_labels.append(label)
    except Exception as e:
        print(f"\nError processing sample: {e}")
        # Skip this sample
        continue

scores = np.array(scores)
true_labels = np.array(true_labels)

print(f"\n✓ Computed scores for {len(scores)} samples")

# ---

# Compute ROC curve to find optimal threshold
fpr, tpr, thresholds = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)

# Find optimal threshold (Youden's index: maximize TPR - FPR)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold: {optimal_threshold:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Make predictions using optimal threshold
predictions = (scores >= optimal_threshold).astype(int)

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average='binary', zero_division=0
)
cm = confusion_matrix(true_labels, predictions)

# Print results
print("\n" + "="*70)
print("BASELINE RESULTS - DetectGPT")
print("="*70)
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"ROC AUC:   {roc_auc*100:.2f}%")
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Human    AI")
print(f"Actual Human    {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"       AI       {cm[1][0]:4d}    {cm[1][1]:4d}")
print("="*70)

# ---

# Save metrics
results = {
    "method": "DetectGPT",
    "scoring_model": SCORING_MODEL,
    "masking_model": MASKING_MODEL,
    "num_perturbations": NUM_PERTURBATIONS,
    "mask_ratio": MASK_RATIO,
    "optimal_threshold": float(optimal_threshold),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist(),
    "num_samples": len(true_labels),
    "num_human": int(np.sum(true_labels == 0)),
    "num_ai": int(np.sum(true_labels == 1)),
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(OUTPUT_DIR, "detectgpt_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Save scores for analysis
np.save(os.path.join(OUTPUT_DIR, "scores.npy"), scores)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), true_labels)

print(f"\n✓ Results saved to: {OUTPUT_DIR}")

# ---

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Score distribution
axes[0].hist(scores[true_labels == 0], bins=30, alpha=0.6, label='Human', color='blue')
axes[0].hist(scores[true_labels == 1], bins=30, alpha=0.6, label='AI', color='red')
axes[0].axvline(optimal_threshold, color='black', linestyle='--', label=f'Threshold={optimal_threshold:.2f}')
axes[0].set_xlabel('DetectGPT Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Score Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. ROC curve
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

# 3. Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Human', 'AI'],
            yticklabels=['Human', 'AI'],
            ax=axes[2])
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')
axes[2].set_title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "detectgpt_analysis.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Visualizations saved to: {OUTPUT_DIR}/detectgpt_analysis.png")

# ---

print("\n" + "="*70)
print("DETECTGPT BASELINE EVALUATION COMPLETE")
print("="*70)
print(f"\n✓ All results saved to Google Drive:")
print(f"  {OUTPUT_DIR}\n")
print(f"Saved files:")
print(f"  1. detectgpt_results.json - Metrics")
print(f"  2. detectgpt_analysis.png - Visualizations")
print(f"  3. scores.npy - Raw scores")
print(f"  4. labels.npy - True labels\n")
print(f"Configuration:")
print(f"  Scoring model: {SCORING_MODEL}")
print(f"  Masking model: {MASKING_MODEL}")
print(f"  Perturbations: {NUM_PERTURBATIONS}")
print(f"  Mask ratio: {MASK_RATIO}\n")
print(f"Final Results:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1 Score:  {f1*100:.2f}%")
print(f"  ROC AUC:   {roc_auc*100:.2f}%\n")
print(f"Dataset: artnitolog/llm-generated-texts")
print(f"Total samples: {len(true_labels)}")
print("="*70)