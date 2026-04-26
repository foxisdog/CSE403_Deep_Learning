from google.colab import drive
drive.mount('/content/drive')

print("✓ Google Drive mounted successfully!")

# ---

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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
    "/content/drive/MyDrive/RoBERTa_Baseline_Results",
    datetime.now().strftime("%Y%m%d_%H%M%S")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
MODEL_NAME = "openai-community/roberta-base-openai-detector"

# Batch size for inference
BATCH_SIZE = 32

print(f"Output directory: {OUTPUT_DIR}")
print(f"Model: {MODEL_NAME}")
print(f"Batch size: {BATCH_SIZE}")

# ---

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✓ Using device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ---

print(f"\nLoading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

print(f"✓ Model loaded successfully")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---

def load_test_data():
    """
    Load dataset and prepare test samples.
    Uses the same dataset as the RNN experiments.

    Returns:
        List of (text, label) tuples
    """
    print(f"\nLoading dataset...")
    ds = load_dataset("artnitolog/llm-generated-texts", split="train")
    print(f"✓ Dataset loaded: {len(ds)} rows")

    # Collect samples (same logic as preprocessing notebook)
    samples = []
    excluded_cols = {'id', 'prompt', 'dataset_name', 'classes'}

    print(f"Collecting samples...")
    for row in tqdm(ds, desc="Processing"):
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

        # Add human and AI samples
        samples.append((row[human_col], 0))  # Human = 0
        selected_ai_col = random.choice(ai_candidates)
        samples.append((row[selected_ai_col], 1))  # AI = 1

    print(f"✓ Total samples: {len(samples)}")
    print(f"  Human: {sum(1 for _, label in samples if label == 0)}")
    print(f"  AI: {sum(1 for _, label in samples if label == 1)}")

    return samples

# Set random seed for reproducibility
random.seed(42)
test_data = load_test_data()

# ---

def predict_batch(texts, batch_size=32):
    """
    Run inference on a batch of texts.

    Args:
        texts: List of text strings
        batch_size: Batch size for inference

    Returns:
        List of predictions (0=Human, 1=AI)
    """
    all_predictions = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch_texts = texts[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy().tolist())

    return all_predictions

print("✓ Inference function defined")

# ---

print("\n" + "="*70)
print("Running Baseline Evaluation")
print("="*70)

# Prepare data
texts = [text for text, _ in test_data]
true_labels = [label for _, label in test_data]

# Run inference
predictions = predict_batch(texts, batch_size=BATCH_SIZE)

print(f"\n✓ Inference complete!")
print(f"  Total samples: {len(predictions)}")

# ---

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average='binary', zero_division=0
)
cm = confusion_matrix(true_labels, predictions)

# Print results
print("\n" + "="*70)
print("BASELINE RESULTS - RoBERTa OpenAI Detector")
print("="*70)
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1 Score:  {f1*100:.2f}%")
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Human    AI")
print(f"Actual Human    {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"       AI       {cm[1][0]:4d}    {cm[1][1]:4d}")
print("="*70)

# ---

# Save metrics
results = {
    "model": MODEL_NAME,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "num_samples": len(test_data),
    "num_human": sum(1 for _, label in test_data if label == 0),
    "num_ai": sum(1 for _, label in test_data if label == 1),
    "batch_size": BATCH_SIZE,
    "timestamp": datetime.now().isoformat()
}

with open(os.path.join(OUTPUT_DIR, "baseline_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {OUTPUT_DIR}/baseline_results.json")

# ---

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Human', 'AI'],
            yticklabels=['Human', 'AI'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - RoBERTa Baseline\nAccuracy: {accuracy*100:.2f}%')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"✓ Confusion matrix saved to: {OUTPUT_DIR}/confusion_matrix.png")

# ---

print("\n" + "="*70)
print("BASELINE EVALUATION COMPLETE")
print("="*70)
print(f"\n✓ All results saved to Google Drive:")
print(f"  {OUTPUT_DIR}\n")
print(f"Saved files:")
print(f"  1. baseline_results.json - Metrics")
print(f"  2. confusion_matrix.png - Visualization\n")
print(f"Final Results:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1 Score:  {f1*100:.2f}%\n")
print(f"Model: {MODEL_NAME}")
print(f"Dataset: artnitolog/llm-generated-texts")
print(f"Total samples: {len(test_data)}")
print("="*70)