# [Cell 2] Drive Mount & Configuration
from google.colab import drive
import os
from datetime import datetime
from huggingface_hub import login
import torch

# 1. Mount Google Drive
drive.mount('/content/drive')
print("✓ Google Drive mounted.")

# 2. User Configuration (Fill this part)
# ==================================================================
HF_TOKEN = "YOUR_HF_TOKEN_HERE"
# 결과가 저장될 Drive 경로 설정
OUTPUT_BASE_DIR = "/content/drive/MyDrive/DetectGPT_Results"
# ==================================================================

# 3. Create Output Directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4. Parameters
N_PERTURBATIONS = 100    # 논문 권장: 100 (시간이 오래 걸림. 테스트 시 줄일 수 있음)
SAMPLES_PER_DOMAIN = 200 # 도메인당 Human 100 + AI 100 (총 1000개 샘플)
BATCH_SIZE = 1           # DetectGPT 특성상 1 유지

# 5. Hugging Face Login
if HF_TOKEN and HF_TOKEN != "여기에_HUGGINGFACE_ACCESS_TOKEN_을_입력하세요":
    try:
        login(token=HF_TOKEN)
        print("✓ Logged in to Hugging Face")
    except Exception as e:
        print(f"Warning: Login failed. {e}")
else:
    print("! Warning: No HF Token provided. Llama-3.2 (Gated Model) loading might fail.")

# 6. Device Info
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nConfiguration:")
print(f"  Device: {device}")
print(f"  Output Dir: {OUTPUT_DIR}")
print(f"  Perturbations: {N_PERTURBATIONS}")

# ---

# [Cell 3] Load Models
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("[1/1] Loading Scoring Model (Llama-3.2-1B-Instruct)...")
try:
    # Fast-DetectGPT는 Scoring Model 하나만 사용합니다.
    scorer_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    scorer_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token=HF_TOKEN)
    print("  ✓ Llama loaded (Backbone for Fast-DetectGPT).")
except Exception as e:
    print(f"  ! Error loading Llama: {e}")
    print("  ! Please check your HF Token and Access permissions.")
    raise e

print("\n✓ Model setup complete. (T5 is not required for Fast-DetectGPT)")

# ---

# [Cell 4] Define Fast-DetectGPT Functions
import torch.nn.functional as F
import numpy as np

def get_fast_detectgpt_score(text, model, tokenizer):
    """
    Compute Fast-DetectGPT Score using Conditional Probability Curvature.
    Score = Average( LogLikelihood(x) - Estimated_LogLikelihood(x_tilde) )
          ~ Average( LogLikelihood(x) - Entropy(x) )
    """
    if not text:
        return 0.0

    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(device)
        input_ids = tokenized.input_ids
        labels = input_ids.clone()

        # Forward pass to get logits
        outputs = model(input_ids)
        logits = outputs.logits

        # Shift logits and labels for Causal LM (predict next token)
        # logit[t] predicts label[t+1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 1. Log-Likelihood of the original text (log p(x))
        # log_softmax gives log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather the log prob of the actual target tokens
        true_token_logprobs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # 2. Conditional Probability Curvature (Approximation using Entropy)
        # We calculate the expected log-likelihood of samples drawn from the model
        # Expectation = Sum( P(v) * log P(v) )  <-- This is negative Entropy
        probs = F.softmax(shift_logits, dim=-1)
        # exp_log_probs = sum( p * log(p) )
        expected_token_logprobs = torch.sum(probs * log_probs, dim=-1)

        # 3. Compute the discrepancy (Fast-DetectGPT Score)
        # Score_i = Original_LL_i - Expected_LL_i
        # This tells us how much more likely the original token is compared to the "average" prediction
        token_scores = true_token_logprobs - expected_token_logprobs

        # 4. Normalize by Variance (Optional but recommended in the paper for stability)
        # var = sum( p * (log_p - mu)^2 )
        # mu = expected_token_logprobs.unsqueeze(-1)
        # var_token_logprobs = torch.sum(probs * (log_probs - mu)**2, dim=-1)
        # std_token_logprobs = torch.sqrt(var_token_logprobs)
        # token_scores = token_scores / (std_token_logprobs + 1e-8)

        # Average over the sequence length
        mean_score = token_scores.mean().item()

        return mean_score

print("✓ Fast-DetectGPT functions defined.")

# ---

# [Cell 5] Load & Prepare Data
from datasets import load_dataset
import random

DOMAINS = ['finance', 'medicine', 'open_qa', 'reddit_eli5', 'wiki_csai']
domain_data = {} # {domain: [(text, label), ...]}

print("Loading Data from Hugging Face (Direct JSONL)...")

for domain in DOMAINS:
    url = f"https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{domain}.jsonl"
    try:
        ds = load_dataset("json", data_files=url, split="train")

        pairs = []
        for row in ds:
            h = row.get('human_answers', [])
            a = row.get('chatgpt_answers', [])
            # 너무 짧은 텍스트 제외 (50자 이상)
            if h and a and len(h[0]) > 50 and len(a[0]) > 50:
                pairs.append((h[0], a[0]))

        # Sampling (Human 50 + AI 50)
        random.seed(42)
        random.shuffle(pairs)
        n_pairs = SAMPLES_PER_DOMAIN // 2
        selected = pairs[:n_pairs]

        samples = []
        for h_text, a_text in selected:
            samples.append((h_text, 0)) # Human = 0
            samples.append((a_text, 1)) # AI = 1

        domain_data[domain] = samples
        print(f"  ✓ {domain}: {len(samples)} samples prepared.")

    except Exception as e:
        print(f"  ! Error loading {domain}: {e}")

print("✓ Data preparation complete.")

# ---

# [Cell 6] Run Inference (Fast-DetectGPT)
from tqdm.auto import tqdm

# 결과 저장 구조
all_results = {
    'y_true': [],
    'y_scores': [],
    'domains': []
}

print(f"Starting Fast-DetectGPT Execution...")
print(f"  Total Samples: {sum(len(v) for v in domain_data.values())}")
print(f"  Mode: Analytical Curvature (No T5 Perturbations needed)")
print("This step will be significantly faster than Baseline.")

for domain in DOMAINS:
    if domain not in domain_data:
        continue

    samples = domain_data[domain]
    print(f"\nProcessing {domain} ({len(samples)} samples)...")

    for text, label in tqdm(samples, desc=domain):
        try:
            # Fast-DetectGPT Score 계산
            score = get_fast_detectgpt_score(text, scorer_model, scorer_tokenizer)

            all_results['y_true'].append(label)
            all_results['y_scores'].append(score)
            all_results['domains'].append(domain)
        except Exception as e:
            print(f"Skipping error sample: {e}")
            continue

print("\n✓ Execution complete.")

# ---

# [Cell 7] Evaluate, Visualize & Save
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# 1. Global Metrics (전체 데이터 통합 평가)
y_true_all = all_results['y_true']
y_scores_all = all_results['y_scores']

fpr, tpr, thresholds = roc_curve(y_true_all, y_scores_all)
global_auc = auc(fpr, tpr)

# Find Optimal Threshold (Youden's J statistic)
optimal_idx = np.argmax(tpr - fpr)
global_thresh = thresholds[optimal_idx]
y_pred_all = [1 if s >= global_thresh else 0 for s in y_scores_all]

print(f"\n{'='*60}")
print(f"FINAL SUMMARY (Fast-DetectGPT / Llama-3.2-1B)")
print(f"{'='*60}")
print(f"Overall AUC:       {global_auc:.4f}")
print(f"Overall Accuracy:  {accuracy_score(y_true_all, y_pred_all)*100:.2f}%")
print(f"Optimal Threshold: {global_thresh:.4f}")
print("-" * 60)

# 2. Domain-Specific Metrics & Plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
domain_metrics_dict = {}

# 도메인별 반복
for idx, domain in enumerate(DOMAINS):
    # 해당 도메인의 인덱스 추출
    indices = [i for i, d in enumerate(all_results['domains']) if d == domain]

    if not indices:
        continue

    d_true = [y_true_all[i] for i in indices]
    d_scores = [y_scores_all[i] for i in indices]

    # 도메인별 ROC
    d_fpr, d_tpr, d_threshs = roc_curve(d_true, d_scores)
    d_auc = auc(d_fpr, d_tpr)

    # 도메인별 Threshold 적용 (Global threshold 사용 시 성능 비교용)
    # 개별 최적화가 필요하면 d_threshs에서 다시 구해야 함
    d_pred = [1 if s >= global_thresh else 0 for s in d_scores]

    # 지표 계산
    acc = accuracy_score(d_true, d_pred)
    prec = precision_score(d_true, d_pred, zero_division=0)
    rec = recall_score(d_true, d_pred, zero_division=0)
    f1 = f1_score(d_true, d_pred, zero_division=0)

    domain_metrics_dict[domain] = {
        "auc": d_auc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    }

    print(f"{domain:<15} | Acc: {acc*100:.2f}% | AUC: {d_auc:.4f} | F1: {f1:.4f}")

    # ROC Plot
    ax = axes[idx]
    ax.plot(d_fpr, d_tpr, color='darkorange', lw=2, label=f'AUC = {d_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_title(f'{domain.upper()}')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

axes[-1].axis('off') # 빈 서브플롯 숨김
plt.tight_layout()

# 3. Save Results
# Save Plot
plot_path = os.path.join(OUTPUT_DIR, "fast_detectgpt_roc.png")
plt.savefig(plot_path)
print(f"\n✓ Plot saved to: {plot_path}")

# Save JSON Data (Config 부분 수정됨)
final_output = {
    "config": {
        "method": "Fast-DetectGPT",
        "scoring_model": "Llama-3.2-1B",
        "perturbation_model": "None (Analytical Curvature)"
    },
    "global_metrics": {
        "auc": global_auc,
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "f1": f1_score(y_true_all, y_pred_all)
    },
    "domain_metrics": domain_metrics_dict
}

json_path = os.path.join(OUTPUT_DIR, "fast_detectgpt_results.json")
with open(json_path, "w") as f:
    json.dump(final_output, f, indent=4)
print(f"✓ JSON saved to: {json_path}")

plt.show()