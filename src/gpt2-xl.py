import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# 1. 디바이스 설정 (Apple Silicon의 경우 'mps', NVIDIA는 'cuda', 그 외 'cpu')
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

device = get_device()
print(f"Using device: {device}")

# 2. 모델과 토크나이저 로드 (처음 실행 시 다운로드로 인해 시간이 좀 걸립니다)
model_id = "gpt2-xl"
print(f"Loading {model_id} model...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# 3. 입력 텍스트 설정
input_text = "say hello"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# 4. 텍스트 생성 (추론)
print("Generating text...")
start_time = time.time()

# 생성 옵션 설정 (결과 품질에 큰 영향을 줍니다)
output = model.generate(
    input_ids,
    max_length=100,          # 생성할 최대 길이 (토큰 수)
    num_return_sequences=1,  # 생성할 문장 개수
    no_repeat_ngram_size=2,  # 2-gram 반복 방지 (말 더듬기 방지)
    do_sample=True,          # 샘플링 사용 (더 자연스러운 문장 생성)
    top_k=50,                # Top-K 샘플링
    top_p=0.95,              # Top-P (Nucleus) 샘플링
    temperature=0.7,         # 창의성 조절 (낮을수록 보수적, 높을수록 창의적)
    pad_token_id=tokenizer.eos_token_id # 경고 방지용 설정
)

end_time = time.time()

# 5. 결과 디코딩 및 출력
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n" + "="*30)
print(f"Input: {input_text}")
print("-" * 30)
print(f"Output:\n{generated_text}")
print("="*30)
print(f"Inference time: {end_time - start_time:.2f} seconds")