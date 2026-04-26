import torch
from transformers import pipeline
from huggingface_hub import login

# login(token="")

# 1. 모델 ID 설정 (Meta 공식 3B Instruct 모델)
model_id = "meta-llama/Llama-3.2-1B-Instruct"



print(f"Loading {model_id} on Apple Silicon...")

# 2. 파이프라인 생성 (모델 다운로드 + 로드 자동 수행)
# device_map="auto": Mac의 GPU(MPS)를 자동으로 찾아서 할당해줍니다.
# torch_dtype=torch.bfloat16: Llama3는 bfloat16에서 성능이 제일 좋고 메모리도 아낍니다.
pipe = pipeline(
    "text-generation",
    model=model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

# 3. 대화 메시지 작성 (이게 바로 QA 형식입니다)
# 시스템(system): 모델의 페르소나 설정
# 유저(user): 질문
messages = [
    {"role": "system", "content": "You are a helpful and smart AI assistant."},
    {"role": "user", "content": "say only hello."}
]

# 4. 추론 실행
print("Thinking...")
outputs = pipe(
    messages,
    max_new_tokens=512,   # 답변 최대 길이
    do_sample=True,       # 적절한 다양성
    temperature=0.7,      # 창의성 조절
    top_p=0.9,
)

# 5. 결과 출력
print("="*30)
# 생성된 답변만 깔끔하게 뽑아냅니다.
print(outputs[0]["generated_text"][-1]["content"])
print("="*30)
