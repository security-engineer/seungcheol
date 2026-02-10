# Fast-DetectGPT: 빠른 Zero-Shot AI 텍스트 탐지

> **논문**: [Fast-DetectGPT: Efficient Zero-Shot Detection via Conditional Probability Curvature](https://arxiv.org/abs/2310.05130)  
> **학회**: ICLR 2024

---

## 📌 핵심 아이디어

기존 DetectGPT는 텍스트를 **수백 번 perturbation** 해야 해서 느렸습니다.  
Fast-DetectGPT는 **단 1번의 forward pass**로 탐지하여 **340배 빠릅니다**.

핵심 개념: **Conditional Probability Curvature**
- AI 생성 텍스트는 확률 분포의 "peak" 근처에 위치
- 사람 텍스트는 확률 분포에서 더 "flat"한 영역에 위치

---

## 🏗️ 동작 원리

```
입력 텍스트
    ↓
Scoring Model (예: GPT-Neo-2.7B)로 토큰 확률 계산
    ↓
Conditional Probability Curvature 계산
    ↓
Threshold 비교 → AI / Human 판정
```

---

## ⚡ 성능 비교

| 방법 | AUROC | 속도 (DetectGPT 대비) |
|------|-------|----------------------|
| DetectGPT | 0.95 | 1x |
| **Fast-DetectGPT** | **0.96** | **340x 빠름** |

---

## 🚀 Docker 사용법

```bash
# 1. 빌드 및 실행
cd Fast-detect-gpt
docker-compose up -d

# 2. 인터랙티브 데모 실행
docker exec -it fast-detect-gpt python scripts/local_infer.py

# 3. GPT-J-6B 샘플링 모델 사용 (더 정확)
docker exec -it fast-detect-gpt python scripts/local_infer.py --sampling_model_name gpt-j-6B
```

---

## 📁 프로젝트 구조

| 파일/폴더 | 설명 |
|-----------|------|
| `scripts/local_infer.py` | 로컬 인터랙티브 데모 |
| `exp_main/` | 5가지 모델 생성 실험 |
| `exp_gpt3to4/` | GPT-3/ChatGPT/GPT-4 실험 |
| `main.sh` | 메인 실험 스크립트 |

---

## 📚 참고

- [GitHub](https://github.com/baoguangsheng/fast-detect-gpt)
- [ArXiv](https://arxiv.org/abs/2310.05130)
