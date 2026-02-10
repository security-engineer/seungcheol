# DetectGPT: Zero-Shot AI 텍스트 탐지

> **논문**: [DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature](https://arxiv.org/abs/2301.11305)  
> **학회**: ICML 2023

---

## 📌 핵심 아이디어

LLM 생성 텍스트는 **확률 곡률 (Probability Curvature)** 특성을 가집니다.
- 원본 텍스트에서 약간 변형(perturbation)하면 로그 확률이 **감소**
- Human 텍스트는 이러한 특성이 약함

---

## 🏗️ 동작 원리

```
입력 텍스트
    ↓
T5 등으로 Perturbation 생성 (100개)
    ↓
각 텍스트의 로그 확률 계산
    ↓
원본 vs 변형 확률 차이 분석
    ↓
AI / Human 판정
```

---

## 🚀 Docker 사용법

```bash
# 1. 환경변수 설정
export OPENAI_API_KEY=sk-xxx

# 2. 빌드 및 실행
cd DetectGPT
docker-compose up -d

# 3. 컨테이너 접속
docker exec -it detect-gpt bash
```

---

## 📁 데이터셋

WritingPrompts 데이터셋 필요:
- **다운로드**: https://www.kaggle.com/datasets/ratthachat/writing-prompts
- **저장 위치**: `data/writingPrompts/`

---

## 📚 참고

- [GitHub](https://github.com/eric-mitchell/detect-gpt)
- [Demo](https://detectgpt.ericmitchell.ai)
