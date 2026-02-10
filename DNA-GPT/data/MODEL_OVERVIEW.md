# DNA-GPT: AI 생성 텍스트 탐지 (Zero-Shot)

> **논문**: [DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text](https://arxiv.org/abs/2305.17359)  
> **학회**: ICLR 2024

---

## 📌 핵심 아이디어

DNA-GPT는 **학습 없이 (Zero-Shot)** AI 생성 텍스트를 탐지합니다.

핵심 관찰:
- **사람 텍스트**: 같은 프롬프트로 이어쓰기해도 **다양한** 결과가 나옴
- **AI 텍스트**: 같은 프롬프트로 이어쓰기하면 **유사한** 결과가 나옴 (원래 생성과 비슷)

이 **재생성 유사도** 차이를 이용해 탐지합니다.

---

## 🏗️ 동작 원리

```
입력 텍스트 → 앞부분을 프롬프트로 사용 → LLM으로 N번 재생성
                                              ↓
                              원본 텍스트와 N-gram 유사도 비교
                                              ↓
                              유사도 높으면 AI, 낮으면 Human
```

---

## 🚀 Docker 사용법

```bash
# 1. 빌드 및 실행
cd DNA-GPT
docker-compose up -d

# 2. Gradio 데모 접속
# 브라우저에서 http://localhost:7860 접속

# 3. OpenAI API 사용 시
OPENAI_API_KEY=sk-xxx docker-compose up -d
```

---

## 📁 프로젝트 구조

| 파일/폴더 | 설명 |
|-----------|------|
| `DNA-GPT-dist.py` | Gradio 데모 (메인 실행 파일) |
| `open_source_models/` | 오픈소스 LLM 기반 탐지 |
| `openai_generate/` | OpenAI API 기반 탐지 |
| `get_data/` | 데이터셋 준비 스크립트 |

---

## 📚 참고

- [GitHub](https://github.com/Xianjun-Yang/DNA-GPT)
- [ArXiv](https://arxiv.org/abs/2305.17359)
