# Lastde: Training-Free AI 텍스트 탐지

> **논문**: [Training-free LLM-generated Text Detection by Mining Token Probability Sequences](https://openreview.net/forum?id=vo4AHjowKi)  
> **학회**: ICLR 2025

---

## 📌 핵심 아이디어

**학습 없이 (Training-free)** 토큰 확률 시퀀스의 패턴만 분석하여 탐지합니다.

두 가지 방법 제공:
- **Lastde**: 토큰 확률 시퀀스의 통계적 특성 분석
- **Lastde++**: 추가 기법을 결합한 향상 버전

---

## 🏗️ 동작 원리

```
입력 텍스트
    ↓
Proxy 모델 (GPT-J, LLaMA 등)로 토큰 확률 계산
    ↓
확률 시퀀스에서 통계적 패턴 추출
    ↓
AI / Human 판정
```

---

## ⚡ 지원 탐지 방법

| 방법 | 설명 |
|------|------|
| Likelihood | 전체 로그 확률 |
| LogRank | 토큰 랭크 기반 |
| Entropy | 엔트로피 분석 |
| DetectLRR | Likelihood Ratio |
| **Lastde** | 토큰 확률 시퀀스 마이닝 |
| **Lastde++** | Lastde 확장 버전 |

---

## 🚀 Docker 사용법

```bash
# 1. 환경변수 설정
export HF_TOKEN=hf_xxx

# 2. 빌드 및 실행
cd Lastde_Detector
docker-compose up -d

# 3. 탐지 실행
docker exec -it lastde-detector bash
cd shell_scripts
./detection_white_box.sh  # White-box
./detection_black_box.sh  # Black-box
```

---

## 📁 프로젝트 구조

| 폴더 | 설명 |
|------|------|
| `py_scripts/` | 탐지 스크립트 |
| `shell_scripts/` | 실험 실행 스크립트 |
| `pretrain_models/` | 모델 가중치 (GPT-J, LLaMA 등) |
| `datasets/` | 실험 데이터셋 |

---

## 📚 참고

- [GitHub](https://github.com/TrustMedia-zju/Lastde_Detector)
- [OpenReview (ICLR 2025)](https://openreview.net/forum?id=vo4AHjowKi)
