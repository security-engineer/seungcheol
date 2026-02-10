# POGER: 10개 단어로 Black-Box AI 텍스트 탐지

> **논문**: [Ten Words Only Still Help: Improving Black-Box AI-Generated Text Detection via Proxy-Guided Efficient Re-Sampling](https://arxiv.org/abs/2402.09199)  
> **학회**: IJCAI 2024

---

## 📌 핵심 아이디어

기존 방법들은 **전체 텍스트**의 확률을 계산해야 해서 비용이 높았습니다.  
POGER는 **10개 단어만** 샘플링해도 효과적으로 탐지할 수 있음을 보였습니다.

핵심: **Proxy-Guided Efficient Re-Sampling**
- 작은 Proxy 모델로 중요한 위치를 빠르게 선정
- 선정된 위치의 토큰만 Black-Box API로 확률 계산
- 비용 절감 + 성능 유지

---

## 🏗️ 동작 원리

```
입력 텍스트
    ↓
Proxy 모델 (GPT-2)로 중요 위치 K개 선정
    ↓
Black-Box API (GPT-3.5/4)로 해당 위치 확률 추정
    ↓
POGER Feature 생성 → 분류기 학습/추론
```

---

## 🚀 Docker 사용법

```bash
# 1. 환경변수 설정
export OPENAI_API_KEY=sk-xxx
export HF_TOKEN=hf_xxx

# 2. 빌드 및 실행
cd POGER
docker-compose up -d

# 3. Feature 추출
docker exec -it poger bash
cd get_feature
python get_poger_feature.py --n 100 --k 10 --delta 1.2 \
    --input ../data/train.jsonl \
    --output ./train_poger_feature.jsonl

# 4. 학습
cd ../POGER
python main.py --cuda --model poger --data-dir ../get_feature
```

---

## 📁 프로젝트 구조

| 파일/폴더 | 설명 |
|-----------|------|
| `POGER/` | 학습/테스트 코드 |
| `get_feature/` | Feature 추출 스크립트 |
| `data/` | 데이터셋 |

---

## 📚 참고

- [GitHub](https://github.com/ICTMCG/POGER)
- [ArXiv](https://arxiv.org/abs/2402.09199)
- [IJCAI 2024](https://www.ijcai.org/proceedings/2024/55)
