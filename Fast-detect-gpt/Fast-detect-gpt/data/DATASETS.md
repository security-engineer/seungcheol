# Fast-DetectGPT - 데이터셋 및 리소스

**GitHub**: https://github.com/baoguangsheng/fast-detect-gpt

---

## 📁 제공 데이터셋

위치: `exp_gpt3to4/data/`

### GPT-3/3.5/4 생성 텍스트 (직접 포함)

| 데이터셋 | Davinci | GPT-3.5-Turbo | GPT-4 |
|----------|:-------:|:-------------:|:-----:|
| PubMed | ✅ | ✅ | ✅ |
| XSum | ✅ | ✅ | ✅ |
| WritingPrompts | ✅ | ✅ | ✅ |

파일 형식: `{dataset}_{model}.raw_data.json`

---

## 🔧 제공 코드

| 스크립트 | 설명 |
|----------|------|
| `main.sh` | 5개 모델 (GPT-2, Neo 등) 실험 |
| `gpt3to4.sh` | GPT-3/3.5/4 실험 |
| `scripts/local_infer.py` | 로컬 인터랙티브 데모 |
| `supervised.sh` | Supervised 학습 |
| `attack.sh` | 공격 실험 |
| `temperature.sh` | Temperature 실험 |

---

## 📥 사용법

```bash
# 로컬 데모 실행
python scripts/local_infer.py

# GPT-3/4 실험 실행
bash gpt3to4.sh
```
