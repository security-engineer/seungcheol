# POGER - 데이터셋 및 리소스

**GitHub**: https://github.com/ICTMCG/POGER

---

## 📁 제공 데이터셋

### Google Drive 다운로드
🔗 https://drive.google.com/drive/folders/1xxdjZedn7le_P1HunCDF_WCuoFYI0-pz

| 데이터셋 | 설명 |
|----------|------|
| Binary AIGT | 이진 분류용 |
| Multiclass AIGT | 다중 클래스 분류용 |
| OOD AIGT | Out-of-Distribution 테스트 |
| POGER Features | 사전추출 Feature 파일 |
| POGER-Mixture Features | 혼합 Feature 파일 |

---

## 🔧 제공 코드

| 폴더 | 설명 |
|------|------|
| `POGER/` | 학습/테스트 메인 코드 |
| `get_feature/` | Feature 추출 스크립트 |
| `get_feature/get_true_prob/` | White-box LLM 확률 추출 |

---

## 📥 사용법

```bash
# 1. Google Drive에서 데이터 다운로드 후 data/ 폴더에 저장

# 2. Feature 추출 (선택, 이미 추출된 Feature 제공됨)
cd get_feature
python get_poger_feature.py --n 100 --k 10 --delta 1.2 \
    --input ../data/train.jsonl \
    --output ./train_poger_feature.jsonl

# 3. 학습
cd POGER
python main.py --cuda --model poger --data-dir ../get_feature
```
