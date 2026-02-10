# DNA-GPT - 데이터셋 및 리소스

**GitHub**: https://github.com/Xianjun-Yang/DNA-GPT

---

## 📁 제공 데이터셋

❌ **직접 데이터셋 미포함** - 생성 스크립트만 제공

---

## 🔧 제공 코드

| 폴더/파일 | 설명 |
|-----------|------|
| `get_data/` | 데이터셋 생성 스크립트 |
| `open_source_models/` | 오픈소스 LLM 탐지 코드 |
| `openai_generate/` | OpenAI API 기반 탐지 |
| `DNA-GPT-dist.py` | Gradio 인터랙티브 데모 |

---

## 📥 데이터 생성 방법

```bash
# OpenAI API를 사용한 데이터 생성
cd openai_generate
python generate_data.py

# 오픈소스 모델 사용
cd open_source_models
python generate_data.py
```

> **Note**: OpenAI API 사용 시 API 키 필요
