# SeqXGPT - 데이터셋 및 리소스

**GitHub**: https://github.com/Jihuai-wpy/SeqXGPT

---

## 📁 제공 데이터셋

위치: `SeqXGPT/dataset/`

| 데이터셋 | 설명 |
|----------|------|
| `SeqXGPT-Bench/` | 문장 수준 탐지용 벤치마크 |
| `document-level detection dataset/` | 문서 수준 탐지용 |
| `OOD sentence-level detection dataset/` | Out-of-Distribution 테스트 |

### 레이블 종류
`gpt2`, `gptneo`, `gptj`, `llama`, `gpt3re`, `human`

---

## 📝 데이터 포맷

```json
{
  "text": "전체 문서 텍스트...",
  "prompt_len": 254,
  "label": "gpt3re"
}
```

- `text`: 전체 문서
- `prompt_len`: Human/AI 경계 위치 (text[:prompt_len]이 Human)
- `label`: 문장별 레이블

---

## 🔧 제공 코드

| 파일 | 설명 |
|------|------|
| `backend_api.py` | LLM 추론 서버 |
| `backend_model.py` | 모델 로딩 |
| `dataset/gen_features.py` | Feature 추출 |
| `SeqXGPT/` | 학습/테스트 코드 |

---

## 📥 다운로드

데이터셋은 GitHub 레포에 직접 포함되어 있습니다.
