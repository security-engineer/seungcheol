# Lastde_Detector - 데이터셋 및 리소스

**GitHub**: https://github.com/TrustMedia-zju/Lastde_Detector

---

## 📁 제공 데이터셋 (가장 풍부)

위치: `datasets/`

| 폴더 | 설명 |
|------|------|
| `human_original_data/` | 원본 Human 텍스트 (XSum 등) |
| `human_llm_data_for_experiment/` | Human+LLM 실험 데이터 |
| `perturbation_data_detectgpt_npr/` | DetectGPT/NPR 실험용 |
| `regeneration_data_dnagpt/` | DNA-GPT 실험용 |
| `paraphrasing_attack_data/` | Paraphrasing 공격 실험 |
| `multi_language_data/` | 다국어 실험 |
| `decoding_strategies_data/` | 디코딩 전략 실험 |
| `response_lengths_data/` | 응답 길이 실험 |

---

## 🔧 제공 코드

| 폴더 | 설명 |
|------|------|
| `py_scripts/baselines/` | 7개 Baseline 탐지기 구현 |
| `shell_scripts/` | 실험 실행 스크립트 |
| `py_scripts/data_generations/` | 데이터 생성 스크립트 |

### 지원 탐지 방법
Likelihood, LogRank, Entropy, DetectLRR, DetectGPT, DetectNPR, DNA-GPT, Fast-DetectGPT, **Lastde**, **Lastde++**

---

## 📥 사용법

```bash
# White-box 탐지
cd shell_scripts
./detection_white_box.sh

# Black-box 탐지
./detection_black_box.sh
```

---

## 📝 결과 저장 위치

`experiment_results/` 폴더 하위:
- `statistic_detection_results/`
- `detectgpt_detection_results/`
- `fast_detectgpt_detection_results/`
- `lastde_doubleplus_detection_results/`
