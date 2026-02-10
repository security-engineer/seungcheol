"""
DNA-GPT 배치 탐지 스크립트
===========================
sample_001.jsonl의 abstract_hwt, abstract_onlyllm, abstract_rag 각 필드 텍스트를
DNA-GPT 알고리즘으로 탐지하여 AI/Human 판정 결과를 출력한다.

사용법:
    # OpenAI API 모드
    python run_detection.py --input datasets/sample_001.jsonl --api-key YOUR_KEY
    python run_detection.py --input datasets/sample_001.jsonl --api-key YOUR_KEY --limit 5

    # 로컬 모델 모드 (LLaMA 3.1 8B 등)
    python run_detection.py --input datasets/sample_001.jsonl --local meta-llama/Llama-3.1-8B-Instruct
    python run_detection.py --input datasets/sample_001.jsonl --local meta-llama/Llama-3.1-8B-Instruct --limit 5
    python run_detection.py --input datasets/sample_001.jsonl --dry-run
"""

import argparse
import json
import os
import sys
import time
import re
from datetime import datetime

import ssl
import nltk
import numpy as np

# ── NLTK 설정 ──
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── rouge_score에서 _create_ngrams 가져오기 ──
import six
from nltk.stem.porter import PorterStemmer
from rouge_score.rouge_scorer import _create_ngrams

# ─────────────────────────────────────────────
#  DNA-GPT 핵심 함수들 (원본에서 추출 + 버그 수정)
# ─────────────────────────────────────────────

_stemmer = PorterStemmer()

try:
    import spacy
    _nlp = spacy.load('en_core_web_sm')
    _stopwords = _nlp.Defaults.stop_words
except Exception:
    _stopwords = set()


def tokenize(text, stemmer=_stemmer, stopwords=_stopwords):
    """텍스트를 토큰 리스트로 변환 (소문자화 + 스테밍 + 불용어 제거)"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", six.ensure_str(text))
    tokens = re.split(r"\s+", text)
    if stemmer:
        tokens = [stemmer.stem(x) if len(x) > 3 else x
                  for x in tokens if x not in stopwords]
    tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", six.ensure_str(x))]
    return tokens


def get_score_ngrams(target_ngrams, prediction_ngrams):
    """두 N-gram 집합의 겹침 비율 계산"""
    intersection_count = 0
    ngram_dict = {}
    for ngram in six.iterkeys(target_ngrams):
        intersection_count += min(target_ngrams[ngram], prediction_ngrams[ngram])
        ngram_dict[ngram] = min(target_ngrams[ngram], prediction_ngrams[ngram])
    target_count = sum(target_ngrams.values())
    return intersection_count / max(target_count, 1), ngram_dict


def get_ngram_info(article_tokens, summary_tokens, _ngram):
    """두 문서의 N-gram 겹침 점수 계산"""
    article_ngram = _create_ngrams(article_tokens, _ngram)
    summary_ngram = _create_ngrams(summary_tokens, _ngram)
    ngram_score, ngram_dict = get_score_ngrams(article_ngram, summary_ngram)
    return ngram_score, ngram_dict, sum(ngram_dict.values())


def N_gram_detector(ngram_n_ratio):
    """N=3~25 N-gram 겹침 점수를 가중 평균으로 계산 (decay weighting: n*log(n))"""
    score = 0
    non_zero = []
    for idx, key in enumerate(ngram_n_ratio):
        if idx in range(3) and ('score' in key or 'ratio' in key):
            score += 0.0 * ngram_n_ratio[key]
            continue
        if 'score' in key or 'ratio' in key:
            score += (idx + 1) * np.log(idx + 1) * ngram_n_ratio[key]
            if ngram_n_ratio[key] != 0:
                non_zero.append(idx + 1)
    return score / (sum(non_zero) + 1e-8)


def truncate_string_by_words(string, max_words):
    """단어 수 기준으로 텍스트 자르기"""
    words = string.split()
    if len(words) <= max_words:
        return string
    return ' '.join(words[:max_words])


# ─────────────────────────────────────────────
#  로컬 모델 로드 + 재생성
# ─────────────────────────────────────────────

def load_local_model(model_path, device_map="auto"):
    """로컬 HuggingFace 모델을 FP16으로 로드.
    device_map='auto'로 VRAM 부족 시 자동으로 RAM에 분배."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"모델 로딩: {model_path} (FP16, device_map={device_map})")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # VRAM 사용량 출력
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 메모리: {allocated:.1f}GB 사용 / {reserved:.1f}GB 예약")

    print("모델 로딩 완료!")
    return model, tokenizer


def local_generate(prefix, model, tokenizer, max_new_tokens=300,
                   temperature=0.7, num_return=1):
    """로컬 모델로 텍스트 재생성."""
    import torch

    # Instruct 모델이면 chat template 사용
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that continues the passage from the sentences provided."},
            {"role": "user",
             "content": f"Continue the following text in around 300 words:\n\n{prefix}"},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = prefix

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    results = []
    for _ in range(num_return):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        # 입력 부분 제거하고 생성된 부분만 추출
        gen_ids = output[0][inputs['input_ids'].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append(gen_text)

    return results


# ─────────────────────────────────────────────
#  OpenAI / 로컬 모델 탐지 (버그 수정 버전)
# ─────────────────────────────────────────────

def detect_text(text, client, model_name="gpt-3.5-turbo",
                truncate_ratio=0.5, threshold=0.00025,
                regen_number=10, max_new_tokens=300,
                temperature=0.7, max_words=350,
                local_model=None, local_tokenizer=None):
    """
    DNA-GPT 탐지: 텍스트를 반으로 잘라 뒷부분을 재생성한 뒤
    원본 suffix와 재생성본의 N-gram 겹침 점수로 AI 여부를 판정.
    local_model이 주어지면 로컬 모델로, 아니면 OpenAI API로 재생성.

    Returns:
        dict: {
            'decision': bool (True=AI, False=Human),
            'score': float (N-gram overlap score),
            'threshold': float,
            'regen_count': int (실제 재생성 수),
        }
    """
    text = truncate_string_by_words(text, max_words)

    if len(text.strip()) < 50:
        return {
            'decision': None,
            'score': 0.0,
            'threshold': threshold,
            'regen_count': 0,
            'error': 'text_too_short'
        }

    # 텍스트를 prefix / suffix로 분리
    split_point = int(truncate_ratio * len(text))
    prefix = text[:split_point]
    suffix = text[split_point:]
    suffix_tokens = tokenize(suffix)

    if len(suffix_tokens) == 0:
        return {
            'decision': None,
            'score': 0.0,
            'threshold': threshold,
            'regen_count': 0,
            'error': 'empty_suffix_tokens'
        }

    # ── 재생성 ──
    regen_texts = []
    try:
        if local_model is not None and local_tokenizer is not None:
            # ── 로컬 모델로 재생성 ──
            regen_texts = local_generate(
                prefix, local_model, local_tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return=regen_number,
            )
        elif model_name in ("gpt-3.5-turbo-instruct",):
            # Completions API (instruct 모델)
            response = client.completions.create(
                model=model_name,
                prompt=prefix,
                max_tokens=max_new_tokens,
                temperature=temperature,
                n=regen_number
            )
            regen_texts = [c.text for c in response.choices]
        else:
            # Chat Completions API (gpt-3.5-turbo, gpt-4 등)
            for _ in range(regen_number):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant that continues the passage from the sentences provided."},
                        {"role": "user",
                         "content": "continues the passage from the current text within in total around 300 words:"},
                        {"role": "assistant",
                         "content": prefix},
                    ],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                regen_texts.append(response.choices[0].message.content)
    except Exception as e:
        return {
            'decision': None,
            'score': 0.0,
            'threshold': threshold,
            'regen_count': 0,
            'error': str(e)
        }

    # ── N-gram 겹침 점수 계산 ──
    gpt_scores = []
    for gen_text in regen_texts:
        gen_text_truncated = truncate_string_by_words(gen_text, max_words - 150)
        gen_tokens = tokenize(gen_text_truncated)
        if len(gen_tokens) == 0:
            continue

        temp1 = {}
        for _ngram in range(1, 25):
            ngram_score, ngram_dict, overlap_count = get_ngram_info(
                suffix_tokens, gen_tokens, _ngram)
            temp1[f'ngram_{_ngram}_score'] = ngram_score / len(gen_tokens)
            temp1[f'ngram_{_ngram}_ngramdict'] = ngram_dict
            temp1[f'ngram_{_ngram}_count'] = overlap_count

        score = N_gram_detector(temp1)
        gpt_scores.append(score)

    if len(gpt_scores) == 0:
        return {
            'decision': None,
            'score': 0.0,
            'threshold': threshold,
            'regen_count': 0,
            'error': 'no_valid_regenerations'
        }

    avg_score = float(np.mean(gpt_scores))

    return {
        'decision': avg_score > threshold,
        'score': avg_score,
        'threshold': threshold,
        'regen_count': len(gpt_scores),
    }


# ─────────────────────────────────────────────
#  데이터 로드 + 배치 실행
# ─────────────────────────────────────────────

def load_data(path):
    """sample_001.jsonl 로드"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_batch(records, client, args):
    """
    각 레코드의 abstract_hwt, abstract_onlyllm, abstract_rag를
    DNA-GPT로 탐지하고 결과를 수집한다.
    """
    results = []
    total = len(records)

    fields = [
        ('abstract_hwt', 'human'),        # 기대: Human (decision=False)
        ('abstract_only_llm', 'ai'),       # 기대: AI (decision=True)
        ('abstract_rag', 'ai'),            # 기대: AI (decision=True)
    ]

    for i, record in enumerate(records):
        paper_id = record.get('paper_id', f'unknown_{i}')
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] Paper: {paper_id}")
        print(f"{'='*60}")

        record_result = {
            'paper_id': paper_id,
            'keyword': record.get('keyword', ''),
        }

        for field_name, expected_label in fields:
            text = record.get(field_name, '')
            if not text or not text.strip():
                print(f"  {field_name}: (빈 텍스트 - 건너뜀)")
                record_result[field_name] = {
                    'decision': None,
                    'score': 0.0,
                    'expected': expected_label,
                    'correct': None,
                    'error': 'empty_text'
                }
                continue

            word_count = len(text.split())
            print(f"  {field_name} ({word_count} words, expected={expected_label})...", end=" ", flush=True)

            if args.dry_run:
                # dry-run 모드: API 호출 없이 데이터 구조만 확인
                result = {
                    'decision': None,
                    'score': 0.0,
                    'threshold': args.threshold,
                    'regen_count': 0,
                    'dry_run': True
                }
            else:
                result = detect_text(
                    text=text,
                    client=client,
                    model_name=args.model_name,
                    truncate_ratio=args.truncate_ratio,
                    threshold=args.threshold,
                    regen_number=args.regen_number,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    max_words=args.max_words,
                    local_model=getattr(args, '_local_model', None),
                    local_tokenizer=getattr(args, '_local_tokenizer', None),
                )

            result['expected'] = expected_label
            if result['decision'] is not None:
                predicted = 'ai' if result['decision'] else 'human'
                result['predicted'] = predicted
                result['correct'] = (predicted == expected_label)
                decision_str = "🤖 AI" if result['decision'] else "👤 Human"
                correct_str = "✅" if result['correct'] else "❌"
                print(f"{decision_str} (score={result['score']:.6f}) {correct_str}")
            else:
                result['predicted'] = None
                result['correct'] = None
                error = result.get('error', result.get('dry_run', ''))
                print(f"⚠️ 판정불가 ({error})")

            record_result[field_name] = result

            # API rate limit 대비 딜레이
            if not args.dry_run:
                time.sleep(args.delay)

        results.append(record_result)

    return results


def print_summary(results):
    """탐지 결과 요약 출력"""
    fields = ['abstract_hwt', 'abstract_only_llm', 'abstract_rag']
    field_labels = {
        'abstract_hwt': 'Human (hwt)',
        'abstract_only_llm': 'AI (only_llm)',
        'abstract_rag': 'AI (rag)',
    }

    print(f"\n{'='*60}")
    print("탐지 결과 요약")
    print(f"{'='*60}")

    total_correct = 0
    total_tested = 0

    for field in fields:
        correct = 0
        tested = 0
        scores = []
        for r in results:
            fr = r.get(field, {})
            if fr.get('correct') is not None:
                tested += 1
                if fr['correct']:
                    correct += 1
                scores.append(fr['score'])

        total_correct += correct
        total_tested += tested

        if tested > 0:
            acc = correct / tested * 100
            avg_score = np.mean(scores) if scores else 0
            print(f"  {field_labels[field]:20s}: {correct}/{tested} 정확 ({acc:.1f}%), "
                  f"평균 score={avg_score:.6f}")
        else:
            print(f"  {field_labels[field]:20s}: 테스트 없음")

    if total_tested > 0:
        overall_acc = total_correct / total_tested * 100
        print(f"\n  {'전체':20s}: {total_correct}/{total_tested} 정확 ({overall_acc:.1f}%)")


def save_results(results, output_path):
    """결과를 JSONL 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            # ngram dict는 직렬화 불가능하므로 제거
            clean_r = {}
            for k, v in r.items():
                if isinstance(v, dict):
                    clean_v = {kk: vv for kk, vv in v.items()
                               if not isinstance(vv, dict) or kk in ('error',)}
                    clean_r[k] = clean_v
                else:
                    clean_r[k] = v
            f.write(json.dumps(clean_r, ensure_ascii=False) + '\n')
    print(f"\n결과 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DNA-GPT 배치 탐지: sample_001.jsonl의 세 필드를 AI/Human으로 판정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # dry-run (API 호출 없이 데이터 구조 확인)
  python run_detection.py --input datasets/sample_001.jsonl --dry-run

  # 로컬 모델 (LLaMA 3.1 8B Instruct, FP16)
  python run_detection.py -i datasets/sample_001.jsonl --local meta-llama/Llama-3.1-8B-Instruct --limit 3

  # 로컬 모델 + 재생성 줄이기 (빠른 테스트)
  python run_detection.py -i datasets/sample_001.jsonl --local meta-llama/Llama-3.1-8B-Instruct --regen 3 --limit 5

  # OpenAI API 모드
  python run_detection.py -i datasets/sample_001.jsonl --api-key sk-... --limit 5
  python run_detection.py -i datasets/sample_001.jsonl --api-key sk-... --model gpt-4
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='sample_001.jsonl 경로')
    parser.add_argument('--local', default=None, metavar='MODEL_PATH',
                        help='로컬 HuggingFace 모델 경로 또는 ID (예: meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--api-key', '-k', default=None,
                        help='OpenAI API key (또는 OPENAI_API_KEY 환경변수)')
    parser.add_argument('--model', dest='model_name', default='gpt-3.5-turbo',
                        help='재생성에 사용할 OpenAI 모델 (기본: gpt-3.5-turbo)')
    parser.add_argument('--limit', '-n', type=int, default=None,
                        help='테스트할 레코드 수 제한')
    parser.add_argument('--regen', dest='regen_number', type=int, default=10,
                        help='재생성 횟수 (기본: 10, 원본: 30)')
    parser.add_argument('--threshold', type=float, default=0.00025,
                        help='AI 판정 임계값 (기본: 0.00025)')
    parser.add_argument('--truncate-ratio', type=float, default=0.5,
                        help='텍스트 분할 비율 (기본: 0.5)')
    parser.add_argument('--max-tokens', type=int, default=300,
                        help='재생성 최대 토큰 수 (기본: 300)')
    parser.add_argument('--max-words', type=int, default=350,
                        help='입력 텍스트 최대 단어 수 (기본: 350)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='재생성 온도 (기본: 0.7)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='API 호출 간 딜레이(초) (기본: 1.0)')
    parser.add_argument('--output', '-o', default=None,
                        help='결과 저장 경로 (기본: results/detection_결과_타임스탬프.jsonl)')
    parser.add_argument('--dry-run', action='store_true',
                        help='API 호출 없이 데이터 구조만 확인')

    args = parser.parse_args()

    # ── 모드 결정 ──
    use_local = args.local is not None

    if not use_local:
        api_key = args.api_key or os.environ.get('OPENAI_API_KEY', '')
        if not api_key and not args.dry_run:
            print("❌ OpenAI API key가 필요합니다.")
            print("   --api-key 옵션 또는 OPENAI_API_KEY 환경변수를 설정하세요.")
            print("   또는 --local 옵션으로 로컬 모델을 사용하세요.")
            sys.exit(1)

    # ── 데이터 로드 ──
    print(f"데이터 로드: {args.input}")
    records = load_data(args.input)
    print(f"전체 레코드: {len(records)}건")

    if args.limit:
        records = records[:args.limit]
        print(f"테스트 대상: {args.limit}건으로 제한")

    # ── 모델 로드 ──
    client = None
    if not args.dry_run:
        if use_local:
            model, tok = load_local_model(args.local)
            args._local_model = model
            args._local_tokenizer = tok
            args.model_name = args.local
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

    # ── 설정 출력 ──
    mode_str = f"로컬 ({args.local})" if use_local else f"OpenAI ({args.model_name})"
    print(f"\n── 설정 ──")
    print(f"  모드:        {mode_str}")
    print(f"  모델:        {args.model_name}")
    print(f"  재생성 횟수:  {args.regen_number}")
    print(f"  임계값:      {args.threshold}")
    print(f"  분할 비율:   {args.truncate_ratio}")
    print(f"  최대 단어:   {args.max_words}")
    print(f"  Dry-run:     {args.dry_run}")
    print(f"  필드: abstract_hwt(Human), abstract_only_llm(AI), abstract_rag(AI)")

    # ── 배치 실행 ──
    start_time = time.time()
    results = run_batch(records, client, args)
    elapsed = time.time() - start_time

    # ── 요약 출력 ──
    print_summary(results)
    print(f"\n소요 시간: {elapsed:.1f}초")

    # ── 결과 저장 ──
    if args.output:
        output_path = args.output
    else:
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'results/detection_{timestamp}.jsonl'

    save_results(results, output_path)


if __name__ == '__main__':
    main()
