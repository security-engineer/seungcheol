# SeqXGPT Docker 사용 가이드

## 📋 사전 요구사항

- Docker Desktop (with WSL2 backend)
- NVIDIA Container Toolkit
- NVIDIA GPU (CUDA 11.3 이상 지원)

## 🚀 빠른 시작

### 1. Docker 이미지 빌드

```bash
docker build -t seqxgpt:latest .
```

### 2. 컨테이너 실행

**GPT-2 추론 서버 시작:**
```bash
docker run --gpus all -p 6006:6006 -v ./models:/app/models -v ./data:/app/data seqxgpt:latest \
    python backend_api.py --port 6006 --timeout 30000 --model=gpt2 --gpu=0
```

**인터랙티브 모드로 실행:**
```bash
docker run --gpus all -it -p 6006:6006 -v ./models:/app/models -v ./data:/app/data seqxgpt:latest /bin/bash
```

### 3. Docker Compose 사용 (권장)

```bash
# 기본 GPT-2 서버 실행
docker-compose up -d seqxgpt

# 모든 서비스 실행 (GPT-2 + GPT-Neo)
docker-compose --profile full up -d

# 로그 확인
docker-compose logs -f seqxgpt

# 종료
docker-compose down
```

## 🔧 지원 모델

| 모델 | 파라미터 | 포트 | 커맨드 |
|------|----------|------|--------|
| GPT-2-xl | 1.5B | 6006 | `--model=gpt2` |
| GPT-Neo | 2.7B | 6007 | `--model=gptneo` |
| GPT-J | 6B | 6008 | `--model=gptj` |
| Llama-3.1-Instruct | 8B | 6009 | `--model=llama` |
| T5 | - | 6010 | `--model=t5` |

## 📁 볼륨 마운트

| 로컬 경로 | 컨테이너 경로 | 설명 |
|-----------|---------------|------|
| `./models` | `/app/models` | 모델 가중치 저장 |
| `./data` | `/app/data` | 입출력 데이터 |
| `./custom` | `/app/custom` | 커스텀 코드 |

## 🛠️ 주요 사용 예시

### Feature 추출

```bash
# 컨테이너 내부에서 실행
python ./dataset/gen_features.py --get_en_features \
    --input_file /app/data/input.jsonl \
    --output_file /app/data/output.jsonl
```

### SeqXGPT 학습

```bash
# 컨테이너 내부에서 SeqXGPT 디렉토리로 이동 후 실행
cd /app/SeqXGPT/SeqXGPT/SeqXGPT
python train.py --config config.yaml
```

## ⚠️ 트러블슈팅

### GPU를 인식하지 못하는 경우
```bash
# NVIDIA Container Toolkit 설치 확인
nvidia-smi
docker run --gpus all nvidia/cuda:11.3.1-base nvidia-smi
```

### 메모리 부족 오류
- `docker-compose.yml`의 `shm_size`를 늘려보세요
- 더 작은 모델 (GPT-2) 사용을 권장합니다

### Windows에서 볼륨 마운트 문제
WSL2 경로를 사용하세요:
```bash
docker run -v /mnt/c/Users/.../models:/app/models seqxgpt:latest
```

## 📚 참고 자료

- [SeqXGPT GitHub](https://github.com/Jihuai-wpy/SeqXGPT)
- [ArXiv Paper](https://arxiv.org/abs/2310.08903)
