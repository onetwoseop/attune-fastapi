# GPU 클라우드 서버용 (CUDA 12.4 + Python 3.10)
# CUDA 버전이 다르면 nvidia/cuda 태그 변경: https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 타임존 설정 (apt 설치 중 interactive 방지)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# python3 → python3.10 심볼릭 링크
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# requirements 먼저 복사 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
