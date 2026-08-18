FROM python:3.10-slim

WORKDIR /app

COPY requirements-docker.txt .
# pip 설치 캐시를 image에 남기지 않아 image 용량을 줄이기 위해 --no-cache-dir 옵션을 사용함
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY gaze_monitoring ./gaze_monitoring
COPY models ./models

CMD ["python", "-m", "gaze_monitoring.inference.single_image_onnx", "--no-display"]