FROM python:3.10-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY gaze_monitoring ./gaze_monitoring
COPY models ./models

CMD ["python", "-m", "gaze_monitoring.inference.realtime_onnx"]