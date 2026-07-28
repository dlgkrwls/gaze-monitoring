from pathlib import Path
from typing import Any
import os
import time

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from model import build_base_model


# =========================================================
# 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
WEIGHT_PATH = BASE_DIR / "model_epoch_100.pth"

CAMERA_INDEX = int(os.getenv("GAZE_CAMERA_INDEX", "0"))
INPUT_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================
# 전처리
# =========================================================

def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# =========================================================
# 가중치 로드
# =========================================================

def load_state_dict(
    weight_path: Path,
    device: torch.device,
) -> dict[str, Any]:

    checkpoint = torch.load(
        weight_path,
        map_location=device,
        weights_only=True,
    )

    state_dict = checkpoint

    # 여러 checkpoint 저장 형식 대응
    if isinstance(checkpoint, dict):
        for checkpoint_key in (
            "state_dict",
            "model_state_dict",
            "model",
        ):
            if (
                checkpoint_key in checkpoint
                and isinstance(checkpoint[checkpoint_key], dict)
            ):
                state_dict = checkpoint[checkpoint_key]
                break

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        # DataParallel로 학습했을 때 붙는 module. 제거
        if key.startswith("module."):
            key = key[len("module.") :]

        cleaned_state_dict[key] = value

    return cleaned_state_dict


# =========================================================
# Gaze 화살표
# =========================================================

def draw_gaze_arrow(
    image: np.ndarray,
    yaw: float,
    pitch: float,
    length: int = 85,
) -> np.ndarray:

    output = image.copy()

    height, width = output.shape[:2]

    start_x = width // 2
    start_y = height // 2

    # 기존 확인 결과에 따라 yaw 부호 반전
    yaw_for_draw = -yaw

    gaze_x = np.cos(pitch) * np.sin(yaw_for_draw)
    gaze_y = np.sin(pitch)

    end_x = int(start_x + length * gaze_x)
    end_y = int(start_y - length * gaze_y)

    cv2.circle(
        output,
        (start_x, start_y),
        5,
        (0, 255, 255),
        -1,
    )

    cv2.arrowedLine(
        output,
        (start_x, start_y),
        (end_x, end_y),
        (0, 0, 255),
        3,
        tipLength=0.25,
    )

    yaw_degree = np.degrees(yaw)
    pitch_degree = np.degrees(pitch)

    cv2.putText(
        output,
        f"Yaw: {yaw_degree:.1f}",
        (7, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        output,
        f"Pitch: {pitch_degree:.1f}",
        (7, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    return output


# =========================================================
# ROI 전처리
# =========================================================

def preprocess_roi(
    roi_bgr: np.ndarray,
    preprocess: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:

    # OpenCV BGR → RGB
    roi_rgb = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(roi_rgb)

    input_tensor = preprocess(pil_image)

    # [3, 224, 224] → [1, 3, 224, 224]
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor.to(device)


# =========================================================
# 메인
# =========================================================

def main() -> None:

    if not WEIGHT_PATH.is_file():
        raise FileNotFoundError(
            f"가중치 파일을 찾을 수 없습니다: {WEIGHT_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Weight: {WEIGHT_PATH}")

    preprocess = build_preprocess()

    # 모델 생성
    model = build_base_model(
        backbone="resnet18",
        pretrained=False,
    )

    state_dict = load_state_dict(
        weight_path=WEIGHT_PATH,
        device=device,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    # Windows에서는 CAP_DSHOW가 웹캠 시작 지연을 줄여주는 경우가 많음
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        raise RuntimeError(
            f"웹캠을 열 수 없습니다. camera index={CAMERA_INDEX}"
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )
    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    previous_time = time.perf_counter()

    try:
        while True:

            success, frame = camera.read()

            if not success:
                print("웹캠 프레임을 읽지 못했습니다.")
                break

            frame_height, frame_width = frame.shape[:2]

            if (
                frame_width < INPUT_SIZE
                or frame_height < INPUT_SIZE
            ):
                raise RuntimeError(
                    f"웹캠 해상도가 너무 작습니다: "
                    f"{frame_width}x{frame_height}"
                )

            # ---------------------------------------------
            # 중앙 224×224 ROI
            # ---------------------------------------------

            center_x = frame_width // 2
            center_y = frame_height // 2

            x1 = center_x - INPUT_SIZE // 2
            y1 = center_y - INPUT_SIZE // 2
            x2 = x1 + INPUT_SIZE
            y2 = y1 + INPUT_SIZE

            roi = frame[y1:y2, x1:x2].copy()

            # ---------------------------------------------
            # 추론
            # ---------------------------------------------

            input_tensor = preprocess_roi(
                roi_bgr=roi,
                preprocess=preprocess,
                device=device,
            )

            with torch.inference_mode():
                prediction = model(input_tensor)

            prediction = (
                prediction
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            yaw = float(prediction[0])
            pitch = float(prediction[1])

            # ---------------------------------------------
            # ROI 안에 gaze 화살표
            # ---------------------------------------------

            roi_result = draw_gaze_arrow(
                image=roi,
                yaw=yaw,
                pitch=pitch,
                length=85,
            )

            # 처리된 ROI를 원본 프레임에 다시 삽입
            frame[y1:y2, x1:x2] = roi_result

            # ROI 외곽선
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # 얼굴 위치 안내
            cv2.putText(
                frame,
                "Place your face inside the box",
                (x1 - 20, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # ---------------------------------------------
            # FPS
            # ---------------------------------------------

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "Q: Quit",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.imshow(
                "Real-time Gaze Estimation",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
