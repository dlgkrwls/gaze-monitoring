from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def build_yunet_detector(
    model_path: Path,
    input_size: Tuple[int, int],
    score_threshold: float = 0.85,
    nms_threshold: float = 0.30,
    top_k: int = 5000,
):
    if not model_path.is_file():
        raise FileNotFoundError(
            "YuNet 모델을 찾지 못했습니다: {}".format(
                model_path
            )
        )

    # Git LFS pointer만 내려받은 경우 방지
    if model_path.stat().st_size < 10_000:
        raise RuntimeError(
            "YuNet ONNX 파일 크기가 너무 작습니다. "
            "실제 ONNX 모델 대신 Git LFS pointer를 "
            "받았을 가능성이 있습니다."
        )

    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError(
            "현재 OpenCV에 FaceDetectorYN이 없습니다. "
            "opencv-python을 업그레이드하세요."
        )

    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        input_size,
        score_threshold,
        nms_threshold,
        top_k,
    )

    return detector


def select_driver_face(
    faces: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    여러 얼굴이 검출되면 가장 큰 얼굴을 운전자로 선택한다.

    YuNet detection 한 행:
    [x, y, width, height,
     right_eye_x, right_eye_y,
     left_eye_x, left_eye_y,
     nose_x, nose_y,
     right_mouth_x, right_mouth_y,
     left_mouth_x, left_mouth_y,
     confidence]
    """

    if faces is None or len(faces) == 0:
        return None

    largest_face = max(
        faces,
        key=lambda face: float(
            face[2] * face[3]
        ),
    )

    return largest_face


def draw_yunet_detection(
    frame: np.ndarray,
    face: np.ndarray,
    draw_landmarks: bool = True,
) -> None:
    """
    YuNet 원본 bbox와 landmark를 전체 프레임에 표시한다.
    """

    x, y, width, height = (
        face[:4]
        .astype(np.int32)
    )

    confidence = float(face[-1])

    # YuNet 원본 bbox: 파란색
    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        2,
    )

    cv2.putText(
        frame,
        "YuNet {:.2f}".format(confidence),
        (x, max(22, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
    )

    if not draw_landmarks:
        return

    landmarks = (
        face[4:14]
        .reshape(5, 2)
        .astype(np.int32)
    )

    landmark_colors = [
        (255, 0, 0),      # right eye
        (0, 0, 255),      # left eye
        (0, 255, 0),      # nose
        (255, 0, 255),    # right mouth
        (0, 255, 255),    # left mouth
    ]

    for landmark, color in zip(
        landmarks,
        landmark_colors,
    ):
        cv2.circle(
            frame,
            tuple(landmark),
            3,
            color,
            -1,
        )


def smooth_bbox(
    previous_bbox: Optional[np.ndarray],
    current_bbox: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    EMA 방식으로 얼굴 bbox 흔들림을 줄인다.
    """

    current_bbox = current_bbox.astype(
        np.float32
    )

    if previous_bbox is None:
        return current_bbox

    return (
        alpha * current_bbox
        + (1.0 - alpha) * previous_bbox
    )


def make_square_face_crop(
    frame: np.ndarray,
    bbox: np.ndarray,
    scale: float,
    output_size: int = 224,
) -> Tuple[
    np.ndarray,
    Tuple[int, int, int, int],
]:
    """
    YuNet bbox를 중심 기준 정사각형 crop으로 변환한다.

    Gaze360 전처리 방식:
    1. bbox 중심 계산
    2. max(width, height) 기준 정사각형
    3. output_size x output_size resize
    """

    x, y, width, height = bbox[:4]

    x = float(x)
    y = float(y)
    width = float(width)
    height = float(height)

    frame_height, frame_width = frame.shape[:2]

    center_x = x + width / 2.0
    center_y = y + height / 2.0

    side = max(width, height) * scale
    side = max(1, int(round(side)))

    x1 = int(round(center_x - side / 2.0))
    y1 = int(round(center_y - side / 2.0))

    x2 = x1 + side
    y2 = y1 + side

    # 프레임 밖으로 나간 영역
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)

    pad_right = max(
        0,
        x2 - frame_width,
    )

    pad_bottom = max(
        0,
        y2 - frame_height,
    )

    clipped_x1 = max(0, x1)
    clipped_y1 = max(0, y1)

    clipped_x2 = min(
        frame_width,
        x2,
    )

    clipped_y2 = min(
        frame_height,
        y2,
    )

    crop = frame[
        clipped_y1:clipped_y2,
        clipped_x1:clipped_x2,
    ].copy()

    if crop.size == 0:
        raise RuntimeError(
            "얼굴 crop 영역이 비어 있습니다."
        )

    # 얼굴이 프레임 경계에 걸린 경우 padding
    if any(
        (
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        )
    ):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )

    crop = cv2.resize(
        crop,
        (
            output_size,
            output_size,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    display_box = (
        clipped_x1,
        clipped_y1,
        clipped_x2,
        clipped_y2,
    )

    return crop, display_box
