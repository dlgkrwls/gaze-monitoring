from typing import Tuple

import cv2
import numpy as np


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

    # 단일 이미지 확인 결과를 반영한 yaw 부호
    yaw_for_draw = -yaw

    gaze_x = (
        np.cos(pitch)
        * np.sin(yaw_for_draw)
    )

    gaze_y = np.sin(pitch)

    end_x = int(
        start_x
        + length * gaze_x
    )

    end_y = int(
        start_y
        - length * gaze_y
    )

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

    cv2.putText(
        output,
        "Yaw: {:.1f}".format(
            np.degrees(yaw)
        ),
        (7, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        output,
        "Pitch: {:.1f}".format(
            np.degrees(pitch)
        ),
        (7, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    return output


def draw_gaze_arrow_on_frame(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    yaw: float,
    pitch: float,
) -> None:
    """전체 결과 영상의 얼굴 위치에 gaze 방향을 직접 표시한다."""
    x1, y1, x2, y2 = box
    start_x = int((x1 + x2) / 2)
    start_y = int((y1 + y2) / 2)

    box_size = max(x2 - x1, y2 - y1)
    length = max(60, int(box_size * 0.75))

    yaw_for_draw = -yaw
    gaze_x = np.cos(pitch) * np.sin(yaw_for_draw)
    gaze_y = np.sin(pitch)

    end_x = int(start_x + length * gaze_x)
    end_y = int(start_y - length * gaze_y)

    cv2.circle(frame, (start_x, start_y), 6, (0, 255, 255), -1)
    cv2.arrowedLine(
        frame,
        (start_x, start_y),
        (end_x, end_y),
        (0, 0, 255),
        4,
        tipLength=0.25,
    )

    cv2.putText(
        frame,
        "Yaw/Pitch: {:.1f}, {:.1f} deg".format(
            np.degrees(yaw),
            np.degrees(pitch),
        ),
        (x1, min(frame.shape[0] - 10, y2 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )


def paste_face_preview(
    frame: np.ndarray,
    face_result: np.ndarray,
    width: int = 224,
) -> None:
    """gaze 화살표가 그려진 얼굴 crop을 결과 영상 우측 상단에 삽입한다."""
    frame_h, frame_w = frame.shape[:2]
    preview = cv2.resize(face_result, (width, width))

    margin = 12
    x1 = max(0, frame_w - width - margin)
    y1 = margin
    x2 = min(frame_w, x1 + width)
    y2 = min(frame_h, y1 + width)

    preview = preview[: y2 - y1, : x2 - x1]
    frame[y1:y2, x1:x2] = preview
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(
        frame,
        "Gaze crop",
        (x1 + 6, min(frame_h - 6, y2 + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )
