from pathlib import Path
from typing import Optional
import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch

from model import build_base_model
from gaze_monitoring.utils.checkpoint import load_checkpoint
from gaze_monitoring.utils.face_detection import (
    build_yunet_detector,
    select_driver_face,
    draw_yunet_detection,
    smooth_bbox,
    make_square_face_crop,
)
from gaze_monitoring.utils.preprocessing import preprocess_face
from gaze_monitoring.utils.visualization import (
    draw_gaze_arrow,
    draw_gaze_arrow_on_frame,
    paste_face_preview,
)
from gaze_monitoring.utils.monitoring import (
    DriverMonitor,
    draw_monitoring_overlay,
)


# ============================================================
# 1. File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

GAZE_WEIGHT_PATH = BASE_DIR / "weights" / "model_epoch_100.pth"

# 기본 입출력 경로. 실행 인자로 덮어쓸 수 있다.
DEFAULT_VIDEO_PATH = BASE_DIR / "input.mp4"
DEFAULT_OUTPUT_VIDEO_PATH = BASE_DIR / "outputs" / "output_gaze_monitoring.mp4"
DEFAULT_OUTPUT_CSV_PATH = BASE_DIR / "outputs" / "output_gaze_monitoring.csv"

YUNET_MODEL_PATH = Path(
    BASE_DIR / "models/face_detection_yunet_2023mar.onnx"
)


# ============================================================
# 2. Camera settings
# ============================================================

CAMERA_INDEX = int(os.getenv("GAZE_CAMERA_INDEX", "0"))

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# ============================================================
# 3. YuNet face detector settings
# ============================================================

YUNET_SCORE_THRESHOLD = 0.85
YUNET_NMS_THRESHOLD = 0.30
YUNET_TOP_K = 5000

# 얼굴 bbox에 추가할 여백
FACE_SCALE = 1.10

# 1: 매 프레임 얼굴 검출
# FPS가 낮으면 2 또는 3으로 변경
DETECT_EVERY_N_FRAMES = 1

# 얼굴 검출이 잠깐 끊겨도 이전 bbox 유지
MAX_MISSED_DETECTIONS = 4

# bbox EMA smoothing
# 낮을수록 부드럽지만 반응이 느림
BBOX_SMOOTHING_ALPHA = 0.45

DRAW_LANDMARKS = True


# ============================================================
# 4. Gaze model settings
# ============================================================

GAZE_INPUT_SIZE = 224

# gaze EMA smoothing
# 낮을수록 부드럽지만 반응이 느림
GAZE_SMOOTHING_ALPHA = 0.30

SHOW_FACE_WINDOW = True


# ============================================================
# 5. Driver monitoring settings
# ============================================================

# 입력 영상 시작 후 이 시간 동안 정면 gaze를 자동 수집한다.
AUTO_CALIBRATION_SEC = 2.0

# 캘리브레이션에 필요한 최소 유효 얼굴 프레임 수
MIN_CALIBRATION_SAMPLES = 5

# 영상 FPS를 읽지 못했을 때 사용할 기본값
DEFAULT_VIDEO_FPS = 30.0

# 정면 기준으로부터 허용할 gaze 각도
# 현재는 프로토타입용 임시 기준
YAW_THRESHOLD_DEG = 15.0
PITCH_THRESHOLD_DEG = 12.0

# 정면 이탈이 이 시간 이상 지속되면 경고
WARNING_AFTER_SEC = 1.5

# 정면 복귀 후 이 시간 이상 유지해야
# 하나의 이탈 이벤트가 종료된 것으로 판단
RECOVERY_AFTER_SEC = 0.5


# ============================================================
# 6. Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run gaze estimation and driver-attention monitoring on a video. "
            "The first two seconds are used for automatic front-gaze calibration."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="input video path",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=DEFAULT_OUTPUT_VIDEO_PATH,
        help="annotated output video path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV_PATH,
        help="frame-level result CSV path",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="process without OpenCV preview windows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = args.video.resolve()
    output_video_path = args.output_video.resolve()
    output_csv_path = args.output_csv.resolve()

    if not GAZE_WEIGHT_PATH.is_file():
        raise FileNotFoundError(
            "Gaze weight가 없습니다: {}".format(GAZE_WEIGHT_PATH)
        )

    if not YUNET_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "YuNet 모델이 없습니다: {}".format(YUNET_MODEL_PATH)
        )

    if not video_path.is_file():
        raise FileNotFoundError(
            "입력 동영상이 없습니다: {}".format(video_path)
        )

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("OpenCV:", cv2.__version__)
    print("Device:", device)
    print("Input video:", video_path)
    print("Output video:", output_video_path)
    print("Output CSV:", output_csv_path)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    gaze_model = build_base_model(
        backbone="resnet18",
        pretrained=False,
    )
    state_dict = load_checkpoint(
        weight_path=GAZE_WEIGHT_PATH,
        device=device,
    )
    gaze_model.load_state_dict(state_dict, strict=True)
    gaze_model = gaze_model.to(device)
    gaze_model.eval()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(
            "동영상을 열 수 없습니다: {}".format(video_path)
        )

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = DEFAULT_VIDEO_FPS

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise RuntimeError("입력 동영상 해상도를 읽지 못했습니다.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        source_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "출력 동영상 writer를 열 수 없습니다: {}".format(
                output_video_path
            )
        )

    face_detector = build_yunet_detector(
        model_path=YUNET_MODEL_PATH,
        input_size=(frame_width, frame_height),
        score_threshold=YUNET_SCORE_THRESHOLD,
        nms_threshold=YUNET_NMS_THRESHOLD,
        top_k=YUNET_TOP_K,
    )
    detector_input_size = (frame_width, frame_height)

    calibration_frames = max(
        1,
        int(round(source_fps * AUTO_CALIBRATION_SEC)),
    )
    monitor = DriverMonitor(
        calibration_frames=calibration_frames,
        yaw_threshold_deg=YAW_THRESHOLD_DEG,
        pitch_threshold_deg=PITCH_THRESHOLD_DEG,
        warning_after_sec=WARNING_AFTER_SEC,
        recovery_after_sec=RECOVERY_AFTER_SEC,
    )
    monitor.start_calibration()

    frame_index = 0
    last_face = None
    missed_detection_count = 0
    smoothed_bbox = None
    smoothed_yaw = None
    smoothed_pitch = None
    latest_yaw = None
    latest_pitch = None
    calibration_finalized = False
    processing_start = time.perf_counter()

    csv_fields = [
        "frame_index",
        "timestamp_sec",
        "face_detected",
        "raw_yaw_deg",
        "raw_pitch_deg",
        "smoothed_yaw_deg",
        "smoothed_pitch_deg",
        "front_yaw_deg",
        "front_pitch_deg",
        "delta_yaw_deg",
        "delta_pitch_deg",
        "status",
        "off_road_duration_sec",
        "warning_count",
    ]

    try:
        with output_csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=csv_fields,
            )
            csv_writer.writeheader()

            while True:
                success, frame = capture.read()
                if not success:
                    break

                frame_index += 1
                video_time_sec = (frame_index - 1) / source_fps

                current_height, current_width = frame.shape[:2]
                current_input_size = (
                    current_width,
                    current_height,
                )
                if current_input_size != detector_input_size:
                    face_detector.setInputSize(current_input_size)
                    detector_input_size = current_input_size

                detection_start = time.perf_counter()
                run_detection = (
                    frame_index % DETECT_EVERY_N_FRAMES == 0
                    or last_face is None
                )

                if run_detection:
                    _, faces = face_detector.detect(frame)
                    selected_face = select_driver_face(faces)

                    if selected_face is not None:
                        last_face = selected_face.copy()
                        missed_detection_count = 0
                    else:
                        missed_detection_count += 1
                        if (
                            missed_detection_count
                            > MAX_MISSED_DETECTIONS
                        ):
                            last_face = None
                            smoothed_bbox = None

                detected_face = last_face
                detection_ms = (
                    time.perf_counter() - detection_start
                ) * 1000.0

                gaze_ms = 0.0
                face_result = None
                square_box = None
                raw_yaw = None
                raw_pitch = None

                if detected_face is not None:
                    current_bbox = detected_face[:4]
                    smoothed_bbox = smooth_bbox(
                        previous_bbox=smoothed_bbox,
                        current_bbox=current_bbox,
                        alpha=BBOX_SMOOTHING_ALPHA,
                    )

                    face_crop, square_box = make_square_face_crop(
                        frame=frame,
                        bbox=smoothed_bbox,
                        scale=FACE_SCALE,
                        output_size=GAZE_INPUT_SIZE,
                    )
                    input_tensor = preprocess_face(
                        face_bgr=face_crop,
                        device=device,
                    )

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    gaze_start = time.perf_counter()

                    with torch.inference_mode():
                        prediction = gaze_model(input_tensor)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    gaze_ms = (
                        time.perf_counter() - gaze_start
                    ) * 1000.0

                    prediction = (
                        prediction.squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    if prediction.size != 2:
                        raise RuntimeError(
                            "Gaze 모델 출력이 2개가 아닙니다: {}".format(
                                prediction.shape
                            )
                        )

                    raw_yaw = float(prediction[0])
                    raw_pitch = float(prediction[1])

                    if smoothed_yaw is None:
                        smoothed_yaw = raw_yaw
                        smoothed_pitch = raw_pitch
                    else:
                        alpha = GAZE_SMOOTHING_ALPHA
                        smoothed_yaw = (
                            alpha * raw_yaw
                            + (1.0 - alpha) * smoothed_yaw
                        )
                        smoothed_pitch = (
                            alpha * raw_pitch
                            + (1.0 - alpha) * smoothed_pitch
                        )

                    latest_yaw = smoothed_yaw
                    latest_pitch = smoothed_pitch
                    monitoring_result = monitor.update(
                        yaw=latest_yaw,
                        pitch=latest_pitch,
                        face_detected=True,
                        current_time=video_time_sec,
                    )

                    face_result = draw_gaze_arrow(
                        image=face_crop,
                        yaw=latest_yaw,
                        pitch=latest_pitch,
                        length=85,
                    )
                    draw_yunet_detection(
                        frame=frame,
                        face=detected_face,
                        draw_landmarks=DRAW_LANDMARKS,
                    )

                    x1, y1, x2, y2 = square_box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        "Gaze input ROI",
                        (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                    # 저장되는 원본 크기 결과 영상에도 gaze 화살표 표시
                    draw_gaze_arrow_on_frame(
                        frame=frame,
                        box=square_box,
                        yaw=latest_yaw,
                        pitch=latest_pitch,
                    )

                    # 우측 상단에 224x224 gaze crop 미리보기 삽입
                    paste_face_preview(
                        frame=frame,
                        face_result=face_result,
                    )
                else:
                    latest_yaw = None
                    latest_pitch = None
                    smoothed_yaw = None
                    smoothed_pitch = None
                    monitoring_result = monitor.update(
                        yaw=None,
                        pitch=None,
                        face_detected=False,
                        current_time=video_time_sec,
                    )

                # 정확히 첫 2초 구간이 끝나는 시점에 수집된 표본으로 확정한다.
                if (
                    not calibration_finalized
                    and video_time_sec >= AUTO_CALIBRATION_SEC
                ):
                    calibration_finalized = True

                    if monitor.calibration_mode:
                        success_calibration = monitor.finalize_calibration(
                            min_samples=MIN_CALIBRATION_SAMPLES
                        )
                        if not success_calibration:
                            raise RuntimeError(
                                "첫 {:.1f}초 동안 유효한 얼굴 프레임이 "
                                "{}개 미만입니다.".format(
                                    AUTO_CALIBRATION_SEC,
                                    MIN_CALIBRATION_SAMPLES,
                                )
                            )

                    monitoring_result = monitor.get_result()

                # 저장 전용 프레임을 명시적으로 만든다.
                # 이후 모든 overlay는 output_frame에 그린 뒤 이 프레임만 저장한다.
                output_frame = frame.copy()

                draw_monitoring_overlay(
                    frame=output_frame,
                    result=monitoring_result,
                    fps=source_fps,
                    detection_ms=detection_ms,
                    gaze_ms=gaze_ms,
                    footer_text="Automatic 2-sec calibration | Q: Quit",
                )

                cv2.putText(
                    output_frame,
                    "Video time: {:.2f} sec".format(video_time_sec),
                    (20, 254),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                # gaze 결과를 저장 프레임에 마지막으로 다시 그린다.
                # 이 위치는 writer.write() 직전이므로 결과 영상에서 누락되지 않는다.
                if (
                    detected_face is not None
                    and square_box is not None
                    and latest_yaw is not None
                    and latest_pitch is not None
                ):
                    draw_gaze_arrow_on_frame(
                        frame=output_frame,
                        box=square_box,
                        yaw=latest_yaw,
                        pitch=latest_pitch,
                    )

                    if face_result is not None:
                        paste_face_preview(
                            frame=output_frame,
                            face_result=face_result,
                        )

                cv2.putText(
                    output_frame,
                    "GAZE RESULT",
                    (max(20, frame_width - 230), frame_height - 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                )

                writer.write(output_frame)

                def deg_or_blank(value: Optional[float]):
                    if value is None:
                        return ""
                    return float(np.degrees(value))

                csv_writer.writerow({
                    "frame_index": frame_index,
                    "timestamp_sec": video_time_sec,
                    "face_detected": detected_face is not None,
                    "raw_yaw_deg": deg_or_blank(raw_yaw),
                    "raw_pitch_deg": deg_or_blank(raw_pitch),
                    "smoothed_yaw_deg": deg_or_blank(latest_yaw),
                    "smoothed_pitch_deg": deg_or_blank(latest_pitch),
                    "front_yaw_deg": deg_or_blank(
                        monitoring_result["front_yaw"]
                    ),
                    "front_pitch_deg": deg_or_blank(
                        monitoring_result["front_pitch"]
                    ),
                    "delta_yaw_deg": (
                        "" if monitoring_result["delta_yaw_deg"] is None
                        else monitoring_result["delta_yaw_deg"]
                    ),
                    "delta_pitch_deg": (
                        "" if monitoring_result["delta_pitch_deg"] is None
                        else monitoring_result["delta_pitch_deg"]
                    ),
                    "status": monitoring_result["status"],
                    "off_road_duration_sec": monitoring_result[
                        "off_road_duration"
                    ],
                    "warning_count": monitoring_result[
                        "warning_count"
                    ],
                })

                if not args.no_display:
                    cv2.imshow(
                        "Driver Gaze Monitoring",
                        frame,
                    )
                    if SHOW_FACE_WINDOW and face_result is not None:
                        cv2.imshow(
                            "Gaze Input 224x224",
                            face_result,
                        )
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if total_frames > 0 and frame_index % 100 == 0:
                    print(
                        "Processed {}/{} frames".format(
                            frame_index,
                            total_frames,
                        )
                    )

        elapsed_sec = time.perf_counter() - processing_start
        print("Processing complete")
        print("Frames:", frame_index)
        print("Warnings:", monitor.warning_count)
        print("Elapsed sec: {:.1f}".format(elapsed_sec))
        print("Saved video:", output_video_path)
        print("Saved CSV:", output_csv_path)

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
