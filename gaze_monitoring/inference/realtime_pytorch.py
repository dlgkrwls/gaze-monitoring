from pathlib import Path
import os
import time

import cv2
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
from gaze_monitoring.utils.visualization import draw_gaze_arrow
from gaze_monitoring.utils.monitoring import (
    DriverMonitor,
    draw_monitoring_overlay,
)


# ============================================================
# 1. File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

GAZE_WEIGHT_PATH = BASE_DIR / "weights" / "model_epoch_100.pth"

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

# C를 누른 후 정면 기준을 계산할 프레임 수
CALIBRATION_FRAMES = 30

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

def main() -> None:

    # --------------------------------------------------------
    # File validation
    # --------------------------------------------------------

    if not GAZE_WEIGHT_PATH.is_file():
        raise FileNotFoundError(
            "Gaze weight가 없습니다: {}".format(
                GAZE_WEIGHT_PATH
            )
        )

    if not YUNET_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "YuNet 모델이 없습니다: {}".format(
                YUNET_MODEL_PATH
            )
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Python-compatible code: 3.8+")
    print("OpenCV:", cv2.__version__)
    print("Device:", device)
    print("Gaze weight:", GAZE_WEIGHT_PATH)
    print("YuNet model:", YUNET_MODEL_PATH)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------
    # Gaze model
    # --------------------------------------------------------

    gaze_model = build_base_model(
        backbone="resnet18",
        pretrained=False,
    )

    state_dict = load_checkpoint(
        weight_path=GAZE_WEIGHT_PATH,
        device=device,
    )

    gaze_model.load_state_dict(
        state_dict,
        strict=True,
    )

    gaze_model = gaze_model.to(device)
    gaze_model.eval()

    # --------------------------------------------------------
    # Webcam
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        1,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        # CAP_DSHOW가 실패하는 환경을 위한 fallback
        camera.release()

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():
        raise RuntimeError(
            "웹캠을 열 수 없습니다. "
            "CAMERA_INDEX={}".format(
                CAMERA_INDEX
            )
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    # --------------------------------------------------------
    # YuNet
    # --------------------------------------------------------

    face_detector = build_yunet_detector(
        model_path=YUNET_MODEL_PATH,
        input_size=(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
        ),
        score_threshold=YUNET_SCORE_THRESHOLD,
        nms_threshold=YUNET_NMS_THRESHOLD,
        top_k=YUNET_TOP_K,
    )

    detector_input_size = (
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
    )

    # --------------------------------------------------------
    # Driver monitor
    # --------------------------------------------------------

    monitor = DriverMonitor(
        calibration_frames=CALIBRATION_FRAMES,

        yaw_threshold_deg=(
            YAW_THRESHOLD_DEG
        ),

        pitch_threshold_deg=(
            PITCH_THRESHOLD_DEG
        ),

        warning_after_sec=(
            WARNING_AFTER_SEC
        ),

        recovery_after_sec=(
            RECOVERY_AFTER_SEC
        ),
    )

    # --------------------------------------------------------
    # Runtime variables
    # --------------------------------------------------------

    frame_index = 0

    last_face = None
    missed_detection_count = 0

    smoothed_bbox = None

    smoothed_yaw = None
    smoothed_pitch = None

    latest_yaw = None
    latest_pitch = None

    previous_frame_time = (
        time.perf_counter()
    )

    try:
        while True:

            success, frame = camera.read()

            if not success:
                print(
                    "웹캠 프레임을 읽지 못했습니다."
                )
                break

            frame_index += 1

            frame_height, frame_width = (
                frame.shape[:2]
            )

            current_input_size = (
                frame_width,
                frame_height,
            )

            # 해상도가 변경된 경우에만 갱신
            if (
                current_input_size
                != detector_input_size
            ):
                face_detector.setInputSize(
                    current_input_size
                )

                detector_input_size = (
                    current_input_size
                )

            # ------------------------------------------------
            # YuNet face detection
            # ------------------------------------------------

            detection_start = (
                time.perf_counter()
            )

            run_detection = (
                frame_index
                % DETECT_EVERY_N_FRAMES
                == 0

                or

                last_face is None
            )

            if run_detection:
                _, faces = face_detector.detect(
                    frame
                )

                selected_face = (
                    select_driver_face(faces)
                )

                if selected_face is not None:
                    last_face = (
                        selected_face.copy()
                    )

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
                time.perf_counter()
                - detection_start
            ) * 1000.0

            gaze_ms = 0.0

            face_result = None
            square_box = None

            # ------------------------------------------------
            # Face crop + gaze inference
            # ------------------------------------------------

            if detected_face is not None:

                current_bbox = (
                    detected_face[:4]
                )

                smoothed_bbox = smooth_bbox(
                    previous_bbox=smoothed_bbox,
                    current_bbox=current_bbox,
                    alpha=BBOX_SMOOTHING_ALPHA,
                )

                # 중요한 점:
                # bbox나 landmark를 그리기 전에
                # 원본 frame에서 먼저 crop해야 한다.
                face_crop, square_box = (
                    make_square_face_crop(
                        frame=frame,
                        bbox=smoothed_bbox,
                        scale=FACE_SCALE,
                        output_size=GAZE_INPUT_SIZE,
                    )
                )

                input_tensor = preprocess_face(
                    face_bgr=face_crop,
                    device=device,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()

                gaze_start = (
                    time.perf_counter()
                )

                with torch.inference_mode():
                    prediction = gaze_model(
                        input_tensor
                    )

                if device.type == "cuda":
                    torch.cuda.synchronize()

                gaze_ms = (
                    time.perf_counter()
                    - gaze_start
                ) * 1000.0

                prediction = (
                    prediction
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )

                if prediction.size != 2:
                    raise RuntimeError(
                        "Gaze 모델 출력이 2개가 아닙니다: "
                        "{}".format(
                            prediction.shape
                        )
                    )

                raw_yaw = float(
                    prediction[0]
                )

                raw_pitch = float(
                    prediction[1]
                )

                # --------------------------------------------
                # Gaze EMA smoothing
                # --------------------------------------------

                if smoothed_yaw is None:
                    smoothed_yaw = raw_yaw
                    smoothed_pitch = raw_pitch

                else:
                    alpha = (
                        GAZE_SMOOTHING_ALPHA
                    )

                    smoothed_yaw = (
                        alpha * raw_yaw
                        + (1.0 - alpha)
                        * smoothed_yaw
                    )

                    smoothed_pitch = (
                        alpha * raw_pitch
                        + (1.0 - alpha)
                        * smoothed_pitch
                    )

                latest_yaw = smoothed_yaw
                latest_pitch = smoothed_pitch

                monitoring_result = monitor.update(
                    yaw=latest_yaw,
                    pitch=latest_pitch,
                    face_detected=True,
                )

                face_result = draw_gaze_arrow(
                    image=face_crop,
                    yaw=latest_yaw,
                    pitch=latest_pitch,
                    length=85,
                )

                # crop이 끝난 뒤 전체 화면에 시각화
                draw_yunet_detection(
                    frame=frame,
                    face=detected_face,
                    draw_landmarks=DRAW_LANDMARKS,
                )

                x1, y1, x2, y2 = square_box

                # Gaze 모델 입력 ROI: 초록색
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
                    (
                        x1,
                        max(22, y1 - 8),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

            # ------------------------------------------------
            # Face not detected
            # ------------------------------------------------

            else:
                latest_yaw = None
                latest_pitch = None

                smoothed_yaw = None
                smoothed_pitch = None

                monitoring_result = monitor.update(
                    yaw=None,
                    pitch=None,
                    face_detected=False,
                )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_frame_time = (
                time.perf_counter()
            )

            elapsed = (
                current_frame_time
                - previous_frame_time
            )

            previous_frame_time = (
                current_frame_time
            )

            if elapsed > 0:
                fps = 1.0 / elapsed
            else:
                fps = 0.0

            # ------------------------------------------------
            # Draw monitoring information
            # ------------------------------------------------

            draw_monitoring_overlay(
                frame=frame,
                result=monitoring_result,
                fps=fps,
                detection_ms=detection_ms,
                gaze_ms=gaze_ms,
                footer_text="C: Calibrate | R: Reset count | Q: Quit",
            )

            # ------------------------------------------------
            # Show windows
            # ------------------------------------------------

            cv2.imshow(
                "Driver Gaze Monitoring",
                frame,
            )

            if (
                SHOW_FACE_WINDOW
                and face_result is not None
            ):
                cv2.imshow(
                    "Gaze Input 224x224",
                    face_result,
                )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            # C: 정면 캘리브레이션
            if key == ord("c"):

                if (
                    latest_yaw is None
                    or latest_pitch is None
                ):
                    print(
                        "얼굴이 검출된 상태에서 "
                        "C를 누르세요."
                    )

                else:
                    monitor.start_calibration()

            # R: Warning Count 초기화
            elif key == ord("r"):
                monitor.reset_warning_count()

            # Q: 종료
            elif key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
