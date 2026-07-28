from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import time

import cv2
import numpy as np
import torch

from model import build_base_model


# ============================================================
# 1. File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

GAZE_WEIGHT_PATH = BASE_DIR / "model_epoch_100.pth"

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

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

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
# 6. Checkpoint loading
# ============================================================

def load_checkpoint(
    weight_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    다양한 PyTorch checkpoint 저장 형식을 처리한다.
    """

    try:
        checkpoint = torch.load(
            weight_path,
            map_location=device,
            weights_only=True,
        )

    # 구형 PyTorch에서는 weights_only 인자가 없을 수 있음
    except TypeError:
        checkpoint = torch.load(
            weight_path,
            map_location=device,
        )

    state_dict = checkpoint

    if isinstance(checkpoint, dict):
        for checkpoint_key in (
            "state_dict",
            "model_state_dict",
            "model",
        ):
            if (
                checkpoint_key in checkpoint
                and isinstance(
                    checkpoint[checkpoint_key],
                    dict,
                )
            ):
                state_dict = checkpoint[checkpoint_key]
                break

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint에서 state_dict를 찾지 못했습니다. "
            "현재 형식: {}".format(type(state_dict))
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        # DataParallel로 학습했을 때 붙는 module. 제거
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    return cleaned_state_dict


# ============================================================
# 7. YuNet functions
# ============================================================

def build_yunet_detector(
    model_path: Path,
    input_size: Tuple[int, int],
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
        YUNET_SCORE_THRESHOLD,
        YUNET_NMS_THRESHOLD,
        YUNET_TOP_K,
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

    if not DRAW_LANDMARKS:
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


# ============================================================
# 8. Face crop functions
# ============================================================

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
) -> Tuple[
    np.ndarray,
    Tuple[int, int, int, int],
]:
    """
    YuNet bbox를 중심 기준 정사각형 crop으로 변환한다.

    Gaze360 전처리 방식:
    1. bbox 중심 계산
    2. max(width, height) 기준 정사각형
    3. 224x224 resize
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
            GAZE_INPUT_SIZE,
            GAZE_INPUT_SIZE,
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


# ============================================================
# 9. Gaze preprocessing
# ============================================================

def preprocess_face(
    face_bgr: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    OpenCV BGR
    → RGB
    → float32 0~1
    → ImageNet normalization
    → HWC에서 CHW
    → batch 추가
    """

    # BGR → RGB
    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # uint8 0~255 → float32 0~1
    face_rgb = (
        face_rgb.astype(np.float32)
        / 255.0
    )

    # ImageNet normalization
    face_rgb = (
        face_rgb - IMAGENET_MEAN
    ) / IMAGENET_STD

    # HWC → CHW
    face_chw = face_rgb.transpose(
        2,
        0,
        1,
    )

    face_chw = np.ascontiguousarray(
        face_chw
    )

    face_tensor = torch.from_numpy(
        face_chw
    )

    # CHW → BCHW
    face_tensor = face_tensor.unsqueeze(0)

    return face_tensor.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )


# ============================================================
# 10. Gaze visualization
# ============================================================

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


# ============================================================
# 11. Driver monitoring
# ============================================================

class DriverMonitor:
    def __init__(
        self,
        calibration_frames: int,
        yaw_threshold_deg: float,
        pitch_threshold_deg: float,
        warning_after_sec: float,
        recovery_after_sec: float,
    ):
        self.calibration_frames = calibration_frames

        self.yaw_threshold_deg = (
            yaw_threshold_deg
        )

        self.pitch_threshold_deg = (
            pitch_threshold_deg
        )

        self.warning_after_sec = (
            warning_after_sec
        )

        self.recovery_after_sec = (
            recovery_after_sec
        )

        self.front_yaw = None
        self.front_pitch = None

        self.calibration_mode = False

        self.calibration_yaws = []
        self.calibration_pitches = []

        self.status = "UNCALIBRATED"

        self.delta_yaw_deg = None
        self.delta_pitch_deg = None

        self.off_road_start_time = None
        self.off_road_duration = 0.0

        self.recovery_start_time = None

        self.warning_count = 0
        self.warning_issued = False

    def start_calibration(self) -> None:
        """
        C키를 누르면 호출된다.
        이후 calibration_frames만큼의 gaze 평균을
        정면 기준으로 저장한다.
        """

        self.calibration_mode = True

        self.calibration_yaws = []
        self.calibration_pitches = []

        self.front_yaw = None
        self.front_pitch = None

        self.delta_yaw_deg = None
        self.delta_pitch_deg = None

        self.off_road_start_time = None
        self.off_road_duration = 0.0

        self.recovery_start_time = None
        self.warning_issued = False

        self.status = "CALIBRATING"

        print(
            "Calibration started. "
            "Look straight ahead."
        )

    def reset_warning_count(self) -> None:
        self.warning_count = 0

        print("Warning count reset.")

    def issue_warning(
        self,
        reason: str,
    ) -> None:
        """
        하나의 이탈 구간에서 Count는 한 번만 증가한다.
        """

        if self.warning_issued:
            return

        self.warning_count += 1
        self.warning_issued = True

        print(
            "Warning #{}: {}".format(
                self.warning_count,
                reason,
            )
        )

    def reset_attention_event(self) -> None:
        self.off_road_start_time = None
        self.off_road_duration = 0.0

        self.recovery_start_time = None
        self.warning_issued = False

    def update(
        self,
        yaw: Optional[float],
        pitch: Optional[float],
        face_detected: bool,
    ) -> Dict[str, Any]:
        current_time = time.perf_counter()

        # ----------------------------------------------------
        # Face lost
        # ----------------------------------------------------

        if (
            not face_detected
            or yaw is None
            or pitch is None
        ):
            self.delta_yaw_deg = None
            self.delta_pitch_deg = None

            self.recovery_start_time = None

            if self.calibration_mode:
                current_count = len(
                    self.calibration_yaws
                )

                self.status = (
                    "CALIBRATION PAUSED "
                    "{}/{}".format(
                        current_count,
                        self.calibration_frames,
                    )
                )

                return self.get_result()

            if (
                self.front_yaw is None
                or self.front_pitch is None
            ):
                self.status = "FACE LOST"
                self.off_road_duration = 0.0

                return self.get_result()

            # 얼굴 검출 실패도 정면 이탈 이벤트로 처리
            if self.off_road_start_time is None:
                self.off_road_start_time = (
                    current_time
                )

            self.off_road_duration = (
                current_time
                - self.off_road_start_time
            )

            if (
                self.off_road_duration
                >= self.warning_after_sec
            ):
                self.status = "WARNING: FACE LOST"

                self.issue_warning(
                    "face lost"
                )

            else:
                self.status = "FACE LOST"

            return self.get_result()

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        if self.calibration_mode:
            self.calibration_yaws.append(yaw)
            self.calibration_pitches.append(pitch)

            current_count = len(
                self.calibration_yaws
            )

            self.status = "CALIBRATING {}/{}".format(
                current_count,
                self.calibration_frames,
            )

            if (
                current_count
                >= self.calibration_frames
            ):
                self.front_yaw = float(
                    np.mean(
                        self.calibration_yaws
                    )
                )

                self.front_pitch = float(
                    np.mean(
                        self.calibration_pitches
                    )
                )

                self.calibration_mode = False

                self.calibration_yaws = []
                self.calibration_pitches = []

                self.reset_attention_event()

                self.delta_yaw_deg = 0.0
                self.delta_pitch_deg = 0.0

                self.status = "ATTENTIVE"

                print(
                    "Calibration complete: "
                    "yaw={:.2f} deg, "
                    "pitch={:.2f} deg".format(
                        np.degrees(
                            self.front_yaw
                        ),
                        np.degrees(
                            self.front_pitch
                        ),
                    )
                )

            return self.get_result()

        # ----------------------------------------------------
        # Not calibrated
        # ----------------------------------------------------

        if (
            self.front_yaw is None
            or self.front_pitch is None
        ):
            self.status = "UNCALIBRATED"

            self.delta_yaw_deg = None
            self.delta_pitch_deg = None

            return self.get_result()

        # ----------------------------------------------------
        # Compare current gaze with calibrated front
        # ----------------------------------------------------

        self.delta_yaw_deg = float(
            np.degrees(
                yaw - self.front_yaw
            )
        )

        self.delta_pitch_deg = float(
            np.degrees(
                pitch - self.front_pitch
            )
        )

        looking_forward = (
            abs(self.delta_yaw_deg)
            <= self.yaw_threshold_deg

            and

            abs(self.delta_pitch_deg)
            <= self.pitch_threshold_deg
        )

        # ----------------------------------------------------
        # Looking forward
        # ----------------------------------------------------

        if looking_forward:

            if self.recovery_start_time is None:
                self.recovery_start_time = (
                    current_time
                )

            recovery_duration = (
                current_time
                - self.recovery_start_time
            )

            if (
                recovery_duration
                >= self.recovery_after_sec
            ):
                self.reset_attention_event()
                self.status = "ATTENTIVE"

            else:
                # 이전에 이탈 중이었다면 짧은 복구 상태 표시
                if self.off_road_start_time is not None:
                    self.status = "RECOVERING"
                else:
                    self.status = "ATTENTIVE"

            return self.get_result()

        # ----------------------------------------------------
        # Looking away
        # ----------------------------------------------------

        self.recovery_start_time = None

        if self.off_road_start_time is None:
            self.off_road_start_time = current_time

        self.off_road_duration = (
            current_time
            - self.off_road_start_time
        )

        if (
            self.off_road_duration
            >= self.warning_after_sec
        ):
            self.status = "WARNING"

            self.issue_warning(
                "driver looking away"
            )

        else:
            self.status = "DISTRACTED"

        return self.get_result()

    def get_result(self) -> Dict[str, Any]:
        return {
            "status": self.status,

            "front_yaw": self.front_yaw,
            "front_pitch": self.front_pitch,

            "delta_yaw_deg": (
                self.delta_yaw_deg
            ),

            "delta_pitch_deg": (
                self.delta_pitch_deg
            ),

            "off_road_duration": (
                self.off_road_duration
            ),

            "warning_count": (
                self.warning_count
            ),

            "calibration_mode": (
                self.calibration_mode
            ),

            "calibration_count": len(
                self.calibration_yaws
            ),
        }


# ============================================================
# 12. Monitoring overlay
# ============================================================

def get_status_color(
    status: str,
) -> Tuple[int, int, int]:

    if status == "ATTENTIVE":
        return (0, 255, 0)

    if status == "RECOVERING":
        return (0, 255, 255)

    if status == "DISTRACTED":
        return (0, 165, 255)

    if status.startswith("WARNING"):
        return (0, 0, 255)

    if status.startswith("CALIBRAT"):
        return (255, 255, 0)

    if status == "FACE LOST":
        return (0, 0, 255)

    return (255, 255, 255)


def draw_monitoring_overlay(
    frame: np.ndarray,
    result: Dict[str, Any],
    fps: float,
    detection_ms: float,
    gaze_ms: float,
) -> None:

    status = result["status"]

    status_color = get_status_color(
        status
    )

    cv2.putText(
        frame,
        "FPS: {:.1f}".format(fps),
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "YuNet: {:.1f} ms".format(
            detection_ms
        ),
        (20, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "Gaze: {:.1f} ms".format(
            gaze_ms
        ),
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "Status: {}".format(status),
        (20, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        status_color,
        2,
    )

    cv2.putText(
        frame,
        "Off-road: {:.1f} sec".format(
            result["off_road_duration"]
        ),
        (20, 146),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        status_color,
        2,
    )

    cv2.putText(
        frame,
        "Warnings: {}".format(
            result["warning_count"]
        ),
        (20, 174),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        status_color,
        2,
    )

    delta_yaw = result["delta_yaw_deg"]
    delta_pitch = result["delta_pitch_deg"]

    if (
        delta_yaw is not None
        and delta_pitch is not None
    ):
        cv2.putText(
            frame,
            "Delta yaw/pitch: "
            "{:.1f}, {:.1f}".format(
                delta_yaw,
                delta_pitch,
            ),
            (20, 202),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    front_yaw = result["front_yaw"]
    front_pitch = result["front_pitch"]

    if (
        front_yaw is not None
        and front_pitch is not None
    ):
        cv2.putText(
            frame,
            "Front yaw/pitch: "
            "{:.1f}, {:.1f}".format(
                np.degrees(front_yaw),
                np.degrees(front_pitch),
            ),
            (20, 228),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )

    frame_height = frame.shape[0]

    cv2.putText(
        frame,
        "C: Calibrate | R: Reset count | Q: Quit",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        2,
    )


# ============================================================
# 13. Main
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
        CAMERA_INDEX,
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
