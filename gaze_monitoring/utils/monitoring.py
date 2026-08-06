import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


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

    def finalize_calibration(
        self,
        min_samples: int = 1,
    ) -> bool:
        """수집된 gaze로 정면 기준을 확정한다."""
        sample_count = len(self.calibration_yaws)

        if sample_count < min_samples:
            self.calibration_mode = False
            self.status = "CALIBRATION FAILED"
            self.front_yaw = None
            self.front_pitch = None
            return False

        self.front_yaw = float(np.mean(self.calibration_yaws))
        self.front_pitch = float(np.mean(self.calibration_pitches))

        self.calibration_mode = False
        self.calibration_yaws = []
        self.calibration_pitches = []

        self.reset_attention_event()
        self.delta_yaw_deg = 0.0
        self.delta_pitch_deg = 0.0
        self.status = "ATTENTIVE"

        print(
            "Calibration complete: yaw={:.2f} deg, pitch={:.2f} deg".format(
                np.degrees(self.front_yaw),
                np.degrees(self.front_pitch),
            )
        )
        return True

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
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        # 웹캠에서는 perf_counter를 사용할 수 있고, 저장 영상 분석에서는
        # 원본 영상 타임스탬프를 전달해 실제 영상 시간 기준으로 판정한다.
        if current_time is None:
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

            if current_count >= self.calibration_frames:
                self.finalize_calibration(min_samples=1)

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
    footer_text: str = "C: Calibrate | R: Reset count | Q: Quit",
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
        footer_text,
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        2,
    )
