import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


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
