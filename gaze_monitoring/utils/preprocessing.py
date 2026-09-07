import cv2
import numpy as np
# import torch
from pathlib import Path
from typing import Optional
IMAGE_SIZE = 224
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



def preprocess_numpy(
    image_path: Optional[Path] = None,
    face_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:

    if image_path is not None:
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"OpenCV로 이미지를 읽지 못했습니다: {image_path}"
            )

    elif face_bgr is not None:
        image = face_bgr

    else:
        raise ValueError(
            "image_path 또는 face_bgr 중 하나는 입력해야 합니다."
        )

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = image.astype(np.float32) / 255.0

    image = (
        image - IMAGENET_MEAN
    ) / IMAGENET_STD

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    image = np.expand_dims(
        image,
        axis=0,
    )

    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    return image