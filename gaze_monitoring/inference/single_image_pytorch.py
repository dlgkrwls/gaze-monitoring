from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from model import build_base_model
from gaze_monitoring.utils.checkpoint import load_checkpoint


BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def draw_gaze_arrow(
    image: np.ndarray,
    yaw: float,
    pitch: float,
    length: int = 90,
) -> np.ndarray:
    """
    yaw, pitch로부터 gaze 방향을 계산해 이미지 중앙에 화살표를 그린다.

    GazeTo2d의 역변환 관계:
        gx = cos(pitch) * sin(yaw)
        gy = sin(pitch)
        gz = -cos(pitch) * cos(yaw)

    OpenCV 이미지 좌표는 아래쪽이 +y이므로 gy 부호를 반대로 사용한다.
    """

    output = image.copy()

    height, width = output.shape[:2]

    start_x = width // 2
    start_y = height // 2

    gaze_x = np.cos(pitch) * np.sin(yaw)
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
        f"Yaw: {yaw_degree:.2f} deg",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        output,
        f"Pitch: {pitch_degree:.2f} deg",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    return output


def main() -> None:
    image_path = BASE_DIR / "assets" / "images" / "test_img.jpg"
    weight_path = BASE_DIR / "weights" / "model_epoch_100.pth"
    output_path = BASE_DIR / "outputs" / "gaze_result.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"이미지가 없습니다: {image_path}")

    if not weight_path.is_file():
        raise FileNotFoundError(f"가중치가 없습니다: {weight_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print(f"Weight: {weight_path}")

    preprocess = build_preprocess()

    # 모델 입력용 RGB PIL 이미지
    pil_image = Image.open(image_path).convert("RGB")

    input_tensor = preprocess(pil_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    model = build_base_model(
        backbone="resnet18",
        pretrained=False,
    )

    state_dict = load_checkpoint(
        weight_path=weight_path,
        device=device,
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        prediction = model(input_tensor)

    prediction = prediction.squeeze(0).cpu().numpy()

    yaw = float(prediction[0])
    pitch = float(prediction[1])

    print(f"Input shape: {tuple(input_tensor.shape)}")
    print(f"Yaw: {yaw:.6f} rad / {np.degrees(yaw):.2f} deg")
    print(f"Pitch: {pitch:.6f} rad / {np.degrees(pitch):.2f} deg")

    # 시각화용 BGR 이미지
    visualization_image = cv2.imread(str(image_path))

    if visualization_image is None:
        raise RuntimeError(f"OpenCV로 이미지를 읽지 못했습니다: {image_path}")

    visualization_image = cv2.resize(
        visualization_image,
        (IMAGE_SIZE, IMAGE_SIZE),
    )

    result = draw_gaze_arrow(
        image=visualization_image,
        yaw=-yaw,
        pitch=pitch,
        length=90,
    )

    cv2.imwrite(str(output_path), result)

    print(f"Saved result: {output_path}")

    cv2.imshow("Gaze Estimation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
