from pathlib import Path
import argparse
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
from gaze_monitoring.utils.preprocessing import preprocess_numpy
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_SIZE = 224


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyTorch gaze inference on one image.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="save the result without opening an OpenCV window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = BASE_DIR / "assets" / "images" / "test_img.jpg"
    weight_path = BASE_DIR / "weights" / "gaze_model.onnx"
    output_path = BASE_DIR / "outputs" / "gaze_result_onnx.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"이미지가 없습니다: {image_path}")

    if not weight_path.is_file():
        raise FileNotFoundError(f"가중치가 없습니다: {weight_path}")


    print(f"Image: {image_path}")
    print(f"Weight: {weight_path}")

    input_tensor = preprocess_numpy(image_path)


    model = ort.InferenceSession(str(weight_path),providers=['CPUExecutionProvider'])
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    prediction = model.run([output_name], {input_name: input_tensor})[0][0]
    print("Prediction:", prediction)
    print("Prediction shape:", prediction.shape)
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

    if not args.no_display:
        cv2.imshow("Gaze Estimation", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
