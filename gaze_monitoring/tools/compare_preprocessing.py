from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = BASE_DIR / "assets" / "images"
ONNX_MODEL_PATH = BASE_DIR / "weights" / "gaze_model.onnx"

IMAGE_SIZE = 224

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def preprocess_torchvision(
    image_path: Path,
) -> np.ndarray:
    """
    기존 PyTorch 기준 전처리.

    PIL RGB
    -> Resize
    -> ToTensor
    -> ImageNet Normalize
    -> batch dimension

    반환:
        shape: (1, 3, 224, 224)
        dtype: float32
    """

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN.tolist(),
                std=IMAGENET_STD.tolist(),
            ),
        ]
    )

    pil_image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = preprocess(
        pil_image
    ).unsqueeze(0)

    return input_tensor.numpy()


def preprocess_numpy(
    image_path: Path,
) -> np.ndarray:
    """
    배포 후보 OpenCV + NumPy 전처리.

    OpenCV BGR
    -> Resize
    -> RGB
    -> float32
    -> 0~1 scaling
    -> ImageNet Normalize
    -> HWC to CHW
    -> batch dimension

    반환:
        shape: (1, 3, 224, 224)
        dtype: float32
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"OpenCV로 이미지를 읽지 못했습니다: "
            f"{image_path}"
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

    image = image.astype(
        np.float32
    )

    image = image / 255.0

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

    # C/C++ 포팅 시에도 연속 메모리 형태가 중요함
    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    return image


def get_image_paths() -> list[Path]:
    """
    assets/images 폴더의 지원 이미지 파일을 모두 반환한다.
    """

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"이미지 폴더가 없습니다: "
            f"{IMAGE_DIR}"
        )

    image_paths = [
        path
        for path in IMAGE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    image_paths.sort()

    return image_paths


def main() -> None:

    if not ONNX_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"ONNX 모델이 없습니다: "
            f"{ONNX_MODEL_PATH}"
        )

    image_paths = get_image_paths()

    if len(image_paths) == 0:
        raise RuntimeError(
            f"비교할 이미지가 없습니다: "
            f"{IMAGE_DIR}"
        )

    print(
        f"Number of images: "
        f"{len(image_paths)}"
    )

    print(
        f"ONNX model: "
        f"{ONNX_MODEL_PATH}"
    )

    print()

    # ONNX Runtime 세션은
    # 이미지마다 다시 만들 필요가 없으므로 한 번만 생성
    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    output_name = (
        session
        .get_outputs()[0]
        .name
    )

    # 전체 이미지의 결과를 저장
    input_max_diffs = []
    input_mean_diffs = []

    yaw_diffs_rad = []
    pitch_diffs_rad = []

    yaw_diffs_deg = []
    pitch_diffs_deg = []

    failed_images = []

    print(
        "=" * 80
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:

            torchvision_input = (
                preprocess_torchvision(
                    image_path
                )
            )

            numpy_input = (
                preprocess_numpy(
                    image_path
                )
            )

            # ------------------------------------------------
            # 전처리 tensor 차이
            # ------------------------------------------------

            input_difference = np.abs(
                torchvision_input
                - numpy_input
            )

            input_max_diff = float(
                input_difference.max()
            )

            input_mean_diff = float(
                input_difference.mean()
            )

            input_max_diffs.append(
                input_max_diff
            )

            input_mean_diffs.append(
                input_mean_diff
            )

            # ------------------------------------------------
            # Torchvision 전처리 -> ONNX 추론
            # ------------------------------------------------

            torchvision_output = (
                session.run(
                    [output_name],
                    {
                        input_name:
                        torchvision_input
                    },
                )[0][0]
            )

            # ------------------------------------------------
            # OpenCV/NumPy 전처리 -> ONNX 추론
            # ------------------------------------------------

            numpy_output = (
                session.run(
                    [output_name],
                    {
                        input_name:
                        numpy_input
                    },
                )[0][0]
            )

            # ------------------------------------------------
            # 모델 출력 차이
            # ------------------------------------------------

            output_difference = np.abs(
                torchvision_output
                - numpy_output
            )

            yaw_diff_rad = float(
                output_difference[0]
            )

            pitch_diff_rad = float(
                output_difference[1]
            )

            yaw_diff_deg = float(
                np.degrees(
                    yaw_diff_rad
                )
            )

            pitch_diff_deg = float(
                np.degrees(
                    pitch_diff_rad
                )
            )

            yaw_diffs_rad.append(
                yaw_diff_rad
            )

            pitch_diffs_rad.append(
                pitch_diff_rad
            )

            yaw_diffs_deg.append(
                yaw_diff_deg
            )

            pitch_diffs_deg.append(
                pitch_diff_deg
            )

            print(
                "  Input max diff : "
                f"{input_max_diff:.8f}"
            )

            print(
                "  Input mean diff: "
                f"{input_mean_diff:.8f}"
            )

            print(
                "  Torchvision output "
                f"(yaw, pitch): "
                f"{torchvision_output}"
            )

            print(
                "  NumPy output       "
                f"(yaw, pitch): "
                f"{numpy_output}"
            )

            print(
                "  Yaw diff   : "
                f"{yaw_diff_rad:.8f} rad / "
                f"{yaw_diff_deg:.4f} deg"
            )

            print(
                "  Pitch diff : "
                f"{pitch_diff_rad:.8f} rad / "
                f"{pitch_diff_deg:.4f} deg"
            )

        except Exception as error:

            failed_images.append(
                image_path.name
            )

            print(
                f"  Failed: {error}"
            )

        print(
            "-" * 80
        )

    # ========================================================
    # 전체 통계
    # ========================================================

    if len(yaw_diffs_deg) == 0:
        raise RuntimeError(
            "성공적으로 처리된 이미지가 없습니다."
        )

    print()
    print(
        "=" * 80
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "Successfully processed:",
        len(yaw_diffs_deg),
    )

    print(
        "Failed:",
        len(failed_images),
    )

    print()

    print(
        "Input preprocessing difference"
    )

    print(
        "  Mean of max diff:",
        np.mean(input_max_diffs),
    )

    print(
        "  Worst max diff:",
        np.max(input_max_diffs),
    )

    print(
        "  Mean diff:",
        np.mean(input_mean_diffs),
    )

    print()

    print(
        "Yaw difference"
    )

    print(
        "  Mean:",
        f"{np.mean(yaw_diffs_rad):.8f} rad / "
        f"{np.mean(yaw_diffs_deg):.4f} deg"
    )

    print(
        "  Max :",
        f"{np.max(yaw_diffs_rad):.8f} rad / "
        f"{np.max(yaw_diffs_deg):.4f} deg"
    )

    print()

    print(
        "Pitch difference"
    )

    print(
        "  Mean:",
        f"{np.mean(pitch_diffs_rad):.8f} rad / "
        f"{np.mean(pitch_diffs_deg):.4f} deg"
    )

    print(
        "  Max :",
        f"{np.max(pitch_diffs_rad):.8f} rad / "
        f"{np.max(pitch_diffs_deg):.4f} deg"
    )

    print()

    # yaw/pitch를 합쳐서 가장 큰 각도 차이도 확인
    all_angle_diffs_deg = (
        yaw_diffs_deg
        + pitch_diffs_deg
    )

    print(
        "Overall angle component difference"
    )

    print(
        "  Mean:",
        f"{np.mean(all_angle_diffs_deg):.4f} deg"
    )

    print(
        "  Max :",
        f"{np.max(all_angle_diffs_deg):.4f} deg"
    )

    if failed_images:
        print()
        print(
            "Failed image list:"
        )

        for filename in failed_images:
            print(
                f"  {filename}"
            )


if __name__ == "__main__":
    main()