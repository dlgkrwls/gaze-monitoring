from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

import cv2

from gaze_monitoring.model import build_base_model
from gaze_monitoring.utils.checkpoint import load_checkpoint
from gaze_monitoring.utils.preprocessing import preprocess_face

BASE_DIR = Path(__file__).resolve().parents[2]


if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[2])
    )


GAZE_WEIGHT_PATH = BASE_DIR / "weights" / "model_epoch_100.pth"
ONNX_MODEL_PATH = BASE_DIR / "weights" / "gaze_model.onnx"

# 비교용 테스트 이미지. local_artifacts/는 개인 데이터를 담는 gitignore된
# 로컬 전용 디렉터리이며, 저장소에는 포함되지 않는다.
TEST_IMAGE_PATH = BASE_DIR / "local_artifacts" / "images" / "8798.jpg"


#  파일 존재 여부 확인

if not GAZE_WEIGHT_PATH.is_file():
    raise FileNotFoundError(
        "PyTorch 가중치 파일을 찾지 못했습니다: "
        f"{GAZE_WEIGHT_PATH}"
    )

if not ONNX_MODEL_PATH.is_file():
    raise FileNotFoundError(
        "ONNX 모델 파일을 찾지 못했습니다: "
        f"{ONNX_MODEL_PATH}"
    )


# PyTorch 모델 불러오기


# 이번 비교는 CPU에서 진행
# PyTorch와 ONNX Runtime을 같은 CPU 환경에서 비교하면
# CUDA 연산 차이를 제외하고 모델 변환 오차만 확인하기 쉬움
device = torch.device("cpu")

torch_model = build_base_model(
    backbone="resnet18",
    pretrained=False,
)

# checkpoint 형식을 안전하게 처리하여 state_dict만 추출
state_dict = load_checkpoint(
    weight_path=GAZE_WEIGHT_PATH,
    device=device,
)

torch_model.load_state_dict(
    state_dict,
    strict=True,
)

torch_model = torch_model.to(device)

# BatchNorm과 Dropout을 추론 모드로 변경
torch_model.eval()


# ============================================================
# 5. ONNX Runtime 세션 생성
# ============================================================

# ONNX 모델은 PyTorch처럼 직접 호출하지 않고
# InferenceSession을 통해 실행한다.
onnx_session = ort.InferenceSession(
    str(ONNX_MODEL_PATH),
    providers=["CPUExecutionProvider"],
)


# ============================================================
# 6. ONNX 입출력 명세 확인
# ============================================================

onnx_input = onnx_session.get_inputs()[0]
onnx_output = onnx_session.get_outputs()[0]

input_name = onnx_input.name
output_name = onnx_output.name

print("ONNX input name:", input_name)
print("ONNX input shape:", onnx_input.shape)
print("ONNX input type:", onnx_input.type)

print("ONNX output name:", output_name)
print("ONNX output shape:", onnx_output.shape)
print("ONNX output type:", onnx_output.type)


# ============================================================
# 7. 동일한 dummy input 생성
# ============================================================

# 모델 입력 형식:
# [batch, channel, height, width]
# [1, 3, 224, 224]
#
# PyTorch 모델과 ONNX 모델에 반드시 동일한 입력을 넣어야 한다.
torch.manual_seed(42)

dummy_input = torch.randn(
    1,
    3,
    224,
    224,
    dtype=torch.float32,
    device=device,
)

print("Dummy input shape:", dummy_input.shape)
print("Dummy input dtype:", dummy_input.dtype)

if not TEST_IMAGE_PATH.is_file():
    raise FileNotFoundError(
        "비교용 테스트 이미지가 없습니다. local_artifacts/images/에 "
        f"직접 준비해야 합니다: {TEST_IMAGE_PATH}"
    )

img = cv2.imread(str(TEST_IMAGE_PATH))
input_tensor = preprocess_face(img, device=device)


# ============================================================
# 8. PyTorch 추론
# ============================================================

with torch.inference_mode():
    torch_prediction = torch_model(
        input_tensor
    )

# ONNX Runtime 출력과 비교하기 위해 NumPy 배열로 변환
torch_prediction_np = (
    torch_prediction
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# 9. ONNX Runtime 추론
# ============================================================

# ONNX Runtime은 기본적으로 NumPy 배열을 입력으로 받는다.
dummy_input_np = (
    input_tensor
    .detach()
    .cpu()
    .numpy()
)

# session.run()은 출력들을 리스트 형태로 반환한다.
# 출력이 하나이므로 [0]으로 첫 번째 출력을 가져온다.
onnx_prediction = onnx_session.run(
    [output_name],
    {
        input_name: dummy_input_np
    },
)[0]


# ============================================================
# 10. 출력 비교
# ============================================================

absolute_difference = np.abs(
    torch_prediction_np
    - onnx_prediction
)

max_absolute_difference = float(
    absolute_difference.max()
)

mean_absolute_difference = float(
    absolute_difference.mean()
)

# atol:
# 절대 오차 허용값
#
# rtol:
# 값의 크기에 비례한 상대 오차 허용값
predictions_close = np.allclose(
    torch_prediction_np,
    onnx_prediction,
    atol=1e-6,
    rtol=1e-5,
)


# ============================================================
# 11. 결과 출력
# ============================================================

print()
print("PyTorch prediction:")
print(torch_prediction_np)

print()
print("ONNX prediction:")
print(onnx_prediction)

print()
print("Absolute difference:")
print(absolute_difference)

print()
print(
    "Max absolute difference:",
    max_absolute_difference,
)

print(
    "Mean absolute difference:",
    mean_absolute_difference,
)

print(
    "Predictions close:",
    predictions_close,
)
