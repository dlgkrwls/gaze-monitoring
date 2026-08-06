from pathlib import Path

import torch

from model import build_base_model
from gaze_monitoring.utils.checkpoint import load_checkpoint


BASE_DIR = Path(__file__).resolve().parents[2]

GAZE_WEIGHT_PATH = BASE_DIR / "weights" / "model_epoch_100.pth"
ONNX_OUTPUT_PATH = BASE_DIR / "weights" / "gaze_model.onnx"


def main() -> None:
    model = build_base_model(backbone="resnet18", pretrained=False)
    state_dict = load_checkpoint(
        GAZE_WEIGHT_PATH,
        torch.device("cpu"),
    )

    model.load_state_dict(state_dict)
    dummy_input = torch.randn(1, 3, 224, 224)
    model.eval()

    with torch.inference_mode():
        predict = model(dummy_input)

    print("Predict type:", type(predict))
    print("Predict shape:", predict.shape)
    print("Predict:", predict)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_OUTPUT_PATH),
        input_names=["input"],
        output_names=["output"],
    )

    print("Saved ONNX model:", ONNX_OUTPUT_PATH)


if __name__ == "__main__":
    main()
