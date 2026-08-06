from pathlib import Path
from typing import Any, Dict

import torch


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
