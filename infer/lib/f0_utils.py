"""rtrvc.py와 pipeline.py의 공통 F0 처리 로직"""

from __future__ import annotations

import numpy as np
import torch


def mel_quantize(
    f0: np.ndarray | torch.Tensor,
    f0_mel_min: float,
    f0_mel_max: float,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """F0 (Hz)를 mel 스케일로 변환하고 [1, 255] 범위로 양자화한다.

    numpy array와 torch Tensor 모두 처리하며, 입력과 같은 타입으로 반환한다.

    Returns:
        f0_coarse: 정수 양자화된 mel-scale F0 (pitch index)
        f0_mel:    실수 mel-scale F0 ([1, 255] 클램프 후)
    """
    use_torch = torch.is_tensor(f0)

    if use_torch:
        f0_mel = 1127 * torch.log(1 + f0 / 700)
    else:
        f0_mel = 1127 * np.log(1 + f0 / 700)

    f0_mel[f0_mel > 0] = (
        (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    )
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255

    if use_torch:
        f0_coarse = torch.round(f0_mel).long()
    else:
        f0_coarse = np.rint(f0_mel).astype(np.int32)

    return f0_coarse, f0_mel


def extract_f0_rmvpe(
    audio: np.ndarray,
    model_path: str,
    use_jit: bool = False,
    is_half: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """RMVPE로 F0를 추출한다.

    rtrvc.py (use_jit=True 가능)와 pipeline.py (use_jit=False 고정)의
    공통 로직을 통합한다. 모델 캐시는 호출부에서 관리한다.

    Args:
        audio:      16kHz mono numpy array
        model_path: rmvpe.pt 경로
        use_jit:    JIT 모델 사용 여부 (rtrvc용, pipeline은 False)
        is_half:    fp16 여부
        device:     torch device 문자열

    Returns:
        f0: numpy array (Hz, 미가공)
    """
    from infer.lib.rmvpe import RMVPE

    model = RMVPE(model_path, is_half=is_half, device=device, use_jit=use_jit)
    f0 = model.infer_from_audio(audio, thred=0.03)

    # DirectML은 추론 후 모델을 즉시 해제 (pipeline.py L156-159 동일)
    if "privateuseone" in str(device):
        del model.model
        del model

    return f0
