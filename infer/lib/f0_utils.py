"""rtrvc.py와 pipeline.py의 공통 F0 처리 로직"""

from __future__ import annotations

import numpy as np
import torch
import torchcrepe


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

    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
        f0_mel_max - f0_mel_min
    ) + 1
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


def extract_f0_crepe(
    audio: torch.Tensor,
    device,
    f0_min: float = 50,
    f0_max: float = 1100,
) -> torch.Tensor:
    """crepe로 F0를 추출한다.

    DML 폴백, 피치시프트, 양자화는 호출부 책임.

    Args:
        audio:  (1, N) float tensor
        device: torch device
        f0_min: 최저 피치 (Hz)
        f0_max: 최고 피치 (Hz)

    Returns:
        f0: raw pitch tensor (피치시프트/양자화 전)
    """
    f0, pd = torchcrepe.predict(
        audio,
        16000,
        160,
        f0_min,
        f0_max,
        "full",
        batch_size=512,
        device=device,
        return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    f0[pd < 0.1] = 0
    return f0
