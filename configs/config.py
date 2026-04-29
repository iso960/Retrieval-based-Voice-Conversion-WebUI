import argparse
import os
import sys
import json
import shutil
import tempfile
from multiprocessing import cpu_count

import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init

        ipex_init()
except Exception:  # pylint: disable=broad-exception-caught
    pass
import logging

logger = logging.getLogger(__name__)


class RealtimeConfig:
    """실시간 API/GUI 세션 설정 파일(config.json)을 관리하는 클래스.

    path 파라미터로 api(configs/config.json)와
    gui(configs/inuse/config.json) 두 경로를 모두 수용한다.
    gui 전용 키(sg_hostapi, sg_wasapi_exclusive, sr_type)는
    DEFAULT에서 None으로 정의되며, api에서 로드 시 None이면 무시된다.
    """

    DEFAULT: dict = {
        # 공통
        "pth_path": "",
        "index_path": "",
        "sg_input_device": "",
        "sg_output_device": "",
        "threhold": -60,
        "pitch": 0,
        "formant": 0.0,
        "index_rate": 0.0,
        "rms_mix_rate": 0.0,
        "block_time": 0.25,
        "crossfade_length": 0.05,
        "extra_time": 2.5,
        "n_cpu": 4,
        "use_jit": False,
        "use_pv": False,
        "f0method": "fcpe",
        # gui 전용 (api에서는 None으로 유지)
        "sg_hostapi": None,
        "sg_wasapi_exclusive": None,
        "sr_type": None,
    }

    def __init__(self, path: str = "configs/config.json"):
        self.path = path
        self.data: dict = self._load()

    def _load(self) -> dict:
        """파일이 없거나 파싱 실패 시 DEFAULT의 복사본을 반환한다."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # DEFAULT 키 중 누락된 것은 기본값으로 보완
            return {**self.DEFAULT, **loaded}
        except Exception:
            return dict(self.DEFAULT)

    def save(self, data: dict) -> None:
        """data를 원자적으로 저장한다 (tmp 파일 → rename)."""
        # formant 키 누락 방지: DEFAULT 값으로 보완
        merged = {**self.DEFAULT, **data}
        dir_ = os.path.dirname(self.path) or "."
        os.makedirs(dir_, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            os.unlink(tmp)
            raise
        self.data = merged

    def update(self, **kwargs) -> None:
        """지정한 키만 업데이트하고 저장한다."""
        self.save({**self.data, **kwargs})


class TrainConfig:
    """v1/v2 학습 하이퍼파라미터 JSON을 관리하는 클래스.

    argparse 의존성 없음 — 학습 subprocess에서도 안전하게 임포트 가능.
    configs/inuse/ 가 없으면 configs/ 에서 복사 후 로드한다.
    """

    VERSION_CONFIG_LIST = [
        "v1/32k.json",
        "v1/40k.json",
        "v1/48k.json",
        "v2/48k.json",
        "v2/32k.json",
    ]

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self.json_config: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        d = {}
        for rel_path in self.VERSION_CONFIG_LIST:
            inuse = os.path.join(self.config_dir, "inuse", rel_path)
            src = os.path.join(self.config_dir, rel_path)
            if not os.path.exists(inuse):
                os.makedirs(os.path.dirname(inuse), exist_ok=True)
                shutil.copy(src, inuse)
            with open(inuse, "r", encoding="utf-8") as f:
                d[rel_path] = json.load(f)
        return d

    def get(self, version: str, sr: str) -> dict:
        """예: get('v2', '40k') → json_config['v2/40k.json']"""
        key = f"{version}/{sr}.json"
        if key not in self.json_config:
            raise KeyError(
                f"TrainConfig: '{key}' not found. Available: {list(self.json_config)}"
            )
        return self.json_config[key]

    def set_fp32(self) -> None:
        """모든 버전 config의 fp16_run을 False로 설정하고 inuse/ 파일에 반영한다."""
        for rel_path in self.VERSION_CONFIG_LIST:
            self.json_config[rel_path]["train"]["fp16_run"] = False
            inuse = os.path.join(self.config_dir, "inuse", rel_path)
            with open(inuse, "r", encoding="utf-8") as f:
                content = f.read().replace("true", "false")
            with open(inuse, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("TrainConfig: overwrite fp16_run=false in %s", rel_path)


# 하위 호환: Config.use_fp32_config()에서 참조하는 전역 목록
version_config_list = TrainConfig.VERSION_CONFIG_LIST


class RuntimeConfig:
    """GPU/디바이스 감지, fp16/fp32 결정, CLI 인자 파싱을 담당하는 클래스.

    argv=None  → sys.argv 파싱 (일반 진입점)
    argv=[]    → argparse 스킵, CLI 속성은 모두 기본값 (학습 subprocess용)

    train_config 를 넘기면 fp32 강제 시 train_config.set_fp32()를 호출해
    inuse/ JSON 파일도 함께 갱신한다. None이면 파일 갱신 없이 is_half만 변경.
    """

    def __init__(
        self,
        argv: list[str] | None = None,
        train_config: TrainConfig | None = None,
    ):
        self._train_config = train_config

        # 디바이스 속성
        self.device: str = "cuda:0"
        self.is_half: bool = True
        self.use_jit: bool = False
        self.n_cpu: int = 0
        self.gpu_name: str | None = None
        self.gpu_mem: int | None = None
        self.instead: str = ""
        self.preprocess_per: float = 3.7

        # CLI 속성 (argv=[] 시 기본값 유지)
        self.python_cmd: str = sys.executable or "python"
        self.listen_port: int = 7865
        self.iscolab: bool = False
        self.noparallel: bool = False
        self.noautoopen: bool = False
        self.dml: bool = False

        if argv is None or argv != []:
            self._parse_args(argv)
        self._detect_device()

        # x_pad 등 패딩 값
        self.x_pad, self.x_query, self.x_center, self.x_max = self._pad_config()

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _parse_args(self, argv: list[str] | None) -> None:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865)
        parser.add_argument("--pycmd", type=str, default=exe)
        parser.add_argument("--colab", action="store_true")
        parser.add_argument("--noparallel", action="store_true")
        parser.add_argument("--noautoopen", action="store_true")
        parser.add_argument("--dml", action="store_true")
        opts = parser.parse_args(argv)
        self.python_cmd = opts.pycmd
        self.listen_port = opts.port if 0 <= opts.port <= 65535 else 7865
        self.iscolab = opts.colab
        self.noparallel = opts.noparallel
        self.noautoopen = opts.noautoopen
        self.dml = opts.dml

    def _force_fp32(self) -> None:
        """is_half 를 False 로 내리고, train_config 가 있으면 파일도 갱신한다."""
        self.is_half = False
        if self._train_config is not None:
            self._train_config.set_fp32()
        self.preprocess_per = 3.0

    def _detect_device(self) -> None:
        if torch.cuda.is_available():
            if self.has_xpu():
                self.device = self.instead = "xpu:0"
                self.is_half = True
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info("Found GPU %s, force to fp32", self.gpu_name)
                self._force_fp32()
            else:
                logger.info("Found GPU %s", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                self.preprocess_per = 3.0
        elif self.has_mps():
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "mps"
            self._force_fp32()
        else:
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "cpu"
            self._force_fp32()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.dml:
            logger.info("Use DirectML instead")
            if not os.path.exists(
                r"runtime\Lib\site-packages\onnxruntime\capi\DirectML.dll"
            ):
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime",
                        r"runtime\Lib\site-packages\onnxruntime-cuda",
                    )
                except Exception:
                    pass
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime-dml",
                        r"runtime\Lib\site-packages\onnxruntime",
                    )
                except Exception:
                    pass
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
            self.is_half = False
        else:
            if self.instead:
                logger.info("Use %s instead", self.instead)
            if not os.path.exists(
                r"runtime\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
            ):
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime",
                        r"runtime\Lib\site-packages\onnxruntime-dml",
                    )
                except Exception:
                    pass
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime-cuda",
                        r"runtime\Lib\site-packages\onnxruntime",
                    )
                except Exception:
                    pass

        logger.info("Half-precision: %s, device: %s", self.is_half, self.device)

    def _pad_config(self) -> tuple[int, int, int, int]:
        if self.is_half:
            x_pad, x_query, x_center, x_max = 3, 10, 60, 65
        else:
            x_pad, x_query, x_center, x_max = 1, 6, 38, 41
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = 1, 5, 30, 32
        return x_pad, x_query, x_center, x_max

    # ------------------------------------------------------------------
    # 정적 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    @staticmethod
    def has_xpu() -> bool:
        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance

    wrapper.instance = None
    return wrapper


@singleton_variable
class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.use_jit = False
        self.n_cpu = 0
        self.gpu_name = None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.dml,
        ) = self.arg_parse()
        self.instead = ""
        self.preprocess_per = 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        d = {}
        for config_file in version_config_list:
            p = f"configs/inuse/{config_file}"
            if not os.path.exists(p):
                shutil.copy(f"configs/{config_file}", p)
            with open(f"configs/inuse/{config_file}", "r") as f:
                d[config_file] = json.load(f)
        return d

    @staticmethod
    def arg_parse() -> tuple:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        parser.add_argument(
            "--dml",
            action="store_true",
            help="torch_dml",
        )
        cmd_opts = parser.parse_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.dml,
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    @staticmethod
    def has_xpu() -> bool:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        else:
            return False

    def use_fp32_config(self):
        for config_file in version_config_list:
            self.json_config[config_file]["train"]["fp16_run"] = False
            with open(f"configs/inuse/{config_file}", "r") as f:
                strr = f.read().replace("true", "false")
            with open(f"configs/inuse/{config_file}", "w") as f:
                f.write(strr)
            logger.info("overwrite " + config_file)
        self.preprocess_per = 3.0
        logger.info("overwrite preprocess_per to %d" % (self.preprocess_per))

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            if self.has_xpu():
                self.device = self.instead = "xpu:0"
                self.is_half = True
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info("Found GPU %s, force to fp32", self.gpu_name)
                self.is_half = False
                self.use_fp32_config()
            else:
                logger.info("Found GPU %s", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                self.preprocess_per = 3.0
        elif self.has_mps():
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "mps"
            self.is_half = False
            self.use_fp32_config()
        else:
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "cpu"
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        if self.dml:
            logger.info("Use DirectML instead")
            if (
                os.path.exists(
                    "runtime\Lib\site-packages\onnxruntime\capi\DirectML.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime",
                        "runtime\Lib\site-packages\onnxruntime-cuda",
                    )
                except Exception:
                    pass
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime-dml",
                        "runtime\Lib\site-packages\onnxruntime",
                    )
                except Exception:
                    pass
            # if self.device != "cpu":
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
            self.is_half = False
        else:
            if self.instead:
                logger.info(f"Use {self.instead} instead")
            if (
                os.path.exists(
                    "runtime\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime",
                        "runtime\Lib\site-packages\onnxruntime-dml",
                    )
                except Exception:
                    pass
                try:
                    os.rename(
                        "runtime\Lib\site-packages\onnxruntime-cuda",
                        "runtime\Lib\site-packages\onnxruntime",
                    )
                except Exception:
                    pass
        logger.info(
            "Half-precision floating-point: %s, device: %s"
            % (self.is_half, self.device)
        )
        return x_pad, x_query, x_center, x_max
