from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, ClassVar, TypeVar
from botsort._botsort import configs as configs

GMC_METHOD = TypeVar("GMC_METHOD", bound=IntEnum) # pyright: ignore[reportInvalidTypeForm]

@dataclass
class BotSortConfig:
    reid_enabled: bool = field(init=False, default=False)
    gmc_enabled: bool = field(init=False, default=False)
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 30
    match_thresh: float = 0.7
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    gmc_method_name: str = field(init=False, default="sparseOptFlow")
    frame_rate: int = 30
    lambda_: float = 0.985

    def setGMCMethod(self, gmcMethod: 'MethodParams'):
        self.gmc_method_name = gmcMethod.name

    def c_init(self) -> dict[str, Any]:
        _c_init = configs.trackerParams(*vars(self).values())
        return _c_init


@dataclass
class MethodParams:
    name: ClassVar[str]
    method: ClassVar[GMC_METHOD]
    CMETHOD: ClassVar[Callable]
    
    c_init: dict[str, Any] = field(init=False)

    def __post_init__(self):
        self.c_init = configs.GMC_Config(
            self.method,
            self.CMETHOD(**vars(self))
        )
    
@dataclass
class ORBConfig(MethodParams):
    name: ClassVar[str] = 'orb'
    method: ClassVar[GMC_METHOD] = configs.GMC_METHOD.ORB
    CMETHOD: ClassVar[Callable] = configs.ORBConfig

    downscale: float = 0.2
    inlier_ratio: float = 0.5
    ransac_conf: float = 0.99
    max_iterations: int = 500


@dataclass
class ECCConfig(MethodParams):
    name: ClassVar[str] = 'ecc'
    method: ClassVar[GMC_METHOD] = configs.GMC_METHOD.ECC
    CMETHOD: ClassVar[Callable] = configs.ECCConfig

    downscale:float = 5.0
    max_iterations:int = 100
    termination_eps:float = 1e-6


@dataclass
class SparseOptFlowConfig(MethodParams):
    name: ClassVar[str] = 'sparseOptFlow'
    method: ClassVar[GMC_METHOD] = configs.GMC_METHOD.SparseOptFlow
    CMETHOD: ClassVar[Callable] = configs.SparseOptFlowConfig

    max_corners: int = 1000
    block_size: int = 3
    ransac_max_iters: int = 500
    quality_level: float = 0.01
    k: float = 0.04
    min_distance: float = 1.0
    downscale: float = 2.0
    inlier_ratio: float = 0.5
    ransac_conf: float = 0.99
    use_harris_detector: bool = False


@dataclass
class OpenCVVideoStabGMCConfig(MethodParams):
    name: ClassVar[str] = "OpenCV_VideoStab"
    method: ClassVar[GMC_METHOD] = configs.GMC_METHOD.OpenCV_VideoStab
    CMETHOD: ClassVar[Callable] = configs.OpenCVVideoStabGMCConfig

    downscale: float = 2.0
    num_features: float = 4000
    detection_masking: bool = True

