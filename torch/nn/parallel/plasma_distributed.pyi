from typing import Any, Optional
from .common_types import _devices_t, _device_t
from ..modules import Module
from ... import device, Tensor

class PlasmaParallel(Module):
    def __init__(self, module: Module, device_ids: Optional[_devices_t] = ..., output_device: Optional[_device_t] = ...,
                 dim: int = ...) -> None: ...