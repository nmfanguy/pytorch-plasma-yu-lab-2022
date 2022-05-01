from asyncio import subprocess
import pyarrow.plasma as plasma
import time
import torch
import subprocess
from ..modules import Module
from itertools import chain
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
    _get_devices_properties
)

class PlasmaParallel(Module):
    r"""DataParallel, but with Plasma.
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(PlasmaParallel, self).__init__()
        torch._C._log_api_usage_once("torch.nn.parallel.PlasmaParallel")

        # CUDA is not available on the cluster, so for now we avoid these lines
        """
        device_type = _get_available_device_type()
        if device_type is None:
            print("WE HIT NONE")
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = _get_all_device_indices()

        if output_device is None:
            output_device = device_ids[0]

       self.device_ids = [_get_device_index(x, True) for x in device_ids]
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)
        """

        self.dim = dim
        self.module = module
        
        self.server = subprocess.Popen(["/home/fanguy/.local/bin/plasma_store", "-m", "1000000000", "-s", "/tmp/plasma"])

        time.sleep(1)
        self.client = plasma.connect("/tmp/plasma")

        obj_id = self.client.put("hello, plasma!")
        obj = self.client.get(obj_id)

        print(f"    put and retrieved {obj}")

    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("PlasmaParallel.forward"):
            if not self.device_ids:
                return self.module(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError("module must have its parameters and buffers "
                                       "on device {} (device_ids[0]) but found one of "
                                       "them on device: {}".format(self.src_device_obj, t.device))
            
            # distribute the input and kwargs to all of the client machines -- i.e., store them in Plasma
            inputs_id = self.client.put(inputs)
            kwargs_id = self.client.put(kwargs)

            if not inputs and not kwargs:
                inputs = ((),)
                kwargs = ({},)

            if len(self.device_ids) == 1:
                return self.module(*inputs[0], **kwargs[0])

            # must make all the workers have access to the shared data
            outputs = self.parallel_apply(replicas, inputs, kwargs)

    def __del__(self):
        self.server.terminate()