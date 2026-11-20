"""Optional GPU telemetry with NVIDIA NVML and OpenGL memory fallbacks."""

from __future__ import annotations

import os
import time
from typing import Any


class GpuTelemetry:
    """Sample system GPU memory/utilisation where the driver exposes it.

    NVML reports device-wide memory and utilisation; graphics-process memory is
    reported separately when the driver supports the process query. Without
    NVML, GL_NVX_gpu_memory_info provides device-wide total/free VRAM only.
    """

    def __init__(self, gl, renderer_name: str, sample_interval: float = 0.75) -> None:
        self.gl = gl
        self.renderer_name = renderer_name
        self.sample_interval = sample_interval
        self.last_sample_time = 0.0
        self.values: dict[str, Any] = {
            "backend": "unavailable",
            "gpu_name": renderer_name,
            "total_bytes": None,
            "used_bytes": None,
            "free_bytes": None,
            "process_bytes": None,
            "utilization_percent": None,
            "memory_utilization_percent": None,
        }
        self._pynvml = None
        self._device = None
        self._nvml_initialised = False
        self._nvx_supported = False
        self._nvx_total_bytes = None
        self._detect_nvx()
        self._initialise_nvml()

    def _detect_nvx(self) -> None:
        gl = self.gl
        try:
            count = int(gl.glGetIntegerv(gl.GL_NUM_EXTENSIONS))
            extensions = {
                gl.glGetStringi(gl.GL_EXTENSIONS, index).decode("ascii", "replace")
                for index in range(count)
            }
            self._nvx_supported = "GL_NVX_gpu_memory_info" in extensions
            if self._nvx_supported:
                total_kb = int(gl.glGetIntegerv(0x9048))  # TOTAL_AVAILABLE_MEMORY_NVX
                self._nvx_total_bytes = total_kb * 1024
        except Exception:
            self._nvx_supported = False
            self._nvx_total_bytes = None

    def _initialise_nvml(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialised = True
            device_count = pynvml.nvmlDeviceGetCount()
            candidates = []
            for index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                candidates.append((str(name), handle))
            renderer = self.renderer_name.lower()
            selected = next(
                ((name, handle) for name, handle in candidates
                 if name.lower() in renderer or renderer in name.lower()),
                None,
            )
            if selected is None and len(candidates) == 1:
                selected = candidates[0]
            if selected is None:
                return
            self._pynvml = pynvml
            self._device = selected[1]
            self.values["backend"] = "NVML"
            self.values["gpu_name"] = selected[0]
        except Exception:
            self._pynvml = None
            self._device = None

    def sample(self, force: bool = False) -> dict[str, Any]:
        now = time.perf_counter()
        if not force and now - self.last_sample_time < self.sample_interval:
            return self.values
        self.last_sample_time = now

        if self._pynvml is not None and self._device is not None:
            try:
                memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._device)
                self.values.update({
                    "backend": "NVML",
                    "total_bytes": int(memory.total),
                    "used_bytes": int(memory.used),
                    "free_bytes": int(memory.free),
                })
                try:
                    utilisation = self._pynvml.nvmlDeviceGetUtilizationRates(self._device)
                    self.values["utilization_percent"] = int(utilisation.gpu)
                    self.values["memory_utilization_percent"] = int(utilisation.memory)
                except Exception:
                    self.values["utilization_percent"] = None
                    self.values["memory_utilization_percent"] = None
                self.values["process_bytes"] = self._process_memory_bytes()
                return self.values
            except Exception:
                # The NVIDIA driver may be absent or unavailable after initialisation.
                self.values["backend"] = "NVML unavailable"

        if self._nvx_supported and self._nvx_total_bytes:
            try:
                available_kb = int(self.gl.glGetIntegerv(0x9049))  # CURRENT_AVAILABLE_VIDMEM_NVX
                free_bytes = available_kb * 1024
                self.values.update({
                    "backend": "GL_NVX_gpu_memory_info",
                    "total_bytes": self._nvx_total_bytes,
                    "used_bytes": max(0, self._nvx_total_bytes - free_bytes),
                    "free_bytes": free_bytes,
                })
            except Exception:
                self.values["backend"] = "VRAM query unavailable"
        return self.values

    def _process_memory_bytes(self) -> int | None:
        nvml = self._pynvml
        if nvml is None or self._device is None:
            return None
        process_queries = (
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        )
        processes = None
        for query_name in process_queries:
            query = getattr(nvml, query_name, None)
            if query is None:
                continue
            try:
                processes = query(self._device)
                break
            except Exception:
                continue
        if processes is None:
            return None
        total = 0
        found = False
        for process in processes:
            if int(getattr(process, "pid", -1)) != os.getpid():
                continue
            used = getattr(process, "usedGpuMemory", None)
            if used is None:
                continue
            try:
                used = int(used)
            except (TypeError, ValueError):
                continue
            if used >= 0 and used < (1 << 63):
                total += used
                found = True
        return total if found else None

    def close(self) -> None:
        if self._nvml_initialised and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialised = False
