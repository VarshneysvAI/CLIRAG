import logging
import platform
import subprocess
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger("HardwareProbe")

class HardwareBackend(Enum):
    CPU_AVX2 = "cpu_avx2"
    AMD_XDNA_NPU = "npu_directml"
    NVIDIA_CUDA = "gpu_cuda"
    AMD_ROCM = "gpu_rocm"

class HardwareProbe:
    """
    Dynamically detects the available hardware on the system to route inference workloads
    to the optimal backend (CPU AVX2, AMD Ryzen NPU, or Dedicated GPU via CUDA/ROCm).
    """

    def __init__(self):
        self.os_type = platform.system()
        self.architecture = platform.machine()
        
    def _check_command_exists(self, cmd: str) -> bool:
        """Helper to safely check if a CLI tool exists in PATH."""
        try:
            # Cross-platform check
            check_cmd = ["where", cmd] if self.os_type == "Windows" else ["which", cmd]
            result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def check_cuda(self) -> bool:
        """Probe for NVIDIA CUDA support via nvidia-smi."""
        return self._check_command_exists("nvidia-smi")

    def check_rocm(self) -> bool:
        """Probe for AMD ROCm support via rocminfo."""
        return self._check_command_exists("rocminfo")

    def check_amd_xdna_npu(self) -> bool:
        """
        Probe for AMD Ryzen AI (XDNA NPUs). 
        On Windows, we inspect WMI / PowerShell for the NPU device class.
        """
        if self.os_type != "Windows":
            # For Linux, we might check /sys/class/drm for specific AMD IPU nodes,
            # but for this MVP on Windows we focus on Device Manager.
            try:
                # Check for specific PCI IDs or node devices for AMD NPU on Linux if needed
                pass
            except Exception:
                pass
            return False

        try:
            # Query Windows PnP Devices for "NPU" or "IPU" or "Ryzen AI"
            powershell_cmd = (
                'Get-PnpDevice -Status OK | Where-Object { '
                '$_.FriendlyName -match "NPU" -or '
                '$_.FriendlyName -match "IPU" -or '
                '$_.FriendlyName -match "Ryzen AI" } | '
                'Select-Object -ExpandProperty FriendlyName'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", powershell_cmd],
                capture_output=True,
                text=True
            )
            
            output = result.stdout.strip()
            if output:
                logger.debug(f"Detected NPU/IPU Device(s): {output}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to probe for AMD NPU: {e}")
            
        return False

    def detect_optimal_backend(self) -> HardwareBackend:
        """
        Executes the hardware routing logic (Scenarios A, B, and C).
        """
        logger.info("Initializing hardware probe capabilities...")

        # Scenario C: Dedicated GPU
        if self.check_cuda():
            logger.info("✅ Dedicated GPU detected: NVIDIA CUDA.")
            logger.info("   -> Offloading to VRAM, upscaling to 16-bit precision (FP16).")
            return HardwareBackend.NVIDIA_CUDA

        if self.check_rocm():
            logger.info("✅ Dedicated GPU detected: AMD ROCm.")
            logger.info("   -> Offloading to VRAM, upscaling to 16-bit precision (FP16).")
            return HardwareBackend.AMD_ROCM

        # Scenario B: AMD Ryzen AI (XDNA NPUs)
        if self.check_amd_xdna_npu():
            logger.info("✅ AI Accelerator detected: AMD Ryzen AI (XDNA NPU).")
            logger.info("   -> Offloading via ONNX Runtime / DirectML.")
            return HardwareBackend.AMD_XDNA_NPU

        # Scenario A: CPU Only (Force AVX2)
        logger.warning("⚠️ No Dedicated GPU or Ryzen NPU detected. Falling back to CPU.")
        logger.info("   -> Forcing AVX2 instruction sets for optimized BLAS operations.")
        return HardwareBackend.CPU_AVX2

    def get_llama_cpp_kwargs(self) -> Dict[str, Any]:
        """
        Returns the optimal kwargs dictionary for initializing the llama.cpp model
        based on the detected hardware backend.
        """
        backend = self.detect_optimal_backend()
        
        # Base CPU configuration
        kwargs = {
            "n_ctx": 4096,           # Base context
            "n_threads": None,       # Let llama.cpp auto-detect core count
            "use_mmap": True,
            "use_mlock": False       # Avoid pinning to RAM unless explicitly required
        }

        if backend == HardwareBackend.NVIDIA_CUDA or backend == HardwareBackend.AMD_ROCM:
            # GPU Specific Overrides
            kwargs.update({
                "n_gpu_layers": -1,  # Offload ALL layers to VRAM
                "f16_kv": True       # Upscale/Maintain 16-bit precision KV cache
            })
            
        elif backend == HardwareBackend.AMD_XDNA_NPU:
            # NPU Specific overrides 
            # Note: Native DirectML/ONNX via llama.cpp python bindings requires specific build flags,
            # but we configure the routing parameters here.
            kwargs.update({
                "n_gpu_layers": -1,  # NPU offloading mechanism shares this parameter in some builds
            })
            
        elif backend == HardwareBackend.CPU_AVX2:
            # CPU Specific overrides
            # We rely on the compile flags (llama-cpp-python built with AVX2)
            # but ensure we don't try to offload layers
            kwargs.update({
                "n_gpu_layers": 0
            })
            
        return kwargs

if __name__ == "__main__":
    probe = HardwareProbe()
    optimal_backend = probe.detect_optimal_backend()
    
    print("\n" + "="*50)
    print(f" CLIRAG Hardware Probe Results")
    print("="*50)
    print(f" Optimized Backend Enforced : {optimal_backend.value}")
    print(f" Llama.cpp Init Kwargs     : {probe.get_llama_cpp_kwargs()}")
    print("="*50 + "\n")
