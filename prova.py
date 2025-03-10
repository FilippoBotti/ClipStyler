import torch
##import numpy
import mamba_ssm
import selective_scan_cuda
##print("Numpy version:", numpy.version.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CudNN version:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
print("CUDA Compatibility:", torch.cuda.get_arch_list())
