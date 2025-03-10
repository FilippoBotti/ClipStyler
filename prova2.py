import torch

# Test semplice di allocazione su GPU
try:
    x = torch.randn(3, 3).cuda()
    print("CUDA funziona correttamente!")
except Exception as e:
    print(f"Errore CUDA: {e}")
print(torch.cuda.get_arch_list())