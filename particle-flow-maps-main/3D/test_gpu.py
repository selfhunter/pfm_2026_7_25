# import torch

# # 获取 GPU 总内存（以 GB 为单位）
# if torch.cuda.is_available():
#     total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
#     # 预留 1GB 给系统
#     safe_mem = max(1.0, total_gpu_mem - 1.0)
#     print(f"可用 GPU 内存: {total_gpu_mem:.2f} GB，使用: {safe_mem:.2f} GB")
# else:
#     safe_mem = 4.0  # 默认值
#     print("未检测到 GPU，使用默认内存值")

# # 初始化 Taichi
# ti.init(arch=ti.cuda, device_memory_GB=safe_mem, debug=False)


# import torch
# if torch.cuda.is_available():
#     torch.backends.cudnn.enabled = True
#     a = torch.randn(1, 3, 224, 224).cuda()
#     b = torch.randn(1, 3, 224, 224).cuda()
#     c = torch.nn.functional.conv2d(a, b)
#     print("cuDNN可能已正确安装并能正常使用")
# else:
#     print("未检测到可用的GPU，无法测试cuDNN")

import sys
import torch
from torch.backends import cudnn
print(sys.version)
print(torch.__version__)
print(torch.cuda.is_available())
print(cudnn.is_available())