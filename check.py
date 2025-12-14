import torch
print(f"torch版本：{torch.__version__}")  # 2.4.1
print("驱动支持的CUDA最大版本：11.7")  # 由驱动版本457.85决定
print("PyTorch内置CUDA版本：", torch.version.cuda)  # 应输出11.7
print("CUDA可用：", torch.cuda.is_available())  # 应输出True
if torch.cuda.is_available():
    # 显卡算力：GTX 16xx为7.5，RTX 30xx为8.6
    capability = torch.cuda.get_device_capability(0)
    print(f"显卡算力：{capability}")
    # Tensor Core要求算力≥7.0（Volta）且是RTX系列，GTX 16xx算力7.5但无Tensor Core
    has_tensor_core = (capability[0] >= 7) and ("RTX" in torch.cuda.get_device_name(0))
    print(f"支持Tensor Core：{has_tensor_core}")  # 你的输出应为False