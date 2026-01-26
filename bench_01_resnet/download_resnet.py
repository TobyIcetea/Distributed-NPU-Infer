import os
import torch
import torchvision.models as models

# 1. 创建 /models 目录（如果不存在）
output_dir = "/ascend-models/bench_01_resnet/"
os.makedirs(output_dir, exist_ok=True)

# 2. 加载预训练的 ResNet-50 模型（处于 eval 模式）
model = models.resnet50(pretrained=True)
model.eval()

# 3. 准备一个示例输入（用于 tracing）
example_input = torch.randn(1, 3, 224, 224)

# 4. 将模型转换为 TorchScript（使用 tracing）
traced_script_module = torch.jit.trace(model, example_input)

# 5. 保存为 .pt 文件
output_path = os.path.join(output_dir, "resnet50_traced.pt")
traced_script_module.save(output_path)

print(f"TorchScript 模型已保存到: {output_path}")