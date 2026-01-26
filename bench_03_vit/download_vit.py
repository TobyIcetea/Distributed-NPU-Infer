import os
import torch
import timm

# 1. 创建输出目录（如果不存在）
output_dir = "/ascend-models/bench_03_vit/"
os.makedirs(output_dir, exist_ok=True)

# 2. 加载预训练的 ViT-Base 模型（timm 中对应名称通常是 'vit_base_patch16_224'）
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# 3. 准备示例输入张量（ViT-Base 输入尺寸为 224x224，batch size = 1）
example_input = torch.randn(1, 3, 224, 224)

# 4. 使用 tracing 将模型转换为 TorchScript
traced_script_module = torch.jit.trace(model, example_input)

# 5. 保存为 .pt 文件
output_path = os.path.join(output_dir, "vit_base_patch16_224_traced.pt")
traced_script_module.save(output_path)

print(f"TorchScript 模型已保存到: {output_path}")