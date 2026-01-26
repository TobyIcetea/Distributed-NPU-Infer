import torch
import torch_npu
import torch.onnx

# 1. 加载 TorchScript 模型
from config import MODEL_PATH, ONNX_MODEL_PATH
model = torch.jit.load(MODEL_PATH)
model.eval()  # 设置为评估模式

# 2. 创建一个示例输入（根据你的模型输入尺寸调整）
dummy_input = torch.randn(1, 3, 224, 224)

# 3. 导出为 ONNX
torch.onnx.export(
    model,
    dummy_input,
    ONNX_MODEL_PATH,
    export_params=True,        # 存储训练好的参数权重
    opset_version=11,          # ONNX 算子集版本（建议 >=11）
    do_constant_folding=True,  # 是否执行常量折叠优化
    input_names=["input"],     # 输入名
    output_names=["output"],   # 输出名
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }  # 可选：支持动态 batch size
)

print(f"ONNX 模型已保存到: {ONNX_MODEL_PATH}")
