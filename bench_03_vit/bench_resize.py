import torch
import torch_npu
import time
from config import DEVICE_TYPE, MODEL_PATH, BATCH_SIZE as CONFIG_BATCH_SIZE, WARMUP_BATCH_COUNT
import torch.nn.functional as F

# === 配置区域 ===
DEVICE = DEVICE_TYPE  # 使用配置文件中的设备配置
BATCH_SIZE = 1   # 推理 BatchSize
WARM_UP = 10     # 预热次数
TEST_STEPS = 50  # 正式测试次数

# 我们的 ViT 模型只吃 224*224
VIT_NATIVE_SIZE = 224

def get_model():
    print(f"正在加载模型: {MODEL_PATH}")
    # 加载配置文件中指定的模型
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model = model.to(DEVICE)
    model.eval()
    return model

def preprocess_for_vit(input_tensor):
    """
    专门为 ViT 准备的预处理函数。
    无论输入多大，强制缩放到 224x224，否则 JIT 模型会崩溃。
    """
    # 如果尺寸已经是 224x224，直接返回，省去计算量
    if input_tensor.shape[2] == VIT_NATIVE_SIZE and input_tensor.shape[3] == VIT_NATIVE_SIZE:
        return input_tensor
    
    # 否则进行双线性插值缩放
    return F.interpolate(input_tensor, size=(VIT_NATIVE_SIZE, VIT_NATIVE_SIZE), 
                         mode='bilinear', align_corners=False)

def benchmark_static(model, height, width):
    """
    静态输入测试：整个过程中输入尺寸固定
    """
    print(f"\n--- 开始测试 [静态输入] 尺寸: {height}x{width} ---")
    raw_input = torch.randn(BATCH_SIZE, 3, height, width).to(DEVICE)
    
    real_input = preprocess_for_vit(raw_input)

    # 1. Warm Up (非常重要，让NPU完成算子编译和缓存)
    print("正在预热 (Warmup)...")
    with torch.no_grad():
        for _ in range(WARM_UP):
            _ = model(real_input)
    torch.npu.synchronize() # 等待计算完成

    # 2. 正式测试
    print("开始计时...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(TEST_STEPS):
            _ = model(real_input)
            torch.npu.synchronize() 
            
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / TEST_STEPS) * 1000 # ms
    fps = (TEST_STEPS * BATCH_SIZE) / total_time
    
    print(f"结果: Latency = {avg_latency:.2f} ms, FPS = {fps:.2f}")
    return fps, avg_latency

def benchmark_dynamic(model):
    """
    动态输入测试：模拟输入尺寸在两个数值间不断切换
    测试 NPU 处理变长输入的鲁棒性
    """
    print(f"\n--- 开始测试 [动态输入] (尺寸切换: 224 <-> 300) ---")
    # 准备两个不同尺寸的 Tensor
    raw_a = torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE)
    raw_b = torch.randn(BATCH_SIZE, 3, 300, 300).to(DEVICE) # 举例一个非标尺寸
    
    # 预处理成模型能吃的样子 (实际上都变成了 224)
    input_a = preprocess_for_vit(raw_a)
    input_b = preprocess_for_vit(raw_b)

    # 也可以先分别预热一下，排除单纯的首次编译时间，专注于切换开销
    with torch.no_grad():
        model(input_a)
        model(input_b)
    torch.npu.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for i in range(TEST_STEPS):
            if i % 2 == 0:
                _ = model(input_a)
            else:
                _ = model(input_b)
            torch.npu.synchronize()
            
    end_time = time.time()
    total_time = end_time - start_time
    fps = (TEST_STEPS * BATCH_SIZE) / total_time
    
    print(f"结果 [动态切换]: FPS = {fps:.2f}")

if __name__ == "__main__":
    # 设置 NPU 设备
    torch.npu.set_device(DEVICE)
    
    model = get_model()
    
    # === 测试项目 A: 不同输入尺寸的静态性能 ===
    # 小尺寸 (112x112)
    benchmark_static(model, 112, 112)
    
    # 中尺寸 (224x224) - 标准
    benchmark_static(model, 224, 224)
    
    # 大尺寸 (448x448)
    benchmark_static(model, 448, 448)
    
    # === 测试项目 B: 动态输入性能 ===
    benchmark_dynamic(model)
    
    print("\n测试完成。")