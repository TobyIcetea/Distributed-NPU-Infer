"""
ResNet ONNX 推理主程序 - 使用 ONNX 模型在 NPU 上进行推理
并将预热阶段的结果进行可视化保存，用于展示。
"""

import logging
import numpy as np
import onnxruntime as ort
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# 确保导入了必要的配置项
from config import (
    LOG_LEVEL, LOG_FORMAT, LOG_FILE, ONNX_MODEL_PATH,
    DATASET_VAL_DIR, BATCH_SIZE, RESIZE_SIZE, IMAGE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD, WARMUP_BATCH_COUNT
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 新增可视化辅助函数
# ==========================================
def denormalize(tensor, mean, std):
    """
    反归一化，将预处理过的张量转回用于显示的图像数据 (H, W, C, [0,1])
    """
    # 将 mean 和 std 转换为适应 (C, H, W) 的形状
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    
    # 反归一化公式: image = tensor * std + mean
    img_np = tensor.numpy() * std + mean
    
    # 将数据限制在 [0, 1] 范围内，防止显示异常
    img_np = np.clip(img_np, 0, 1)
    
    # 转换维度顺序从 (C, H, W) 到 (H, W, C) 以便 matplotlib 显示
    return np.transpose(img_np, (1, 2, 0))

def visualize_warmup_batch(inputs_tensor, true_labels_idx, pred_labels_idx, class_names, batch_idx, output_dir="vis_results"):
    """
    可视化一个预热 Batch 的推理结果，生成网格拼图并保存。
    图上标示真实类别和预测类别，预测正确标绿，错误标红。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    batch_size_current = inputs_tensor.size(0)
    # 为了展示美观，每个 Batch 最多只选前 16 张图进行拼图
    num_to_vis = min(batch_size_current, 16)
    
    # 计算网格的行数和列数 (例如 16张图就是 4x4)
    rows = int(np.ceil(np.sqrt(num_to_vis)))
    cols = int(np.ceil(num_to_vis / rows))
    
    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3.5))
    # 展平 axes 数组方便遍历，处理只有 1 张图的特殊情况
    axes = axes.flatten() if num_to_vis > 1 else [axes]
    
    logger = logging.getLogger(__name__)

    for i in range(num_to_vis):
        ax = axes[i]
        img_tensor = inputs_tensor[i]
        true_idx = true_labels_idx[i].item()
        pred_idx = pred_labels_idx[i]

        # 1. 反归一化图像
        img_np = denormalize(img_tensor, NORMALIZE_MEAN, NORMALIZE_STD)

        # 2. 获取类别名称并截断（防止名字太长）
        # 如果 class_names 为 None，则直接显示索引
        true_name = class_names[true_idx][:15] if class_names else str(true_idx)
        pred_name = class_names[pred_idx][:15] if class_names else str(pred_idx)

        # 3. 判断对错并设置颜色
        is_correct = true_idx == pred_idx
        color = 'green' if is_correct else 'red'
        # 设置标题内容：真实值 vs 预测值
        title_text = f"T: {true_name}\nP: {pred_name}"

        # 4. 绘图
        ax.imshow(img_np)
        # 设置标题，使用对应颜色，字体稍小
        ax.set_title(title_text, color=color, fontsize=9, pad=3)
        # 关闭坐标轴显示
        ax.axis('off')

    # 隐藏多余的空白子图
    for i in range(num_to_vis, len(axes)):
        axes[i].axis('off')

    # 调整布局紧凑
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(output_dir, f"warmup_vis_batch_{batch_idx}.png")
    # dpi=100 保证清晰度适中，文件不会太大
    plt.savefig(save_path, dpi=100)
    plt.close(fig) # 关闭画布释放内存
    logger.info(f"预热 Batch {batch_idx} 可视化结果已保存至: {save_path}")
# ==========================================


def setup_logging():
    """配置日志系统"""
    # 确保日志文件目录存在
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'), # mode='w' 每次覆盖，方便调试
            logging.StreamHandler()
        ],
        force=True # 强制重新配置，防止多次调用冲突
    )
    return logging.getLogger(__name__)

def get_data_transforms():
    """
    获取数据预处理转换
    
    返回:
        transforms.Compose: 数据预处理转换组合
    """
    preprocess = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    return preprocess

def load_dataset():
    """
    加载imagenet-mini数据集的验证部分
    
    返回:
        dataset: 加载的数据集
        dataloader: 数据加载器
    """
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    # 获取数据预处理转换
    preprocess = get_data_transforms()
    
    # 加载数据集
    try:
        # ImageFolder 会自动根据文件夹名称生成类别映射
        dataset = datasets.ImageFolder(root=DATASET_VAL_DIR, transform=preprocess)
        # num_workers 设置为 4 或 8，根据 CPU 核数调整
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # 预热时shuffle一下可以看到不同类别的图
        return dataset, dataloader
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        exit(1)

def load_onnx_model():
    """
    加载 ONNX 模型并配置 NPU 执行提供者
    
    返回:
        session: ONNX Runtime 会话
    """
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    # 配置 ONNX Runtime 选项
    options = ort.SessionOptions()
    # options.enable_profiling = True # 如果需要性能分析可以打开

    # 创建 ONNX Runtime 会话，使用 CANN 执行提供者进行 NPU 推理
    try:
        session = ort.InferenceSession(
            ONNX_MODEL_PATH,
            sess_options=options,
            providers=[
                (
                    "CANNExecutionProvider",
                    {
                        "device_id": 0,
                        # "arena_extend_strategy": "kNextPowerOfTwo", # 根据实际情况调整
                        # "npu_mem_limit": 20 * 1024 * 1024 * 1024,
                        "op_select_impl_mode": "high_performance",
                        # "optypelist_for_implmode": "Gelu",
                        "enable_cann_graph": True
                    },
                ),
                "CPUExecutionProvider", # 作为 fallback
            ]
        )
    except Exception as e:
         logger.error(f"模型加载失败，请检查路径和 NPU 环境: {e}")
         exit(1)
    
    # 获取输入输出信息
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    logger.info(f"ONNX 模型加载成功: {ONNX_MODEL_PATH}")
    logger.info(f"输入名称: {input_info.name}, 形状: {input_info.shape}, 类型: {input_info.type}")
    logger.info(f"输出名称: {output_info.name}, 形状: {output_info.shape}, 类型: {output_info.type}")
    
    return session

# 修改函数签名，接收 class_names
def run_onnx_inference(session, dataloader, class_names=None):
    """
    执行 ONNX 模型推理并统计性能指标，同时在预热阶段进行可视化。
    
    参数:
        session: ONNX Runtime 会话
        dataloader: 数据加载器
        class_names: 类别名称列表 (list of strings)
        
    返回:
        dict: 包含推理结果和性能指标的字典
    """
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    logger.info("开始 ONNX 推理...")
    start_time = time.time()
    
    # 初始化统计变量
    correct = 0
    total = 0
    batch_count = 0
    
    # 用于统计吞吐量的变量
    warmup_time = 0
    formal_start_time = None
    formal_batch_times = []  # 记录每个正式batch的处理时间
    formal_batch_sizes = []  # 记录每个正式batch的大小
    
    # 获取输入输出名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 创建一个迭代器用于手动控制数据加载
    dataloader_iter = iter(dataloader)

    # -------------------------------------------
    # 预热阶段 (Warmup Phase) + 可视化
    # -------------------------------------------
    logger.info(f"开始预热，共 {WARMUP_BATCH_COUNT} 个 Batch。前 3 个 Batch 将进行可视化。")
    
    for i in range(WARMUP_BATCH_COUNT):
        try:
            # 获取数据
            inputs, labels = next(dataloader_iter)
        except StopIteration:
            logger.warning("数据集数据不足以完成预热，提前结束预热。")
            break

        batch_count += 1
        batch_start_time = time.time()
        
        # 将 PyTorch 张量转换为 NumPy 数组
        # 确保数据类型匹配模型输入 (通常是 float32)
        inputs_np = inputs.numpy().astype(np.float32)
        
        # 创建 IO 绑定 (对于 NPU 性能至关重要)
        io_binding = session.io_binding()
        
        # 将输入数据绑定到 NPU 设备
        # 注意：需要确保 inputs_np 是连续的内存，否则可能会报错
        if not inputs_np.flags['C_CONTIGUOUS']:
            inputs_np = np.ascontiguousarray(inputs_np)

        input_ortvalue = ort.OrtValue.ortvalue_from_numpy(inputs_np, device_type="cann", device_id=0)
        io_binding.bind_ortvalue_input(name=input_name, ortvalue=input_ortvalue)
        # 绑定输出到 NPU
        io_binding.bind_output(output_name, device_type="cann", device_id=0)
        
        # 执行推理
        session.run_with_iobinding(io_binding)
        
        # 获取输出结果并转回 CPU numpy
        outputs = io_binding.get_outputs()[0].numpy()
        
        # 计算预测结果
        predicted = np.argmax(outputs, axis=1)
        
        # --- 插入可视化逻辑 ---
        # 仅对前 3 个预热 Batch 进行可视化，避免产生太多文件
        if i < 3: 
            logger.info(f"正在生成预热 Batch {i+1} 的可视化结果...")
            visualize_warmup_batch(
                inputs_tensor=inputs, # 传入原始 Tensor 用于反归一化
                true_labels_idx=labels,
                pred_labels_idx=predicted,
                class_names=class_names,
                batch_idx=i + 1
            )
        # ----------------------

        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        warmup_time += batch_time
        
        # 打印预热进度
        if (i + 1) % 2 == 0 or (i + 1) == WARMUP_BATCH_COUNT:
             logger.info(f"预热进度: {i+1}/{WARMUP_BATCH_COUNT}, Batch耗时: {batch_time*1000:.2f}ms")

    
    # -------------------------------------------
    # 正式推理阶段 (Formal Inference Phase)
    # -------------------------------------------
    # 计算剩余需要推理的 Batch 数
    remaining_batches = len(dataloader) - batch_count
    
    if remaining_batches > 0:
        formal_pbar = tqdm(
            range(remaining_batches), 
            desc="正式推理", 
            unit="batch",
            position=0,
        )
        
        formal_start_time = time.time() # 记录正式推理开始时间

        for _ in formal_pbar:
            try:
                inputs, labels = next(dataloader_iter)
            except StopIteration:
                break

            batch_start_time = time.time()
            
            inputs_np = inputs.numpy().astype(np.float32)
            if not inputs_np.flags['C_CONTIGUOUS']:
                inputs_np = np.ascontiguousarray(inputs_np)
            
            # IO Binding
            io_binding = session.io_binding()
            input_ortvalue = ort.OrtValue.ortvalue_from_numpy(inputs_np, device_type="cann", device_id=0)
            io_binding.bind_ortvalue_input(name=input_name, ortvalue=input_ortvalue)
            io_binding.bind_output(output_name, device_type="cann", device_id=0)
            
            # 推理
            session.run_with_iobinding(io_binding)
            
            # 获取结果
            outputs = io_binding.get_outputs()[0].numpy()
            
            # 统计
            predicted = np.argmax(outputs, axis=1)
            total += labels.size(0)
            correct += (predicted == labels.numpy()).sum().item()
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_size = labels.size(0)
            
            formal_batch_times.append(batch_time)
            formal_batch_sizes.append(batch_size)
        
        formal_pbar.close()
    else:
        logger.warning("没有剩余数据进行正式推理，仅完成了预热。")

    end_time = time.time()

    # 计算总体指标
    total_time = end_time - start_time
    # 如果没有进行正式推理，formal_time 为 0
    formal_time = end_time - formal_start_time if formal_start_time else 0
    accuracy = 100 * correct / total if total > 0 else 0

    # 计算吞吐量（仅基于正式batch）
    if formal_batch_times:
        # 使用 sum(formal_time) 更准确，避免计时误差累积
        actual_formal_time = sum(formal_batch_times)
        total_formal_images = sum(formal_batch_sizes)
        # 吞吐量 = 总图片数 / 总处理时间
        throughput = total_formal_images / actual_formal_time if actual_formal_time > 0 else 0
        avg_batch_time = actual_formal_time / len(formal_batch_times)
        avg_batch_size = total_formal_images / len(formal_batch_sizes)
    else:
        throughput = 0
        avg_batch_time = 0
        avg_batch_size = 0

    # 返回结果
    results = {
        "model_type": "ONNX",
        "device_used": "NPU (CANN)",
        "total_images": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "total_inference_time": total_time,
        "warmup_batch_count": WARMUP_BATCH_COUNT,
        "warmup_time": warmup_time,
        "formal_inference_time": formal_time,
        "formal_batch_count": len(formal_batch_times),
        "formal_images_processed": sum(formal_batch_sizes) if formal_batch_sizes else 0,
        "throughput_images_per_sec": throughput,
        "avg_batch_time": avg_batch_time,
        "avg_batch_size": avg_batch_size,
        "batch_count": batch_count
    }
    
    # 输出最终结果
    logger.info(f"ONNX 推理完成 - 总耗时: {total_time:.2f}s, 预热: {warmup_time:.2f}s")
    if formal_time > 0:
         logger.info(f"正式推理耗时: {formal_time:.2f}s, 准确率: {accuracy:.2f}%, 吞吐量: {throughput:.2f} 图片/秒")
    else:
         logger.info(f"准确率 (预热阶段): {accuracy:.2f}%")
    
    return results

def save_results(results):
    """保存推理结果到 JSON 文件"""
    # 获取日志记录器
    logger = logging.getLogger(__name__)
    
    import json
    # 确保文件名不冲突，可以加上时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"onnx_inference_results_{timestamp}.json"
    try:
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"性能指标已保存到 {filename}")
    except IOError as e:
        logger.error(f"保存结果文件失败: {e}")

def print_results(results):
    """打印推理结果"""
    print("\n===== ONNX 推理结果 =====")
    print(f"模型类型: {results['model_type']}")
    print(f"使用设备: {results['device_used']}")
    print(f"总图片数: {results['total_images']}")
    print(f"正确预测数: {results['correct_predictions']}")
    print(f"准确率: {results['accuracy']:.2f}%")
    print(f"总推理时间: {results['total_inference_time']:.2f}s")
    print("-------------------------")
    print(f"预热批次: {results['warmup_batch_count']}")
    print(f"预热时间: {results['warmup_time']:.2f}s")
    print("-------------------------")
    if results['formal_batch_count'] > 0:
        print(f"正式推理时间: {results['formal_inference_time']:.2f}s")
        print(f"正式批次: {results['formal_batch_count']}")
        print(f"正式处理图片数: {results['formal_images_processed']}")
        # 高亮显示关键指标
        print(f"\033[1;32m吞吐量: {results['throughput_images_per_sec']:.2f} 图片/秒\033[0m")
        print(f"平均批次时间: {results['avg_batch_time']*1000:.2f}ms")
        print(f"平均批次大小: {results['avg_batch_size']:.1f}")
    else:
        print("未进行正式推理。")
    print("=========================\n")

def main():
    """主函数 - 协调各模块完成 ONNX 推理任务"""
    # 设置日志
    logger = setup_logging()
    
    # 加载数据集
    dataset, dataloader = load_dataset()
    # 获取类别名称列表，ImageFolder 的 classes 属性就是类别名列表
    class_names = dataset.classes
    logger.info(f"数据集加载成功: {len(dataset)} 张图片, {len(dataloader)} 个batch")
    logger.info(f"类别总数: {len(class_names)}")
    
    # 加载 ONNX 模型
    session = load_onnx_model()
    
    # 执行推理，传入 class_names
    results = run_onnx_inference(session, dataloader, class_names)
    
    # 保存结果
    save_results(results)
    
    # 打印结果
    print_results(results)

if __name__ == "__main__":
    main()