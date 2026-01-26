"""
ResNet ONNX 推理主程序 - NPU 性能基准测试
"""

import logging
import numpy as np
import onnxruntime as ort
import time
import threading
import subprocess
import re
import json
from tqdm import tqdm
from config import (
    LOG_LEVEL, LOG_FORMAT, LOG_FILE, ONNX_MODEL_PATH,
    DATASET_VAL_DIR, BATCH_SIZE, RESIZE_SIZE, IMAGE_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD, WARMUP_BATCH_COUNT
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 核心配置区域 ---
# NPU 物理设备 ID
NPU_SMI_ID = 768

MODEL_DISPLAY_NAME = "EfficientNet-B3"
CURRENT_MODEL_GFLOPS = 1.8

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

class NPUPowerMonitor:
    """NPU 功耗监控器 (独立线程)"""
    def __init__(self, smi_device_id, interval=0.1):
        self.smi_device_id = smi_device_id
        self.interval = interval
        self.running = False
        self.power_readings = []
        self.thread = None

    def _get_npu_power(self):
        try:
            cmd = ['npu-smi', 'info', '-t', 'power', '-i', str(self.smi_device_id)]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0: return 0.0
            match = re.search(r"NPU Real-time Power\(W\)\s*:\s*([\d\.]+)", result.stdout)
            return float(match.group(1)) if match else 0.0
        except Exception: return 0.0

    def _monitor_loop(self):
        while self.running:
            power = self._get_npu_power()
            if power > 0: self.power_readings.append(power)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.power_readings = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=2.0)

    def get_average_power(self):
        if not self.power_readings: return 0.0
        return sum(self.power_readings) / len(self.power_readings)

def get_data_transforms():
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

def load_dataset():
    logger = logging.getLogger(__name__)
    preprocess = get_data_transforms()
    try:
        dataset = datasets.ImageFolder(root=DATASET_VAL_DIR, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=8)
        return dataset, dataloader
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        exit(1)

def load_onnx_model():
    logger = logging.getLogger(__name__)
    options = ort.SessionOptions()
    try:
        session = ort.InferenceSession(
            ONNX_MODEL_PATH,
            sess_options=options,
            providers=[
                ("CANNExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "npu_mem_limit": 20 * 1024 * 1024 * 1024,
                    "op_select_impl_mode": "high_performance",
                    "optypelist_for_implmode": "Gelu",
                    "enable_cann_graph": True
                }),
                "CPUExecutionProvider",
            ]
        )
        logger.info(f"ONNX 模型加载成功: {ONNX_MODEL_PATH}")
        return session
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        exit(1)

def run_onnx_inference(session, dataloader):
    logger = logging.getLogger(__name__)
    power_monitor = NPUPowerMonitor(smi_device_id=NPU_SMI_ID, interval=0.1)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 1. 预热
    logger.info(f"开始预热 ({WARMUP_BATCH_COUNT} batches)...")
    dataloader_iter = iter(dataloader)
    for _ in range(WARMUP_BATCH_COUNT):
        inputs, _ = next(dataloader_iter)
        inputs_np = inputs.numpy()
        io_binding = session.io_binding()
        input_ortvalue = ort.OrtValue.ortvalue_from_numpy(inputs_np, device_type="cann", device_id=0)
        io_binding.bind_ortvalue_input(name=input_name, ortvalue=input_ortvalue)
        io_binding.bind_output(output_name, device_type="cann", device_id=0)
        session.run_with_iobinding(io_binding)

    # 2. 正式推理
    logger.info("预热完成，开始正式推理与功耗监控...")
    power_monitor.start()
    
    correct = 0
    total = 0
    formal_total_images = 0
    
    # 跳过预热部分
    dataloader_iter = iter(dataloader)
    for _ in range(WARMUP_BATCH_COUNT):
        next(dataloader_iter)
        
    start_time = time.time()
    
    formal_steps = len(dataloader) - WARMUP_BATCH_COUNT
    for inputs, labels in tqdm(dataloader_iter, total=formal_steps, unit="batch", desc="Inference"):
        inputs_np = inputs.numpy()
        
        io_binding = session.io_binding()
        input_ortvalue = ort.OrtValue.ortvalue_from_numpy(inputs_np, device_type="cann", device_id=0)
        io_binding.bind_ortvalue_input(name=input_name, ortvalue=input_ortvalue)
        io_binding.bind_output(output_name, device_type="cann", device_id=0)
        
        session.run_with_iobinding(io_binding)
        
        outputs = io_binding.get_outputs()[0].numpy()
        predicted = np.argmax(outputs, axis=1)
        
        batch_size = labels.size(0)
        formal_total_images += batch_size
        total += batch_size
        correct += (predicted == labels.numpy()).sum().item()

    end_time = time.time()
    
    power_monitor.stop()
    avg_power = power_monitor.get_average_power()
    
    # 3. 计算指标
    duration = end_time - start_time
    accuracy = 100 * correct / total
    throughput = formal_total_images / duration
    
    total_gflops = CURRENT_MODEL_GFLOPS * formal_total_images
    actual_tflops = (total_gflops / duration) / 1000
    efficiency = (actual_tflops * 1000) / avg_power if avg_power > 0 else 0

    results = {
        "model_name": MODEL_DISPLAY_NAME,
        "model_flops_G": CURRENT_MODEL_GFLOPS,
        "accuracy_percent": round(accuracy, 2),
        "throughput_img_per_sec": round(throughput, 2),
        "avg_power_watts": round(avg_power, 2),
        "actual_compute_tflops": round(actual_tflops, 4),
        "efficiency_gflops_per_watt": round(efficiency, 2)
    }
    
    return results

def save_and_print_results(results):
    # 保存 JSON
    with open("inference_report.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # 打印简报
    print("\n" + "="*50)
    print(f"      Ascend NPU 推理性能报告")
    print("="*50)
    print(f" 模型:           {results['model_name']}")
    print(f" FLOPs:          {results['model_flops_G']} GFLOPs")
    print(f" 准确率:         {results['accuracy_percent']} %")
    print(f" 吞吐量 (FPS):   {results['throughput_img_per_sec']}")
    print(f" 平均功耗:       {results['avg_power_watts']} W")
    print("-" * 50)
    print(f" 实际算力:       {results['actual_compute_tflops']} TFLOPS")
    print(f" 能效比:         {results['efficiency_gflops_per_watt']} GFLOPS/W")
    print("="*50 + "\n")

def main():
    setup_logging()
    
    # 加载资源
    dataset, dataloader = load_dataset()
    session = load_onnx_model()
    
    # 推理
    results = run_onnx_inference(session, dataloader)
    
    # 输出
    save_and_print_results(results)

if __name__ == "__main__":
    main()