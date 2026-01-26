"""
ResNet 推理主程序 - 简化为调用各模块的主程序
"""

import logging
from config import LOG_LEVEL, LOG_FORMAT, LOG_FILE
from model_loader import load_model
from data_loader import load_dataset
from inference_engine import run_inference
from result_saver import save_results, print_results
from npu_monitor import NPUPowerMonitor

def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():

    # 设置NPU功耗监控
    monitor = NPUPowerMonitor(device_id=768, interval=1.0)
    monitor.start()

    # 设置日志
    logger = setup_logging()
    
    # 加载模型
    model, device = load_model()
    
    # 加载数据集
    dataset, dataloader = load_dataset()
    
    # 执行推理
    results = run_inference(model, device, dataloader)
    
    # 保存结果
    save_results(results)
    
    # 打印结果
    print_results(results)

    # 停止监控并画图
    monitor.stop()
    monitor.analyze_and_plot(save_path="./power_log.png")

if __name__ == "__main__":
    main()
