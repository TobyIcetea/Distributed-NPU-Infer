"""
配置模块 - 集中管理所有配置参数
"""

# 设备配置
DEVICE_TYPE = "npu:0"

# 根据精度类型自动选择模型路径
USE_FP16 = False
MODEL_PATH = "/ascend-models/bench_04_swin_transformer/swin_base_patch4_window7_224_traced.pt"
MODEL_FP16_PATH = "/ascend-models/bench_04_swin_transformer/swin_base_patch4_window7_224_fp16_traced.pt"
ONNX_MODEL_PATH = "/ascend-models/bench_04_swin_transformer/swin_base_patch4_window7_224.onnx"

# 数据集配置
DATASET_DIR = "/ascend-datasets/imagenet-mini"
DATASET_VAL_DIR = DATASET_DIR + '/val'
BATCH_SIZE = 64

# 数据预处理配置
IMAGE_SIZE = 224
RESIZE_SIZE = 256
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# 推理配置
WARMUP_BATCH_COUNT = 5

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = 'inference.log'

# 结果输出配置
RESULTS_FILE = "inference_results.json"
