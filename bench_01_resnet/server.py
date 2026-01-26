import time
import logging
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Response
from contextlib import asynccontextmanager
from config import ONNX_MODEL_PATH, BATCH_SIZE, IMAGE_SIZE, LOG_LEVEL

# 设置日志
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量
model_session = None
input_name = None
output_name = None
dummy_input = None

def load_onnx_model():
    """加载模型并配置 CANN (NPU)"""
    logger.info("正在加载 ONNX 模型到 NPU...")
    options = ort.SessionOptions()
    
    # 核心 NPU 配置
    providers = [
        (
            "CANNExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "npu_mem_limit": 20 * 1024 * 1024 * 1024,
                "op_select_impl_mode": "high_performance",
                "optypelist_for_implmode": "Gelu",
                "enable_cann_graph": True
            },
        ),
        "CPUExecutionProvider",
    ]
    
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=options, providers=providers)
    return session

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    生命周期管理：启动时加载模型，关闭时释放
    """
    global model_session, input_name, output_name, dummy_input
    
    # 1. 加载模型
    try:
        model_session = load_onnx_model()
        input_info = model_session.get_inputs()[0]
        output_info = model_session.get_outputs()[0]
        input_name = input_info.name
        output_name = output_info.name
        
        logger.info(f"模型加载成功。输入: {input_name}, 输出: {output_name}")

        # 2. 预生成 Dummy 数据 (避免每次请求都 malloc，专注测推理计算)
        # Shape: [Batch, Channel, Height, Width]
        dummy_input = np.random.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
        
        # 3. 预热 (Warmup) - 跑 10 次让 NPU 唤醒并分配内存
        logger.info("开始预热 (Warmup)...")
        for _ in range(10):
            model_session.run(None, {input_name: dummy_input})
        logger.info("预热完成，服务就绪。")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise e
        
    yield
    
    # 关闭时清理
    logger.info("服务正在关闭...")

# 初始化 API
app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict():
    """
    推理接口
    这里我们不接受图片上传，而是直接使用内存中的 Dummy 数据，
    目的是对 NPU 施加最纯粹的计算压力，排除网络传输图片的干扰。
    """
    try:
        start_time = time.time()
        
        # --- 推理核心 ---
        # 使用 IO Binding 也可以，但在 HTTP 服务中直接 run 兼容性更好
        # 且对于 ResNet50 这种计算密集型，run 的 overhead 可以接受
        _ = model_session.run(None, {input_name: dummy_input})
        # ----------------
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # 返回 JSON
        return {
            "status": "success", 
            "latency_ms": latency_ms
        }
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}