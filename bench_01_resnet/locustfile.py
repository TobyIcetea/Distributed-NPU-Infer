from locust import HttpUser, task, between

class NpuInferenceUser(HttpUser):
    # 模拟用户思考时间：
    # between(0.01, 0.05) 表示每个用户发完请求后等待 0.01~0.05秒
    # wait_time = between(0.01, 0.05) 
    wait_time = between(0, 0.01) 

    @task
    def inference_request(self):
        # 发送 POST 请求
        with self.client.post("/predict", catch_response=True) as response:
            if response.status_code == 200:
                json_data = response.json()
                if json_data.get("status") == "error":
                    response.failure(f"Inference Logic Error: {json_data.get('message')}")
            else:
                response.failure(f"HTTP Error: {response.status_code}")