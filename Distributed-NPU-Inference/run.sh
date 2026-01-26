#!/bin/bash

# 获取节点的名称和算力的对应信息，并保存为 npu-levels.json 文件
bash generate-npu-levels.sh

# 切分视频
python3 ./split-video.py     --input full-videos/racing.mp4     --output ./segments/     --npu-json npu-levels.json

# 将要处理的视频分发给工作节点的 /root/workdir/data/input-file 目录
python3 auto-scp.py \
  --input npu-levels.json \
  --source-dir ./segments \
  --dest-dir /root/workdir/data/input-file/

# 让 k8s 启动 pod 去执行任务，并且执行完任务之后清理文件、Job
python3 patch-k8s.py     --json ./npu-levels.json     --yaml ./ascend-infer-job.yaml

# 收集结果，并将结果进行合并
python3 collect_and_merge_video.py --input-dir ./segments/ --json npu-levels.json --output-dir ./full-videos
