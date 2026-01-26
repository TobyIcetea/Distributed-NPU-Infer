# Distributed-NPU-Infer

## 项目介绍

本项目开发于 2025 年初，旨在利用 Kubernetes 和 Docker 技术，结合 NPU 开发板构建一个分布式的视频推理系统。

项目采用 YOLOv5 模型进行目标检测。系统接收视频输入，利用分布式计算能力对视频进行推理处理，并输出带有检测框的视频结果。

核心实现逻辑基于 MapReduce 思想：系统根据各开发板（Worker 节点）的算力差异，将输入视频动态切分为不同大小的片段（Split/Map），分发至对应节点并行处理；待所有节点推理完成后，系统自动回收结果并合并为完整视频（Reduce）。通过这种方式，充分利用了边缘设备的异构算力，提高了视频推理效率。

## 项目结构

```bash
├── ascend-infer-job.yaml           # k8s 中 infer-job 的 yaml 文件
├── auto-scp.py                     # 分发视频文件到 worker 节点的脚本
├── collect_and_merge_video.py      # 收集 worker 节点推理结果，并合并视频的脚本
├── generate-npu-levels.sh          # 查询 Atlas 节点算力
├── main.py                         # worker 节点中推理脚本
├── patch-k8s.py                    # 修改 k8s 中 infer-job 的 yaml 文件
├── README.md                       # 项目说明文档
├── resources                       # 项目资源目录
├── run.sh                          # 一键运行推理脚本，用于查询算力、切分视频、分发视频、收集结果、合并结果
├── segments                        # 视频切分后的片段目录
└── split-video.py                  # 视频切分脚本
```

## 项目运行前置条件

1. **Kubernetes 集群**：需要部署 Kubernetes 集群，且 Worker 节点需为 Ascend（昇腾）开发板，主机名需匹配 `atlas*` 模式。
2. **运行环境镜像**：Ascend 开发板的容器运行时（默认为 containerd）中需预加载 `ascend-infer` 镜像。该镜像应包含 CANN 驱动、torch_npu 等必要的推理依赖环境。

## 运行项目

### 1. 查询 Atlas 节点算力

有两种查询的方式，使用第二种可以将 worker 节点的算力查询结果保存到一个 json 文件中。

（1）在 Atlas 节点上直接查询：

```bash
(base) root@atlas:~/workdir/data/output-file# npu-smi info -t nve-level -i 0 -c 0
        nve level                      : 20T_1.6GHz
```

（2）在 k8s 环境中检测所有 worker 节点的算力：

```bash
bash generate-npu-levels.sh
```

执行结果如下所示：

```json
{
  "atlas1": 8,
  "atlas": 20
}
```

### 2. 按照算力切分视频

在 master 节点上执行如下脚本：

```bash
python3 ./split-video.py --input full-videos/racing.mp4 --output ./segments/ --npu-json npu-levels.json
```

之后就可以在 `--output` 目录下生成两个文件：`racing_1.mp4` 和 `racing_2.mp4`。并且修改 json 文件为如下格式：

```json
{
  "atlas1": {
    "power": 8,
    "filename": "racing_1.mp4"
  },
  "atlas": {
    "power": 20,
    "filename": "racing_2.mp4"
  }
}
```

### 3. 按照 json 分发视频文件

在 master 节点上执行：

```bash
python3 auto-scp.py --input npu-levels.json --source-dir ./segments  --dest-dir /root/workdir/data/input-file/
```

之后可以在 worker 节点的 `/root/workdir/data/input-file/` 目录下看到两个文件：`racing_1.mp4` 和 `racing_2.mp4`。

### 4. worker 节点中推理

推理脚本参加 `main.py` 文件。调用时，通过参数 `--input` 指定输入文件的目录，`--output` 指定输出文件所在的目录。例如：

```bash
python3 ./main.py --input /root/workdir/data/input-file/ --output /root/workdir/data/output-file/
```

在实验所用容器环境中，使用：

```bash
python3 ./main.py --input video-data/input-file/ --output video-data/output-file/
```

### 5. k8s 中 infer-job 部署

首先在本地打包好业务包，打包命令参考：

```bash
tar zcvf yolov5.tgz coco_names.txt main.py requirements.txt yolo.om det_utils.py
```

使用如下命令，让 k8s 启动 pod 去执行任务，并且执行完任务之后清理文件、Job 等资源。

```bash
python3 patch-k8s.py --json ./npu-levels.json --yaml ./ascend-infer-job.yaml
```

### 6. 收集并合并视频

执行如下命令：

```bash
python3 collect_and_merge_video.py --input-dir ./segments/ --json npu-levels.json --output-dir ./full-videos
```

之后就可以根据 `input-dir` 和 `json` 参数中指定的所有文件生成一个完整的 `output_racing.mp4` 到 `--output-dir` 目录下。

## 总体的 run 脚本

将上述所有脚本合并到一个 `run.sh` 脚本中，可以一键执行分布式推理任务：

```bash
bash run.sh
```

执行 `run.sh` 之前的项目目录树：

```bash
.
├── ascend-infer-job.yaml
├── auto-scp.py
├── collect_and_merge_video.py
├── full-videos
│   └── racing.mp4
├── generate-npu-levels.sh
├── npu-levels.json
├── patch-k8s.py
├── run.sh
├── segments
└── split-video.py
```

执行 `run.sh` 之后的项目目录树：

```bash
.
├── ascend-infer-job.yaml
├── auto-scp.py
├── collect_and_merge_video.py
├── full-videos
│   ├── racing.mp4
│   └── racing_output.mp4
├── generate-npu-levels.sh
├── npu-levels.json
├── patch-k8s.py
├── run.sh
├── segments
│   ├── output_racing_1.mp4
│   ├── output_racing_2.mp4
│   ├── racing_1.mp4
│   └── racing_2.mp4
└── split-video.py
```

