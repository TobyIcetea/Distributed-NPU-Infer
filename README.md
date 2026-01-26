# Omni-Infer-Ascend

Omni-Infer-Ascend 是一个基于昇腾（Ascend）NPU 的推理项目集合，其中包含了多个不同的推理项目，且不同的项目都有各自的 README.md 文件。

## 项目总览

目前其中包含的项目如下：

| 项目名称 | 项目介绍 | 设备 |
| -------- | -------- | ---- |
| Distributed-NPU-Inference | 分布式 yolov5 推理项目 | 310B * N |
| bench_01_resnet | ResNet 模型推理项目 | 310P |
| bench_02_efficientnet | EfficientNet 模型推理项目 | 310P |
| bench_03_vit | ViT 模型推理项目 | 310P |
| bench_04_swin_transformer | Swin Transformer 模型推理项目 | 310P |

## 项目设备

### 310B

310B 设备主要是基于 Atlas 200I DK A2 完成的。

![310B](./images/Atlas200IDKA2.png)

### 310P

310P 设备主要是使用的香橙派 AI Studio 完成的。

![310P](./images/OriginPiAIStudio.png)

## 推理结果展示

### Distributed-NPU-Inference

拆分：

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

分布式推理并合并：

![yolov5-infer](./images/yolov5-infer.png)

### bench_01_resnet

![resnet-infer](./bench_01_resnet/vis_results/warmup_vis_batch_1.png)

### bench_02_efficient_net

![efficientnet-infer](./bench_02_efficientnet/vis_results/warmup_vis_batch_1.png)

### bench_03_vit

![vit-infer](./bench_03_vit/vis_results/warmup_vis_batch_1.png)

### bench_04_swin_transformer

![swin-transformer-infer](./bench_04_swin_transformer/vis_results/warmup_vis_batch_1.png)




