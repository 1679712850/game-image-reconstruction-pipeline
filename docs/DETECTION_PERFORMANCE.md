# 检测与分割性能优化验证

默认开启的优化保留类别、提示词分组、窗口、精度、阈值、候选顺序、去重规则、SAM 候选评分和 QA 门槛。GPU 批处理已实现但保持 `batch_size: 1`，因为真实模型对比证实改变 batch 会产生浮点数值差异。

## 实现与边界

- **窗口输入复用**：同一窗口只裁剪一次。首次调用完整 Grounding DINO processor，随后调用相同 processor 的 text-only API，复用 `pixel_values`/`pixel_mask`。缓存只存一个窗口，离开窗口、卸载模型、改变 processor/device/降级比例时释放或失效，不共享不同图像的输入。
- **局部补检调度**：`p1.detection_prefetch_regions: 4` 最多预取四个区域的首次检测，随后按原 region/attempt 顺序执行 SAM 和 QA。先前区域的验收结果仍决定后续去重、邻居提示词和 ID。首次检测失败的早期区域先完成其原有重试，再提交后面的预取结果。预取准入使用 detector 返回数量和调用数量的保守上界（包含 OOM 重试），保证不会抢占早期区域的对象名额/重试预算。预算临界、未知 detector 或显式设为 `1` 时保留串行路径。不会为了组成批次提前执行本应跳过的重试。
- **SAM 评分**：同一裁剪多个 mask 复用 Canny 和膨胀后的边缘图，逐候选的形态学、覆盖度和分数公式保持不变。
- **融合索引**：按类别和空间单元索引已有候选。查询结果按原插入顺序检查同一 `same_object` 判定；合并后更新空间索引。巨型框使用有界退化路径。补检未追加匹配观察时复用第一次融合，否则重新融合原始观察，避免改变贪心匹配顺序。
- **可选 batch**：`models.yaml:grounding.batch_size` 支持 1–8。生产调度仅在 CUDA 上启用大于 1 的批次，并仅在同一窗口内合并张量尺寸和文本 token 长度完全一致的组，不增加 padding，也不调整浮点精度。OOM 先二分批次，失败的单样本回到原有逐组隔离/资源恢复。扫描预算继续按逻辑提示词请求计数，物理批调用由 profiler 的 `_infer_batch` 事件记录。缓存包含配置和组顺序，命中时不加载模型。

批处理不跨窗口重排，因为后续窗口的类别规划会使用前面窗口的候选；贸然并行可能改变检测范围。要求数值一致时保持 `batch_size: 1`。

## 对比证据

[局部 CPU 基准](../benchmarks/reports/detection_optimization/microbench.json) 使用固定随机种子，先预热，再取多次测量的中位数：

| 项目 | 原实现 | 优化后 | 一致性检查 |
|---|---:|---:|---|
| 1800 个合成候选融合 | 1.947 秒 | 0.046 秒 | 保留/拒绝结果、顺序、合并记录全部一致 |
| 512×512 图像的 3 个 SAM 候选评分 | 8.706 毫秒 | 3.484 毫秒 | 每个候选的每项分数一致，Canny 次数 3 → 1 |

这是有利于空间索引的稀疏合成场景，不代表整条流水线能加速 42 倍。密集重叠候选仍可能需要较多比较。

[真实模型对比](../benchmarks/reports/detection_optimization/real_model_check.json) 使用本地 Grounding DINO tiny、CPU、真实图片 `sect_ruins.png` 的 768×768 裁剪和 3 组提示词：

- 原串行输入与复用后的每个输入张量完全相同。
- 输出分别为 107、40、33 个候选，共 180 个；开启输入复用后，候选输出逐值完全相同。
- 批处理的数量和标签相同，但最大置信度差为 `0.0002546`，最大框坐标差为 `0.05994` 像素，因此没有默认启用。
- JSON 中的模型调用耗时为单次诊断，基线先运行包含预热成本，不能作为加速比例。没有在 CUDA 上测量吞吐或完整复杂图像耗时。

补检差分测试使用保留的原实现，覆盖空检测后重试、重复区域、重复候选、异常、对象上限和预算耗尽，逐项比较最终对象、ID、history、预算和 RGBA 像素。在三个区域的受控案例中，DINO/SAM 阶段由六次交替执行变成两个连续阶段。随机融合测试另覆盖框扩张跨单元、同分排序、巨型框与输入不被修改。

## 复现

```bash
.venv/bin/python -m unittest tests.test_detection_optimization -v
.venv/bin/python -m benchmarks.detection_speed --output /tmp/detection-speed.json
# 需要 .cache/models 中已有 DINO tiny；只读本地模型，不下载。
.venv/bin/python -m benchmarks.check_detection_inputs --output /tmp/dino-check.json
```

如需排查输入缓存或调度差异，可设置 `grounding.reuse_image_inputs: false`、`p1.detection_prefetch_regions: 1`。无需降低检测阈值、减少类别或关闭多尺度。
