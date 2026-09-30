# 工程化优化交付报告

本轮保留现有生产 Pipeline，完成可重复真实 ROI 评测、状态/指标拆分、检测预算与类别规划、Schema 1.2、模型指纹任务缓存及 PSD 内存优化。工程回归通过；真实质量验收仍为 **WARN**：标注集只有 3 个目标和 1 个粗粒度可见 mask，尚未经独立人工复核。

## 1. Architecture Review

修改前已审计 README、P0 检测验收边界、P1/P2、入口/配置加载、whole-image/tiled/multiscale、merge/filter、SAM/mask 后处理、QA/retry、高清、重建/residual、导出/PSD、Schema、模型/资源/cache。

生产链路仍是 `build_graph` → `GroundingService.detect_p0` → `P0DetectionPipeline` → 全图/切片分组推理 → 坐标恢复/去重 → SAM → QA/重试 → ownership/terrain → 候选选择/补全 → 可选高清 → 重建/导出。Benchmark 直接调用相同 graph，也支持重评生产 manifest。

已存在并复用：重叠切片、多尺度、小目标/边缘恢复、候选 QA、源图归属、残差层、完整资产画布、模型生命周期与 profiler。P1 原来的 ownership coverage 已排除 residual，但不能代表 QA 通过的语义完成度。保留了开始工作前已有的 amodal 等未提交改动。

## 2. Changes Implemented

| 优先级 | 本轮实现 |
|---|---|
| P0-A | 原图坐标 ROI GT、bbox/mask/semantic region/ignore 共存；JSON/Markdown/可视化；尺寸召回、FP/重复/碎片、边界质量、独立维度回归门禁；真实模型评测记录 |
| P0-B | QA-passed semantic 与 predicted ownership 分开；residual 不计语义；基础资产就绪、复核、拒绝及 partial/completed 状态；终端不再以重建分数宣告拆分成功 |
| P0-C | 基础状态、增强状态、交付复核分离；高清默认关闭；required 才传播交付失败；统一倍率函数 |
| P0-D | 跨轮预算与 checkpoint 恢复、有限全局/区域/剩余区域扫描；场景/邻近证据分组；调用来源/组成本/候选 provenance；真实回归后保留显式场景 prompt 类别 |
| P0-E | 复用 Pydantic 配置；兼容别名/优先级/弃用警告；Schema 1.2 和 1.1 归一化；visible/full 几何；同步 README/P0/P1/P2 |
| P1-A | 每任务/模型配置缓存廉价指纹；不读取权重内容求 hash |
| P1-B | PSD 内容边界裁剪+偏移、逐层解码、临时文件通道缓存、块写入、原子替换、预算预估/警告或拒绝 |
| P1-C | 新增类型化配置/GT 结构与模块边界；提取高清/边缘恢复逻辑；复用结构化节点日志与资源统计 |

## 3. Files Changed

以下是本轮核心文件地图，不把所有原有未提交改动都归为本轮新增：

| 范围 | 主要文件 |
|---|---|
| Benchmark | `benchmarks/{schema,metrics,pipeline_metrics,report,runner}.py`；`scenes/`、`annotations/`、`configs/`、`reports/` |
| 指标/状态 | `app/asset_status.py`、`cv/completion_metrics.py`、`nodes/{upscale_objects,reconstruct_scene,export}.py`、`main.py` |
| 检测 | `detection/{budget,p0_pipeline,global_detector,tile_detector,grouped_detector}.py`、`taxonomy/planner.py`、`nodes/{detect_instances,scene_loop}.py`、`retry/detection_retry.py` |
| 配置/Schema | `app/{config,detection_config}.py`、`config/pipeline.yaml`、`schemas/{object,scene,migration}.py`、`agent/{graph,state}.py` |
| 性能/导出 | `services/{model_fingerprint,runtime,cache_manager,model_manager,execution,profiler,upscale_service}.py`、`exporters/{json_exporter,psd_exporter}.py` |
| 测试 | `tests/test_benchmark_metrics.py`、`test_engineering_contracts.py`、`test_schema_migration.py` 及已有高清/Schema/集成断言 |
| 文档 | `README.md`、`docs/{DETECTION_P0,P1_DECOMPOSITION,P2_CANDIDATES_RESOURCES,AMODAL_RECONSTRUCTION,ENGINEERING_EVALUATION,IMPLEMENTATION_REPORT}.md` |

## 4. Config Migration

| 旧配置/行为 | 当前兼容方式 |
|---|---|
| 根层配置 | 继续支持；也支持 `pipeline: {...}`，冲突时嵌套值胜出并警告 |
| `detection.tiled` | 迁移到 `detection.tiling`；旧别名警告，canonical 优先 |
| `detection.multiscale` | 迁移到 `detection.multi_scale`；同上 |
| `models.grounding` 旧 discovery 字段 | P0 不采用时警告；旧 detect/detect_round API 与 P1 legacy 分支仍按原契约使用 |
| 默认高清开启 | 默认 `upscale.enabled:false`；显式 true 保留，auto 检查可用性；默认 `required:false` |
| 对象上限 | 常规生产仍可配置；新 Benchmark 强制 `scene_loop.max_objects:null`；重评历史产物明确输出截断/未知 |

检测新增 `detection.budget` 和 `category_planner`；导出新增 `export.memory_budget_mb` 与 `low_memory`。倍率沿用实际策略：长边 <128 为 4x，其余为 2x。详细 Active/Deprecated/Unused 消费关系见 [工程指南](ENGINEERING_EVALUATION.md)。

## 5. Schema Migration

采用 **1.2 增量升级**，保留旧字段、路径和 signed placement 契约。新增基础/增强状态、completion metrics、visible/full geometry。完整资产使用最终接受的 placement，不能使用失败的补全提案。

```python
from schemas.scene import SceneManifest
manifest = SceneManifest.model_validate_json(open("old_scene.json").read())
assert manifest.schema_version == "1.2"
```

也可直接调用 `schemas.migration.normalize_manifest`。未知版本明确报错。历史高清污染状态不会被迁移器无依据地恢复成 ready；strict 1.1 消费方需要升级以接受新增字段。

## 6. New Metrics

- Detection：类别一致、一对一 bbox 匹配的 Recall/Precision；未匹配但重合已匹配 GT 的 Duplicate；同一 GT 内多片段的 Fragmentation 启发式；尺寸分桶与漏检列表。
- Asset：mask IoU、Dice、Boundary F、completeness、contamination；基础 asset-ready、review、failed 比例。未标注 mask 时不假装存在 mask 质量。
- Scene：QA 通过的源图可见语义像素并集/有效源图像素；另报包括待复核候选的 predicted coverage；unassigned=1−semantic；residual 是 unassigned 的子集，不是额外相加的分区。
- Fidelity：重建相似度独立呈现，不能证明语义正确；GT semantic-region mask 评分与预测覆盖率分开。
- Performance：运行时、采样峰值 RSS、设备分配/可用 CUDA allocator 峰值；无法测量的 VRAM 为 null。

## 7. Benchmark Usage

从仓库根目录执行（本机现有虚拟环境和本地模型）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python -m benchmarks.runner \
  --annotation benchmarks/annotations/sect_ruins_courtyard.json \
  --pipeline benchmarks/configs/sect_ruins_safe.yaml \
  --models benchmarks/configs/sect_ruins_models.yaml \
  --config benchmarks/configs/default.yaml \
  --output benchmarks/reports/my_current \
  --baseline benchmarks/reports/baseline/benchmark_report.json
```

输出 `benchmark_report.json`、`benchmark_report.md`、`visualizations/` 和生产产物。FAIL 返回退出码 2；draft、Mock、截断未知、缺失 mask 等不能得到完整验收 PASS。模型路径需按其他机器的本地安装调整。离线重评使用 `--manifest benchmarks/reports/safe/pipeline/scene.json`；大型 pipeline 产物仅保留本机，不纳入 Git，其他机器需先运行生产命令。

## 8. Example Report

真实 DINO tiny + SAM2.1 CPU；最终配置、整张 1536×1024 场景：

```text
Pipeline completed (partial)
Detected objects: 122
Asset-ready:       31
Needs review:      87
Rejected:           4
Semantic coverage: 42.29%
Unassigned:        57.71%
Residual:          48.83% (included in unassigned)
Reconstruction:   100.00%
Runtime:          187.20 s
Peak RAM:        2264.48 MiB
Peak VRAM:       unavailable (CPU)
```

同次运行的标注 ROI：Recall 1.0、Precision 1.0、Duplicate 0、Fragmentation 0、mask IoU .586、Boundary F .808；只有 3 个 bbox/1 个 mask，不得外推为整图 Recall。

基线和最终策略均为 19 次调用。最终单次运行耗时较低，但 RAM 较高且采样方式有修正，**不能据此宣称性能提升**。激进 13-call 策略 Recall 降至 0，另一个策略 Precision 降至 .375，均由门禁判定 FAIL 并保留报告。默认已回到保留显式类别的策略，见 [全部真实对比](../benchmarks/reports/README.md)。

## 9. Test Results

```bash
.venv/bin/python -m unittest discover -s tests -q
# Ran 177 tests in 6.452s — OK
git diff --check
# no errors
.venv/bin/python main.py --input input/test.png \
  --output /tmp/engineering-cli-delivery --mock --max-rounds 1
# completed (partial), reconstruction 100%, metrics/export present
```

覆盖预算/恢复、类别保留与扩展组裁剪、配置兼容/优先级/警告、HD disabled/missing/failed/required、Schema 迁移与最终几何、ROI/ignore/mask 指标、残差排除、任务指纹、PSD crop/offset/预算。首次建立基线时即使没有 comparison baseline，资源失败也会强制 gate FAIL。真实重建回归构造验证 similarity=1.0 时 semantic=.215、unassigned=.785。Mock CLI 仅验证契约，不用于上述真实质量结论。

## 10. Remaining Risks

- 真实 GT 仍是 draft 小 ROI；须补建筑/植被/水边/遮挡等代表性区域，并人工复核细 mask。此次不声称已解决整图漏检或获得可直接入库的全部资产。
- 自动类别裁剪有真实 recall/precision 风险；保留显式 prompt 类别是保守默认，扩展 taxonomy 的真实节省和质量仍需更大基线验证。
- QA 规则不能代替人类语义/资产检查；fragmentation 是启发式；mask 平均分基于匹配且标注了 mask 的对象。
- 未执行 CUDA/Real-ESRGAN 视觉质量或 8K 数百层峰值压力验收；PSD 仍需合成画布、单张解码和临时磁盘空间，预算不是 OS 硬上限。
- 模型文件在运行中原地改变需要重启任务；不透明第三方 detector 内部调用不能全部计入统一预算。
- 基线曾遇到 RAM 估计 guard，完整报告来自调整 tiny 模型估计后的重跑；首轮 profiler 的 Torch import race 已修复，但性能对比仅供观察。

真实推理结果之后的最终改动为导出几何/诊断及回归测试，不影响已保存检测预测；这些改动完成了全量测试，但未重复执行昂贵的真实模型运行。
