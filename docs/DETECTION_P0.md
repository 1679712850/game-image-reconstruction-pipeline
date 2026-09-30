# P0 高召回检测

LangGraph 主流程使用 `GroundingService.detect_p0`，实现整图 + 重叠切片 + 语义分组、
候选池、跨窗口融合、截断复检、SAM mask 融合和诊断。生成式补全、Qwen、高清化效果未改变。

## 模块与边界

| 模块 | 责任 |
| --- | --- |
| `app/detection_config.py` | 独立、严格验证的 detection 配置 |
| `detection/global_detector.py` | 保留整图上下文的分组扫描 |
| `detection/tile_detector.py` | 多尺度滑窗调度和 Tile 元数据 |
| `detection/grouped_detector.py` | 语义分组、局部图指令、逐组异常隔离 |
| `detection/candidate_adapter.py` | 坐标契约、标签归一化、无效候选审计、颜色直方图 |
| `detection/p0_pipeline.py` | 检测阶段编排、扩展区域复检、候选统计 |
| `tiling/` | 重叠窗口、归一化/像素坐标恢复 |
| `taxonomy/` | 10 个分组、细分类别、英文/中文别名、提示词分组 |
| `fusion/` | bbox/中心距离/面积比例/颜色证据融合、分割后 mask 并集 |
| `postprocess/` | 人工切片边缘识别、按大小过滤、小目标复核池 |
| `diagnostics/` | 标框、网格与重叠区、扫描覆盖、JSON 与 HTML 报告 |
| `schemas/detection_candidate.py` | 原始候选及 lineage；临时 mask 不进入 checkpoint |

候选内部 bbox 为**原图绝对像素 xyxy**，右下边界 exclusive。
模型适配器必须显式说明 `bbox_format=xyxy|xywh`、`coordinate_space=pixel|normalized`。
默认 dict 兼容旧 xywh 输出；四元素数组默认像素 xyxy，不根据数值猜归一化坐标。
转入已有 SceneObject 时显式转换为绝对像素 xywh；crop、pivot、重建协议保持兼容。

## 配置与兼容性

`config/pipeline.yaml:detection` 是 graph 的唯一检测策略入口。默认：

- 整图和 Tile 均开启，1024 px、25% overlap，额外执行 1536 px 尺度；相同窗口去重。
- `min_tile_size=64` 检查配置尺寸；原图小于 Tile 时仍完整扫描小图。
- `expand_categories=true` 将完整 taxonomy 加入场景分析类别；关闭时只扫描配置/审查建议类别。
- 每组最多 6 类；按语义组划分，再切成短提示词。
- 原始模型候选门槛 0.1；small/medium/large 后处理门槛为 0.25/0.4/0.5。
- small：框面积严格小于原图面积 0.1%；large：框面积至少 5%。
- 扩展区域 padding 192，单轮最多复检 100 个边缘候选。

`config/models.yaml` 继续管理模型、设备、文本短语、文本阈值等。
P0 路径不使用其中的旧 tile/NMS/max_detections 选项提前删除候选；
文本阈值不高于候选门槛，避免低分框全部变成空标签。
直接调用旧 `detect` / `detect_round` 的集成保留旧接口行为；graph 默认走新路径。
自定义覆盖 `detect_round` 的注入适配器继续受支持。

Grounding DINO 接收短名词短语，不是指令型 VLM。局部图完整指令随 scan context
保存，供 VLM 适配器使用；DINO 实际使用细类别短语和分组扫描，未声称其能理解长指令。
模型加载错误仍明确失败。单个 Tile/分组推理错误写入 scans，并继续其它扫描。

`platform` 默认归 terrain，`spirit_stone` 默认归 cultivation_prop，避免重复分组。
`flag` 与 `formation_flag` 分开；`gravestone` 统一为 `tombstone`。
细类别如 `stone_pillar` 保留同名 subtype，旧宽类别和显式自定义类别仍可使用。

## 融合、恢复与保守过滤

先按同类别、IoU、中心距离和面积比例判断重复。
边界碎片额外要求不同窗口、重叠/包含比例和颜色直方图相似证据。
不会仅因同类别且空间邻近，就合并两个无重叠对象。颜色证据是轻量启发式，未引入 embedding 模型。
完全不可见的缝隙两侧物体不凭空关联；它们进入复检/复核路径。

完整观察优先于截断观察。两个片段的框并集仍保留 `is_truncated=true`，
只有完整重新观察才清除截断标记。原图自身边缘不算人工 Tile 截断。
复检只接纳与父候选有足够空间对应的结果。未恢复或预算耗尽的候选进入
`review_candidate_pool`，不直接生成半个对象的 PNG。

独立窗口观察可有界增加融合置信度；同窗口不同提示词不重复计分。
每次模型观察的原始置信度、模型 ID、Tile、窗口、标签和截断边均保存在 observations。
known small-object 类别的低分 Tile 结果留在复核池，并可进入分割；其 QA 保持人工复核。
SAM 后对同类、空间重叠且 mask IoU 达标的对象做 mask 并集并合并 lineage。

旧 scene loop 的 max_objects 仍是预算；因此诊断中 after_filter 可能大于最终对象数。
超出预算也记录 `max_objects`，不会静默消失。分割推理失败按对象隔离；空 mask、
过滤、跨窗口/跨轮重复都保留原因。规则 QA 的 pass 不代表语义正确。

## 产物与检查方式

实体 PNG 沿用现有 `assets/`、`assets_hd/` 和 `masks/`，保持下游兼容。
环境特效写入 `effects/`、`effects_hd/`，manifest 中与实体 `objects` 分开存入
`environment_effects`。当前 manifest schema_version 为 1.2；读取器接受 1.1 并通过 migration helper 归一化。

`diagnostics/` 自动生成：

- global_detection.png、tile_detection.png、merged_detection.png、filtered_detection.png
- tile_grid.png（重叠区着色、失败 Tile 标红）、coverage_map.png
- candidates.json（含无效候选）、filtered.json（删除原因）、review_candidate_pool.json
- summary.json、small_object_report.json、scans.json、objects.json、report.html

覆盖图红色表示未成功扫描，蓝色表示已扫描但无候选，黄色表示候选密度。
全图扫描成功会覆盖整张图；仍应结合 scans 中的失败分组与 Tile 图判断局部失败。
小目标统计按融合后的唯一候选计算，global/tile 计数可能交集；tile_only 是缺少 global 观察的数量。
这些是诊断数据，**不是召回率/准确率**。召回提升需要带标注地图的前后对照。

报告在检测后先输出，在最终 export 时补入分割/过滤结果和多轮历史。
每个最终对象的 `source_candidates` 可回溯到 candidates.json，`observations` 包含原始置信度。
诊断可用 `detection.diagnostics.enabled=false` 关闭；lineage 与复核数据仍保留在状态/manifest。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python main.py --input input/test.png --output output/p0_mock --mock --max-rounds 1
```

P0 回归覆盖重叠覆盖、归一化坐标、英文/中文别名、分组全目录、相邻物体、
跨窗口置信度、互补碎片、真实适配器低分候选保留、截断复检、失败 Tile、
小目标复核、mask 融合、逐对象分割失败、lineage 导出与特效分离。

真实离线 CPU 小目标冒烟结果在 `output/p0_small_props_validation/`，使用仓库的
1536×1024 宗门废墟地图、DINO tiny 和 SAM 2.1。为控制验证时间，实际只扫描
stone_lantern / flag / box / jar / sign / tombstone，1 轮、20 对象预算、复检预算 3。
配置副本、run.log、debug/run.json 和完整报告均保留。

该次运行整图 93 条、Tile 238 条，共 331 条原始候选；融合后 91 条。
46 个融合后小目标候选中，28 个有全图观察，46 个有 Tile 观察，其中 18 个仅来自 Tile。
0 / 3 Tile 失败。本轮没有未解决的 Tile 截断候选，因此真实图未触发扩展复检；
扩展复检由注入的边缘对象回归测试验证。
20 对象预算下，SAM 再融合 1 对重复后输出 19 个对象记录；13 个待复核。
报告仍可见柱子/建筑被模型识别成石灯等误判，不能据此宣称小目标召回率已量化达标。

## Detection budget and benchmark (2026-09-30)

`detection.budget` is shared by global discovery, regional tiled detection and gap-fill recovery. `category_planner` ranks global/neighboring detections and scene categories to choose a bounded set of groups per tile; exhausted calls are recorded as `budget_exhausted`. The three pass purposes are distinct and do not re-run a full category matrix.

`benchmarks.runner` calls the production manifest or graph and reports one-to-one GT recall/precision, duplicate rate, fragmentation rate, miss rate, size buckets, and optional mask IoU/Dice/boundary F-score. The seed annotation is draft and must be reviewed before it becomes an acceptance gate:

```bash
.venv/bin/python -m benchmarks.runner \
  --annotation benchmarks/annotations/sect_ruins_courtyard.json \
  --output benchmarks/reports/sect_ruins \
  --manifest output/real_test/scene.json \
  --config benchmarks/configs/default.yaml
```

The benchmark does not use `reconstruction_score` as decomposition quality. Its pipeline section keeps semantic coverage, residual pixels and asset-ready counts separate.

See [ENGINEERING_EVALUATION.md](ENGINEERING_EVALUATION.md) for current budgets, schema/config migration and real ROI acceptance limits. Earlier six-class/20-object results above remain historical smoke evidence.
