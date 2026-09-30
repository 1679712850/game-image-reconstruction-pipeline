# 验证记录（2026-09-30）

## V1 框架验收

**V1 真实 LangGraph 端到端验收已通过。** 依赖已安装到仓库 `.venv`，
使用真实 `StateGraph` 和 `graph.invoke(...)`，正常、强制重试、
零重试预算三次 CLI 运行均正常返回并打印 `Done. Reached END.`。

2026-09-29 的依赖安装阻塞已解除。本记录替代之前的待验收状态；
`output/node_smoke/` 仍保留为早期独立节点验证产物，不作为本次 graph 验收依据。

## 依赖检查

Python 3.12.14，`python -m pip check` 返回 `No broken requirements found.`。

| 依赖 | 实测版本 |
|---|---|
| langgraph | 1.2.12 |
| langchain | 1.4.3 |
| pydantic | 2.13.5 |
| pillow | 12.3.0 |
| numpy | 2.5.3 |
| opencv-python | 5.0.0.93 |
| pyyaml | 6.0.3 |
| python-dotenv | 1.2.3 |

以上是 V1 框架验收时的基础依赖。后续真实模型接入已安装额外视觉依赖并下载权重，见本文末尾。

## 自动化测试

实际执行：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：**38 项测试全部通过，无 skip、无 failure、无 error。**

其中 7 项真实 LangGraph 集成测试验证：

- 正常 graph 输出可移动的 scene.json 及全部引用资产。
- 强制 Mock failure 进入一次 retry，再次 QA 后继续。
- 零 retry budget 直接导出 manual_review。
- 持续空 mask 在预算耗尽后终止，不产生伪资产。
- 无检测对象时仍正常导出空场景。
- crop padding 和可选 upscale / reconstruction 配置生效。
- InMemorySaver 在 QA 后暂停，恢复后到达 END，checkpoint.next 为空。

其余 31 项覆盖 bbox、padding、空 mask、RGB/alpha、重建坐标与混合、
pivot、纹理倍率、schema、router、配置和独立节点/服务。

## 实际 CLI Demo

输入 `input/test.png` 为 640 × 480 合成测试地图。

```bash
.venv/bin/python main.py --input input/test.png --output output/test --mock
.venv/bin/python main.py --input input/test.png --output output/test_retry --mock --exercise-retry
.venv/bin/python main.py --input input/test.png --output output/no_retry --mock --exercise-retry --max-retry 0
```

| 运行 | 对象数 | retry_count | 未解决对象 | 结果 |
|---|---:|---:|---|---|
| output/test | 4 | 0 | 无 | END |
| output/test_retry | 4 | 1 | 无 | END |
| output/no_retry | 4 | 0 | tree_001（manual_review） | END |

三次 CLI 均返回 exit code 0。每次均生成：

```text
assets/                 4 个紧凑 RGBA PNG
assets_hd/              4 个高清 PNG
masks/                  4 个原图尺寸灰度 mask
debug/run.json          completed=true、实际节点访问顺序和最终 State
scene.json
reconstruction.png
```

正常路径访问 12 个业务节点；强制重试路径额外访问一次
retry_objects 和一次 qa_objects。图共有 13 个业务节点，另有 START / END。

## 产物检查

对三组实际 CLI 产物逐一完成校验：

- scene.json 通过 Pydantic SceneManifest 校验，所有引用文件存在。
- 所有 assets 均为 RGBA，宽高严格小于测试原图尺寸。
- 原始资产尺寸与 crop_bbox / logical_size 一致。
- 高清尺寸与 texture_size / texture_scale 一致。
- 裁剪后 RGB 与原图对应区域逐像素相同，Alpha 与 mask 对应区域逐像素相同。
- mask 为原图尺寸，pivot / global z_order 的坐标关系正确。
- reconstruction 为 640 × 480 RGBA，已实际打开检查。
- 每次 exported_assets 的 14 个引用文件均存在（debug/run.json 另行保存）。

| 对象 | Logical size | Texture size |
|---|---:|---:|
| tree_001 | 121 × 200 | 363 × 600 |
| rock_001 | 134 × 94 | 402 × 282 |
| building_001 | 153 × 147 | 459 × 441 |
| mountain_001 | 288 × 152 | 576 × 304 |

三次运行 reconstruction_score 均为约 0.48869。该分数是全画布 RGBA
像素差异指标；矩形 Mock mask 保留了选区内背景，未提取区域保持透明。
这符合 V1 Mock 行为，不能解释为已经完成真实语义分割或完整地图复原。

## 后续模型能力 TODO

V1 框架验收已完成。以下继续保留为后续扩展，不影响当前 Mock 运行：

- Real-ESRGAN 真实服务适配器（Grounding DINO / SAM 2 已完成接入，见下文）。
- Qwen 模型配置与真实分层/编辑验收（代码适配已完成，见本文末尾）。
- 遮挡分析、Qwen 编辑后的新轮廓分割与 inpainting 补全闭环。
- CV + VLM QA、retry diagnosis 和策略执行。
- 持久化 checkpoint、人工交互 UI、PSD / Godot 导出。


## Grounding DINO + SAM 2 接入验收（2026-09-30）

**真实模型已接入，在线首次加载与离线 CPU 重跑均到达 END。**
不是固定 bbox、矩形 mask 或替身模型的集成演示。

### 实测依赖与模型

| 组件 | 版本 / 权重 |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.14.0 |
| torchvision | 0.29.0 |
| Transformers | 5.17.0 |
| huggingface-hub | 1.33.0 |
| SAM-2 官方包 | 1.0；源码提交 2b90b9f5ceec907a1c18123530e92e794ad901a4 |
| Grounding DINO | IDEA-Research/grounding-dino-tiny |
| DINO 权重 snapshot | a2bb814dd30d776dcf7e30523b00659f4f141c71 |
| SAM 2.1 | facebook/sam2.1-hiera-tiny |
| SAM 权重 snapshot | de431c4043854a71d8101e17995dfe596bf101a5 |

模型存放在仓库 `.cache/models/`，该目录被 gitignore 排除。
`pip check` 无依赖冲突。真实模型依赖与基础 requirements.txt 分离。

### 实际运行

```bash
.venv/bin/python main.py --input input/test.png --output output/real_test --real
.venv/bin/python main.py --input input/test.png --output output/real_test --real --offline --device cpu
```

第二次离线运行覆盖同名产物，`debug/run.json` 保存对应的 offline=true、cpu 配置。
两次均返回 exit code 0，实际执行 Grounding DINO、SAM 2 和既有 LangGraph 节点。
结果为 **6 个候选、1 次重试、4 个规则 QA pass、2 个 manual_review**。

| 对象 | 置信度（约） | 紧框 PNG 尺寸 | 状态 |
|---|---:|---|---|
| building_001 | 0.574 | 153 × 145 | pass |
| rock_001 | 0.485 | 132 × 94 | pass |
| road_001 | 0.453 | 640 × 480 | pass |
| mountain_001 | 0.362 | 199 × 130 | pass |
| mountain_002 | 0.333 | 161 × 151 | manual_review |
| bridge_001 | 0.324 | 640 × 480 | manual_review |

当前 QA 最低置信度 0.35，高于检测框阈值 0.30，因此两个低置信度候选
在有限次重新分割后仍保留人工复核，不提高原始置信度或强行通过。

逐项校验：所有 PNG / mask / JSON 引用文件存在；mask 为 640 × 480、
uint8 0/255；alpha 与 mask 对应区域逐像素一致；RGB 与原图对应区域
逐像素一致；高清尺寸、pivot、global z_order 及重建尺寸正确。
全部 6 个 mask 均为非矩形轮廓（紧框内占比约 0.51–0.72）。
建筑透明效果另在 `debug/building_checkerboard.png` 中人工检查。
校验明细保存为 `output/real_test/debug/validation.json`。

`scene.json.backends` 标记真实 Grounding DINO、官方 SAM 2、配置类别与 Lanczos。
未宣称已接入 VLM、生成式超分或遮挡补全。

### 回归测试与边界

自动测试现为 **50 项全部通过**。新增检测框裁剪、NMS、无效结果、标签归一化、
SAM bbox 坐标/最佳候选/形状校验/资源复位、配置路径、设备选择及懒加载测试。
新增适配器单测使用注入的模型替身验证接口；真实神经网络推理由上述两次 CLI 单独验证。
原 Mock CLI 也重新执行并正常生成 4 个对象、零重试，未受影响。

效果限制：合成图仍有道路/桥梁大范围误检及树木漏检，规则 QA 无法判断全部
语义错误。大范围 mask 的真实紧框覆盖整图，不能人为缩小以制造“紧凑”结果。
本次验收证明模型加载、数据接口及完整流程正确，不构成真实游戏地图准确率评估。
CPU 已实测，CUDA/MPS 尚未实测；MPS 为显式可选路径。

## Qwen Layered / Image Edit 代码接入验收（2026-09-30）

**本节验收代码、Mock 与接口契约，不代表 Qwen 真实模型推理已通过。**
未下载任何 Qwen 权重，未安装 `requirements-qwen.txt`。
`config/models.yaml` 的两个 `model_path` 均保持 `null`；
两个可选节点在 `config/pipeline.yaml` 中默认关闭。

实现了本地 `QwenImageLayeredPipeline` / `QwenImageEditPipeline` 适配，
只从包含 `model_index.json` 的本地目录加载，强制 `local_files_only=True`。
新增 `decompose_layers`、`complete_objects` 两个可选节点，
以及 `decomposed_layers`、`edit_requests`、`object_edits` 三个状态字段。
生成候选独立导出至 `layers/`、`assets_edited/`、`edit_masks/`，
不替换原始实例、坐标与重建结果。清单扩展至 schema version `1.1`。

### 自动化测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：**65 项全部通过**，其中新增 15 项覆盖：

- 空/无效本地模型路径在导入模型包前失败；mock 不加载模型。
- 通过注入的管线工厂确认 `local_files_only=True`、dtype、设备与实例复用。
- 分层输出的嵌套格式、RGBA alpha、画布大小和源坐标恢复。
- 编辑输入校验、mask 外像素不变、原 alpha/尺寸不变、错误输出拒绝。
- 显式请求的相对路径、唯一对象 ID、未知对象和禁用阶段校验。
- 两个可选节点连同 retry、checkpoint 暂停/恢复和便携导出到达 END。

真实适配器调用契约测试采用注入的管线替身；没有调用 Qwen 神经网络。
`compileall` 与 `git diff --check` 通过。

### 实际 CLI 与产物

```bash
.venv/bin/python main.py --input input/test.png --output output/qwen_baseline --mock
.venv/bin/python main.py --input input/test.png --output output/qwen_mock_demo \
  --mock --decompose-layers --edit-requests input/qwen_demo_edits/requests.json \
  --exercise-retry
```

测试请求及 crop 尺寸白色 mask 已在 `input/qwen_demo_edits/` 生成，
重新生成方式见 README。两次 CLI 均 exit code 0 并到达 END。
完整 Demo 输出：4 个实例、1 个 `mock_passthrough` 图层、
1 个 `mock_noop` 编辑候选、1 次重试、17 个导出引用文件。
图层与原图 RGBA、编辑候选与源 crop 逐像素一致；所有引用文件存在，
`scene.json` 通过 Pydantic 校验。
明细：`output/qwen_mock_demo/debug/validation.json`。

错误路径也已实测：

```bash
.venv/bin/python main.py --input input/test.png --output output/qwen_unconfigured \
  --real --offline --decompose-layers
```

预期 exit code 1：`QwenImageLayeredPipeline: model_path is empty`。
在任何业务节点执行前终止，不触发 Grounding/SAM/Qwen 加载或下载。

### 未验收与后续工作

待配置本地 Qwen 模型后验证 Diffusers 实际版本兼容性、推理输出格式、
设备资源需求及生成质量。当前 Image Edit 的 mask 是本地合成约束，
模型只接收 RGB 和提示词；原 alpha 不变，因此不支持生成新的完整物体轮廓。
遮挡诊断、生成后的 SAM 重分割、图层语义/所有权映射和候选采纳仍待实现。
本轮没有改变检测提示词、阈值或增加切片检测，不宣称提升漏检召回率。
