# 验证记录（2026-09-30）

## 当前结论

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

仅安装了轻量框架依赖及其传递依赖，没有下载视觉模型权重。

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

- Grounding DINO / SAM 2 / Real-ESRGAN 真实服务适配器。
- Qwen-Image-Layered 语义层图像。
- 遮挡分析、Qwen-Image-Edit / inpainting 补全。
- CV + VLM QA、retry diagnosis 和策略执行。
- 持久化 checkpoint、人工交互 UI、PSD / Godot 导出。
