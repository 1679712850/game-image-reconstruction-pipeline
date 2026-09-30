# 遮挡重建与独立高清修复

## 执行链路

`visible crop → vision occlusion analysis → semantic amodal polygons → transparent expanded canvas → Qwen edit → SAM mask selection → edge matting/cleanup → completion QA → bounded retry/candidate ranking → Real-ESRGAN → HD QA → RGBA PNG`

- 源 alpha 只表示可见区域。真实 Image Edit 输出 RGB，任何完成候选都必须重新进行前景分割；缺少 segmenter 会拒绝候选。
- 视觉 reviewer 同时接收原场景、透明目标 crop、对象类别、遮挡关系和邻居框，输出遮挡方向、面积、边界推理、对称/重复结构、置信度和完整轮廓多边形。坐标相对源 crop，可超出 `[0,1]`，受 `[-2,3]` 的硬预算限制。
- amodal 多边形与可见证据取并集，绝不把 dilation 当成真实轮廓。语义服务缺失/失败时，只保存标记 `evidence=unavailable` 的可见提示，置信度为零，不能通过自动入库。
- 画布与 mask 解耦。默认 30% context；遮挡率 ≥30% 使用 60%；最少 32px。新增画布为 `(0,0,0,0)`，允许越出原图。贴边输出按 1.5 的指数增加 padding，受边长/像素总量与重试次数限制。
- 原请求 mask 存为 `requested_edit_mask.png`；重建实际 edit mask 允许模型在上下左右补全。QwenImageEditPipeline 无原生 mask conditioning；完整轮廓、方向和坐标以 prompt 引导，SAM 使用 amodal bbox + 可见正点 + 画布角点负点选择输出 mask。
- 新 alpha 来自生成图上的 SAM。按可见保留率、SAM 分数、背景外溢和边缘接触评分选择候选。清理微小孤岛/孔洞，保护细结构；局部色彩 trimap 估计软边缘，不做整体 erosion。当前不是独立的神经 matting 模型。

## QA 与坐标

补全必须满足新增面积和原可见轮廓外的有效像素均 >1%，可见区域保留率 ≥90%，总面积不超过视觉预测的 `max_expansion_ratio`，且目标不贴画布边缘。视觉 QA 另外检查突然截断、遮挡物残留、透视、形状、材质、风格和颜色。

失败按 `resources.max_generation_retry` 重试；不重复请求不可用的视觉 QA。遮挡率达到 `candidates.severe_occlusion` 默认创建 3 个候选（可配 2～4），每个候选、每次修正使用不同且可复现的 prompt/seed。低于 65% 重建置信度或可见比例 <20% 必须人工复核，显式编辑请求也不能跳过此门槛。

补全按扩展画布坐标原位输出，不缩回源 bbox、不按旧 alpha 剪裁。`crop_bbox`/`bbox_full` 可以为负坐标。独立 PNG 和 PSD 图层保留完整像素，场景预览与场景空间 mask 仅在合成时裁到原场景。

新增元数据包括 `bbox_visible`、`bbox_full`、`original_crop_path`、`expanded_crop_path`、`amodal_mask_path`、`edit_mask_path`、`reconstructed_mask_path`、`reconstruction`、`completion_qa`、`reconstruction_confidence`、`hd_qa`、`asset_library_eligible`。`visible_mask_path` 不再被最终 ownership 覆盖，最终可见性另存 `accepted_visible_mask_path`。JSON 保留全部候选/失败证据供复核；消费最终资产库时必须筛选 `asset_library_eligible=true`。

## Stage B：真实高清后端

真实模式默认 `real_esrgan`，使用本地 `RealESRGAN_x4plus.pth` 与兼容官方 checkpoint 命名的 RRDBNet（64 features、23 blocks、growth 32）。神经网络按带重叠上下文的 tiles 做 4x 推理，然后按需要得到 2x 输出；不是 resize + sharpen。Stage B 仅对已通过完成度 QA 的补全对象执行。

模型只修复 RGB；透明区域底色由邻近前景延伸以避免黑边；**Stage A 重建完成后的 alpha**按目标分辨率采样并锁定。这个 alpha 与最初的 visible mask 不同。输出还要经过独立视觉 QA，确认没有颜色、纹理、构件或美术风格改变。Real-ESRGAN 本身不接受文本提示，`HD_PROMPT` 为其他提示词式后端提供契约，当前视觉 QA 使用等价约束。

配置：

```yaml
# pipeline.yaml
object_completion:
  enabled: true
scene_loop:
  reviewer: llm
amodal:
  context_padding: 0.30
  severe_padding: 0.60
  minimum_padding: 32
  confidence_threshold: 0.65
  severe_candidates: 3

# models.yaml
qwen_image_edit:
  model_path: /absolute/path/to/local/Qwen-Image-Edit
scene_reviewer:
  model: your-vision-model
  base_url: https://your-compatible-endpoint/v1
  api_key_env: VLM_API_KEY
upscale:
  backend: real_esrgan
  checkpoint: /absolute/path/to/RealESRGAN_x4plus.pth
  tile: 256
  tile_pad: 16
  max_output_pixels: 67108864
```

安装可选 `requirements-qwen.txt`、`requirements-llm.txt`、`requirements-sam2.txt`、`requirements-upscale.txt`；本代码不自动下载 Qwen/高清模型。缺少权重或推理失败会保留 native 资产，记录 `hd_qa.status=unavailable`，不静默换成插值。明确选择 `backend: lanczos` 或 Mock 时输出 `preview_only`，不能宣称已完成高清修复。

## 验证范围

`tests/test_amodal_reconstruction.py` 使用可控生成/视觉/分割替身，覆盖树、山体、建筑、柱子的扩张、旧轮廓失败重试、画布扩张、严重遮挡多候选、低置信度/缺分割阻止入库、场景外 PNG/PSD、微小孔洞/噪点/半透明/细线保护，以及高清轮廓、缺权重和 tiled stitching。已有 Qwen 适配测试已改为验证 RGB 输出不携带旧 alpha。

这些是软件契约测试，不是预训练模型视觉质量验收。当前项目 Qwen/高清 checkpoint 仍为空，未安装 Diffusers，高清推理只依赖可选 PyTorch；真实树根、山体纹理、屋顶透视及高清效果须在配置模型后，以实际遮挡图像复验。此限制在每个候选/资产元数据中可见。

## Visible/full geometry and optional enhancement

Visible source geometry is represented by `visible_bbox`/`visible_mask`; an inferred or edited complete object uses `full_asset_bbox`/`full_asset_canvas` and its reconstructed mask. A full asset may extend outside the source crop. Upscale is an optional enhancement: unavailable weights produce `enhancement.status=unavailable` or `skipped` while the QA-derived `base_asset_status` remains factual. Set `upscale.required=true` only when HD delivery is an acceptance requirement.
