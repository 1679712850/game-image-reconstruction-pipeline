# P2：候选回接与资源执行

## 主流程

assign_ownership → select_candidates（启用生成时名称为 complete_objects）→ upscale_objects → build_metadata → reconstruct_scene → export

所有可见实例都有一个 CandidateRegistry。原始分割以 original 进入；局部边缘缺损优先局部修复，严重遮挡/污染进入完整对象生成。关闭 object_completion 时仍选择原始候选，不加载 Qwen。开启 layer_decomposition 后，有局部 alpha 重叠证据的非 Mock 分层结果也进入 Registry；整场景背景层不会硬当成独立对象。

生成按轮批量执行：全部生成 → 全部新轮廓分割 → 全部 QA → 修改 prompt 后重试。单个实例生成/分割/QA 失败保留原因，其他实例继续。REJECT 结果不会作为 fallback；有评分证据的 RETRY 完整生成、局部修复、原分割、可恢复的原 crop+mask 按优先级回退并标记人工复核。不存在有效像素的实例保留空资产和失败原因，不制造矩形假分割。

## QA 与选择

生成候选以 shape/style/perspective/scale/color/edge/semantic 的 .20/.20/.15/.15/.10/.10/.10 加权评分排序；同时检查 lighting、background_leak、occlusion_reconstruction_quality。严重类型/结构/风格失败直接 REJECT；可修复失败 RETRY；完整评分、各项门槛和低背景泄漏全部达标才 ACCEPT。相同分数优先原图，其次优先较早候选，不按最后生成时间取胜。

原图沿用上游分割 QA，规则基线为正常 .80、有缺损迹象 .65、原 QA 不通过 .40；这是保留来源像素的选择基线，不代表完成了视觉语义评分。未变化的 Mock 候选不会获得额外分数。

生成图的视觉分数来自 SceneReviewService.review_candidate 的结构化 LLM 比较。scene_loop.reviewer: llm 配合 models.yaml.scene_reviewer 使用已有视觉端点；规则后端不编造风格、透视、语义分数，生成结果保留 RETRY，最终回退原始资产并标记人工复核。单次 Qwen、VLM 推理结果可缓存，修改 retry prompt 会产生不同缓存键。

重试受 resources.max_generation_retry 约束，并根据 perspective/color/lighting/shape/style/semantic/edge 失败原因追加不同指令。生成预算不与原始 SAM 分割重试预算混用。

## 几何与下游资产

SceneObject.accepted_asset 是最终权威路径；asset_path 和 manifest 的 asset 同步指向它。Registry 保留全部候选、原图、mask、bbox、QA、失败理由和选择记录。source_asset_path/source_mask_path/source_crop_bbox 保留 P1 可见裁剪前的分割证据。

完整生成结果先裁到有效 alpha，再按源物体比例归一化，恢复原 ground anchor；局部编辑裁透明边时只改变 crop offset，不重新缩放未编辑像素。placement 保留原 bbox、center、anchor、scale、rotation、z_order、depth、mask_position。RGB 生成图必须经 SAM 恢复 alpha，不能直接把旧 SAM mask 贴在生成图上。

最终 asset_mask_path/full_mask_path/mask_path 从已采纳 alpha 重建；accepted_ownership 用新 mask 和显式 occludes → z_order 顺序生成最终可见归属。原 ownership 仍是源图分割覆盖诊断，二者定义明确区分。PNG 合成、PSD 图层、高清纹理均消费 accepted_asset。P1 的 terrain/residual 层保留其来源属性。重建差异是源图像素差异诊断，修复隐藏区域后不要求与源图完全一致。

PSD 导出为标准 PSD v1：RGBA 图层、原场景偏移、Unicode 层名、正常 alpha 混合和 merged preview。不依赖 Photoshop；单边最多 30000 像素、最多 32767 图层；不实现 PSB、复杂混合模式或图层组。

## 资源与缓存配置

~~~yaml
object_completion:
  enabled: false
candidates:
  automatic: true
  accept_threshold: 0.80
  retry_threshold: 0.55
  severe_occlusion: 0.35
  export_psd: true
resources:                         # 所有内存值单位 GiB
  soft_vram: 20
  max_vram: 22
  max_ram: 64
  gpu_models_max_resident: 1
  max_generation_retry: 2
  max_oom_retry: 2
  keep_alive: {sam: true, image_edit: false}
  estimated_vram: {grounding: 2, sam: 3, image_edit: 18, layered: 18}
  estimated_ram: {}                # 未指定时按显存估计的 2 倍预估 CPU 模型内存
cache:
  enabled: true
  directory: null                  # output_dir/cache；可指定绝对共享路径
  pipeline_version: p2-v1
  prompt_version: p2-candidate-v1
~~~

ExecutionRuntime 统一拦截图中服务的加载和推理。模型由 ModelManager 懒加载；状态为 NOT_LOADED/CPU/GPU/OFFLOADED，记录 last_used 与实测显存增量。加载前根据显存估算、CUDA free/allocated、soft/hard limit 和 GPU 常驻数量执行 LRU；活跃模型不被驱逐。keep_alive 模型可转 CPU，其他模型释放所有服务持有的权重引用。gc/empty_cache/ipc_collect 只发生在卸载、阶段切换和 OOM 恢复，不在每个小操作执行。

模型实际峰值和机器资源差异很大，estimated_vram/estimated_ram 必须按自己的权重校准；资源预算是加载准入和运行恢复机制，不是操作系统级内存硬隔离。

OOM 先卸载非活跃模型，再切 CPU、降低输入分辨率并有限重试。当前适配器的图像 batch 已是 1。DINO 降低 processor 分辨率；局部 SAM 同步缩放 prompts 并把结果恢复原大小；Qwen 降低输入/Layered 分辨率。Diffusers 支持时启用 VAE tiling/slicing。Qwen dtype 可显式配置 fp16/bf16；不会强制把不支持半精度的 CPU 算子改为 fp16。

可在 models.yaml.qwen_image_edit / qwen_layered 配置 quantized_model_path 指向预先量化好的完整本地 Diffusers 管线；第二次 OOM 恢复可切换该管线。不自动下载/量化权重。这不是任意模型的通用 tile diffusion 或动态量化器。

检测/分割等节点耗尽资源恢复后可继续导出 resource_failures；生成按实例回退。模型依赖缺失、无效配置、文件系统失败仍如实报错，不当成成功结果。同一图的节点串行使用设备；同时运行多个图应使用独立 ServiceBundle 和输出目录。

结果缓存按输入内容 SHA256、适配器、模型配置/revision、本地权重文件版本、prompt、推理参数、pipeline_version、prompt_version 区分。命中前不加载模型。目录分 detection/segmentation/vlm/generation/qa/upscale；JSON + 无 pickle 的 NPY + 校验和文件 blob，原子写入，损坏当 miss。跨输出目录命中时恢复本次输出目录中的文件，限制路径不能逃出输出目录。OOM 降级或失败的结果不会以完整质量键缓存。修改提示词模板或算法时应提升版本；Hub revision 建议固定到 commit，不要依赖可变 main。

## 产物与统计

- scene.json：accepted_asset、placement、provenance、candidate_registry、accepted_ownership、resource_failures。
- metadata/candidates.json：可移动路径的完整注册表。
- debug/candidates/<id>_contact_sheet.png：Original、Candidates、Accepted、QA 原因及坐标摘要。
- candidates/<id>/、accepted_masks/、metadata/accepted_owner.npy。
- scene.psd、reconstruction.png、高清纹理与 PNG 层。
- performance_report.json：节点 start/end/duration，模型 load/inference/postprocess，RSS/system RAM、VRAM before/after/peak、实例耗时、retry_count/duration/GPU time、cache hit/miss/rate。
- timeline.html：节点时间轴与 RAM、VRAM、GPU utilization 曲线。

内存采样间隔 200 ms，并在 span 边界取样；CUDA load/inference 另读 allocator peak。CPU 采样无法保证捕获极短峰值；GPU utilization 取不到时明确 unavailable/null。GPU event time 只在 CUDA 上可用，CPU 不会冒充 0 秒 GPU 工作。同一图 checkpoint 暂停不会后台继续采样。报告中模型推理为服务调用耗时减去已单独记录的加载/后处理时间；未单独标记的少量输入准备仍包含在该服务耗时中。

## 验证边界

自动测试和 Mock CLI 验证候选回接、坐标/alpha、PNG/PSD 一致性、重试与缓存/资源恢复。资源测试使用可控内存快照和模型替身验证 CUDA 决策分支；没有在本机运行真实 CUDA/Qwen。Qwen 真实输出质量、视觉端点质量、各权重的实际显存峰值仍需要对应环境验收。
