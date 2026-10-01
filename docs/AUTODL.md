# AutoDL 部署配置与存储规划

核对日期：2026-10-01。适用于 Qwen3-VL-32B-Instruct、Grounding DINO Base、SAM 2.1 Large、Qwen-Image-Layered、原版 Qwen-Image-Edit 与 RealESRGAN_x4plus 这六个非量化模型。完整下载清单和命令见 [AutoDL 完整部署命令](DEPLOYMENT.md)。这里没有改用 MoE、Edit Plus 或 SAM 3。

## 1. 租用实例时怎么选

**推荐：单张 96GB GPU（H200 141GB 余量更充裕）、实例可用 RAM 至少 256 GiB、24–32 vCPU、数据盘总容量 300GB 起；持续生产选 400GB 或更大。** 只做一次性测试时 250GB 是可行下限，但空间余量较小。此容量规划尚未在 AutoDL 上完成真实 CUDA 峰值验收；GPU 库存、价格与每卡 CPU/RAM 分配以租用页面为准。

| 选项 | 建议 | 说明 |
| --- | --- | --- |
| 实例类型 | 普通容器实例，优先本地 SSD/NVMe 数据盘 | 本文使用 `/root/autodl-tmp` 数据盘；容器实例 Pro 的存储规则不同 |
| GPU 充裕档位 | 1×H200 141GB | 为 32B VLM 和完整 Diffusers 管线留更多显存余量，不要求六个模型同时驻留 |
| GPU 推荐档位 | 1×96GB 级 GPU，例如 RTX PRO 6000 Blackwell 96GB 或 H20 96GB | 优先满足显存，吞吐与租价另行比较；按下文调整资源预算，仍需实测 |
| 80GB GPU | 仅作为受限试部署 | 32B 本体约 62 GiB，仍需 KV cache、图像激活和工作空间；不承诺完整配置稳定运行 |
| CPU | 建议 24–32 vCPU，至少按 16 vCPU 试运行 | 图片切片、后处理、模型加载与 PSD 导出需要 CPU |
| RAM | **实例实际可用 ≥256 GiB**；有大图/并发需求选 384–512 GiB | 不要误用物理整机总内存；容器内存限制可能低于 `free -h` 看到的宿主机容量 |
| 数据盘 | **250GB 最低，300GB 推荐，长期使用 400GB+** | 是最终总容量，不是另加容量；以控制台扩容后的总额和 `df` 核对 |
| 基础镜像 | NVIDIA CUDA 环境，Ubuntu 22.04/24.04，Python 3.11/3.12 | 以官方镜像实际选项为准；项目依赖在独立环境安装，不能假定镜像预装版本满足要求 |
| 运行方式 | 每卡一次一个任务，batch=1，阶段间卸载模型 | 当前代码没有 tensor parallel、多卡 `device_map` 或逐层 CPU offload |

不要选 24GB/32GB/48GB GPU 来直接运行这套完整加载方案。**2×48GB 不会自动变成一张 96GB，2×80GB 也不会让当前单卡加载器自动使用 160GB。** 如一张卡已能装下模型，多卡可在未来用于独立任务吞吐，但这不是当前任务的必要支出。

AutoDL 官方说明 CPU 和 RAM 随 GPU 数量按主机的每卡配额分配，且内存超限可能直接杀进程。应挑选单卡配额足够的主机；这里的 256 GiB 是容器实际内存预算要求，不是“只要租 H200 就一定有”的平台承诺。[A1][A2]

### GPU 快速选择

保留这六个模型、使用官方非量化权重且不改加载代码时，**96GB 单卡是优先比较租价的档位，H200 141GB 是更充裕的选择，并非硬性最低要求**。没有这些模型在目标实例的实际峰值记录，不能把任何档位承诺为所有分辨率都不会 OOM。

| 单卡显存 / 例子 | 对当前完整方案的判断 |
| --- | --- |
| 24GB：RTX 3090 / 4090 | 无法完整容纳 32B VLM，不能直接运行整套方案 |
| 32GB：RTX 5090 | 同样装不下 32B VLM；模型数量串行也解决不了单个模型过大 |
| 48GB：A6000 / L40S | 32B VLM 本体约 62 GiB，完整 Qwen 图像管线约 54 GiB，均超出容量 |
| 80GB：A100 / H100 80GB | 可作为小输入、短上下文、batch=1 的试部署目标；需要实测并调整准入估算，不能直接套用 H200 预算 |
| **96GB：RTX PRO 6000 Blackwell / H20 的 96GB 型号** | **推荐先比较这一档价格与速度**；给权重和激活留更多空间 |
| **141GB：H200** | **显存余量更充裕**，适合较大输入与减少显存调参；仍按单任务串行加载 |

约 62 GiB / 54 GiB 仅是根据浮点权重文件估算的容量基线，不包含 KV cache、图像激活、attention 工作区和框架分配；实际显存以加载及推理实测为准。磁盘占用 171.43 GiB 不代表需要同等显存。

仅测试 DINO、SAM 或超分组件时显存需求会小得多，但当前完整 `--real` 流程即使关闭分层和补全，仍需要 32B VLM 做场景分析。因此不能把组件冒烟测试的 GPU 配置当成完整流水线配置。Mock 自检无需 GPU。

可降低显存而保持非量化的 CPU offload、多卡分片属于可选代码改造；当前适配器未实现这些方案。租卡时应按现有单卡容量选型，而非先租多张小卡再期待显存自动合并。

## 2. 存储需要多少

用户所列模型共 **171.43 GiB，约 184.07 GB**。这是权重和模型配置本身，不能直接当成整台实例的磁盘需求。

| 数据盘用途 | 本次规划预算 | 依据 |
| --- | ---: | --- |
| 六个模型的单份完整文件 | 171.43 GiB | 完整部署指南核验的 ModelScope 文件清单 |
| Python 环境及依赖 | 10–20 GiB | 估算，含 PyTorch/CUDA 依赖；使用独立 venv，不保留 pip 下载缓存 |
| 下载临时空间 | 10–20 GiB | 顺序下载，下载完成即清理失败分片和缓存；不保留第二套模型 |
| 输入、输出、PSD、推理缓存 | 10–30 GiB | 一张图或少量测试图；PSD 使用磁盘 spool，不等于固定占用 100 GiB |
| 空闲余量 | 20–30 GiB | 处理重试、文件系统波动和临时文件 |
| **精简部署可用空间** | **约 221–271 GiB** | 模型 171.43 GiB + 上述运行空间；不含第二套模型 |

按十进制 GB 换算，下列是容量选择，不是云盘价格或平台对单位的保证：

| 最终数据盘总容量 | 约合 GiB | 使用建议 |
| --- | ---: | --- |
| 200GB | 186.26 | 只比模型总量多约 14.83 GiB，无法稳定安装和运行 |
| **250GB** | **232.83** | **精简试部署下限，须监控空间**；只保留一套模型、关闭 pip 缓存、顺序下载、少量输出 |
| **300GB** | **279.40** | **推荐测试和单人使用**；扣除模型后还有约 108 GiB（116GB）空间 |
| **400GB** | **372.53** | **推荐长期使用**；可保留更多输出和一次失败重试 |
| 500GB | 465.66 | 适合多批结果或临时保留一个版本 |
| 1TB / 1000GB | 931.32 | 只有需要多套模型版本、大量数据或多人共用时才建议 |

如果控制台以 GiB 分配或显示，按实际单位和 `df` 核对，不要只看“GB”字样。**检查的是可用空间，不是磁盘标称总容量。** 600GB 数据盘上已有 300GB 文件，就不能按新空盘预算继续下载。

普通容器官方说明默认系统盘 30GB、数据盘 50GB 起；租用时需选择可扩容到目标容量的主机。数据盘路径 `/root/autodl-tmp`，不要把 184GB 模型下载到 `/root/.cache`、`/root/models` 或其他系统盘目录。[A1][A3]

此前的环境 30 GiB、临时 60 GiB、输出 100 GiB 是保守预留，不能当成最低需求。实际占用须在目标环境测量；上述精简预算也不是实测保证。

节省空间可采用以下方式：

- 所有 `pip install` 命令加 `--no-cache-dir`，或在安装前设置 `PIP_NO_CACHE_DIR=1`，减少重复 wheel 缓存。已安装的 CUDA 库仍需保留。
- ModelScope 直接下载到最终 `local_dir`，只保留一套权重，不额外复制到第二个缓存目录；按顺序下载，限制并发分片。临时峰值取决于 SDK 版本和下载并发，不是必然再占 60 GiB。
- 少量地图先给输出留 10–30 GiB，用实际样本的 `du` 结果估算批量容量。按需关闭 `cache.enabled` 或 `candidates.export_psd`；这些选项分别影响重复运行速度和是否导出 PSD，不改变权重。
- 当前 PSD 写入使用临时通道文件与待完成 PSD，导出时两者会同时占盘；临时文件正常完成后释放。因此要看导出峰值，不能只看最终 PNG 大小。
- 清理历史输出前先备份需要的产物。不要直接删除 ModelScope/Hugging Face 缓存：先确认最终模型目录文件不依赖该缓存的符号链接，且没有下载任务正在使用它。

```bash
# 在执行完整部署指南中的 pip 安装命令前设置；新终端需重新设置。
export PIP_NO_CACHE_DIR=1

# 安装完成后查看并清理 pip 下载缓存，不删除已安装依赖。
PIP_NO_CACHE_DIR=false python -m pip cache info
PIP_NO_CACHE_DIR=false python -m pip cache purge

# 下载和运行过程中检查实际占用；先执行第 3 节环境初始化。
du -sh "$MODEL_ROOT" "$DEPLOY_ROOT/cache" "$DEPLOY_ROOT/outputs"
df -h "$DEPLOY_ROOT"
```

250GB 只适合低端预算成立的情况：小环境、顺序下载、少量输出、无重复模型。若 SDK 保留额外完整副本、大图输出增长或依赖超出预算，应先扩容。300GB 覆盖上述预算的大部分单人测试场景；600GB/1TB 属于更多版本和长期产物保留的可选容量。

容器实例 Pro 官方文档说明没有独立数据盘，系统盘可扩容上限与普通容器不同；即便使用同名 `/root/autodl-tmp` 路径，也不能假定它背后有 600GB 独立容量。本方案优先普通容器，选择 Pro 时先核对其当前容量上限。[A4]

## 3. AutoDL 目录和环境初始化

以 [完整部署指南第 3 节](DEPLOYMENT.md#3-autodl-数据盘初始化与-github-源码下载) 为唯一初始化命令入口，依次执行数据盘目录创建、`env.sh` 写入、GitHub 克隆和 venv 创建，再执行该指南第 4–6 节的依赖安装、ModelScope 下载及配置生成。所有部署目录均在 `/root/autodl-tmp/game-reconstruction` 下。

| 内容 | 路径 |
| --- | --- |
| 源码 | `/root/autodl-tmp/game-reconstruction/source` |
| Python 环境 | `/root/autodl-tmp/game-reconstruction/venv` |
| 模型 | `/root/autodl-tmp/game-reconstruction/models` |
| 部署配置 | `/root/autodl-tmp/game-reconstruction/config` |
| 缓存 / 临时文件 | `/root/autodl-tmp/game-reconstruction/cache` / `tmp` |
| 地图 / 输出 | `/root/autodl-tmp/game-reconstruction/input` / `outputs` |
| 核验及版本记录 | `/root/autodl-tmp/game-reconstruction/deployment-records` |

每个新终端执行：

```bash
source /root/autodl-tmp/game-reconstruction/env.sh
source "$DEPLOY_ROOT/venv/bin/activate"
cd "$SOURCE_ROOT"
```

镜像已有的系统 Python 和工具继续作为基础运行时；新增 Python 包安装在数据盘 venv。`env.sh` 已设置 ModelScope、Hugging Face、PyTorch、CUDA、Triton、pip 及临时文件目录，并默认不保留 pip 下载缓存。无需在系统盘新建项目、虚拟环境或模型目录。

## 4. 显存与内存预算

完整部署指南默认生成 96GB 单卡配置；H200 或其他实例可按下表调整 `$CONFIG_ROOT` 下的两个 pipeline 配置。预算单位为 GiB，是程序准入阈值，不能增加硬件容量，也不保证真实峰值。

| 配置项 | H200 141GB、RAM ≥256 GiB | 96GB GPU、RAM ≥256 GiB |
| --- | ---: | ---: |
| `resources.soft_vram` | 115 | 84 |
| `resources.max_vram` | 125 | 88 |
| `resources.max_ram` | 220 | 220 |
| `resources.gpu_models_max_resident` | 1 | 1 |

保留完整部署指南完整的 `estimated_vram` / `estimated_ram` 及 `keep_alive: false` 配置；它们用于 32B / 完整图像管线估算。96GB GPU 的可用显存需要通过 `nvidia-smi` 确认，首次先跑基础阶段，再开启生成；出现 OOM 时减少输入分辨率和上下文并重新测量。80GB 卡不能直接复用这里 80GiB 的 VLM 估算而期望始终在 GPU 上执行。

内存按容器实得配额取值：`max_ram=220` 要求实际配额高于 220 GiB 且留足余量；如果控制台标称“256GB”但实际仅 238 GiB，可将 `max_ram` 从 220 调到 200，并据实检查加载余量。128GB 内存实例不是本方案的稳妥默认选择，不能简单把 `max_ram` 写大来绕过限制。

显存需求看最大的同时活跃模型及中间张量，**不是把 171.43 GiB 磁盘权重相加后要求同等 GPU 显存**。代码当前让模型分阶段加载与释放，磁盘则必须能保存六个模型。

## 5. 启动及 AutoDL 运维要点

先按照完整部署指南生成 `$CONFIG_ROOT` 下的三个部署 YAML，上传真实地图到 `/root/autodl-tmp/game-reconstruction/input/map.png`。

```bash
source /root/autodl-tmp/game-reconstruction/env.sh
source "$DEPLOY_ROOT/venv/bin/activate"
cd "$SOURCE_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 首先验收 VLM + DINO + SAM，输出在数据盘。
python main.py --input "$INPUT_ROOT/map.png" --output "$OUTPUT_ROOT/base-001" \
  --real --device cuda --offline --max-rounds 1 \
  --models-config "$CONFIG_ROOT/models.deploy.yaml" \
  --config "$CONFIG_ROOT/pipeline.deploy-base.yaml"

# 基础验收通过后，再执行完整分层、补全、高清。
python main.py --input "$INPUT_ROOT/map.png" --output "$OUTPUT_ROOT/full-001" \
  --real --device cuda --offline \
  --models-config "$CONFIG_ROOT/models.deploy.yaml" \
  --config "$CONFIG_ROOT/pipeline.deploy.yaml"
```

- 下载大模型可以先使用平台无卡模式；该模式资源很少，适合顺序下载，不适合载入 Qwen 模型、GPU 验收或多任务安装。[A5]
- 普通容器关机保留数据，但数据盘不包含在保存镜像中；释放实例和长期闲置的数据保留规则应以官方说明核对。重要产物和版本记录备份到本地或文件存储。[A1][A6]
- 扩容数据盘的费用不会仅因为 GPU 关机就停止；收费按平台当前规则与控制台明细核对，不在本指南固定报价。[A3]
- 每次任务使用新输出目录，按需要管理历史产物。不要为了腾空间误删模型唯一副本或哈希清单。

## 6. 核验来源与验收边界

- [A1 AutoDL：配置环境与目录](https://www.autodl.com/docs/env/)
- [A2 AutoDL：GPU、CPU、RAM 选型](https://www.autodl.com/docs/gpu/)
- [A3 AutoDL：本地数据盘与扩容](https://www.autodl.com/docs/local_disk/)
- [A4 AutoDL：容器实例 Pro](https://www.autodl.com/docs/instance_pro/)
- [A5 AutoDL：无卡模式](https://www.autodl.com/docs/save_money/)
- [A6 AutoDL：实例数据保留](https://www.autodl.com/docs/instance_data/)
- [NVIDIA：H200 显存规格](https://www.nvidia.com/en-au/data-center/h200/)
- [NVIDIA：GPU 型号与显存](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)

171.43 GiB 是此前核验的模型下载总量。环境、缓存、输出和硬件档位是部署规划值；真实用量需要在选定 AutoDL 实例通过 `df`、容器配额、`nvidia-smi` 和项目性能报告复核。本次仅编写与验证指南，不代表已完成云端部署或模型质量验收。
