# AutoDL 部署命令：GitHub 源码与 ModelScope 官方非量化模型

命令更新：2026-10-01；模型容量核对：2026-09-30。适用源码基线：`07ac9cc63968b7e94ef6275852c486879b707c83`。

AutoDL 的实例选择、250GB 最低 / 300GB 推荐数据盘规划、目录布局和启动步骤见 [AutoDL 部署配置](AUTODL.md)（2026-10-01 补充）。

本文命令统一在 AutoDL 普通容器的 Bash 中执行，所有本次部署创建的源码、环境、配置、模型、缓存、临时文件、输入与输出均位于 `/root/autodl-tmp/game-reconstruction`。镜像已有的系统 Python、系统工具和宿主机驱动继续使用平台提供的版本。源码从 [GitHub](https://github.com/1679712850/game-image-reconstruction-pipeline) 获取，权重通过 ModelScope 下载到本地。Qwen 使用官方 BF16 权重；DINO、SAM、Real-ESRGAN 保持原始权重及当前适配器精度。不使用 INT8、INT4、AWQ、GPTQ、GGUF、FP8、量化加载器或第三方微调模型。BF16 是正常的浮点推理精度，不是本文所排除的低比特量化。

## 1. 先明确“最好模型”与当前代码边界

**当前仓库不能直接部署所有最新旗舰模型。** 选用大模型也不能保证在所有游戏场景上效果最好，最终需要用自己的标注集比较召回率、分割和补全质量。以下把官方高规格目标与当前可运行方案分开说明，避免把兼容版写成最新最强版。

| 能力 | 官方高规格目标 / 升级候选 | 当前源码可直接配置的方案 | 差异与限制 |
| --- | --- | --- | --- |
| 场景理解与视觉 QA | Qwen3-VL-235B-A22B-Instruct | **Qwen3-VL-32B-Instruct**；另可选 Qwen2.5-VL-72B-Instruct | 235B 是 MoE，`qwen3_vl_moe` 被当前验证器拒绝；当前加载器只允许 `qwen3_vl` / `qwen2_5_vl`，并调用单设备 `.to(device)`。32B 为 Qwen3-VL 高规格稠密版本，不宣称优于所有 72B 任务 |
| 开放词汇检测 | 官方 Grounding DINO 系列；新架构须另行适配 | **IDEA-Research/grounding-dino-base** | 使用 Transformers 兼容的官方 Base，替换项目默认 Tiny；不能只换路径就接入 DINO-X / 新检测 API |
| 提示分割 | Meta SAM 3 | **SAM 2.1 Hiera Large** | 当前使用 `sam2.build_sam2`，Large 是 SAM 2.1 系列最大规格；SAM 3 不是可直接替换的 checkpoint |
| RGBA 图层分解 | Qwen-Image-Layered | **Qwen/Qwen-Image-Layered** | 完整 Diffusers 管线，包含 transformer、text_encoder、VAE 等 |
| 对象编辑与遮挡补全 | Qwen-Image-Edit-2511 | **Qwen/Qwen-Image-Edit** | 2511 的 `model_index.json` 声明 `QwenImageEditPlusPipeline`，当前代码固定 `QwenImageEditPipeline`；原版是兼容方案，不能称为最新编辑模型 |
| 纹理超分 | 作者原版 RealESRGAN_x4plus | **RealESRGAN_x4plus.pth** | 当前 RRDBNet 是 23-block x4plus 架构；不是所有名为 Real-ESRGAN 的权重都兼容 |

第 2–8 节为**不改业务代码的完整高规格兼容部署**。第 9 节给出旗舰模型下载与扩容方案；如果必须所有环节都使用上述最新官方旗舰，需要先完成该节列出的代码改造，不能把“成功下载”当成“流水线已支持”。

本项目是 CLI 批处理程序，尚无 HTTP 服务、Web UI、鉴权或队列。默认一次运行处理一张地图，推荐每张 GPU 同时只运行一个任务。

## 2. 硬件与存储规划

### 2.1 硬件

以下显存是容量规划估算，**不是本项目实测峰值**。输入分辨率、图像数、token 数、生成图层数、候选数量及框架版本都会影响峰值。表中运行按 batch=1、模型串行切换计算。

| 部署档位 | GPU | CPU / 内存 | 磁盘 | 说明 |
| --- | --- | --- | --- | --- |
| Mock 环境自检 | 无 GPU | 4–8 核，16 GiB RAM | 20 GiB 可用 | 验证代码、导出及坐标，不验证任何真实模型 |
| 32B 兼容方案试部署 | 单卡 80 GB，例如 A100/H100 80GB | 16 核，256 GiB RAM | 新空数据盘推荐 300GB | 32B 原始权重约 62 GiB，长上下文可能 OOM；需要实测，不能承诺全规格通过 |
| **32B 完整兼容方案推荐** | **单卡 96GB 起；H200 141GB 余量更充裕** | **24–32 核，256 GiB RAM** | **少量测试推荐 300GB，长期产物按需扩容** | 有空间容纳 VLM / 图像生成激活和切换开销；仍需逐阶段验收 |
| 旧代 72B 稠密可选方案 | 单卡约 180 GB 级或更高，如 B200 | 32 核，512 GiB RAM | ≥600 GiB 可用，建议 1 TB NVMe | BF16 权重本身约 137 GiB；H200 141GB 余量不足。当前代码不能用两张 80GB 自动拼接 |
| 最新旗舰目标，需改造 | VLM 例如 8×80GB，另配 ≥96GB 生成 GPU | 48–64 核，512 GiB–1 TiB RAM | 建议 2 TB NVMe，≥1.2 TiB 可用 | 235B BF16 权重约 439 GiB，需要多卡分片；MoE 激活 22B 不等于仅加载 22B |

推荐 Ubuntu 24.04 LTS、Python 3.12、支持 BF16 的 NVIDIA GPU、与 PyTorch CUDA wheel 匹配的驱动。下文采用 CUDA 12.8 wheel；驱动建议 R570 或更高，并满足具体 GPU 的驱动要求。`nvidia-smi` 显示的 CUDA 版本是驱动支持上限，不等于已安装 CUDA Toolkit。

24/48GB 单卡不能按下文的完整加载方式容纳 32B 和 Qwen 图像完整管线。资源配置的 `max_vram` 只是调度预算，不会压缩权重，也不会开启 tensor parallel。CPU 回退可能很慢；macOS/MPS 不作为本次高规格非量化部署目标。

### 2.2 模型实际下载容量

下表来自核对日 ModelScope 文件列表的 `Size` 汇总，统一使用 **GiB = bytes / 1024³**。下载过滤方式与第 5 节一致；版本变化后以实际文件清单为准。

| 模型 | 下载内容 | 大小 |
| --- | --- | ---: |
| Qwen3-VL-32B-Instruct | 完整 Transformers 目录 | 62.14 GiB |
| grounding-dino-base | 完整目录，排除重复的 `pytorch_model.bin`，保留 safetensors | 0.87 GiB |
| sam2.1-hiera-large | 仅官方 SAM 2 原生 `sam2.1_hiera_large.pt` | 0.84 GiB |
| Qwen-Image-Layered | 完整 Diffusers 目录 | 53.76 GiB |
| Qwen-Image-Edit | 完整 Diffusers 目录 | 53.76 GiB |
| RealESRGAN_x4plus | 仅指定 `.pth` | 0.062 GiB / 63.94 MiB |
| **完整兼容方案合计** | 一份模型，不重复备份 | **171.43 GiB，约 184.07 GB** |

Qwen 图像模型每份约包含 38.05 GiB transformer、15.45 GiB text_encoder、0.24 GiB VAE 及少量配置。不能只下载 20B 主干就认为整个管线已安装；两个模型的编码器也不要未经验证就共享或覆盖。

以下是保留多批产物和模型版本的保守预算，并非启动最低需求。单套模型、少量测试的精简预算见 [AutoDL 存储规划](AUTODL.md#2-存储需要多少)，推荐先按 300GB 数据盘规划，再依据实测扩容。

长期使用时可另行预留：

- Python 虚拟环境、CUDA wheel、pip 缓存：约 15–30 GiB；系统与工具另留 30–50 GiB。
- 下载临时文件、重试和模型版本切换：预留 50–180 GiB；保存第二套版本时，模型空间近似翻倍。
- 输入、mask、候选、RGBA、PSD、诊断及推理缓存：先预留 100–200 GiB，按生产批量增加。默认输出缓存不会自动轮转。
- 4096×4096 RGBA 的未压缩数据约 64 MiB；300 张同尺寸单通道 mask 已约 4.69 GiB。PNG 压缩率、局部裁图、重试数和 PSD 临时数据使单场景大小差异很大，不能固定估成几 MB。
- 仅 VLM + DINO + SAM 的基础模型合计约 63.85 GiB；开启完整功能后才需要 171.43 GiB，但本文下载步骤按完整方案准备。

按净下载 100 Mbit/s 估算，184 GB 模型的理想传输时间约 4.1 小时；1 Gbit/s 约 25 分钟。镜像限速、分片检查和重试会增加时间。

## 3. AutoDL 数据盘初始化与 GitHub 源码下载

先在 AutoDL 控制台扩容数据盘（少量测试推荐总容量 300GB），选择已提供 Python 3.11/3.12、venv、Git、curl 与编译工具的镜像。使用普通容器的 `/root/autodl-tmp`；不要在本地电脑执行下列初始化命令。无需 `sudo apt`、修改系统 Python 或安装宿主机驱动。

目录统一如下：

```text
/root/autodl-tmp/game-reconstruction/
├── env.sh              # 每次打开终端时加载
├── source/             # GitHub 源码和仓库内测试产物
├── venv/               # 本项目 Python 环境与所有新增依赖
├── config/             # 生成的部署 YAML
├── models/             # ModelScope 完整模型目录
├── cache/              # SDK、PyTorch、CUDA、Triton 等缓存
├── tmp/                # pip 构建与 Python 临时文件
├── input/              # 上传的地图及测试图
├── outputs/            # 分阶段结果、运行日志及每任务推理缓存
└── deployment-records/ # commit、依赖清单和权重核验文件
```

### 3.1 初始化目录与持久化环境变量

```bash
# 只检查平台已挂载的数据盘；不要把未挂载路径当成空白数据盘。
test -d /root/autodl-tmp || { echo '未找到 AutoDL 数据盘'; exit 1; }
df -h / /root/autodl-tmp
python --version
command -v git curl gcc g++
python -c 'import sys, venv; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "请选择 Python 3.11/3.12 镜像"'

mkdir -p /root/autodl-tmp/game-reconstruction
cat > /root/autodl-tmp/game-reconstruction/env.sh <<'ENV'
export DEPLOY_ROOT=/root/autodl-tmp/game-reconstruction
export SOURCE_ROOT="$DEPLOY_ROOT/source"
export MODEL_ROOT="$DEPLOY_ROOT/models"
export CONFIG_ROOT="$DEPLOY_ROOT/config"
export INPUT_ROOT="$DEPLOY_ROOT/input"
export OUTPUT_ROOT="$DEPLOY_ROOT/outputs"
export RECORD_ROOT="$DEPLOY_ROOT/deployment-records"
export XDG_CACHE_HOME="$DEPLOY_ROOT/cache"
export XDG_CONFIG_HOME="$DEPLOY_ROOT/cache/xdg-config"
export XDG_DATA_HOME="$DEPLOY_ROOT/cache/xdg-data"
export XDG_STATE_HOME="$DEPLOY_ROOT/cache/xdg-state"
export PIP_CACHE_DIR="$DEPLOY_ROOT/cache/pip"
export PIP_NO_CACHE_DIR=1
export PYTHONUSERBASE="$DEPLOY_ROOT/cache/python-user"
export HF_HOME="$DEPLOY_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export MODELSCOPE_CACHE="$DEPLOY_ROOT/cache/modelscope"
export TORCH_HOME="$DEPLOY_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$DEPLOY_ROOT/cache/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$DEPLOY_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$DEPLOY_ROOT/cache/triton"
export CUDA_CACHE_PATH="$DEPLOY_ROOT/cache/cuda"
export MPLCONFIGDIR="$DEPLOY_ROOT/cache/matplotlib"
export TMPDIR="$DEPLOY_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export CUDA_VISIBLE_DEVICES=0
ENV
source /root/autodl-tmp/game-reconstruction/env.sh
mkdir -p "$MODEL_ROOT" "$CONFIG_ROOT" "$INPUT_ROOT" "$OUTPUT_ROOT" "$RECORD_ROOT" "$TMPDIR" \
  "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$PIP_CACHE_DIR" "$PYTHONUSERBASE" \
  "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$HF_DATASETS_CACHE" "$MODELSCOPE_CACHE" \
  "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"
```

`env.sh` 只设置路径与缓存，不设置离线模式，下载步骤可以正常联网。不要修改 `HOME` 或通过系统级软链接重定向整个用户目录。首次初始化后，所有安装命令均使用数据盘上的 venv。

### 3.2 获取源码并创建数据盘虚拟环境

```bash
source /root/autodl-tmp/game-reconstruction/env.sh
# 首次部署执行；如果 source 已存在，应进入原目录核对，不覆盖已有工作。
git clone https://github.com/1679712850/game-image-reconstruction-pipeline.git "$SOURCE_ROOT"
cd "$SOURCE_ROOT"

# 锁定已审计的源码版本；如远端尚未包含此 commit，需维护者先推送。
git checkout --detach 07ac9cc63968b7e94ef6275852c486879b707c83
git rev-parse HEAD > "$RECORD_ROOT/source-commit.txt"

python -m venv "$DEPLOY_ROOT/venv"
source "$DEPLOY_ROOT/venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
```

私有仓库先完成 GitHub 登录或 SSH 授权。若选择其他源码版本，记录实际 commit 并重新核对适配器。不要把本机已有 `.venv`、模型或输出重复复制进 `source`。

### 3.3 每次新开终端恢复环境

以下三行也是后续各节命令的前置步骤：

```bash
source /root/autodl-tmp/game-reconstruction/env.sh
source "$DEPLOY_ROOT/venv/bin/activate"
cd "$SOURCE_ROOT"
```

## 4. 安装运行环境

```bash
nvidia-smi

# 固定相互匹配的 PyTorch / torchvision；适用于支持 CUDA 12.8 的驱动。
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 防止后续依赖解析意外替换 CUDA wheel。
python -m pip freeze | grep -E '^(torch|torchvision)==' > "$RECORD_ROOT/torch-constraints.txt"

python -m pip install -c "$RECORD_ROOT/torch-constraints.txt" \
  -r requirements.txt -r requirements-vision.txt -r requirements-vlm.txt \
  -r requirements-qwen.txt -r requirements-upscale.txt 'modelscope>=1.30,<2'

# 不编译 SAM 的可选 CUDA 扩展，模型推理仍在 GPU；不要求安装 nvcc。
# 官方 SAM 2 源码 commit 已固定在 requirements-sam2.txt。
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation \
  -c "$RECORD_ROOT/torch-constraints.txt" -r requirements-sam2.txt

python -m pip check
python -m pip freeze > "$RECORD_ROOT/requirements-resolved.txt"
```

项目当前要求 `transformers>=5.9,<6`、`diffusers>=0.37,<1`。这里固定了 PyTorch 配对，其余按仓库范围解析，并保存实际版本；这不是一份已经在 CUDA 服务器验证过的全依赖锁文件。若官方 Python 索引没有所需版本，应检查软件源同步情况和源码版本，勿用旧依赖强行安装。

本地视觉分析和 QA 不需要 API key，也不需要 `requirements-llm.txt`；仅显式使用外部 HTTP Scene QA 时才安装后者。

```bash
python - <<'PY'
import torch, torchvision, transformers, diffusers
from diffusers import QwenImageLayeredPipeline, QwenImageEditPipeline
from sam2.build_sam import build_sam2
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('transformers:', transformers.__version__, 'diffusers:', diffusers.__version__)
assert torch.cuda.is_available(), '未找到 CUDA GPU，请检查驱动和 wheel'
assert torch.cuda.is_bf16_supported(), '本方案要求 BF16 GPU'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY
```

## 5. 使用 ModelScope 下载模型

### 5.1 来源与官方权重约束

Qwen 使用 `Qwen` 仓库。DINO 和 SAM 使用与作者发布名称一致的 ModelScope 仓库；国内 Hub 上的同名空间本身不是发布者身份的充分证明，严格来源审计应同时对照上游发布记录或权重哈希。

Real-ESRGAN 使用 [`lllyasviel/Annotators`](https://modelscope.cn/models/lllyasviel/Annotators) 中的 `RealESRGAN_x4plus.pth`：它是**镜像仓库，不是 Real-ESRGAN 作者账号**。下文下载后与作者 [GitHub Release](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0) 比较 SHA-256，只有完全一致才用于部署。这样权重主要由 ModelScope 提供，并保持作者原始模型。

不要使用 `AI-ModelScope/Real-ESRGAN` 的 `RealESRGAN_x4.pth` 代替：该模型卡说明它在自定义数据集上训练，并非这里要求的原版 x4plus。也不要把其他权重改名为 x4plus 来绕过校验。

### 5.2 一次性下载完整目录

模型固定放在 `/root/autodl-tmp/game-reconstruction/models`，不会写入源码仓库或系统盘。先按第 3.3 节恢复环境。

```bash
source /root/autodl-tmp/game-reconstruction/env.sh
# 若在同一终端刚跑过离线推理，下载前取消离线标记。
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$MODEL_ROOT"
df -h "$MODEL_ROOT"

python - <<'PY'
import os
from pathlib import Path
from modelscope import snapshot_download

root = Path(os.environ['MODEL_ROOT']).resolve()
models = [
    ('Qwen/Qwen3-VL-32B-Instruct', 'Qwen3-VL-32B-Instruct', {}),
    ('IDEA-Research/grounding-dino-base', 'grounding-dino-base',
     {'ignore_file_pattern': ['pytorch_model.bin']}),
    ('facebook/sam2.1-hiera-large', 'sam2.1-hiera-large',
     {'allow_file_pattern': ['sam2.1_hiera_large.pt']}),
    ('Qwen/Qwen-Image-Layered', 'Qwen-Image-Layered', {}),
    ('Qwen/Qwen-Image-Edit', 'Qwen-Image-Edit', {}),
    ('lllyasviel/Annotators', 'Real-ESRGAN',
     {'allow_file_pattern': ['RealESRGAN_x4plus.pth']}),
]
for repo, directory, filters in models:
    print('Downloading:', repo, flush=True)
    snapshot_download(repo, revision='master', local_dir=str(root / directory), **filters)
print('Downloaded to:', root)
PY

du -sh "$MODEL_ROOT"/*
```

ModelScope 此处使用 `master`，不要照抄 Hugging Face 的 `main`。`master` 会变化；正式上线应固定发布 tag/支持的 revision，并保存下载文件哈希。下载中断后重跑同一命令复用 SDK 已有文件，不要先清空整个目录。Qwen 必须保留完整 tokenizer、processor、配置和所有 safetensors 分片。

### 5.3 原版权重验证与离线交付清单

以下仅额外下载约 64 MiB 的作者原版作核验，不把未知第三方训练权重当成官方模型。编写指南时访问作者 Release 的权重下载超时，尚未完成本地两份文件的哈希比对；部署时必须让下面的断言通过，才能确认此镜像与作者文件一致。ModelScope 文件列表公布的该镜像 SHA-256 为 `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`，它本身不能代替上游核验。

```bash
curl --fail --location --retry 3 \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  --output "$RECORD_ROOT/RealESRGAN_x4plus.official.pth"

python - <<'PY'
import hashlib, os
from pathlib import Path

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

mirror = Path(os.environ['MODEL_ROOT']) / 'Real-ESRGAN/RealESRGAN_x4plus.pth'
official = Path(os.environ['RECORD_ROOT']) / 'RealESRGAN_x4plus.official.pth'
a, b = digest(mirror), digest(official)
assert a == b, f'原版核验失败：ModelScope={a}, GitHub={b}'
print('Official x4plus SHA256:', a)
PY

# 全量读取约 171 GiB，可能需要数分钟；作为发布与迁移验证清单。
(cd "$MODEL_ROOT" && find . -type f ! -name SHA256SUMS -print0 | \
  sort -z | xargs -0 sha256sum > SHA256SUMS)
```

离线拷贝时同时保存源码 commit、`requirements-resolved.txt`、模型目录及 `SHA256SUMS`，目标机器执行 `(cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS)`。完全隔离网络的服务器还需在同架构 Linux/Python 环境预先准备数据盘上的 `"$DEPLOY_ROOT/wheelhouse"` 依赖目录，包含由固定 SAM 2 源码构建的 wheel；只搬模型不足以完成离线安装。

## 6. 生成与当前加载器匹配的部署配置

下面从现有配置生成独立文件，保留类别与检测策略，写入绝对模型路径。示例默认预算适用于 **96GB 单卡 / 256 GiB RAM**；H200 141GB 可将脚本中的 `soft_vram=84, max_vram=88` 改为 `115, 125`；其他硬件应根据 `nvidia-smi` 与实测调整，预算单位均为 GiB。

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml
from app.models import load_models
from app.config import load_config

root = Path(os.environ['MODEL_ROOT']).resolve()
source = Path(os.environ['SOURCE_ROOT']).resolve()
config_dir = Path(os.environ['CONFIG_ROOT']).resolve()
config_dir.mkdir(parents=True, exist_ok=True)
models = yaml.safe_load((source / 'config/models.yaml').read_text())
models.update(device='cuda', local_files_only=True, cache_dir=str(root))
models['grounding']['model_id'] = str(root / 'grounding-dino-base')
models['sam'].update(
    model_config_name='configs/sam2.1/sam2.1_hiera_l.yaml',
    checkpoint=str(root / 'sam2.1-hiera-large/sam2.1_hiera_large.pt'),
    repo_id='facebook/sam2.1-hiera-large', filename='sam2.1_hiera_large.pt',
)
models['qwen_vl'].update(
    model_path=str(root / 'Qwen3-VL-32B-Instruct'), dtype='bfloat16',
    image_long_edge=1536, max_images=4, max_input_tokens=16384, max_new_tokens=2048,
)
models['scene_reviewer']['provider'] = 'local'
models['qwen_layered'].update(model_path=str(root / 'Qwen-Image-Layered'), dtype='bfloat16')
models['qwen_image_edit'].update(model_path=str(root / 'Qwen-Image-Edit'), dtype='bfloat16')
models['upscale'].update(
    backend='real_esrgan', checkpoint=str(root / 'Real-ESRGAN/RealESRGAN_x4plus.pth'),
    tile=256, tile_pad=16,
)
model_file = config_dir / 'models.deploy.yaml'
model_file.write_text(yaml.safe_dump(models, allow_unicode=True, sort_keys=False))

pipeline = yaml.safe_load((source / 'config/pipeline.yaml').read_text())
# 每个任务的推理缓存随该任务保存在数据盘 outputs/<任务>/cache。
pipeline['cache']['directory'] = None
pipeline['mock'] = False
pipeline['scene_loop']['reviewer'] = 'llm'
pipeline['layer_decomposition']['enabled'] = True
pipeline['object_completion']['enabled'] = True
pipeline['upscale'].update(enabled=True, required=True)
pipeline['resources'].update(
    soft_vram=84, max_vram=88, max_ram=220, gpu_models_max_resident=1,
    keep_alive={'sam': False, 'grounding': False, 'qwen_vl': False,
                'image_edit': False, 'layered': False},
    estimated_vram={'qwen_vl': 80, 'grounding': 4, 'sam': 6,
                    'image_edit': 75, 'layered': 80, 'upscale': 3},
    estimated_ram={'qwen_vl': 100, 'grounding': 6, 'sam': 8,
                   'image_edit': 90, 'layered': 90, 'upscale': 4},
)
# estimates 是调度准入估算，不是通过设置即可保证的真实上限。
full_file = config_dir / 'pipeline.deploy.yaml'
full_file.write_text(yaml.safe_dump(pipeline, allow_unicode=True, sort_keys=False))

pipeline['layer_decomposition']['enabled'] = False
pipeline['object_completion']['enabled'] = False
pipeline['upscale'].update(enabled=False, required=False)
base_file = config_dir / 'pipeline.deploy-base.yaml'
base_file.write_text(yaml.safe_dump(pipeline, allow_unicode=True, sort_keys=False))

load_models(model_file)
load_config(full_file)
load_config(base_file)
print('Generated and schema-validated:', model_file, full_file, base_file)
PY
```

注意事项：

- `sam.checkpoint` 必须实际存在，且 Large 权重要对应 `hiera_l.yaml`，不能沿用 Tiny 配置。
- `grounding.model_id` 在这里填**绝对本地目录**。ModelScope 下载的文件不会自动出现在 Hugging Face cache；继续填远程 repo ID 可能触发其他下载或离线报错。
- Qwen 的 `model_path` 是完整本地目录。`MODEL_ROOT` 是本指南脚本变量，不会自动替换 YAML；生成脚本已经写入绝对路径。
- 不设置 `quantized_model_path`。当前适配器没有多卡 `device_map` 或逐层 offload；`gpu_models_max_resident: 1` 只限制不同模型的驻留数量。
- 不沿用默认 `qwen_vl: 18`、`image_edit: 18` 等显存估算；它们不足以描述本方案。
- `estimated_ram` 是加载准入估算，另有进程 RSS 和系统可用内存检查。RAM 配得太小会在加载前报告 `CPU RAM budget exhausted`。
- 单卡 96GB 可从 `soft_vram: 84`、`max_vram: 88` 试起，并实际测量余量。80GB 卡可能无法容纳上述 80GiB 估算，从而回退 CPU；必须先缩小输入并完成实测，不能只虚增预算。

## 7. 启动与分阶段验收

### 7.1 无权重冒烟检查

```bash
python -m app.demo --output "$INPUT_ROOT/deploy-smoke.png"
python main.py --input "$INPUT_ROOT/deploy-smoke.png" --output "$OUTPUT_ROOT/deploy-mock" --mock
python -m unittest discover -s tests -v
```

这一步不证明真实模型兼容性、语义召回率或补全效果。

### 7.2 基础真实流水线

先将一张实际游戏地图上传到 `/root/autodl-tmp/game-reconstruction/input/map.png`。首次建议使用约 1024–1536 像素地图，先把 VLM、DINO、SAM、QA 与导出跑通。

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python main.py --input "$INPUT_ROOT/map.png" --output "$OUTPUT_ROOT/deploy-base" \
  --real --device cuda --offline \
  --models-config "$CONFIG_ROOT/models.deploy.yaml" \
  --config "$CONFIG_ROOT/pipeline.deploy-base.yaml" \
  --max-rounds 1
```

`--offline` 只限制权重加载；此配置还显式设置 `scene_reviewer.provider: local`，从而不需要外部视觉 API。`--scene-reviewer rules` 仅切换 QA，**不会取消真实场景分析对 Qwen-VL 的需求**。

### 7.3 完整分层、补全与高清

```bash
python main.py --input "$INPUT_ROOT/map.png" --output "$OUTPUT_ROOT/deploy-full" \
  --real --device cuda --offline \
  --models-config "$CONFIG_ROOT/models.deploy.yaml" \
  --config "$CONFIG_ROOT/pipeline.deploy.yaml"
```

完整配置开启图层生成、对象补全、神经超分，高清为 required；基础配置则关闭这些阶段。无需添加不存在的 `--upscale` / `--complete-objects` 参数，相关开关来自 YAML。

单次任务可能执行多轮检测、多个补全候选及视觉 QA。完整模型串行卸载会增加加载时间，这是当前实现的资源取舍。先前输出目录可能含旧文件；每次验收使用新的输出目录，消费者以本次 `scene.json` 为准。

另一个终端可查看显存：

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv --loop=2
```

以下相对文件均位于本次 `$OUTPUT_ROOT/<任务名>/` 中；PSD 临时文件和默认推理缓存也在该任务目录。验收至少检查：

1. `debug/run.json` 的模型绝对路径、后端、启用阶段；确认无 Mock / Lanczos 替代。
2. `performance_report.json`、`timeline.html` 的模型加载、CPU 回退、OOM 重试、RAM/VRAM 峰值；同时观察系统监控，采样峰值可能漏掉瞬时尖峰。
3. `metadata/summary.json`、`diagnostics/report.html` 的语义覆盖、未分配区域、待人工复核数量；不要用 residual 带来的高重建相似度代替语义质量。
4. `scene.json`、`assets/`、`assets_hd/`、`scene.psd` 和重建预览；确认高清 QA、透明边缘、pivot、遮挡顺序及图层坐标符合预期。
5. 至少抽检树木、山体、建筑、半透明特效和严重遮挡对象。生成结果可能因 QA 被拒绝，流程跑完不等于全部资产可用。

## 8. 常见部署问题

| 现象 | 检查 / 处理 |
| --- | --- |
| Qwen 本地模型不支持 / `model_type` 错误 | 当前只接受 Qwen2.5-VL / Qwen3-VL 稠密 Instruct，235B MoE 需要改造 |
| 编辑权重能读但管线报错 | 检查 `model_index.json`；2511 使用 Plus，不是当前原版接口 |
| 模型明明下载了仍联网 | 确认 YAML 使用本地绝对路径，SAM checkpoint 已填写，且启用 `--offline` |
| CPU 占满、GPU 空闲 | 检查资源预算与 CPU 回退记录；当前 24/48GB 不能完整加载这些大模型 |
| CUDA OOM | 先排除其他任务；减少图像边长、上下文、layered 分辨率/层数或超分 tile。权重本体装不下时必须增加单卡显存或改造分片，不采用量化 |
| `CPU RAM budget exhausted` | 检查系统可用 RAM、RSS、`estimated_ram` 和 `max_ram`；预算不代表机器真实可用内存 |
| SAM 安装失败 | 先装 torch、setuptools/wheel，再执行禁用可选 CUDA 扩展的 SAM 安装命令 |
| x4plus SHA-256 不一致 | 停止该阶段，检查镜像和官方文件是否完整；不得通过改名或去掉校验继续 |
| 输出很大 / 磁盘耗尽 | 查看 `du -sh "$OUTPUT_ROOT"/*` 和模型目录；备份需要的产物后按运行目录管理保留策略，不要误删唯一权重或验收证据 |

## 9. 所有环节使用旗舰模型时的升级部署方案

### 9.1 先规划额外空间，再下载官方完整权重

| 目标 | ModelScope 仓库 | 本次文件列表容量 | 当前可接入性 |
| --- | --- | ---: | --- |
| Qwen3-VL 旗舰 Instruct | `Qwen/Qwen3-VL-235B-A22B-Instruct` | 438.98 GiB | MoE 校验、多卡加载与生命周期尚待实现 |
| 新版图像编辑 | `Qwen/Qwen-Image-Edit-2511` | 53.76 GiB | 需要 Plus 管线适配与输入输出验收 |
| SAM 3 | `facebook/sam3` | 原生 `sam3.pt` 约 3.21 GiB；全仓库约 6.42 GiB | 需要独立 SAM 3 适配器、官方依赖及许可检查 |
| 旧代大稠密 VLM，非最新旗舰 | `Qwen/Qwen2.5-VL-72B-Instruct` | 136.75 GiB | 现有模型类型允许，但当前要求单卡完整加载 |

235B + Edit-2511 + Layered + SAM 3 原生权重 + DINO Base + x4plus，模型本体合计约 **550.64 GiB**。如果同时保留前文兼容部署，Layered、DINO、x4plus 可复用，合计约 **667.38 GiB**；仍需留缓存、环境和输出空间。不要按照“激活参数 22B × 2 bytes”计算 MoE 存储。

```bash
# 先完成第 3.3 节环境恢复。这是升级模型下载，不会自动使当前 main.py 支持它们。
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python - <<'PY'
import os
from pathlib import Path
from modelscope import snapshot_download
root = Path(os.environ['MODEL_ROOT']).resolve()
for repo, filters in [
    ('Qwen/Qwen3-VL-235B-A22B-Instruct', {}),
    ('Qwen/Qwen-Image-Edit-2511', {}),
    ('facebook/sam3', {'allow_file_pattern': ['sam3.pt', 'LICENSE', 'README.md']}),
]:
    snapshot_download(repo, revision='master',
                      local_dir=str(root / repo.split('/')[-1]), **filters)
PY
```

### 9.2 旗舰接入所需工程工作

1. **VLM 多卡与 MoE**：扩展 `services/qwen_vl_backend.py` 的模型类型检查，采用官方 Transformers 支持的分片加载或独立推理服务；取消对分片模型的整体 `.to(device)`，正确放置输入；修改 `ModelManager` 对 `.to('cpu')`、单卡快照和卸载的假设。若改用服务端，还要接入场景分析，当前 API 配置仅覆盖 Scene QA。
2. **Edit Plus**：依据 `model_index.json` 选择 `QwenImageEditPlusPipeline`，核对单图/多图输入、尺寸及 RGB 返回契约，保留当前 SAM 恢复 alpha、候选 QA 和坐标回接逻辑。
3. **SAM 3**：安装官方独立依赖，新增符合 `SAMService` 提示框、点提示、多候选、局部坐标及 mask 返回契约的适配器；不能把 `sam3.pt` 交给 `build_sam2`。
4. **独立验收**：分别测量显存/内存、断网推理、资源释放和真实场景质量，再进行完整 graph 验收与生产版本锁定。

本仓库目前没有实现这些改造，因此本文不提供冒充“开箱即用”的旗舰 `main.py` 启动命令。若暂时不改代码，应明确使用前述兼容方案；若严格要求最新旗舰，则以上改造属于正式部署前置条件。

## 10. 官方资料与核验记录

- [项目源码](https://github.com/1679712850/game-image-reconstruction-pipeline)
- [PyTorch 官方版本与 CUDA wheel 配对](https://pytorch.org/get-started/previous-versions/)
- [ModelScope 下载 SDK](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py)
- [Qwen3-VL 官方源码](https://github.com/QwenLM/Qwen3-VL)、[32B ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct)、[235B ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Instruct)
- [Qwen2.5-VL-72B ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct)
- [Grounding DINO 官方源码](https://github.com/IDEA-Research/GroundingDINO)、[Base ModelScope](https://modelscope.cn/models/IDEA-Research/grounding-dino-base)
- [Meta SAM 2 官方源码](https://github.com/facebookresearch/sam2)、[SAM 2.1 Large ModelScope](https://modelscope.cn/models/facebook/sam2.1-hiera-large)、[SAM 3 官方源码](https://github.com/facebookresearch/sam3)、[SAM 3 ModelScope](https://modelscope.cn/models/facebook/sam3)
- [Qwen-Image 官方源码](https://github.com/QwenLM/Qwen-Image)、[Layered](https://modelscope.cn/models/Qwen/Qwen-Image-Layered)、[原版 Edit](https://modelscope.cn/models/Qwen/Qwen-Image-Edit)、[Edit-2511](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2511)
- [Real-ESRGAN 作者源码](https://github.com/xinntao/Real-ESRGAN)、[作者原版 x4plus Release](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0)、[ModelScope 镜像](https://modelscope.cn/models/lllyasviel/Annotators)

容量核验接口：`https://modelscope.cn/api/v1/models/{namespace}/{name}/repo/files?Revision=master&Recursive=true`。这些数据是仓库下载字节数，不是显存需求；正式运行应保存所用 revision、权重哈希、软件版本与性能报告。

本指南编写时已核查本地源码接口、ModelScope 文件列表和部分模型配置；未下载完整 Qwen 权重，也未在 NVIDIA 服务器执行上述完整部署。现有 CPU 检测/分割历史验证不能替代本方案的 CUDA、生成模型和完整质量验收。
