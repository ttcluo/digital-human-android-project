#===============================================================================
# 本地Windows环境配置脚本
# 用于配置本地开发环境
# 
# 使用方法: .\setup-local.ps1
#===============================================================================

# 配置
$ErrorActionPreference = "Stop"

# 颜色定义
function Write-LogInfo { Write-Host "[INFO] $args" -ForegroundColor Green }
function Write-LogWarn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-LogError { Write-Host "[ERROR] $args" -ForegroundColor Red }

Write-LogInfo "========================================="
Write-LogInfo "开始配置本地开发环境"
Write-LogInfo "========================================="

# 检查Python
Write-LogInfo "检查Python安装..."
try {
    $pythonVersion = python --version 2>&1
    Write-LogInfo "Python版本: $pythonVersion"
} catch {
    Write-LogError "Python未安装，请先安装Python 3.10+"
    exit 1
}

# 检查CUDA (可选)
Write-LogInfo "检查CUDA..."
try {
    $nvccVersion = nvcc --version 2>&1
    Write-LogInfo "CUDA版本: $nvccVersion"
} catch {
    Write-LogWarn "CUDA未安装，将使用CPU版本PyTorch"
}

# 创建虚拟环境
Write-LogInfo "创建Python虚拟环境..."
python -m venv venv

# 激活虚拟环境
Write-LogInfo "激活虚拟环境..."
& .\venv\Scripts\Activate.ps1

# 升级pip
Write-LogInfo "升级pip..."
python -m pip install --upgrade pip

# 安装PyTorch
Write-LogInfo "安装PyTorch..."
try {
    # 尝试安装CUDA版本
    pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
    Write-LogInfo "PyTorch CUDA版本安装成功"
} catch {
    Write-LogWarn "CUDA版本安装失败，尝试安装CPU版本..."
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    Write-LogInfo "PyTorch CPU版本安装成功"
}

# 安装项目依赖
Write-LogInfo "安装项目依赖..."
pip install numpy==1.23.5
pip install opencv-python
pip install transformers
pip install soundfile
pip install librosa
pip install tqdm
pip install pillow
pip install scipy
pip install matplotlib
pip install scikit-learn
pip install pyyaml
pip install tensorboard
pip install onnxruntime

# 安装开发工具
Write-LogInfo "安装开发工具..."
pip install black
pip install flake8
pip install mypy

# 配置Git
Write-LogInfo "配置Git..."
git config --global user.name "Digital Human Developer"
git config --global user.email "dev@digital-human.local"
git config --global core.editor code
git config --global init.defaultBranch main

# 克隆Ultralight-Digital-Human项目
Write-LogInfo "克隆Ultralight-Digital-Human项目..."
if (-Not (Test-Path "Ultralight-Digital-Human")) {
    git clone https://github.com/anliyuan/Ultralight-Digital-Human.git Ultralight-Digital-Human
} else {
    Write-LogWarn "Ultralight-Digital-Human目录已存在"
}

# 创建数据目录
Write-LogInfo "创建数据目录..."
New-Item -ItemType Directory -Force -Path data\raw | Out-Null
New-Item -ItemType Directory -Force -Path data\processed | Out-Null
New-Item -ItemType Directory -Force -Path data\splits | Out-Null
New-Item -ItemType Directory -Force -Path checkpoints\trained_models | Out-Null
New-Item -ItemType Directory -Force -Path checkpoints\quantized_models | Out-Null
New-Item -ItemType Directory -Force -Path logs | Out-Null
New-Item -ItemType Directory -Force -Path outputs | Out-Null

# 停用虚拟环境
deactivate

Write-LogInfo "========================================="
Write-LogInfo "本地开发环境配置完成!"
Write-LogInfo "========================================="
Write-LogInfo "激活虚拟环境: .\venv\Scripts\Activate.ps1"
Write-LogInfo "启动训练: python -m src.training.train --config configs\train_config.yaml"
Write-LogInfo "========================================="
