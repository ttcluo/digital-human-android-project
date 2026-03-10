#!/bin/bash
#===============================================================================
# Ubuntu服务器环境配置脚本
# 用于配置GPU训练环境
# 
# 使用方法: sudo bash setup-server.sh
# 
# 支持的Ubuntu版本: 20.04, 22.04
#===============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
CUDA_VERSION="11.7"
CUDNN_VERSION="8"
PYTHON_VERSION="3.10"
PROJECT_DIR="/opt/digital-human"
LOG_FILE="server-setup.log"

#===============================================================================
# 函数定义
#===============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[WARN] $1" >> "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $1" >> "$LOG_FILE"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
    echo "[STEP] $1" >> "$LOG_FILE"
}

# 检查是否为root用户
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "请使用root用户运行此脚本"
        log_error "使用方法: sudo bash $0"
        exit 1
    fi
}

# 检查Ubuntu版本
check_ubuntu_version() {
    log_step "检查Ubuntu版本..."
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        VERSION="$VERSION_ID"
        
        if [[ "$VERSION" != "20.04" && "$VERSION" != "22.04" && "$VERSION" != "24.04" ]]; then
            log_warn "此脚本已在Ubuntu 20.04/22.04/24.04上测试，其他版本可能需要手动配置"
        fi
        
        log_info "检测到Ubuntu $VERSION"
    else
        log_error "无法检测Ubuntu版本"
        exit 1
    fi
}

# 更新系统包
update_system() {
    log_step "更新系统包..."
    apt update
    apt upgrade -y
    log_info "系统包更新完成"
}

# 安装基础依赖
install_base_dependencies() {
    log_step "安装基础依赖..."
    
    apt install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        vim \
        htop \
        tmux \
        zip \
        unzip \
        software-properties-common \
        gnupg \
        ca-certificates \
        lsb-release \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1-mesa-glx
    
    log_info "基础依赖安装完成"
}

# 安装NVIDIA驱动
install_nvidia_driver() {
    log_step "安装NVIDIA驱动..."
    
    # 添加NVIDIA驱动仓库
    add-apt-repository ppa:graphics-drivers/ppa -y
    
    # 查找推荐的驱动版本
    ubuntu_drivers=$(ubuntu-drivers devices 2>/dev/null | grep "recommended" | awk '{print $3}')
    
    if [ -z "$ubuntu_drivers" ]; then
        # 默认使用535驱动
        ubuntu_drivers="535"
        log_warn "使用默认驱动版本: $ubuntu_drivers"
    fi
    
    log_info "安装NVIDIA驱动 $ubuntu_drivers..."
    apt install -y nvidia-driver-$ubuntu_drivers
    
    log_info "NVIDIA驱动安装完成"
    log_warn "请重启系统使驱动生效"
}

# 安装CUDA Toolkit
install_cuda() {
    log_step "安装CUDA Toolkit $CUDA_VERSION..."
    
    # 下载CUDA安装脚本
    wget -q https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}.0/local_installers/cuda_11.7.0_515.43.04_linux.run
    
    # 安装CUDA
    chmod +x cuda_11.7.0_515.43.04_linux.run
    ./cuda_11.7.0_515.43.04_linux.run --silent --toolkit --override
    
    # 清理安装脚本
    rm -f cuda_11.7.0_515.43.04_linux.run
    
    # 配置环境变量
    echo "" >> /etc/profile
    echo "# CUDA" >> /etc/profile
    echo "export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}" >> /etc/profile
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> /etc/profile
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> /etc/profile
    
    # 立即加载环境变量
    export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    log_info "CUDA Toolkit安装完成"
}

# 安装cuDNN
install_cudnn() {
    log_step "安装cuDNN v${CUDNN_VERSION}..."
    
    # 下载cuDNN（需要NVIDIA账户）
    # 这里使用apt方式安装
    apt install -y libcudnn8 libcudnn8-dev
    
    # 验证安装
    if [ -f "$CUDA_HOME/lib64/libcudnn.so.${CUDNN_VERSION}" ]; then
        log_info "cuDNN安装完成"
    else
        log_warn "cuDNN验证失败，请手动检查"
    fi
}

# 安装Miniconda
install_conda() {
    log_step "安装Miniconda..."
    
    # 下载Miniconda
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    
    # 安装Miniconda
    bash /tmp/miniconda.sh -b -p /opt/miniconda
    
    # 配置环境变量
    echo "" >> /etc/profile
    echo "# Miniconda" >> /etc/profile
    echo "export CONDA_HOME=/opt/miniconda" >> /etc/profile
    echo "export PATH=\$CONDA_HOME/bin:\$PATH" >> /etc/profile
    
    # 立即加载
    export PATH=/opt/miniconda/bin:$PATH
    
    # 初始化conda
    /opt/miniconda/bin/conda init bash
    
    # 清理
    rm -f /tmp/miniconda.sh
    
    log_info "Miniconda安装完成"
}

# 创建Python环境
create_python_env() {
    log_step "创建Python环境..."
    
    # 加载conda
    source /opt/miniconda/etc/profile.d/conda.sh
    
    # 创建环境
    conda create -n digital-human python=${PYTHON_VERSION} -y
    
    # 激活环境
    conda activate digital-human
    
    # 安装PyTorch（GPU版本）
    log_info "安装PyTorch with CUDA ${CUDA_VERSION}..."
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y
    
    # 安装其他依赖
    log_info "安装项目依赖..."
    pip install \
        numpy==1.23.5 \
        opencv-python \
        transformers \
        soundfile \
        librosa \
        tqdm \
        pillow \
        scipy \
        matplotlib \
        scikit-learn \
        tensorboard \
        wandb
    
    # 退出环境
    conda deactivate
    
    log_info "Python环境创建完成"
}

# 安装Docker（可选）
install_docker() {
    log_step "安装Docker..."
    
    # 卸载旧版本
    apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
    
    # 安装依赖
    apt install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # 添加Docker GPG密钥
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # 添加Docker仓库
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # 安装Docker
    apt update
    apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # 启动Docker
    systemctl start docker
    systemctl enable docker
    
    # 添加用户到docker组
    usermod -aG docker $SUDO_USER
    
    log_info "Docker安装完成"
}

# 配置SSH
configure_ssh() {
    log_step "配置SSH..."
    
    # 生成SSH密钥（如果不存在）
    if [ ! -f ~/.ssh/id_rsa ]; then
        ssh-keygen -t rsa -b 4096 -C "digital-human-server" -N ""
        log_info "SSH密钥已生成"
    fi
    
    # 配置SSH客户端
    echo "Host github.com" >> ~/.ssh/config
    echo "    HostName github.com" >> ~/.ssh/config
    echo "    User git" >> ~/.ssh/config
    echo "    IdentityFile ~/.ssh/id_rsa" >> ~/.ssh/config
    echo "    StrictHostKeyChecking no" >> ~/.ssh/config
    
    chmod 600 ~/.ssh/config
    
    log_info "SSH配置完成"
    log_info "公钥内容:"
    cat ~/.ssh/id_rsa.pub
    log_info "请将上面的公钥添加到GitHub账户"
}

# 配置Git
configure_git() {
    log_step "配置Git..."
    
    # 设置全局Git配置
    git config --global user.name "Digital Human Developer"
    git config --global user.email "dev@digital-human.local"
    git config --global core.editor vim
    git config --global init.defaultBranch main
    
    # 配置Git LFS
    apt install -y git-lfs
    git lfs install
    
    log_info "Git配置完成"
}

# 创建项目目录
create_project_dir() {
    log_step "创建项目目录..."
    
    mkdir -p $PROJECT_DIR
    mkdir -p $PROJECT_DIR/data/{raw,processed,splits}
    mkdir -p $PROJECT_DIR/checkpoints/{trained_models,quantized_models}
    mkdir -p $PROJECT_DIR/logs
    mkdir -p $PROJECT_DIR/outputs
    
    # 设置权限
    chown -R $SUDO_USER:$SUDO_USER $PROJECT_DIR
    
    log_info "项目目录创建完成: $PROJECT_DIR"
}

# 安装模型检查点
install_model_checkpoints() {
    log_step "安装预训练模型检查点..."
    
    # 克隆Ultralight-Digital-Human项目
    cd $PROJECT_DIR
    git clone https://github.com/anliyuan/Ultralight-Digital-Human.git ultralight-dh
    
    # 下载人脸检测模型
    mkdir -p $PROJECT_DIR/models
    cd $PROJECT_DIR/models
    
    # SCRFD人脸检测模型
    wget -q https://github.com/nicechuan/face_detection_and_alignment/releases/download/v1.0/scrfd_2.5g_kps.onnx
    
    # 设置权限
    chown -R $SUDO_USER:$SUDO_USER $PROJECT_DIR/models
    
    log_info "预训练模型下载完成"
}

# 验证安装
verify_installation() {
    log_step "验证安装..."
    
    # 加载conda
    source /opt/miniconda/etc/profile.d/conda.sh
    conda activate digital-human
    
    echo "================================"
    echo "验证信息:"
    echo "================================"
    
    # 检查Python版本
    echo -n "Python版本: "
    python --version
    
    # 检查PyTorch
    echo -n "PyTorch版本: "
    python -c "import torch; print(torch.__version__)"
    
    # 检查CUDA
    echo -n "CUDA版本: "
    python -c "import torch; print(torch.version.cuda)"
    
    # 检查GPU
    echo -n "GPU可用: "
    python -c "import torch; print(torch.cuda.is_available())"
    
    if torch.cuda.is_available():
        echo -n "GPU数量: "
        python -c "import torch; print(torch.cuda.device_count())"
        echo -n "GPU名称: "
        python -c "import torch; print(torch.cuda.get_device_name(0))"
    fi
    
    echo "================================"
    
    conda deactivate
    
    log_info "验证完成"
}

# 生成系统报告
generate_report() {
    log_step "生成系统报告..."
    
    cat > $PROJECT_DIR/system_info.txt << EOF
================================
Digital Human 训练服务器配置报告
================================

系统信息:
- 操作系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
- 内核版本: $(uname -r)
- 主机名: $(hostname)

软件版本:
- CUDA: $CUDA_VERSION
- cuDNN: $CUDNN_VERSION
- Python: $PYTHON_VERSION
- PyTorch: 1.13.1

GPU信息:
$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "GPU信息无法获取，请确保驱动已安装")

项目目录: $PROJECT_DIR

配置完成时间: $(date)
================================
EOF
    
    log_info "系统报告已生成: $PROJECT_DIR/system_info.txt"
}

# 主函数
main() {
    log_info "========================================="
    log_info "开始配置Ubuntu服务器环境"
    log_info "========================================="
    
    # 初始化日志文件
    echo "Server Setup Log - $(date)" > "$LOG_FILE"
    
    # 检查权限
    check_root
    
    # 检查系统
    check_ubuntu_version
    
    # 安装组件
    update_system
    install_base_dependencies
    
    # GPU相关（需要重启后生效）
    install_nvidia_driver
    install_cuda
    install_cudnn
    
    # Python环境
    install_conda
    create_python_env
    
    # Docker（可选）
    # install_docker
    
    # 开发工具配置
    configure_ssh
    configure_git
    
    # 项目目录
    create_project_dir
    install_model_checkpoints
    
    # 验证
    verify_installation
    generate_report
    
    log_info "========================================="
    log_info "服务器环境配置完成!"
    log_info "========================================="
    log_info "重要提示:"
    log_info "1. 请重启服务器使NVIDIA驱动生效"
    log_info "2. 重启后运行: conda activate digital-human"
    log_info "3. 验证GPU: python -c 'import torch; print(torch.cuda.is_available())'"
    log_info "========================================="
}

# 执行主函数
main "$@"
