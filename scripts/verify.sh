#!/bin/bash
# Digital Human 验证脚本
# 用于快速验证服务器环境和项目配置

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }
print_header() { echo -e "\n${BOLD}=== $1 ===${NC}"; }

# 获取项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_DIR}/verification-$(date +%Y%m%d-%H%M%S).log"

# 主函数
main() {
    print_header "Digital Human 项目验证"
    print_info "项目目录: ${PROJECT_DIR}"
    print_info "验证时间: $(date)"
    
    echo "验证日志: ${LOG_FILE}"
    
    # 检查Python环境
    print_header "1. 检查Python环境"
    
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version)
        print_success "Python3 已安装: ${python_version}"
    else
        print_error "Python3 未安装"
        return 1
    fi
    
    # 检查conda环境
    print_header "2. 检查Conda环境"
    
    if conda info --envs | grep -q "digital-human"; then
        print_success "Conda环境 'digital-human' 存在"
        print_info "激活环境: conda activate digital-human"
    else
        print_warning "Conda环境 'digital-human' 不存在"
        print_info "创建环境: conda create -n digital-human python=3.10"
    fi
    
    # 检查PyTorch
    print_header "3. 检查PyTorch和CUDA"
    
    cat > /tmp/check_pytorch.py << 'EOF'
import torch
import sys

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
else:
    print("警告: CUDA不可用，将使用CPU模式")
    sys.exit(0)
EOF
    
    if python3 /tmp/check_pytorch.py 2>&1; then
        print_success "PyTorch检查完成"
    else
        print_error "PyTorch检查失败"
        print_info "安装PyTorch: pip install torch torchvision torchaudio"
    fi
    
    rm -f /tmp/check_pytorch.py
    
    # 检查项目依赖
    print_header "4. 检查项目依赖"
    
    if [ -f "${PROJECT_DIR}/requirements-server.txt" ]; then
        print_success "requirements-server.txt 存在"
        
        # 检查主要依赖
        for dep in torch numpy opencv-python transformers; do
            if python3 -c "import $dep" 2>/dev/null; then
                print_success "$dep 已安装"
            else
                print_warning "$dep 未安装"
            fi
        done
    else
        print_warning "requirements-server.txt 不存在"
    fi
    
    # 检查项目结构
    print_header "5. 检查项目结构"
    
    required_dirs=(
        "src"
        "src/models"
        "src/training"
        "src/inference"
        "src/utils"
        "configs"
        "scripts"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [ -d "${PROJECT_DIR}/${dir}" ]; then
            print_success "${dir}/"
        else
            print_error "${dir}/ (缺失)"
        fi
    done
    
    # 运行Python验证脚本
    print_header "6. 运行详细验证"
    
    if [ -f "${PROJECT_DIR}/scripts/verify-installation.py" ]; then
        print_info "运行完整验证脚本..."
        cd "${PROJECT_DIR}"
        python3 scripts/verify-installation.py
        
        if [ $? -eq 0 ]; then
            print_success "详细验证完成"
        else
            print_warning "详细验证发现一些问题，请查看上方输出"
        fi
    else
        print_warning "verify-installation.py 不存在"
    fi
    
    # 生成报告
    print_header "7. 验证完成"
    
    print_success "验证流程完成"
    print_info "项目路径: ${PROJECT_DIR}"
    print_info "激活环境: conda activate digital-human"
    print_info "运行测试: python3 scripts/verify-installation.py"
    print_info "开始训练: bash scripts/train-eval.sh train"
    
    echo "验证完成时间: $(date)" >> "${LOG_FILE}"
    
    return 0
}

# 执行主函数
main "$@" 2>&1 | tee -a "${LOG_FILE}"

exit_code=${PIPESTATUS[0]}

if [ ${exit_code} -eq 0 ]; then
    print_success "\n✅ 验证完成！项目配置正确。"
else
    print_warning "\n⚠️  验证发现一些问题，请查看上方输出。"
fi

exit ${exit_code}