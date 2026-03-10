#!/bin/bash
#===============================================================================
# 训练和评估脚本
# 用于在服务器上训练和评估模型
# 
# 使用方法: bash train-eval.sh [mode]
# 
# mode: train(训练), eval(评估), both(训练+评估)
#===============================================================================

set -e

# 配置
CONFIG_FILE="./configs/train_config.yaml"
CHECKPOINT_DIR="./checkpoints/trained_models"
OUTPUT_DIR="./outputs"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查参数
MODE=${1:-train}

# 激活conda环境
source /opt/miniconda/etc/profile.d/conda.sh
conda activate digital-human

#===============================================================================
# 训练函数
#===============================================================================

train() {
    log_info "开始训练模型..."
    
    # 检查数据目录
    if [ ! -d "./data/processed" ]; then
        log_error "数据目录不存在: ./data/processed"
        log_info "请先运行数据预处理"
        exit 1
    fi
    
    # 创建输出目录
    mkdir -p ${CHECKPOINT_DIR}
    mkdir -p ${OUTPUT_DIR}
    
    # 启动训练
    python -m src.training.train \
        --config ${CONFIG_FILE} \
        --save_dir ${CHECKPOINT_DIR} \
        --log_dir ./logs \
        --device cuda \
        2>&1 | tee ${OUTPUT_DIR}/train.log
    
    log_info "训练完成!"
}

#===============================================================================
# 评估函数
#===============================================================================

evaluate() {
    log_info "开始评估模型..."
    
    # 检查检查点
    if [ ! -f "${CHECKPOINT_DIR}/best_model.pth" ]; then
        log_error "模型检查点不存在: ${CHECKPOINT_DIR}/best_model.pth"
        exit 1
    fi
    
    # 启动评估
    python -m src.training.evaluate \
        --checkpoint ${CHECKPOINT_DIR}/best_model.pth \
        --data_dir ./data/processed \
        --output_dir ${OUTPUT_DIR}/evaluation \
        --device cuda \
        2>&1 | tee ${OUTPUT_DIR}/eval.log
    
    log_info "评估完成!"
}

#===============================================================================
# 主函数
#===============================================================================

main() {
    log_info "========================================="
    log_info "Digital Human 训练评估脚本"
    log_info "========================================="
    log_info "模式: ${MODE}"
    log_info "Python环境: $(which python)"
    log_info "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
    log_info "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
    
    if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
        log_info "GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())')"
        log_info "GPU名称: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    fi
    
    case $MODE in
        train)
            train
            ;;
        eval)
            evaluate
            ;;
        both)
            train
            evaluate
            ;;
        *)
            log_error "未知模式: ${MODE}"
            echo "用法: $0 [train|eval|both]"
            exit 1
            ;;
    esac
    
    log_info "========================================="
    log_info "所有任务完成!"
    log_info "========================================="
}

main
