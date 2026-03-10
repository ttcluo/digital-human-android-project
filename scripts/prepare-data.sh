#!/bin/bash
#===============================================================================
# 数据准备脚本
# 用于自动化准备数字人训练数据
#
# 使用方法: bash prepare-data.sh <video_path> [asr_mode]
#
# asr_mode: hubert(默认) | wenet
#===============================================================================

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

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <video_path> [asr_mode]"
    echo "  video_path: 视频文件路径"
    echo "  asr_mode: hubert(默认) | wenet"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/video.mp4 hubert"
    echo "  $0 /path/to/video.mp4 wenet"
    exit 1
fi

VIDEO_PATH="$1"
ASR_MODE="${2:-hubert}"

# 获取项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
RAW_DIR="${DATA_DIR}/raw"
PROCESSED_DIR="${DATA_DIR}/processed"
ULTRALIGHT_DIR="/data/luochuan/Ultralight-Digital-Human"

#===============================================================================
# 检查函数
#===============================================================================

check_requirements() {
    print_header "检查依赖"
    
    # 检查视频文件
    if [ ! -f "${VIDEO_PATH}" ]; then
        print_error "视频文件不存在: ${VIDEO_PATH}"
        exit 1
    fi
    print_success "视频文件: ${VIDEO_PATH}"
    
    # 检查FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        print_error "FFmpeg 未安装"
        print_info "安装命令: apt-get install ffmpeg"
        exit 1
    fi
    print_success "FFmpeg 已安装"
    
    # 检查Ultralight-Digital-Human目录
    if [ ! -d "${ULTRALIGHT_DIR}" ]; then
        print_error "Ultralight-Digital-Human 目录不存在: ${ULTRALIGHT_DIR}"
        print_info "请先克隆 Ultralight-Digital-Human 项目"
        exit 1
    fi
    print_success "Ultralight-Digital-Human 已安装"
    
    # 检查ASR模式
    if [ "${ASR_MODE}" != "hubert" ] && [ "${ASR_MODE}" != "wenet" ]; then
        print_error "无效的ASR模式: ${ASR_MODE}"
        print_info "支持的ASR模式: hubert, wenet"
        exit 1
    fi
    print_success "ASR模式: ${ASR_MODE}"
    
    # 检查WeNet模型（如果使用wenet）
    if [ "${ASR_MODE}" == "wenet" ]; then
        if [ ! -f "${ULTRALIGHT_DIR}/data_utils/encoder.onnx" ]; then
            print_warning "WeNet encoder.onnx 不存在"
            print_info "请从 https://drive.google.com/file/d/1e4Z9zS053JEWl6Mj3W9Lbc9GDtzHIg6b/view?usp=drive_link 下载"
            print_info "并放到 ${ULTRALIGHT_DIR}/data_utils/ 目录"
        else
            print_success "WeNet encoder.onnx 已存在"
        fi
    fi
}

#===============================================================================
# 准备目录
#===============================================================================

prepare_directories() {
    print_header "准备目录"
    
    # 创建目录
    mkdir -p "${RAW_DIR}"
    mkdir -p "${PROCESSED_DIR}"
    mkdir -p "${PROCESSED_DIR}/train_data"
    mkdir -p "${PROCESSED_DIR}/train_data/imgs"
    mkdir -p "${PROCESSED_DIR}/train_data/landmarks"
    
    print_success "目录创建完成"
    print_info "数据目录: ${DATA_DIR}"
}

#===============================================================================
# 检查视频
#===============================================================================

check_video() {
    print_header "检查视频"
    
    # 获取视频信息
    print_info "视频信息:"
    ffprobe -v error -select_streams v:0 \
        -show_entries stream=avg_frame_rate,width,height,r_frame_rate \
        -of csv=p=0 "${VIDEO_PATH}" 2>/dev/null | while IFS=',' read -r fps width height r_frame_rate; do
        
        print_info "  分辨率: ${width}x${height}"
        print_info "  帧率: ${fps}"
        
        # 检查帧率
        if [ "${ASR_MODE}" == "hubert" ]; then
            if [ "${fps}" != "25/1" ] && [ "${fps}" != "25" ]; then
                print_warning "视频帧率不是25fps，建议重新采样"
                print_info "  转换命令: ffmpeg -i ${VIDEO_PATH} -vf \"fps=25\" ${VIDEO_PATH%.mp4}_25fps.mp4"
            else
                print_success "帧率符合要求 (25fps)"
            fi
        elif [ "${ASR_MODE}" == "wenet" ]; then
            if [ "${fps}" != "20/1" ] && [ "${fps}" != "20" ]; then
                print_warning "视频帧率不是20fps，建议重新采样"
                print_info "  转换命令: ffmpeg -i ${VIDEO_PATH} -vf \"fps=20\" ${VIDEO_PATH%.mp4}_20fps.mp4"
            else
                print_success "帧率符合要求 (20fps)"
            fi
        fi
    done
    
    # 获取视频时长
    DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${VIDEO_PATH}" 2>/dev/null)
    DURATION_MIN=$(echo "scale=1; ${DURATION} / 60" | bc -l 2>/dev/null || echo "unknown")
    print_info "  时长: ${DURATION}秒 (${DURATION_MIN}分钟)"
    
    # 检查视频时长
    DURATION_INT=${DURATION%.*}
    if [ ${DURATION_INT} -lt 60 ]; then
        print_warning "视频时长少于1分钟，建议3-5分钟"
    elif [ ${DURATION_INT} -gt 600 ]; then
        print_warning "视频时长大于10分钟，建议3-5分钟"
    else
        print_success "视频时长合适"
    fi
}

#===============================================================================
# 复制视频到数据目录
#===============================================================================

copy_video() {
    print_header "复制视频"
    
    VIDEO_NAME=$(basename "${VIDEO_PATH}")
    TARGET_PATH="${RAW_DIR}/${VIDEO_NAME}"
    
    if [ "${VIDEO_PATH}" != "${TARGET_PATH}" ]; then
        cp "${VIDEO_PATH}" "${TARGET_PATH}"
        print_success "视频已复制到: ${TARGET_PATH}"
    else
        print_info "视频已在数据目录中"
    fi
    
    # 更新视频路径
    VIDEO_PATH="${TARGET_PATH}"
}

#===============================================================================
# 数据预处理
#===============================================================================

preprocess_data() {
    print_header "数据预处理"
    
    # 进入Ultralight-Digital-Human目录
    cd "${ULTRALIGHT_DIR}"
    
    print_info "运行数据处理脚本..."
    print_info "这可能需要几分钟到几十分钟，取决于视频长度"
    
    # 运行数据处理
    python data_utils/process.py "${VIDEO_PATH}" --asr "${ASR_MODE}"
    
    if [ $? -eq 0 ]; then
        print_success "数据预处理完成"
    else
        print_error "数据预处理失败"
        exit 1
    fi
}

#===============================================================================
# 整理训练数据
#===============================================================================

organize_data() {
    print_header "整理训练数据"
    
    VIDEO_NAME=$(basename "${VIDEO_PATH}")
    VIDEO_DIR="${RAW_DIR}"
    
    # 检查预处理结果
    if [ ! -d "${VIDEO_DIR}/full_body_img" ]; then
        print_error "图像目录不存在: ${VIDEO_DIR}/full_body_img"
        exit 1
    fi
    
    if [ ! -d "${VIDEO_DIR}/landmarks" ]; then
        print_error "关键点目录不存在: ${VIDEO_DIR}/landmarks"
        exit 1
    fi
    
    if [ ! -f "${VIDEO_DIR}/aud.wav" ]; then
        print_error "音频文件不存在: ${VIDEO_DIR}/aud.wav"
        exit 1
    fi
    
    # 复制到训练目录
    print_info "复制图像..."
    cp -r "${VIDEO_DIR}/full_body_img/"* "${PROCESSED_DIR}/train_data/imgs/"
    
    print_info "复制关键点..."
    cp -r "${VIDEO_DIR}/landmarks/"* "${PROCESSED_DIR}/train_data/landmarks/"
    
    print_info "复制音频特征..."
    # 音频特征文件通常在 data_utils/ 目录生成
    if [ -f "${ULTRALIGHT_DIR}/data_utils/aud.npy" ]; then
        cp "${ULTRALIGHT_DIR}/data_utils/aud.npy" "${PROCESSED_DIR}/train_data/audio_features.npy"
        print_success "音频特征已复制"
    elif [ -f "${VIDEO_DIR}/aud.npy" ]; then
        cp "${VIDEO_DIR}/aud.npy" "${PROCESSED_DIR}/train_data/audio_features.npy"
        print_success "音频特征已复制"
    else
        print_warning "音频特征文件未找到，需要手动处理"
        print_info "运行: python ${ULTRALIGHT_DIR}/data_utils/hubert.py --wav ${VIDEO_DIR}/aud.wav"
    fi
    
    # 创建训练列表
    print_info "创建训练列表..."
    cd "${PROCESSED_DIR}/train_data"
    
    # 获取文件数量
    IMG_COUNT=$(ls imgs/ | wc -l)
    print_info "共 ${IMG_COUNT} 帧图像"
    
    # 创建列表文件
    > train_list.txt
    for i in $(seq 0 $((IMG_COUNT - 1))); do
        echo "imgs/${i}.jpg|audio_features.npy|landmarks/${i}.lms" >> train_list.txt
    done
    
    print_success "训练列表已创建: train_list.txt"
}

#===============================================================================
# 验证数据
#===============================================================================

verify_data() {
    print_header "验证数据"
    
    cd "${PROCESSED_DIR}/train_data"
    
    # 检查图像数量
    IMG_COUNT=$(find imgs/ -name "*.jpg" | wc -l)
    LM_COUNT=$(find landmarks/ -name "*.lms" | wc -l)
    
    print_info "图像数量: ${IMG_COUNT}"
    print_info "关键点数量: ${LM_COUNT}"
    
    if [ ${IMG_COUNT} -ne ${LM_COUNT} ]; then
        print_warning "图像数量和关键点数量不匹配"
    else
        print_success "数据完整性检查通过"
    fi
    
    # 检查音频特征
    if [ -f "audio_features.npy" ]; then
        print_success "音频特征文件存在"
        
        # 检查特征形状
        python -c "import numpy as np; features = np.load('audio_features.npy'); print(f'音频特征形状: {features.shape}')"
    else
        print_warning "音频特征文件不存在"
    fi
    
    # 检查样本
    if [ ${IMG_COUNT} -gt 0 ]; then
        print_info "数据样本:"
        head -3 train_list.txt | while read line; do
            print_info "  ${line}"
        done
    fi
}

#===============================================================================
# 生成报告
#===============================================================================

generate_report() {
    print_header "数据准备报告"
    
    cd "${PROCESSED_DIR}/train_data"
    
    IMG_COUNT=$(find imgs/ -name "*.jpg" | wc -l)
    
    cat > data_preparation_report.txt << EOF
================================
数据准备报告
================================

准备时间: $(date)
视频文件: ${VIDEO_PATH}
ASR模式: ${ASR_MODE}

数据统计:
- 图像数量: ${IMG_COUNT} 帧
- 关键点数量: $(find landmarks/ -name "*.lms" | wc -l)
- 音频特征: $(if [ -f audio_features.npy ]; then echo "已生成"; else echo "缺失"; fi)

目录结构:
${PROCESSED_DIR}/train_data/
├── imgs/           # 视频帧图像
├── landmarks/      # 人脸关键点
├── audio_features.npy  # 音频特征
└── train_list.txt  # 训练列表

下一步:
1. 检查数据质量
2. 修改配置文件: ${PROJECT_DIR}/configs/train_config.yaml
3. 开始训练: bash ${PROJECT_DIR}/scripts/train-eval.sh train
================================
EOF
    
    print_success "报告已生成: ${PROCESSED_DIR}/train_data/data_preparation_report.txt"
    
    # 显示报告
    cat data_preparation_report.txt
}

#===============================================================================
# 主函数
#===============================================================================

main() {
    print_header "数字人数据准备"
    print_info "视频: ${VIDEO_PATH}"
    print_info "ASR: ${ASR_MODE}"
    print_info "项目: ${PROJECT_DIR}"
    
    # 执行步骤
    check_requirements
    prepare_directories
    check_video
    copy_video
    preprocess_data
    organize_data
    verify_data
    generate_report
    
    # 完成
    print_header "数据准备完成！"
    print_success "✅ 所有步骤已完成"
    print_info ""
    print_info "数据路径: ${PROCESSED_DIR}/train_data"
    print_info "训练命令: bash ${PROJECT_DIR}/scripts/train-eval.sh train"
    print_info ""
    print_info "请检查数据质量，然后可以开始训练！"
}

# 执行主函数
main "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    print_success "数据准备成功！"
    exit 0
else
    print_error "数据准备失败"
    exit $exit_code
fi
