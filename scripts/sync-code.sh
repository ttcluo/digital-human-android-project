#!/bin/bash
#===============================================================================
# 代码同步脚本
# 用于本地开发环境与服务器之间的代码同步
# 
# 使用方法: bash sync-code.sh <branch> <server_ip> [action]
# 
# action: push(推送到服务器), pull(从服务器拉取), sync(双向同步)
#===============================================================================

set -e

# 配置
PROJECT_NAME="digital-human-android-project"
REMOTE_USER="root"
REMOTE_PORT="22"
DEFAULT_BRANCH="main"

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
if [ $# -lt 2 ]; then
    echo "用法: $0 <branch> <server_ip> [action]"
    echo "  branch: Git分支名"
    echo "  server_ip: 服务器IP地址"
    echo "  action: push(默认) | pull | sync"
    exit 1
fi

BRANCH=$1
SERVER_IP=$2
ACTION=${3:-push}

REMOTE_PATH="/opt/digital-human/${PROJECT_NAME}"
LOG_FILE="sync-${BRANCH}-$(date +%Y%m%d-%H%M%S).log"

#===============================================================================
# 主函数
#===============================================================================

main() {
    log_info "========================================="
    log_info "开始代码同步"
    log_info "========================================="
    log_info "分支: ${BRANCH}"
    log_info "服务器: ${SERVER_IP}"
    log_info "操作: ${ACTION}"
    
    # 记录日志
    echo "同步开始: $(date)" > "$LOG_FILE"
    
    case $ACTION in
        push)
            push_to_server
            ;;
        pull)
            pull_from_server
            ;;
        sync)
            sync_bidirectional
            ;;
        *)
            log_error "未知操作: ${ACTION}"
            exit 1
            ;;
    esac
    
    log_info "========================================="
    log_info "代码同步完成!"
    log_info "========================================="
}

# 推送到服务器
push_to_server() {
    log_info "步骤1: 在本地提交并推送代码..."
    
    # 检查Git状态
    if [ ! -d ".git" ]; then
        log_error "当前目录不是Git仓库"
        exit 1
    fi
    
    # 添加所有更改
    git add -A
    
    # 检查是否有更改
    if git diff --staged --quiet; then
        log_warn "没有需要提交的更改"
    else
        # 提交更改
        echo "输入提交信息 (输入q取消):"
        read commit_msg
        
        if [ "$commit_msg" = "q" ] || [ -z "$commit_msg" ]; then
            log_warn "取消提交"
        else
            git commit -m "$commit_msg"
            log_info "提交完成"
        fi
    fi
    
    # 推送到远程
    log_info "推送到远程仓库..."
    git push origin ${BRANCH}
    
    log_info "步骤2: 在服务器上拉取代码..."
    
    # SSH到服务器并拉取
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${SERVER_IP} << EOF
        cd ${REMOTE_PATH}
        git fetch origin
        git checkout ${BRANCH}
        git pull origin ${BRANCH}
        echo "服务器代码已更新"
EOF
    
    echo "Push完成: $(date)" >> "$LOG_FILE"
}

# 从服务器拉取
pull_from_server() {
    log_info "步骤1: 从远程仓库拉取最新代码..."
    git fetch origin
    git checkout ${BRANCH}
    git pull origin ${BRANCH}
    
    log_info "步骤2: 推送到服务器..."
    
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${SERVER_IP} << EOF
        cd ${REMOTE_PATH}
        git fetch origin
        git checkout ${BRANCH}
        git pull origin ${BRANCH}
        echo "服务器代码已更新"
EOF
    
    echo "Pull完成: $(date)" >> "$LOG_FILE"
}

# 双向同步
sync_bidirectional() {
    log_info "执行双向同步..."
    
    # 获取服务器最新代码
    log_info "步骤1: 获取服务器最新代码..."
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${SERVER_IP} << EOF
        cd ${REMOTE_PATH}
        git fetch origin
        git pull origin ${BRANCH}
EOF
    
    # 合并服务器更改
    log_info "步骤2: 合并服务器更改..."
    git merge origin/${BRANCH}
    
    # 推送合并结果
    log_info "步骤3: 推送合并结果..."
    git push origin ${BRANCH}
    
    # 服务器拉取
    log_info "步骤4: 服务器拉取最新代码..."
    ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${SERVER_IP} << EOF
        cd ${REMOTE_PATH}
        git pull origin ${BRANCH}
EOF
    
    echo "Sync完成: $(date)" >> "$LOG_FILE"
}

# 执行主函数
main
