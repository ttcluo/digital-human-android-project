#!/bin/bash
# Git提交和推送脚本
# 确保代码更改正确提交并推送到远程仓库

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
DEFAULT_BRANCH="main"

# 检查参数
COMMIT_MESSAGE=""
BRANCH=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--message)
            COMMIT_MESSAGE="$2"
            shift 2
            ;;
        -b|--branch)
            BRANCH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  -m, --message <消息>  提交信息"
            echo "  -b, --branch <分支>   目标分支 (默认: main)"
            echo "  --dry-run             只显示将要执行的命令，不实际执行"
            echo "  -h, --help            显示帮助信息"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

# 如果没有提供分支，使用默认分支
if [ -z "$BRANCH" ]; then
    BRANCH="$DEFAULT_BRANCH"
fi

# 如果没有提供提交信息，提示输入
if [ -z "$COMMIT_MESSAGE" ]; then
    print_info "请输入提交信息 (输入q取消):"
    read -r user_input
    
    if [ "$user_input" = "q" ] || [ -z "$user_input" ]; then
        print_warning "取消提交"
        exit 0
    fi
    
    COMMIT_MESSAGE="$user_input"
fi

# 主函数
main() {
    print_header "Git提交和推送"
    print_info "项目目录: ${PROJECT_DIR}"
    print_info "目标分支: ${BRANCH}"
    print_info "提交信息: ${COMMIT_MESSAGE}"
    
    if [ "$DRY_RUN" = true ]; then
        print_warning "干运行模式 - 只显示命令，不实际执行"
    fi
    
    # 检查是否在Git仓库中
    print_header "1. 检查Git仓库"
    
    if [ ! -d "${PROJECT_DIR}/.git" ]; then
        print_error "当前目录不是Git仓库"
        print_info "初始化Git仓库: git init"
        exit 1
    fi
    
    cd "${PROJECT_DIR}"
    
    # 检查远程仓库配置
    print_header "2. 检查远程仓库"
    
    if ! git remote | grep -q "origin"; then
        print_warning "远程仓库 'origin' 未配置"
        print_info "请先配置远程仓库: git remote add origin <url>"
        exit 1
    fi
    
    remote_url=$(git remote get-url origin)
    print_success "远程仓库: ${remote_url}"
    
    # 检查当前分支
    current_branch=$(git branch --show-current)
    print_info "当前分支: ${current_branch}"
    
    if [ "$current_branch" != "$BRANCH" ]; then
        print_warning "当前分支 (${current_branch}) 与目标分支 (${BRANCH}) 不同"
        print_info "切换到目标分支: git checkout ${BRANCH}"
        
        if [ "$DRY_RUN" = false ]; then
            git checkout "$BRANCH"
        fi
    fi
    
    # 检查未暂存的更改
    print_header "3. 检查更改"
    
    git_status=$(git status --porcelain)
    
    if [ -z "$git_status" ]; then
        print_warning "没有需要提交的更改"
        exit 0
    fi
    
    print_info "以下文件有更改:"
    echo "$git_status"
    
    # 添加更改
    print_header "4. 添加更改到暂存区"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git add -A"
    else
        git add -A
        print_success "更改已添加到暂存区"
    fi
    
    # 提交更改
    print_header "5. 提交更改"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git commit -m \"${COMMIT_MESSAGE}\""
    else
        git commit -m "${COMMIT_MESSAGE}"
        print_success "提交完成"
        
        # 显示提交信息
        print_info "提交哈希: $(git rev-parse --short HEAD)"
    fi
    
    # 拉取远程更改（避免冲突）
    print_header "6. 拉取远程更改"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git pull origin ${BRANCH} --rebase"
    else
        git pull origin "$BRANCH" --rebase
        print_success "远程更改已拉取并合并"
    fi
    
    # 推送到远程仓库
    print_header "7. 推送到远程仓库"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git push origin ${BRANCH}"
    else
        git push origin "$BRANCH"
        print_success "代码已推送到远程仓库"
    fi
    
    # 显示最终状态
    print_header "8. 最终状态"
    
    if [ "$DRY_RUN" = false ]; then
        print_info "当前分支状态:"
        git status --short
        
        print_info "最近提交:"
        git log --oneline -1
        
        print_success "✅ 提交和推送完成！"
        print_info "远程仓库URL: ${remote_url}"
        print_info "分支: ${BRANCH}"
        print_info "提交哈希: $(git rev-parse --short HEAD)"
    else
        print_success "✅ 干运行完成 - 所有命令已检查"
    fi
    
    return 0
}

# 执行主函数
main "$@"

exit_code=$?

if [ $exit_code -ne 0 ]; then
    print_error "提交和推送失败"
    exit $exit_code
fi