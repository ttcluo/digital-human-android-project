#!/bin/bash
# Git仓库初始化脚本
# 用于初始化Git仓库并连接到远程仓库

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

# 默认配置
DEFAULT_USER_NAME="ttcluo"
DEFAULT_USER_EMAIL="ttcluo@163.com"
DEFAULT_BRANCH="main"
DEFAULT_REMOTE_URL="https://github.com/ttcluo/digital-human-android-project.git"

# 检查参数
USER_NAME=""
USER_EMAIL=""
REMOTE_URL=""
GITHUB_TOKEN=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --user-name)
            USER_NAME="$2"
            shift 2
            ;;
        --user-email)
            USER_EMAIL="$2"
            shift 2
            ;;
        --remote-url)
            REMOTE_URL="$2"
            shift 2
            ;;
        --github-token)
            GITHUB_TOKEN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --user-name <名称>    Git用户名 (默认: $DEFAULT_USER_NAME)"
            echo "  --user-email <邮箱>   Git用户邮箱 (默认: $DEFAULT_USER_EMAIL)"
            echo "  --remote-url <URL>    远程仓库URL (默认: $DEFAULT_REMOTE_URL)"
            echo "  --github-token <token> GitHub个人访问令牌"
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

# 使用默认值
if [ -z "$USER_NAME" ]; then
    USER_NAME="$DEFAULT_USER_NAME"
fi

if [ -z "$USER_EMAIL" ]; then
    USER_EMAIL="$DEFAULT_USER_EMAIL"
fi

if [ -z "$REMOTE_URL" ]; then
    REMOTE_URL="$DEFAULT_REMOTE_URL"
fi

# 如果提供了GitHub token，插入到URL中
if [ -n "$GITHUB_TOKEN" ]; then
    # 从URL中提取路径部分
    if [[ "$REMOTE_URL" =~ ^https://github.com/(.+)$ ]]; then
        REPO_PATH="${BASH_REMATCH[1]}"
        REMOTE_URL="https://${GITHUB_TOKEN}@github.com/${REPO_PATH}"
        print_info "已使用GitHub token更新远程URL"
    fi
fi

# 主函数
main() {
    print_header "Git仓库初始化"
    print_info "项目目录: ${PROJECT_DIR}"
    print_info "Git用户: ${USER_NAME} <${USER_EMAIL}>"
    print_info "远程仓库: ${REMOTE_URL}"
    
    if [ "$DRY_RUN" = true ]; then
        print_warning "干运行模式 - 只显示命令，不实际执行"
    fi
    
    cd "${PROJECT_DIR}"
    
    # 检查是否已经是Git仓库
    print_header "1. 检查当前状态"
    
    if [ -d ".git" ]; then
        print_warning "项目已经是Git仓库"
        
        if [ "$DRY_RUN" = false ]; then
            print_info "当前Git配置:"
            git config --list | grep -E "user\.|remote\.origin" || true
        fi
        
        # 询问是否继续
        print_info "是否继续重新初始化? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_info "取消初始化"
            exit 0
        fi
        
        if [ "$DRY_RUN" = false ]; then
            # 备份旧的.git目录
            backup_dir=".git-backup-$(date +%Y%m%d-%H%M%S)"
            mv .git "$backup_dir"
            print_warning "旧的.git目录已备份到: ${backup_dir}"
        fi
    fi
    
    # 初始化Git仓库
    print_header "2. 初始化Git仓库"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git init"
    else
        git init
        print_success "Git仓库已初始化"
    fi
    
    # 配置Git用户
    print_header "3. 配置Git用户"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git config --global user.name \"${USER_NAME}\""
        print_info "将执行: git config --global user.email \"${USER_EMAIL}\""
    else
        git config --global user.name "${USER_NAME}"
        git config --global user.email "${USER_EMAIL}"
        git config --global core.autocrlf false
        git config --global core.safecrlf true
        git config --global init.defaultBranch "${DEFAULT_BRANCH}"
        
        print_success "Git用户配置完成"
        print_info "用户名: $(git config --global user.name)"
        print_info "用户邮箱: $(git config --global user.email)"
    fi
    
    # 添加文件到暂存区
    print_header "4. 添加文件到暂存区"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git add ."
    else
        git add .
        print_success "所有文件已添加到暂存区"
        
        # 显示添加的文件
        print_info "已添加的文件数量:"
        git status --short | wc -l
    fi
    
    # 提交更改
    print_header "5. 提交初始更改"
    
    COMMIT_MESSAGE="Initial commit: Digital Human Android Project"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git commit -m \"${COMMIT_MESSAGE}\""
    else
        git commit -m "${COMMIT_MESSAGE}"
        print_success "初始提交完成"
        print_info "提交哈希: $(git rev-parse --short HEAD)"
    fi
    
    # 重命名主分支
    print_header "6. 设置默认分支"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git branch -M ${DEFAULT_BRANCH}"
    else
        git branch -M "${DEFAULT_BRANCH}"
        print_success "主分支已设置为: ${DEFAULT_BRANCH}"
    fi
    
    # 添加远程仓库
    print_header "7. 添加远程仓库"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git remote add origin \"${REMOTE_URL}\""
    else
        git remote add origin "${REMOTE_URL}"
        print_success "远程仓库已添加"
        
        print_info "远程仓库列表:"
        git remote -v
    fi
    
    # 推送到远程仓库
    print_header "8. 推送到远程仓库"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "将执行: git push -u origin ${DEFAULT_BRANCH}"
    else
        print_info "正在推送到远程仓库..."
        git push -u origin "${DEFAULT_BRANCH}"
        print_success "代码已推送到远程仓库"
    fi
    
    # 验证配置
    print_header "9. 验证配置"
    
    if [ "$DRY_RUN" = false ]; then
        print_info "Git仓库配置验证:"
        
        echo "1. 当前分支: $(git branch --show-current)"
        echo "2. 远程仓库: $(git remote get-url origin)"
        echo "3. 最近提交: $(git log --oneline -1)"
        echo "4. 文件状态: $(git status --short | wc -l) 个文件已跟踪"
        
        print_success "✅ Git仓库初始化完成！"
        print_info ""
        print_info "重要信息:"
        print_info "1. 项目目录: ${PROJECT_DIR}"
        print_info "2. 远程仓库: ${REMOTE_URL}"
        print_info "3. 默认分支: ${DEFAULT_BRANCH}"
        print_info "4. 提交哈希: $(git rev-parse --short HEAD)"
        print_info ""
        print_info "后续操作:"
        print_info "1. 查看Git状态: git status"
        print_info "2. 添加更改: git add <file>"
        print_info "3. 提交更改: git commit -m \"message\""
        print_info "4. 推送更改: git push"
        print_info "5. 拉取更新: git pull"
    else
        print_success "✅ 干运行完成 - 所有命令已检查"
    fi
    
    return 0
}

# 执行主函数
main "$@"

exit_code=$?

if [ $exit_code -ne 0 ]; then
    print_error "Git仓库初始化失败"
    exit $exit_code
fi