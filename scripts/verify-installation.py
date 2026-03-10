#!/usr/bin/env python3
"""
Digital Human 安装验证脚本
用于验证服务器环境和项目代码是否正确安装

用法:
    python3 scripts/verify-installation.py
"""

import sys
import os
import subprocess
import importlib
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 颜色定义
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")

def check_python_version():
    """检查Python版本"""
    print_header("1. 检查Python环境")
    
    version = sys.version_info
    print_info(f"Python版本: {sys.version}")
    
    if version.major == 3 and version.minor >= 10:
        print_success(f"Python版本符合要求 (>= 3.10)")
    else:
        print_error(f"Python版本过低 (需要 >= 3.10)")
        return False
    return True

def check_pytorch():
    """检查PyTorch安装"""
    print_header("2. 检查PyTorch和CUDA")
    
    try:
        import torch
        print_info(f"PyTorch版本: {torch.__version__}")
        
        # 检查CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print_success(f"CUDA可用: True")
            print_info(f"CUDA版本: {torch.version.cuda}")
            print_info(f"GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print_info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print_warning("CUDA不可用，将使用CPU")
            
        # 检查PyTorch版本
        version_parts = torch.__version__.split('+')[0].split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major >= 2 or (major == 1 and minor >= 13):
            print_success("PyTorch版本符合要求 (>= 1.13.1)")
        else:
            print_error(f"PyTorch版本过低 (需要 >= 1.13.1)")
            return False
            
    except ImportError as e:
        print_error(f"PyTorch导入失败: {e}")
        return False
    except Exception as e:
        print_error(f"检查PyTorch时出错: {e}")
        return False
        
    return True

def check_dependencies():
    """检查核心依赖"""
    print_header("3. 检查核心依赖")
    
    dependencies = [
        ("numpy", ">=2.2.6"),
        ("opencv-python", ">=4.7.0"),
        ("Pillow", ">=9.5.0"),
        ("librosa", ">=0.10.0"),
        ("soundfile", ">=0.12.0"),
        ("transformers", ">=4.30.0"),
        ("onnx", ">=1.14.0"),
        ("onnxruntime", ">=1.15.0"),
        ("tensorboard", ">=2.13.0"),
        ("tqdm", ">=4.65.0"),
        ("PyYAML", ">=6.0"),
        ("scipy", ">=1.9.0"),
        ("scikit-learn", ">=1.2.0"),
    ]
    
    all_ok = True
    for dep, version_req in dependencies:
        try:
            module = importlib.import_module(dep.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{dep}: {version}")
        except ImportError:
            print_error(f"{dep}: 未安装")
            all_ok = False
        except Exception as e:
            print_warning(f"{dep}: 检查版本失败 ({e})")
    
    return all_ok

def check_project_structure():
    """检查项目结构"""
    print_header("4. 检查项目结构")
    
    required_dirs = [
        "src",
        "src/models",
        "src/training", 
        "src/inference",
        "src/utils",
        "configs",
        "scripts",
        "android/app/src/main/cpp",
    ]
    
    required_files = [
        "requirements.txt",
        "requirements-server.txt",
        "README.md",
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/unet_light.py",
        "src/models/syncnet_improved.py",
        "src/utils/video_utils.py",
        "src/training/trainer.py",
    ]
    
    all_ok = True
    
    # 检查目录
    print_info("检查目录结构:")
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print_success(f"  {dir_path}/")
        else:
            print_error(f"  {dir_path}/ (缺失)")
            all_ok = False
    
    # 检查文件
    print_info("\n检查关键文件:")
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            print_success(f"  {file_path}")
        else:
            print_error(f"  {file_path} (缺失)")
            all_ok = False
    
    return all_ok

def check_code_imports():
    """检查代码导入"""
    print_header("5. 检查代码导入")
    
    import_checks = [
        ("src.models.unet_light", "LightUNet"),
        ("src.models.syncnet_improved", "ImprovedSyncNet"),
        ("src.utils.video_utils", "resize_video"),
        ("src.training.trainer", "Trainer"),
    ]
    
    all_ok = True
    
    for module_name, class_name in import_checks:
        try:
            module = importlib.import_module(module_name)
            
            if class_name:
                if hasattr(module, class_name):
                    print_success(f"{module_name}.{class_name}")
                else:
                    print_error(f"{module_name}.{class_name} (未找到)")
                    all_ok = False
            else:
                print_success(f"{module_name}")
                
        except ImportError as e:
            print_error(f"{module_name}: 导入失败 ({e})")
            all_ok = False
        except Exception as e:
            print_error(f"{module_name}: 检查失败 ({e})")
            all_ok = False
    
    return all_ok

def check_git_status():
    """检查Git状态"""
    print_header("6. 检查Git状态")
    
    try:
        # 检查是否在Git仓库中
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_success("项目是Git仓库")
            
            # 检查远程仓库
            result = subprocess.run(
                ["git", "remote", "-v"],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            
            if "origin" in result.stdout:
                print_success("远程仓库已配置")
            else:
                print_warning("远程仓库未配置")
                
            # 检查未提交的更改
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip():
                print_warning("有未提交的更改")
                print_info("未提交文件:")
                for line in result.stdout.strip().split('\n'):
                    if line:
                        print_info(f"  {line}")
            else:
                print_success("工作区干净")
                
        else:
            print_warning("项目不是Git仓库")
            
    except FileNotFoundError:
        print_warning("Git未安装")
    except Exception as e:
        print_warning(f"检查Git状态失败: {e}")
    
    return True  # Git状态不影响功能

def generate_report():
    """生成验证报告"""
    print_header("7. 生成验证报告")
    
    report_file = project_root / "verification-report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Digital Human 项目验证报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"验证时间: {subprocess.check_output(['date']).decode().strip()}\n")
        f.write(f"项目路径: {project_root}\n\n")
        
        # Python信息
        f.write("Python环境:\n")
        f.write(f"  版本: {sys.version}\n")
        
        try:
            import torch
            f.write(f"\nPyTorch环境:\n")
            f.write(f"  版本: {torch.__version__}\n")
            f.write(f"  CUDA可用: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"  GPU数量: {torch.cuda.device_count()}\n")
                for i in range(torch.cuda.device_count()):
                    f.write(f"  GPU {i}: {torch.cuda.get_device_name(i)}\n")
        except:
            f.write("  PyTorch: 未安装\n")
        
        # 项目结构
        f.write("\n项目结构检查:\n")
        for dir_path in ["src", "configs", "scripts", "android"]:
            full_path = project_root / dir_path
            if full_path.exists():
                f.write(f"  {dir_path}/: 存在\n")
            else:
                f.write(f"  {dir_path}/: 缺失\n")
    
    print_success(f"验证报告已生成: {report_file}")
    return True

def main():
    """主函数"""
    print_header("Digital Human 项目验证")
    print_info(f"项目路径: {project_root}")
    print_info(f"开始时间: {subprocess.check_output(['date']).decode().strip()}")
    
    results = []
    
    # 执行检查
    results.append(("Python版本", check_python_version()))
    results.append(("PyTorch环境", check_pytorch()))
    results.append(("核心依赖", check_dependencies()))
    results.append(("项目结构", check_project_structure()))
    results.append(("代码导入", check_code_imports()))
    results.append(("Git状态", check_git_status()))
    results.append(("生成报告", generate_report()))
    
    # 汇总结果
    print_header("验证结果汇总")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    for name, success in results:
        if success:
            print_success(f"{name}: 通过")
        else:
            print_error(f"{name}: 失败")
    
    print(f"\n{Colors.BOLD}总体结果: {passed}/{total} 项检查通过{Colors.END}")
    
    if passed == total:
        print_success("所有检查通过！项目环境配置正确。")
        return 0
    else:
        print_warning(f"有 {total - passed} 项检查未通过，请根据提示修复问题。")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n验证被用户中断")
        sys.exit(130)
    except Exception as e:
        print_error(f"验证过程中发生未预期错误: {e}")
        traceback.print_exc()
        sys.exit(1)