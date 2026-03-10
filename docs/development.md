# 开发指南

本指南将帮助您设置开发环境并开始开发。

## 环境要求

### 本地开发环境（Windows）
- Python 3.10+
- CUDA 11.7 (可选，用于GPU支持)
- 4GB+ RAM
- 10GB+ 可用磁盘空间

### 服务器环境（Ubuntu）
- Ubuntu 20.04/22.04
- NVIDIA GPU (建议RTX 3080+)
- CUDA 11.7
- 16GB+ RAM

### Android开发
- Android Studio Arctic Fox+
- JDK 17
- Android NDK 25.1.8937393

## 本地开发环境设置

### Windows

1. 克隆项目：
```bash
git clone https://github.com/your-repo/digital-human-android-project.git
cd digital-human-android-project
```

2. 运行配置脚本：
```powershell
.\scripts\setup-local.ps1
```

3. 激活虚拟环境：
```powershell
.\venv\Scripts\Activate.ps1
```

4. 验证安装：
```bash
python -c "import torch; print(torch.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

## 模型开发

### 创建新模型

在 `src/models/` 目录下创建新的模型文件：

```python
from src.models.unet_light import LightUNet

# 创建模型实例
model = LightUNet(
    in_channels=3,
    out_channels=3,
    base_channels=32,
    depth=4,
    audio_feature_dim=256
)
```

### 训练模型

1. 配置训练参数：
编辑 `configs/train_config.yaml`

2. 开始训练：
```bash
python -m src.training.train --config configs/train_config.yaml
```

### 测试推理

```python
from src.inference.mobile_inference import MobileDigitalHumanInference

engine = MobileDigitalHumanInference(
    model_path="checkpoints/best_model.pth",
    device="cuda"
)

# 执行推理
result = engine.infer(reference_image, audio_features)
```

## Android开发

### 构建Android应用

1. 在Android Studio中打开 `android/` 目录

2. 同步Gradle：
```
File -> Sync Project with Gradle Files
```

3. 构建APK：
```
Build -> Build Bundle(s) / APK(s) -> Build APK(s)
```

### 集成自定义模型

1. 导出ONNX模型：
```python
from src.inference.model_export import ModelExporter

ModelExporter.export_to_android(
    model=model,
    output_dir="android/app/src/main/assets"
)
```

2. 复制到assets目录：
```bash
cp model.onnx android/app/src/main/assets/
```

3. 重新构建应用

## 代码规范

### Python

- 使用Black进行代码格式化
- 使用flake8进行代码检查
- 使用类型注解

```bash
# 格式化代码
black src/

# 检查代码
flake8 src/
```

### Kotlin

- 遵循Kotlin编码规范
- 使用Kotlin Android Extensions

## 测试

### 单元测试
```bash
pytest tests/unit_tests/
```

### 集成测试
```bash
pytest tests/integration_tests/
```

### 性能测试
```bash
python -m src.inference.benchmark --model_path checkpoints/best_model.pth
```

## 调试

### Python调试

使用IDE调试器（如VSCode）进行调试：

```json
{
    "name": "Python: Current File",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal"
}
```

### Android调试

使用Logcat查看日志：
```bash
adb logcat -s DigitalHuman:D *:S
```

## 下一步

- [部署指南](./deployment.md) - 了解如何部署模型
- [API参考](./api-reference.md) - 查看API文档
