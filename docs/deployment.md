# 部署指南

本指南介绍如何将数字人模型部署到不同平台。

## 服务器部署

### 环境要求

- Ubuntu 20.04/22.04
- NVIDIA GPU (RTX 3080+)
- 16GB+ RAM
- 100GB+ 存储空间

### 部署步骤

1. **配置服务器环境**：
```bash
# 复制项目到服务器
git clone https://github.com/your-repo/digital-human-android-project.git
cd digital-human-android-project

# 运行服务器配置脚本
sudo bash scripts/setup-server.sh
```

2. **启动训练服务**：
```bash
# 激活环境
conda activate digital-human

# 启动训练
bash scripts/train-eval.sh train
```

3. **使用Systemd管理服务**：
```ini
# /etc/systemd/system/digital-human.service
[Unit]
Description=Digital Human Training Service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/digital-human
ExecStart=/opt/miniconda/bin/python -m src.training.train
Restart=always

[Install]
WantedBy=multi-user.target
```

## Android部署

### 部署步骤

1. **准备模型**：
```python
# 导出ONNX模型
from src.inference.model_export import ModelExporter

ModelExporter.export_to_android(
    model=model,
    output_dir="./android/app/src/main/assets",
    input_size=256
)
```

2. **构建APK**：
```bash
cd android
./gradlew assembleRelease
```

3. **安装APK**：
```bash
adb install app/build/outputs/apk/release/app-release.apk
```

### 性能优化

1. **模型量化**：
```python
from src.inference.quantize_model import ModelQuantizer

quantized = ModelQuantizer.quantize_dynamic(model)
```

2. **使用NNAPI加速**：
在Android代码中启用：
```kotlin
val sessionOptions = SessionOptions()
sessionOptions.enableNnapi()
```

## Docker部署

### 构建Docker镜像

```bash
docker build -t digital-human:latest .
```

### 运行容器

```bash
# 训练模式
docker run -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  digital-human:latest \
  bash scripts/train-eval.sh train

# 推理模式
docker run -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  digital-human:latest \
  python -m src.inference.server
```

## 云端部署（可选）

### 使用腾讯云

1. 创建CVM实例
2. 配置GPU驱动
3. 部署模型服务

### 使用其他云服务

- AWS EC2 (GPU实例)
- Google Cloud Platform
- Azure VM

## 监控和维护

### 监控指标

- GPU利用率
- 内存使用
- 推理延迟
- 吞吐量

### 日志管理

使用ELK Stack或类似的日志系统收集和分析日志。

## 故障排除

### 常见问题

1. **CUDA不可用**
   - 检查NVIDIA驱动安装
   - 验证PyTorch CUDA版本

2. **内存不足**
   - 减小batch size
   - 使用梯度累积

3. **Android推理慢**
   - 使用量化模型
   - 启用NNAPI加速
