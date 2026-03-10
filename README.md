# Ultralight Digital Human - Android端实时数字人推理系统

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Android-24+-green.svg"></a>
  <br>
</p>

基于Ultralight-Digital-Human项目的Android端实时数字人推理系统开发工作流。

## 项目简介

本项目提供了一个完整的数字人训练和部署解决方案：

- **轻量化模型**：基于Ultralight-Digital-Human的改进U-Net架构
- **移动端部署**：支持Android设备的实时推理
- **完整训练流程**：从数据准备到模型部署的端到端解决方案

## 功能特性

- 轻量化U-Net模型，支持移动端实时运行
- ASR音频编码器（HuBERT/WeNet）
- 唇形同步网络
- 完整的训练和推理代码
- Android移动端推理引擎（C++ + Kotlin）
- ONNX模型导出和量化
- GitHub Actions CI/CD流水线

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/digital-human-android-project.git
cd digital-human-android-project
```

### 2. 本地环境配置（Windows）

```powershell
# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 3. 服务器环境配置（Ubuntu）

```bash
# 运行服务器配置脚本（需要root权限）
sudo bash scripts/setup-server.sh
```

### 4. 数据预处理

```bash
# 准备训练视频
cd Ultralight-Digital-Human/data_utils
python process.py YOUR_VIDEO_PATH --asr hubert
```

### 5. 模型训练

```bash
# 在服务器上训练
bash scripts/train-eval.sh train
```

### 6. 模型导出

```python
from src.inference.mobile_inference import MobileDigitalHumanInference
from src.inference.model_export import ModelExporter

# 导出ONNX模型
ModelExporter.export_to_onnx(model, "digital_human.onnx")
```

### 7. Android部署

1. 将ONNX模型复制到 `android/app/src/main/assets/`
2. 使用Android Studio构建项目

## 项目结构

```
digital-human-android-project/
├── src/                      # 核心Python代码
│   ├── models/              # 模型定义
│   ├── training/            # 训练代码
│   ├── inference/           # 推理代码
│   └── utils/              # 工具函数
├── configs/                 # 配置文件
├── scripts/                 # 自动化脚本
├── android/                 # Android应用
│   ├── app/src/main/
│   │   ├── cpp/           # C++推理引擎
│   │   ├── java/          # Kotlin代码
│   │   └── assets/        # 模型文件
│   └── build.gradle
├── .github/workflows/      # CI/CD配置
├── requirements.txt         # Python依赖
└── README.md
```

## 技术栈

- **Python**: 3.10+
- **PyTorch**: 1.13.1
- **Android**: API 24+
- **ONNX Runtime**: 1.16.3
- **CUDA**: 11.7

## 文档

- [开发指南](./docs/development.md)
- [部署指南](./docs/deployment.md)
- [API参考](./docs/api-reference.md)

## License

Apache License 2.0
