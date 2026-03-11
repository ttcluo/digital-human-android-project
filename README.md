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

# 安装完成后验证环境
bash scripts/verify.sh
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

## 阶段性结果：端侧 U-Net 性能与效果

基于单人 5 分钟训练视频（WeNet 特征），我们在服务器与 Android 端分别验证了原始 Ultralight U-Net 与端侧轻量 OnDeviceUNet 的性能与效果。

- **服务器 A800（PyTorch 直接前向）**
  - 原始 U-Net（输入 160×160）：约 **7.15 ms/帧（≈ 140 FPS）**
  - OnDeviceUNet（输入 128×128，通道 [8,16,32,48,64]）：约 **7.3 ms/帧（≈ 137 FPS）**
- **Android 端（ONNX Runtime FP32，真机）**
  - 原始 U-Net（160×160，对应 `unet_wenet_160.onnx`）：稳态约 **104 ms/帧（≈ 9.6 FPS）**
  - OnDeviceUNet（128×128，对应 `unet_ondevice_128.onnx`）：稳态多次实测约 **10–13 ms/帧（≈ 80–100 FPS）**

在视觉上，通过 `run_compare_ultra_ondevice_wenet.py` 生成的左右对比视频（左：原 U-Net，右：OnDeviceUNet）显示：

- 嘴型开合时机与形状几乎一致；
- 面部细节与整体稳定性在当前数据规模下未出现明显退化。

结论：在保持主观效果基本不变的前提下，**端上 U-Net 推理耗时从百毫秒级降到十毫秒级**，为后续完整端侧实时数字人（含音频编码 + 渲染）提供了充足的算力空间。

## 环境验证

### 快速验证
```bash
# 运行验证脚本
bash scripts/verify.sh

# 或运行详细验证
python3 scripts/verify-installation.py
```

### 代码同步流程

1. **本地开发**
   ```bash
   # 修改代码后提交
   git add .
   git commit -m "描述更改"
   git push origin main
   ```

2. **服务器拉取更新**
   ```bash
   # SSH到服务器
   ssh root@server_ip
   
   # 进入项目目录
   cd /data/luochuan/digital-human-android-project
   
   # 拉取最新代码
   git pull origin main
   
   # 验证更新
   bash scripts/verify.sh
   ```

### 自动化同步
```bash
# 使用同步脚本
bash scripts/sync-code.sh main server_ip push
```

## 文档

- [开发指南](./docs/development.md)
- [部署指南](./docs/deployment.md)
- [API参考](./docs/api-reference.md)

## License

Apache License 2.0
