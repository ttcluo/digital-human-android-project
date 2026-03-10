# 数字人项目 - 数据准备指南

## 📋 概述

本指南详细说明如何准备训练数字人模型所需的数据集。

## 🎯 数据要求

### 视频要求
- **时长**: 3-5分钟（建议）
- **帧率**: 
  - 使用 HuBERT: 25fps
  - 使用 WeNet: 20fps
- **分辨率**: 建议 512x512 或更高
- **内容要求**:
  - 每帧都必须包含完整人脸
  - 人脸清晰可见，无遮挡
  - 光线均匀，避免过曝或过暗
  - 背景简单，避免复杂背景干扰

### 音频要求
- **采样率**: 16kHz
- **质量要求**:
  - 声音清晰，无杂音
  - 无回声（避免在空旷房间录制）
  - 建议使用外接麦克风，而非设备自带麦克风
  - 避免背景噪声

### 推荐录制建议
- **前20秒**: 不说话，但可以做小幅度动作（用于流式推理素材）
- **说话内容**: 清晰、自然，包含各种口型变化
- **头部动作**: 自然的小幅度动作，避免过度运动

## 🛠️ 数据准备步骤

### 第一步: 准备视频文件

1. 创建数据目录:
```bash
cd /data/luochuan/digital-human-android-project
mkdir -p data/raw data/processed
```

2. 将您的训练视频放入 `data/raw/` 目录:
```bash
# 示例
cp /path/to/your/video.mp4 data/raw/
```

### 第二步: 安装数据处理依赖

确保已安装以下依赖:
```bash
conda activate digital-human

# 安装FFmpeg（如果尚未安装）
# Ubuntu: apt-get install ffmpeg
# CentOS: yum install ffmpeg

# 安装Python依赖
pip install opencv-python transformers soundfile librosa
```

### 第三步: 下载预训练模型

如果使用 WeNet 作为音频编码器，需要下载预训练模型:

```bash
cd /data/luochuan/digital-human-android-project

# 下载 WeNet encoder.onnx
# 从 https://drive.google.com/file/d/1e4Z9zS053JEWl6Mj3W9Lbc9GDtzHIg6b/view?usp=drive_link
# 下载后放到 digital-human-android-project/data_utils/

mkdir -p data_utils
# 将下载的 encoder.onnx 放入 data_utils/ 目录
```

### 第四步: 数据预处理

#### 方式一: 使用 Ultralight-Digital-Human 的原始工具

```bash
# 进入 Ultralight-Digital-Human 目录
cd /data/luochuan/Ultralight-Digital-Human

# 运行数据处理（使用 HuBERT）
python data_utils/process.py /data/luochuan/digital-human-android-project/data/raw/your_video.mp4 --asr hubert

# 或使用 WeNet
python data_utils/process.py /data/luochuan/digital-human-android-project/data/raw/your_video.mp4 --asr wenet
```

**处理后的文件结构**:
```
data/raw/
├── your_video.mp4          # 原始视频
├── aud.wav                  # 提取的音频（16kHz）
├── full_body_img/           # 提取的视频帧
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── landmarks/               # 人脸关键点
    ├── 0.lms
    ├── 1.lms
    └── ...
```

#### 方式二: 手动数据处理

如果自动化工具出现问题，可以手动处理:

**1. 提取音频:**
```bash
cd /data/luochuan/digital-human-android-project/data/raw/
ffmpeg -i your_video.mp4 -f wav -ar 16000 aud.wav
```

**2. 提取视频帧:**
```bash
mkdir -p full_body_img
ffmpeg -i your_video.mp4 -vf "fps=25" full_body_img/%d.jpg
```

**3. 提取音频特征 (HuBERT):**
```bash
# 使用 HuBERT 提取特征
cd /data/luochuan/Ultralight-Digital-Human/data_utils
python hubert.py --wav /data/luochuan/digital-human-android-project/data/raw/aud.wav
```

**4. 检测人脸关键点:**
```bash
# 检测关键点
python get_landmark.py /data/luochuan/digital-human-android-project/data/raw/
```

### 第五步: 准备训练数据集

将处理好的数据整理为训练格式:

```bash
cd /data/luochuan/digital-human-android-project

# 创建训练数据目录
mkdir -p data/processed/train_data

# 将处理好的数据复制到训练目录
cp -r data/raw/full_body_img data/processed/train_data/imgs
cp -r data/raw/landmarks data/processed/train_data/landmarks
cp data/raw/aud.wav data/processed/train_data/

# 如果使用 HubERT，复制特征文件
# 特征文件通常在 data_utils/ 目录生成
```

### 第六步: 准备训练列表文件

创建训练数据列表文件 `data/processed/train_data/train_list.txt`:

```
# 格式: 图像路径|音频特征路径|关键点路径
imgs/0.jpg|audio_features.npy|landmarks/0.lms
imgs/1.jpg|audio_features.npy|landmarks/1.lms
imgs/2.jpg|audio_features.npy|landmarks/2.lms
...
```

## 📁 最终数据结构

完成数据准备后，项目结构应该是:

```
digital-human-android-project/
├── data/
│   ├── raw/
│   │   └── your_video.mp4          # 原始视频
│   └── processed/
│       └── train_data/
│           ├── imgs/               # 视频帧
│           │   ├── 0.jpg
│           │   ├── 1.jpg
│           │   └── ...
│           ├── landmarks/          # 人脸关键点
│           │   ├── 0.lms
│           │   ├── 1.lms
│           │   └── ...
│           ├── audio_features.npy  # HuBERT/WeNet 音频特征
│           └── train_list.txt      # 训练列表
├── configs/
│   └── train_config.yaml          # 训练配置
├── src/
│   ├── models/
│   ├── training/
│   └── utils/
└── scripts/
    └── train-eval.sh              # 训练脚本
```

## 🔧 配置文件设置

修改训练配置文件以适应您的数据:

```bash
vim configs/train_config.yaml
```

关键配置项:

```yaml
# 数据配置
data:
  dataset_path: "./data/processed/train_data"  # 数据路径
  audio_sample_rate: 16000
  image_size: 256                               # 图像尺寸
  video_fps: 25                                 # 视频帧率
  train_split: 0.9                              # 训练集比例
  num_workers: 4                                # 数据加载线程数

# 模型配置
model:
  name: "light_unet_v1"
  base_channels: 32
  depth: 4
  audio_feature_dim: 256                        # HuBERT: 1024, WeNet: 256
  use_syncnet: true

# GPU配置（您有8个GPU）
training:
  batch_size: 4                                 # 每个GPU的批次大小
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]           # 使用全部8个GPU
  distributed: true
```

## 🚀 开始训练

数据准备好后，开始训练:

```bash
# 激活环境
conda activate digital-human

# 开始训练
bash scripts/train-eval.sh train
```

## 📊 数据质量检查

### 检查提取的图像
```bash
# 查看提取的帧
ls data/processed/train_data/imgs/ | head -10

# 随机查看一张图片
python -c "import cv2; import matplotlib.pyplot as plt; img = cv2.imread('data/processed/train_data/imgs/0.jpg'); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.show()"
```

### 检查关键点
```bash
# 查看关键点文件
head -5 data/processed/train_data/landmarks/0.lms
```

### 检查音频特征
```bash
# 检查音频特征形状
python -c "import numpy as np; features = np.load('data/processed/train_data/audio_features.npy'); print(f'特征形状: {features.shape}')"
```

## 🔍 常见问题

### Q1: 视频帧率不符合要求怎么办？
```bash
# 转换视频帧率到25fps
ffmpeg -i input.mp4 -vf "fps=25" output_25fps.mp4
```

### Q2: 如何检查视频质量？
```bash
# 检查视频信息
ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate,width,height -of csv=p=0 input.mp4
```

### Q3: 关键点检测失败怎么办？
- 检查人脸是否清晰可见
- 确保光线充足均匀
- 调整检测阈值或更换检测模型

### Q4: 音频特征提取失败？
- 确认音频采样率为16kHz
- 检查音频文件是否损坏
- 确认有足够的GPU内存

### Q5: 如何准备多个视频？
```bash
# 批量处理多个视频
for video in /path/to/videos/*.mp4; do
    python data_utils/process.py "$video" --asr hubert
done
```

## 💡 最佳实践

1. **视频质量**: 使用高质量摄像头录制，避免压缩过度的视频
2. **音频质量**: 使用专业麦克风，在安静环境录制
3. **数据多样性**: 包含不同的表情、口型、头部动作
4. **时长控制**: 3-5分钟足够，过长会增加训练时间
5. **预处理检查**: 在训练前仔细检查预处理结果
6. **备份数据**: 原始视频和预处理结果都要备份

## 📝 训练数据示例

如果您没有现成的视频，可以:

1. **使用手机录制**: 3-5分钟自拍视频，说话清晰
2. **下载示例数据**: 从项目提供的链接下载示例视频
3. **使用公开数据集**: 如 LRS2、LRS3 等唇读数据集

## 🔗 相关资源

- [FFmpeg文档](https://ffmpeg.org/documentation.html)
- [HuBERT论文](https://arxiv.org/abs/2106.07447)
- [WeNet项目](https://github.com/wenet-e2e/wenet)
- [数字人项目README](./README.md)

## 📞 技术支持

如遇到问题:
1. 检查日志文件: `logs/` 目录
2. 验证数据格式是否符合要求
3. 确认所有依赖已正确安装
4. 参考 [github-css-troubleshooting.md](../github-css-troubleshooting.md)

---

**准备好数据后，运行 `bash scripts/train-eval.sh train` 开始训练！** 🚀