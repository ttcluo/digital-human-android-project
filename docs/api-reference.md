# API参考

本文件提供完整的API文档。

## Python API

### 模型模块 (`src/models/`)

#### LightUNet

轻量化U-Net模型，用于数字人生成。

```python
from src.models.unet_light import LightUNet

model = LightUNet(
    in_channels: int = 3,        # 输入通道数
    out_channels: int = 3,        # 输出通道数
    base_channels: int = 32,     # 基础通道数
    depth: int = 4,              # 网络深度
    audio_feature_dim: int = 256 # 音频特征维度
)
```

**方法：**

- `forward(x, audio_features)` - 前向传播
- `get_num_params()` - 获取参数量
- `get_flops(input_size)` - 估算FLOPs

#### ASREncoder

音频特征编码器。

```python
from src.models.asr_encoder import ASREncoder

encoder = ASREncoder(
    encoder_type: str = "hubert",  # "hubert" 或 "wenet"
    feature_dim: int = 256,
    sample_rate: int = 16000,
    frame_length: int = 25
)
```

**方法：**

- `forward(audio_waveform, video_frames)` - 提取音频特征
- `extract_features_offline(audio_path, device)` - 离线特征提取
- `get_frame_features(audio_features, frame_idx)` - 获取单帧特征

### 训练模块 (`src/training/`)

#### Trainer

模型训练器。

```python
from src.training.trainer import Trainer

trainer = Trainer(
    model: nn.Module,           # 模型
    train_loader: DataLoader,   # 训练数据加载器
    val_loader: DataLoader,     # 验证数据加载器
    criterion: nn.Module,      # 损失函数
    optimizer: optim.Optimizer, # 优化器
    device: str = "cuda",       # 设备
    save_dir: str,              # 保存目录
    log_dir: str,               # 日志目录
    use_tensorboard: bool = True,
    use_wandb: bool = False
)
```

**方法：**

- `train(num_epochs, start_epoch, resume_from)` - 开始训练
- `train_epoch()` - 训练一个epoch
- `validate()` - 验证模型
- `save_checkpoint(is_best)` - 保存检查点
- `load_checkpoint(checkpoint_path)` - 加载检查点

#### DataLoaderWrapper

数据加载器包装器。

```python
from src.training.data_loader import DataLoaderWrapper

loader = DataLoaderWrapper(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 256,
    train_split: float = 0.9
)

train_loader = loader.get_train_loader()
val_loader = loader.get_val_loader()
```

### 推理模块 (`src/inference/`)

#### MobileDigitalHumanInference

移动端推理引擎。

```python
from src.inference.mobile_inference import MobileDigitalHumanInference

engine = MobileDigitalHumanInference(
    model_path: str,
    device: str = "cpu",
    input_size: int = 256,
    audio_feature_dim: int = 256
)
```

**方法：**

- `infer(reference_image, audio_features)` - 执行推理
- `stream_infer(reference_images, audio_stream)` - 流式推理
- `export_to_tflite(output_path)` - 导出TFLite
- `export_to_onnx(output_path)` - 导出ONNX
- `get_model_info()` - 获取模型信息

#### ModelExporter

模型导出工具。

```python
from src.inference.model_export import ModelExporter

# 导出TorchScript
ModelExporter.export_to_torchscript(model, "model.pt")

# 导出ONNX
ModelExporter.export_to_onnx(model, "model.onnx")

# 导出Android
ModelExporter.export_to_android(model, "./android/assets")
```

#### ModelQuantizer

模型量化工具。

```python
from src.inference.quantize_model import ModelQuantizer

# 动态量化
quantized = ModelQuantizer.quantize_dynamic(model)

# 静态量化
quantized = ModelQuantizer.quantize_static(model, calibration_data)

# 模型剪枝
pruned = ModelQuantizer.prune_model(model, amount=0.3)

# 获取模型大小
size_info = ModelQuantizer.get_model_size(model)
```

### 工具模块 (`src/utils/`)

#### AudioProcessor

音频处理工具。

```python
from src.utils.audio_utils import AudioProcessor

processor = AudioProcessor(sample_rate=16000)

audio, sr = processor.load_audio("audio.wav")
mfcc = processor.extract_mfcc(audio)
spec = processor.extract_spectrogram(audio)
```

#### VideoProcessor

视频处理工具。

```python
from src.utils.video_utils import VideoProcessor

processor = VideoProcessor(fps=25)

frames, fps = processor.read_video("video.mp4")
processor.write_video(frames, "output.mp4")
processor.extract_frames("video.mp4", "./frames")
```

#### MetricsCalculator

评估指标计算器。

```python
from src.utils.metrics import MetricsCalculator

calculator = MetricsCalculator()
metrics = calculator.compute_all(pred, target)

# metrics包含:
# - psnr: 峰值信噪比
# - ssim: 结构相似性
# - lpips: 感知相似性
# - mse: 均方误差
```

## Android API

### DigitalHumanEngine

Android推理引擎。

```kotlin
val engine = DigitalHumanEngine.getInstance()

// 初始化
val success = engine.initialize(
    context = this,
    modelFileName = "digital_human.onnx",
    numThreads = 4,
    useGpu = false,
    inputSize = 256
)

// 执行推理
val output = engine.infer(inputBitmap, audioFeatures)

// 释放资源
engine.release()
```

### AudioProcessor

Android音频处理器。

```kotlin
val processor = AudioProcessor()

// 初始化
processor.initialize()

// 开始录音
processor.onAudioDataCallback = { features ->
    // 处理音频特征
}
processor.startRecording()

// 停止录音
processor.stopRecording()

// 释放
processor.release()
```

## 配置文件格式

### train_config.yaml

```yaml
model:
  name: "light_unet_v1"
  base_channels: 32
  depth: 4

training:
  batch_size: 8
  num_epochs: 200
  learning_rate: 0.001
  gpu_ids: [0, 1]

data:
  dataset_path: "./data/processed"
  image_size: 256

loss:
  reconstruction_weight: 1.0
  sync_weight: 0.5
```

### inference_config.yaml

```yaml
model:
  checkpoint_path: "./checkpoints/best_model.pth"

inference:
  device: "cuda"
  batch_size: 1

input:
  reference_images_dir: "./data/reference"
  audio_path: "./data/input/audio.wav"

output:
  output_dir: "./outputs"
  fps: 25
```
