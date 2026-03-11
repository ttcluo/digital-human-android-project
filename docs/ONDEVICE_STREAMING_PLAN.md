## 端侧实时数字人总体规划（无服务器流式）

### 1. 目标与约束

- **目标**：
  - 在 Android 端本地完成：音频采集 → 音频编码 → 图像生成 → 渲染。
  - 实现端到端实时数字人，端侧帧率目标 ≥ 20 FPS，端到端延迟 \< 100 ms（理想 \< 50 ms）。
- **约束**：
  - 不依赖服务器（服务器仅用于离线训练与模型导出）。
  - 模型需可导出 ONNX，并由现有 C++/JNI 推理引擎加载。
  - 仅使用端上常见加速路径（NNAPI / GPU / CPU），避免自定义算子。

---

### 2. 当前基线能力回顾

- **训练侧**：
  - 已基于 Ultralight-Digital-Human + WeNet 在服务器上完成训练与推理验证。
  - A800 上实测 U-Net（160×160，WeNet 模式）前向延迟约 **7.15 ms/帧 ≈ 140 FPS**。
- **音频侧**：
  - 现有 WeNet encoder.onnx + `wenet_infer.py` 支持离线/流式特征，但为 ASR 设计，端侧成本偏高。
- **Android 侧**：
  - 已有 C++ + ONNX Runtime 推理引擎与 Kotlin 封装，可加载 ONNX 模型并在端侧运行。

结论：**U-Net 本身已经足够轻；端上流式的主要问题在于音频编码器和端侧集成方式。**

---

### 3. 架构蓝图：纯端侧流式链路

1. **音频采集层（Android）**
   - 使用 `AudioRecord` 以 16kHz 单声道连续采样。
   - 按固定窗口（如 20–40ms）切分为帧，形成音频 chunk 队列。

2. **端侧音频编码器（新设计）**
   - 输入：最近一段音频帧（例如 200–400ms 的滑动窗口）。
   - 输出：低维音频表现特征（例如 `[T', D]`，`D` 在 32–64 范围）。
   - 模型结构：
     - 方案一：`Conv1d`/`DepthwiseConv1d` + 残差块（时间卷积网络 TCN）。
     - 方案二：Mel 频谱 + 小型 2D CNN。
   - 目标：远低于 WeNet/HuBERT 的参数量与 FLOPs，仅保留“嘴型驱动”所需的信息。

3. **视频生成 U-Net（轻量版）**
   - 参考 Ultralight `Model` 结构，保持 Mobile 化设计：
     - 编码器：倒残差块 + 深度可分离卷积，分辨率约 128–160。
     - 解码器：双线性上采样 + depthwise 卷积。
   - 多尺度音频融合：
     - 在 bottleneck 和中间层用 FiLM/条件卷积注入音频特征。
   - 适配接口：
     - 输入：`[B, 6, H, W]`（参考帧 + mask 帧）+ `[B, D_audio, H_a, W_a]` 或 `[B, T', D]`。
     - 输出：`[B, 3, H, W]`，输出范围 [-1, 1] 或 [0, 1]。

4. **渲染与同步（Android/Kotlin + C++）**
   - JNI 接口：
     - `pushAudio(short[] pcm)`：持续推送音频帧到 C++ 层；
     - `pullFrame()`：从 C++ 层拉取最新一帧合成图像（或在 C++ 中回调到 Java 层）。
   - 渲染：
     - 使用 `SurfaceView`/`TextureView` 或 OpenGL，每次收到新帧即刷新。
   - 同步策略：
     - 音频编码器与 U-Net 前向时间固定可预估（例如总计 30–50ms），通过固定延迟缓冲实现嘴型同步。

---

### 4. 训练侧设计（服务器上）

1. **数据格式**
   - 与当前 Ultralight 几乎一致：
     - 视频帧：`full_body_img/*.jpg`；
     - 人脸关键点：`landmarks/*.lms`；
     - 音频：`aud.wav` (16kHz)。
   - 但音频特征由新的端侧编码器生成，而不是 WeNet/HuBERT。

2. **模型拆分**
   - 音频编码器模块 `AudioEncoderLite`：
     - PyTorch 实现，输入原始波形或 Mel 频谱，输出 `[T', D]`。
   - 视频生成模块 `LightUNetOnDevice`：
     - 与端侧 U-Net 完全同构，方便权重迁移。

3. **损失与训练目标**
   - 图像重建损失（L1 / SmoothL1）。
   - 感知损失（VGG19 特征）用于细节、整体结构。
   - 可选：基于已有 SyncNet 或简单对比损失的唇形同步约束。

4. **导出与量化**
   - 将联合模型拆分导出：
     - 导出 `AudioEncoderLite` ONNX；
     - 导出 `LightUNetOnDevice` ONNX。
   - 对端侧模型进行：
     - 动态/静态量化（如 INT8、FP16）。
     - 确保所有算子被 ONNX Runtime / NNAPI 支持。

---

### 5. 端侧集成步骤（Android）

1. **集成音频编码器 ONNX 模型**
   - 在 C++ 层加载 `audio_encoder.onnx`：
     - 输入：浮点 PCM / Mel 特征。
     - 输出：音频特征 embedding。
   - 将音频特征缓存到一个环形缓冲区，供 U-Net 使用。

2. **集成 U-Net ONNX 模型**
   - 替换当前 Android 项目中使用的 U-Net ONNX：
     - 保持输入/输出张量结构与 C++ 接口匹配。
   - 启用 NNAPI / GPU：
     - 在 ORT SessionOptions 中启用适当的 Execution Provider。

3. **实时 pipeline**
   - 在 Kotlin 中：
     - 启动音频录制线程（采集 PCM）。
     - 定时调用 JNI，把新 audio chunk 送到 C++。
     - 渲染线程定时从 C++ 获取最新帧绘制到 UI。

---

### 6. 阶段性里程碑

1. **阶段一：端上 U-Net 验证**
   - 将当前 Ultralight 训练好的 U-Net 导出到 Android。
   - 在端上用固定音频特征（预先计算）驱动，验证端上 FPS。

2. **阶段二：新音频编码器 + U-Net 训练**
   - 在服务器上实现并训练 `AudioEncoderLite + LightUNetOnDevice`。
   - 评估服务器端单帧耗时与画质、唇形效果。

3. **阶段三：端上联调**
   - 导出新模型 ONNX，集成到 Android C++ 引擎。
   - 真机上连通：音频采集 → 编码器 → U-Net → 渲染。

4. **阶段四：性能优化**
   - 对高频算子进行 profile，减少不必要的通道/层数。
   - 调整分辨率（如 128×128 ↔ 160×160）权衡画质与帧率。

