"""
Digital Human - 数字人项目

这是一个完整的数字人训练和部署解决方案，包含：
- 轻量化U-Net模型
- ASR音频编码器
- 唇形同步网络
- Android移动端推理引擎
"""

__version__ = "1.0.0"
__author__ = "Digital Human Team"

# 核心模块
from . import models
from . import training
from . import inference
from . import utils

__all__ = [
    "models",
    "training", 
    "inference",
    "utils",
]