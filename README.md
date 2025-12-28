"""
远程声音控制系统使用指南
Remote Voice Control System User Guide
"""

# 远程声音控制系统

## 系统概述

本项目实现了一个完整的远程声音控制系统，包含以下11个主要处理阶段：

### 处理流程图
```
┌─────────────────────────────────────────────────────────────────┐
│                    远程声音控制系统架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 语音采样 (Signal Sampling)                                  │
│        ↓                                                         │
│  2. 量化处理 (Signal Quantization)                              │
│        ↓                                                         │
│  3. 信源编码 - Huffman (Source Encoding)                        │
│        ↓                                                         │
│  4. 信道编码 - BCH(15,7) (Channel Encoding)                     │
│        ↓                                                         │
│  5. 调制 - BPSK (Modulation)                                    │
│        ↓                                                         │
│  6. 信道传输 - AWGN (Channel Transmission)                      │
│        ↓                                                         │
│  7. 解调 - BPSK (Demodulation)                                  │
│        ↓                                                         │
│  8. 信道解码 - BCH (Channel Decoding)                           │
│        ↓                                                         │
│  9. 信源解码 - Huffman (Source Decoding)                        │
│        ↓                                                         │
│  10. 语音识别 (Speech Recognition)                             │
│        ↓                                                         │
│  11. 控制系统执行 - PID (Control System)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 主要模块说明

### 1. signal_processing.py - 信号处理模块
**功能：** 音频采样和量化
- `SignalProcessing` 类处理音频文件的读取、采样和量化
- 支持自定义采样频率（默认44100 Hz）和量化位数（默认8位）

**主要方法：**
```python
processor = SignalProcessing(fs=44100, quantization_bits=8)
sampled_signal, fs = processor.signal_sampling('audio.flac')
quantized_signal = processor.signal_quantization(sampled_signal)
bit_sequence = processor.quantization_to_bits(quantized_signal)
```

### 2. source_encoding.py - 信源编码模块
**功能：** Huffman无损压缩编码
- 构建Huffman树进行信号压缩
- 提供编码和解码功能
- 计算压缩率、熵等统计信息

**主要方法：**
```python
encoder = SourceEncoding()
encoded_bits, code_dict, stats = encoder.encode(quantized_signal)
decoded_signal = encoder.decode(encoded_bits)
```

**统计输出：**
- 压缩比率 (Compression Ratio)
- 信息熵 (Entropy)
- 平均码长 (Average Code Length)
- 编码效率 (Coding Efficiency)

### 3. channel_encoding.py - 信道编码模块
**功能：** BCH纠错编码
- 实现(15,7) BCH编码器和解码器
- 能纠正3个比特错误
- 支持硬判决解码

**主要方法：**
```python
coder = BCHCoder(n=15, k=7)
encoded_bits = coder.encode(source_codes)
decoded_bits = coder.decode(received_bits)
```

**参数说明：**
- n: 码字长度（比特）
- k: 信息位长度（比特）
- t: 纠错能力 = (n-k)/2

### 4. modulation.py - 调制解调模块
**功能：** BPSK调制和解调，信道模型
- 二进制相移键控(BPSK)调制
- 高斯白噪声(AWGN)信道模型
- 硬判决和软判决解调

**主要方法：**
```python
modem = Modulation()
modulated = modem.bpsk_modulate(bits)
received = modem.channel_transmission(modulated, snr_db=20)
demodulated = modem.bpsk_demodulate(received)
stats = modem.get_error_statistics(original_bits, received_bits)
```

**信道模型：**
- AWGN（高斯白噪声）
- Rayleigh衰落
- 脉冲噪声

### 5. speech_recognition.py - 语音识别模块
**功能：** 命令识别
- 提取MFCC和频谱特征
- 简单的命令分类器
- 启发式识别方法

**主要方法：**
```python
sys = SpeechRecognitionSystem(fs=44100, num_commands=4)
command, confidence = sys.recognize_command(signal)
```

**支持的命令：**
- Forward（前进）
- Backward（后退）
- Stop（停止）
- Rotate（旋转）

### 6. control_system.py - 控制系统模块
**功能：** PID控制和系统仿真
- PID控制器
- 直流电动机模型
- 性能指标计算

**主要方法：**
```python
control_sys = ControlSystem(system_type='motor')
control_sys.execute_command('forward', duration=1.0)
metrics = control_sys.get_performance_metrics()
```

**系统指标：**
- 稳态误差 (Steady-State Error)
- 超调量 (Overshoot)
- 上升时间 (Rise Time)
- 平均误差 (Mean Error)

### 7. main.py - 主程序
**功能：** 集成所有模块，实现完整流程

**主要类：**
```python
system = RemoteVoiceControlSystem(config)
result = system.process_audio_file('audio.flac', snr_db=20)
results = system.process_multiple_snr_levels('audio.flac', snr_levels=[0, 10, 20, 30])
system.save_results(result, 'output.json')
```

## 使用方法

### 环境配置

**所需的Python包：**
```bash
pip install numpy scipy soundfile matplotlib
```

### 运行系统

**基本使用：**
```bash
cd src
python main.py
```

**自定义配置：**
```python
from main import RemoteVoiceControlSystem

config = {
    'fs': 44100,
    'quantization_bits': 8,
    'channel_code_n': 15,
    'channel_code_k': 7,
    'snr_db': 20,
    'num_commands': 4,
}

system = RemoteVoiceControlSystem(config=config)
result = system.process_audio_file('audio.flac', snr_db=20)
```

### 测试各个模块

**测试信号处理：**
```python
from signal_processing import SignalProcessing

processor = SignalProcessing(fs=44100, quantization_bits=8)
sampled, fs = processor.signal_sampling('../data/command.flac')
quantized = processor.signal_quantization(sampled)
```

**测试Huffman编码：**
```python
from source_encoding import SourceEncoding
import numpy as np

encoder = SourceEncoding()
test_signal = np.array([0, 1, 2, 1, 0, 3, 1, 0] * 10)
encoded, dict_table, stats = encoder.encode(test_signal)
decoded = encoder.decode(encoded)
```

**测试BCH编码：**
```python
from channel_encoding import BCHCoder
import numpy as np

coder = BCHCoder(n=15, k=7)
test_bits = np.array([1, 0, 1, 1, 0, 1, 0] * 10)
encoded = coder.encode(test_bits)
decoded = coder.decode(encoded)
```

**测试调制解调：**
```python
from modulation import Modulation
import numpy as np

modem = Modulation()
test_bits = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 10)
modulated = modem.bpsk_modulate(test_bits)
received = modem.channel_transmission(modulated, snr_db=20)
demodulated = modem.bpsk_demodulate(received)
```

## 输出文件说明

### 日志文件
- `remote_voice_control.log` - 系统运行日志

### 结果文件
- `results_YYYYMMDD_HHMMSS.json` - 单次处理结果
- `multi_snr_results.json` - 多SNR处理结果

### 结果JSON格式示例
```json
{
  "timestamp": "2024-01-01T12:00:00.000000",
  "audio_path": "../data/command.flac",
  "snr_db": 20,
  "overall_success": true,
  "stages": {
    "sampling": {...},
    "quantization": {...},
    "source_encoding": {...},
    "channel_encoding": {...},
    "modulation": {...},
    "channel_transmission": {...},
    "demodulation": {...},
    "channel_decoding": {...},
    "source_decoding": {...},
    "speech_recognition": {...},
    "control_execution": {...}
  },
  "summary": {...}
}
```

## 性能指标

### 1. 信源编码（Huffman）
- **压缩比率**：原始比特数 / 编码后比特数
- **信息熵**：每符号的平均信息量
- **编码效率**：熵 / 平均码长

### 2. 信道编码（BCH）
- **码率**：k / n
- **纠错能力**：t = (n-k)/2 = 4 (对于15,7码)

### 3. 通信信道（AWGN）
- **比特错误率（BER）**：错误比特数 / 总比特数
- **信噪比（SNR）**：信号功率 / 噪声功率

### 4. 控制系统
- **稳态误差**：最终位置与目标位置的偏差
- **超调量**：超过目标值的百分比
- **上升时间**：从10%到90%的时间

## 参数调整指南

### 提高准确率的方法

1. **增加信噪比（SNR）**
   - 增加SNR会降低BER，提高识别准确率
   - 推荐SNR ≥ 15 dB

2. **优化量化位数**
   - 增加`quantization_bits`可提高信号保真度
   - 权衡：位数越高，编码比特越多

3. **改进信源编码**
   - Huffman编码对非均匀分布的符号效果好
   - 可考虑使用其他编码如算术编码

4. **增强信道编码**
   - 降低码率k/n可提高纠错能力
   - 权衡：码率低则总比特数增加

5. **优化PID参数**
   - Kp：增加可提高响应速度
   - Ki：增加可消除稳态误差
   - Kd：增加可减少超调

## 常见问题

### Q: 找不到音频文件
A: 确保`data/command.flac`文件存在，或修改代码中的文件路径

### Q: 模块导入错误
A: 确保所有模块文件在`src`目录中

### Q: 识别命令失败
A: 检查SNR值、量化位数、BCH参数是否合适

### Q: 控制系统无响应
A: 调整PID参数或检查命令字符串格式

## 扩展功能

### 添加新的命令
在`control_system.py`的`_parse_command`方法中添加：
```python
'new_command': target_value,
```

### 支持更多控制系统
实现新的系统类（如RoboticArm）继承基类，替换`control_system.py`中的系统模型

### 集成真实语音识别
替换`speech_recognition.py`中的启发式方法为深度学习模型（如CNN、LSTM）

## 参考资源

- **信息论**：Shannon熵、信息编码
- **通信原理**：调制解调、信道编码、BPSK
- **控制理论**：PID控制、系统稳定性分析
- **信号处理**：MFCC、频谱分析、窗函数

## 许可证

MIT License

## 作者

School Project - Remote Voice Control System

## 更新日志

- **v1.0** (2024): 初始版本
  - 完成11个处理阶段
  - 实现Huffman编码、BCH编码、BPSK调制
  - 集成PID控制系统
  - 支持多SNR测试
