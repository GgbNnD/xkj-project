# 远程声音控制系统 - 快速启动指南

## 快速安装 (5分钟)

### 1. 安装依赖
```bash
cd /home/cells/school/xkj/project
pip install -r requirements.txt
```

### 2. 验证安装
```bash
python test_system.py
```

### 3. 运行系统
```bash
python src/main.py
```

---

## 常见命令

### 运行主程序
```bash
cd src
python main.py
```

### 运行测试
```bash
python test_system.py
```

### 查看示例
```bash
python examples.py
```

### 单个模块测试
```bash
# 测试信号处理
cd src
python -c "from signal_processing import SignalProcessing; print('OK')"

# 测试Huffman编码
python -c "from source_encoding import SourceEncoding; print('OK')"

# 测试BCH编码
python -c "from channel_encoding import BCHCoder; print('OK')"

# 测试调制
python -c "from modulation import Modulation; print('OK')"
```

---

## 目录结构说明

```
project/
├── README.md                    # 主文档
├── ARCHITECTURE.md              # 架构设计文档
├── QUICKSTART.md               # 本文件
├── requirements.txt            # Python依赖
├── test_system.py              # 测试脚本
├── examples.py                 # 使用示例
│
├── src/
│   ├── main.py                 # 主程序
│   ├── signal_processing.py    # 信号处理
│   ├── source_encoding.py      # Huffman编码
│   ├── channel_encoding.py     # BCH编码
│   ├── modulation.py           # BPSK调制
│   ├── speech_recognition.py   # 语音识别
│   └── control_system.py       # PID控制
│
├── data/
│   ├── command.flac            # 命令音频文件
│   ├── ChannelEncode.txt       # 参考数据
│   ├── ChannelDecode.txt       # 参考数据
│   ├── SourceEncode.txt        # 参考数据
│   ├── SourceDecode.txt        # 参考数据
│   ├── demodulation.txt        # 参考数据
│   └── SpeechRecognition.txt   # 参考数据
│
└── 伪代码/
    └── 伪代码.tex              # MATLAB参考代码
```

---

## Python版本要求

- Python 3.7+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- soundfile >= 0.10.0
- matplotlib >= 3.4.0 (可选，用于绘图)

---

## 第一次运行检查列表

- [ ] Python已安装 (`python --version`)
- [ ] 依赖已安装 (`pip list | grep numpy`)
- [ ] 音频文件存在 (`ls data/command.flac`)
- [ ] src目录中所有模块都存在
- [ ] 运行测试通过 (`python test_system.py`)

---

## 常见问题快速解决

### 问题1: "找不到模块"
**解决**：确保在项目根目录运行，或添加src到Python路径
```bash
export PYTHONPATH="${PYTHONPATH}:./src"
```

### 问题2: "找不到音频文件"
**检查**：
```bash
ls -la data/command.flac
```

### 问题3: "导入错误"
**解决**：重新安装依赖
```bash
pip install --upgrade -r requirements.txt
```

### 问题4: "权限被拒绝"
**解决**：更改文件权限
```bash
chmod +x src/main.py
```

---

## 输出说明

### 日志文件
运行后会自动生成：
- `remote_voice_control.log` - 详细日志

### 结果文件
处理后会生成：
- `results_YYYYMMDD_HHMMSS.json` - 处理结果
- `multi_snr_results.json` - 多SNR结果

查看结果：
```bash
cat results_*.json | python -m json.tool
```

---

## 性能预期

### 处理时间
- 单个音频文件：10-30秒
- 多SNR处理(7个level)：2-3分钟

### 准确率
- SNR=20dB时：识别准确率~80%
- SNR=30dB时：识别准确率~95%
- SNR<10dB时：准确率快速下降

### 系统资源
- 内存使用：<500MB
- CPU占用：单核100%（处理期间）

---

## 下一步

### 初学者路径
1. 阅读 README.md
2. 运行 test_system.py
3. 查看 examples.py
4. 修改参数进行实验

### 高级用户路径
1. 阅读 ARCHITECTURE.md
2. 研究源代码实现
3. 自定义配置参数
4. 扩展新功能

### 研究人员路径
1. 分析各模块性能
2. 进行对比实验
3. 改进算法实现
4. 撰写论文/报告

---

## 核心概念速记

| 概念 | 说明 | 范围 |
|------|------|------|
| **采样** | 连续信号→离散信号 | 44.1kHz |
| **量化** | 连续幅度→离散幅度 | 8-bit |
| **Huffman** | 无损信源压缩 | 可变码长 |
| **BCH(15,7)** | 纠错编码 | 码率7/15 |
| **BPSK** | 二进制调制 | 0/1→±1 |
| **AWGN** | 噪声信道 | SNR参数 |
| **MFCC** | 语音特征 | 13维向量 |
| **PID** | 控制算法 | Kp/Ki/Kd |

---

## 快速参数调整

### 提高识别准确率
```python
config = {
    'snr_db': 25,              # ↑SNR
    'quantization_bits': 12,   # ↑量化位数
    'channel_code_k': 5,       # ↓码率(更强纠错)
}
```

### 减少延迟
```python
config = {
    'fs': 16000,               # ↓采样率
    'quantization_bits': 4,    # ↓量化位数
}
```

### 降低功耗
```python
config = {
    'quantization_bits': 4,    # ↓量化位数
    'num_commands': 2,         # ↓命令数
}
```

---

## 文档导航

| 文档 | 内容 | 适合 |
|------|------|------|
| README.md | 完整用户指南 | 所有人 |
| ARCHITECTURE.md | 技术设计细节 | 开发者 |
| QUICKSTART.md | 快速开始 | 新手 |
| 代码注释 | 实现细节 | 研究者 |

---

## 获取帮助

### 查看帮助信息
```python
# 在Python中
from signal_processing import SignalProcessing
help(SignalProcessing.signal_quantization)
```

### 查看日志
```bash
# 最后100行
tail -100 remote_voice_control.log

# 搜索错误
grep ERROR remote_voice_control.log
```

### 调试模式
```python
# 在main.py中修改日志级别
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 成功指示

系统正常运行的标志：
- ✓ test_system.py 所有测试通过
- ✓ 生成了 results_*.json 文件
- ✓ 日志中没有ERROR字样
- ✓ 识别了命令并执行了控制

---

**准备好了？开始吧！**
```bash
python src/main.py
```
