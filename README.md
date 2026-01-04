# 远程声音控制系统 (Remote Voice Control System)

这是一个基于 Python 实现的远程声音控制系统仿真项目。系统集成了语音信号采集、数字通信传输（BPSK调制解调）、语音识别以及控制系统执行等完整流程。

## 项目功能

*   **语音信号处理**: 录音、量化、Huffman信源编码。
*   **数字通信仿真**:
    *   信道编码：Reed-Solomon (RS) 编码。
    *   调制解调：BPSK 调制 + 升余弦滚降 (RRC) 成型滤波。
    *   信道模拟：AWGN 高斯白噪声信道。
    *   接收端：相干解调 + 积分判决。
*   **语音识别**: 基于 MFCC 特征和 GMM/SVM 的命令词识别。
*   **控制系统**: PID 控制器驱动的直流电机模型，支持可视化和阶跃响应分析。
*   **图形用户界面 (GUI)**: 提供模型训练和系统仿真两个独立的 GUI 工具。

## 环境依赖

请确保安装以下 Python 库：

```bash
pip install numpy scipy matplotlib sounddevice soundfile scikit-learn
```

*注意：Linux 系统下使用 `sounddevice` 可能需要安装 `libportaudio2` (例如: `sudo apt-get install libportaudio2`)。*

## 快速开始

### 1. 训练语音模型 (`src/train_gui.py`)

在运行主程序之前，建议先录制并训练适合您声音的语音模型。

1.  运行训练工具：
    ```bash
    python src/train_gui.py
    ```
2.  **录制数据**:
    *   在界面上方选择要录制的命令（前进、后退、停止、旋转）。
    *   点击并按住 **"按住说话 (Hold to Record)"** 按钮进行录音。
    *   松开按钮结束录音。
    *   点击 **"保存样本 (Save Sample)"** 将录音保存到训练集。
    *   建议每个命令录制 5-10 个样本。
3.  **训练模型**:
    *   录制完成后，点击右侧的 **"训练模型 (Train Model)"** 按钮。
    *   系统将提取特征并训练分类器，模型会自动保存到 `data/speech_model.pkl`。

### 2. 运行控制系统 (`src/gui_app.py`)

这是系统的主仿真界面，展示了从语音输入到控制执行的全过程。

1.  运行主程序：
    ```bash
    python src/gui_app.py
    ```
2.  **语音输入**:
    *   点击并按住 **"按住说话 (Hold to Speak)"** 按钮输入语音命令。
    *   或者点击 **"选择文件"** 加载预录制的音频文件。
3.  **参数设置**:
    *   **信噪比 (SNR)**: 调节滑动条改变信道噪声水平，观察误码率 (BER) 的变化。
    *   **识别模式**: 默认使用本地训练的模型。
4.  **结果分析**:
    *   **波形显示**: 界面左侧显示原始录音波形和经过通信传输、解码后的波形。
    *   **详细分析**: 点击 **"详细分析 (Analysis)"** 按钮，查看各阶段的信号波形、频谱图、语谱图以及 Huffman 编码效率和误码率统计。
    *   **控制响应**: 界面右侧的小车模型会根据识别到的命令（前进、后退、旋转）进行动画演示。
    *   **阶跃响应**: 点击 **"阶跃响应 (Step Resp)"** 按钮，查看 PID 控制系统的阶跃响应曲线。

## 项目结构

```
project-python/
├── data/               # 数据文件（模型、录音样本等）
├── src/                # 源代码
│   ├── main.py         # 系统主逻辑类
│   ├── gui_app.py      # 主仿真 GUI
│   ├── train_gui.py    # 模型训练 GUI
│   ├── modulation.py   # BPSK 调制解调模块
│   ├── control_system.py # PID 控制系统模块
│   ├── signal_processing.py # 信号处理模块
│   ├── speech_recognition_system.py # 语音识别模块
│   └── ...
└── README.md           # 项目说明文档
```

## 关键技术参数

*   **采样率**: 44.1 kHz
*   **载波频率**: 10 kHz (可配置)
*   **调制方式**: BPSK (双极性不归零码 + RRC成型)
*   **滚降系数**: $\alpha = 0.3$
*   **信道编码**: RS(255, 191)
*   **控制算法**: PID (Kp, Ki, Kd 可调)
