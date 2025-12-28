"""
远程声音控制系统 - 模型训练工具 (GUI版)
Model Training Tool (GUI Version) for Remote Voice Control System
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import logging
import queue
import os
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# 导入系统模块
from speech_recognition_system import SpeechRecognitionSystem, CommandClassifier
from signal_processing import SignalProcessing

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("语音模型训练工具 (Model Training Tool)")
        self.root.geometry("1000x700")
        
        # 初始化配置
        self.fs = 44100
        self.commands = ["前进", "后退", "停止", "旋转"]
        self.data_dir = "data/training"
        self.model_path = "data/speech_model.pkl"
        
        # 确保数据目录存在
        for cmd in self.commands:
            os.makedirs(os.path.join(self.data_dir, cmd), exist_ok=True)
            
        # 状态变量
        self.is_recording = False
        self.recording_thread = None
        self.processing_queue = queue.Queue()
        self.selected_command = tk.StringVar(value=self.commands[0])
        self.last_recorded_signal = None
        
        # 初始化识别系统（用于特征提取）
        self.recognition_system = SpeechRecognitionSystem(fs=self.fs)
        self.signal_processor = SignalProcessing(fs=self.fs, quantization_bits=8)
        
        # 创建界面
        self._create_layout()
        
        # 启动更新线程
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # 刷新样本计数
        self._refresh_counts()
        
    def _create_layout(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：控制区
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 右侧：信息区
        right_panel = ttk.Frame(main_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # --- 左侧内容 ---
        
        # 1. 命令选择
        cmd_frame = ttk.LabelFrame(left_panel, text="1. 选择要录制的命令", padding="10")
        cmd_frame.pack(fill=tk.X, pady=5)
        
        for cmd in self.commands:
            rb = ttk.Radiobutton(cmd_frame, text=cmd, variable=self.selected_command, value=cmd)
            rb.pack(side=tk.LEFT, padx=10)
            
        # 2. 录音控制
        rec_frame = ttk.LabelFrame(left_panel, text="2. 录制语音", padding="10")
        rec_frame.pack(fill=tk.X, pady=5)
        
        self.btn_record = ttk.Button(rec_frame, text="按住说话 (Hold to Record)")
        self.btn_record.pack(side=tk.LEFT, padx=5)
        self.btn_record.bind('<ButtonPress-1>', self._start_recording)
        self.btn_record.bind('<ButtonRelease-1>', self._stop_recording)
        
        self.btn_play = ttk.Button(rec_frame, text="回放 (Playback)", command=self._play_recording, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_save = ttk.Button(rec_frame, text="保存样本 (Save)", command=self._save_sample, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)
        
        self.btn_discard = ttk.Button(rec_frame, text="丢弃 (Discard)", command=self._discard_sample, state=tk.DISABLED)
        self.btn_discard.pack(side=tk.LEFT, padx=5)
        
        self.lbl_status = ttk.Label(rec_frame, text="就绪", foreground="gray")
        self.lbl_status.pack(side=tk.LEFT, padx=10)
        
        # 3. 波形显示
        plot_frame = ttk.LabelFrame(left_panel, text="波形预览", padding="5")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Audio Waveform")
        self.ax.set_ylim(-1, 1)
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- 右侧内容 ---
        
        # 1. 样本统计
        stats_frame = ttk.LabelFrame(right_panel, text="样本统计", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.count_labels = {}
        for cmd in self.commands:
            f = ttk.Frame(stats_frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=cmd).pack(side=tk.LEFT)
            lbl = ttk.Label(f, text="0", font=("Arial", 10, "bold"))
            lbl.pack(side=tk.RIGHT)
            self.count_labels[cmd] = lbl
            
        # 2. 训练控制
        train_frame = ttk.LabelFrame(right_panel, text="3. 模型训练", padding="10")
        train_frame.pack(fill=tk.X, pady=5)
        
        self.btn_train = ttk.Button(train_frame, text="开始训练模型", command=self._train_model)
        self.btn_train.pack(fill=tk.X, pady=5)
        
        # 3. 日志
        log_frame = ttk.LabelFrame(right_panel, text="日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.txt_log = tk.Text(log_frame, height=10, width=30, font=("Courier", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        
    def _log(self, message):
        self.txt_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.txt_log.see(tk.END)
        
    def _refresh_counts(self):
        """刷新样本计数"""
        for cmd in self.commands:
            cmd_dir = os.path.join(self.data_dir, cmd)
            count = len([f for f in os.listdir(cmd_dir) if f.endswith('.wav')])
            self.count_labels[cmd].configure(text=str(count))
            
    def _start_recording(self, event=None):
        if self.is_recording:
            return
        self.is_recording = True
        self.lbl_status.configure(text="正在录音...", foreground="red")
        self.btn_save.configure(state=tk.DISABLED)
        self.btn_discard.configure(state=tk.DISABLED)
        self.btn_play.configure(state=tk.DISABLED)
        
        self.recording_thread = threading.Thread(target=self._record_task)
        self.recording_thread.start()
        
    def _stop_recording(self, event=None):
        if not self.is_recording:
            return
        self.is_recording = False
        self.lbl_status.configure(text="处理中...", foreground="orange")
        
    def _record_task(self):
        max_duration = 5.0 # 最长5秒
        
        try:
            # 强制停止之前的流
            sd.stop()
            
            recording = sd.rec(int(max_duration * self.fs), samplerate=self.fs, channels=1, dtype='float64')
            
            start_time = time.time()
            while self.is_recording and (time.time() - start_time < max_duration):
                time.sleep(0.05)
            
            sd.stop()
            
            # 截取实际长度
            actual_duration = time.time() - start_time
            # 稍微多截取一点点以防万一，或者直接用 actual_duration
            # 实际上 sd.rec 是非阻塞的，我们需要知道它录了多少
            # 但由于我们是手动停止的，我们只能估算。
            # 更好的方法是只保留有效部分，或者我们假设用户按住的时间就是有效时间
            
            actual_samples = int(actual_duration * self.fs)
            # 确保不越界
            actual_samples = min(actual_samples, len(recording))
            
            recorded_signal = recording[:actual_samples].flatten()
            
            self.processing_queue.put(recorded_signal)
            
        except Exception as e:
            self._log(f"录音错误: {e}")
            self.root.after(0, lambda: self.lbl_status.configure(text="录音失败", foreground="red"))

    def _update_loop(self):
        while self.running:
            try:
                if not self.processing_queue.empty():
                    signal = self.processing_queue.get()
                    self._process_recorded_audio(signal)
                time.sleep(0.1)
            except Exception as e:
                print(f"Update loop error: {e}")
                
    def _process_recorded_audio(self, signal):
        self.last_recorded_signal = signal
        
        # 更新波形
        self.ax.clear()
        self.ax.plot(signal)
        self.ax.set_title(f"Recorded: {len(signal)/self.fs:.2f}s")
        self.ax.set_ylim(-1, 1)
        self.canvas.draw()
        
        # 更新按钮状态
        self.btn_save.configure(state=tk.NORMAL)
        self.btn_discard.configure(state=tk.NORMAL)
        self.btn_play.configure(state=tk.NORMAL)
        self.lbl_status.configure(text="录音完成，请保存或丢弃", foreground="blue")
        
    def _play_recording(self):
        if self.last_recorded_signal is not None:
            try:
                sd.play(self.last_recorded_signal, self.fs)
            except Exception as e:
                self._log(f"回放失败: {e}")
                
    def _save_sample(self):
        if self.last_recorded_signal is None:
            return
            
        cmd = self.selected_command.get()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, cmd, f"{cmd}_{timestamp}.wav")
        
        try:
            sf.write(filename, self.last_recorded_signal, self.fs)
            self._log(f"已保存: {cmd}")
            self._refresh_counts()
            
            # 重置状态
            self.last_recorded_signal = None
            self.ax.clear()
            self.ax.grid(True)
            self.ax.set_ylim(-1, 1)
            self.canvas.draw()
            self.btn_save.configure(state=tk.DISABLED)
            self.btn_discard.configure(state=tk.DISABLED)
            self.btn_play.configure(state=tk.DISABLED)
            self.lbl_status.configure(text="就绪", foreground="green")
            
        except Exception as e:
            self._log(f"保存失败: {e}")
            messagebox.showerror("错误", f"保存文件失败: {e}")
            
    def _discard_sample(self):
        self.last_recorded_signal = None
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_ylim(-1, 1)
        self.canvas.draw()
        self.btn_save.configure(state=tk.DISABLED)
        self.btn_discard.configure(state=tk.DISABLED)
        self.btn_play.configure(state=tk.DISABLED)
        self.lbl_status.configure(text="已丢弃", foreground="gray")
        
    def _train_model(self):
        # 检查是否有足够的数据
        total_samples = 0
        for cmd in self.commands:
            cmd_dir = os.path.join(self.data_dir, cmd)
            count = len([f for f in os.listdir(cmd_dir) if f.endswith('.wav')])
            total_samples += count
            if count < 3:
                messagebox.showwarning("警告", f"命令 '{cmd}' 的样本太少 ({count})，建议至少录制5个。")
                return
        
        self.btn_train.configure(state=tk.DISABLED, text="训练中...")
        self.root.update()
        
        # 在新线程中训练，避免卡死UI
        threading.Thread(target=self._train_task).start()
        
    def _train_task(self):
        try:
            self._log("开始提取特征 (包含数据增强)...")
            features = []
            labels = []
            
            for cmd_idx, cmd in enumerate(self.commands):
                cmd_dir = os.path.join(self.data_dir, cmd)
                files = [f for f in os.listdir(cmd_dir) if f.endswith('.wav')]
                
                for f in files:
                    filepath = os.path.join(cmd_dir, f)
                    try:
                        signal, _ = sf.read(filepath)
                        
                        # 1. 原始信号特征
                        mfcc = self.recognition_system.feature_extractor.extract_mfcc_features(signal)
                        if mfcc.shape[0] > 0:
                            # feat = np.mean(mfcc, axis=0) # LSTM 不需要平均
                            features.append(mfcc)
                            labels.append(cmd_idx)
                            
                        # 2. 数据增强：模拟量化噪声 (模拟GUI中的处理流程)
                        # 量化
                        quantized = self.signal_processor.signal_quantization(signal)
                        # 转换为比特再转回 (模拟完整编解码过程)
                        bits = self.signal_processor.quantization_to_bits(quantized)
                        s_min, s_max = np.min(quantized), np.max(quantized)
                        reconstructed = self.signal_processor.bits_to_quantization(bits, s_min, s_max)
                        
                        mfcc_aug = self.recognition_system.feature_extractor.extract_mfcc_features(reconstructed)
                        if mfcc_aug.shape[0] > 0:
                            # feat_aug = np.mean(mfcc_aug, axis=0) # LSTM 不需要平均
                            features.append(mfcc_aug)
                            labels.append(cmd_idx)
                            
                    except Exception as e:
                        self._log(f"处理文件错误 {f}: {e}")
            
            if not features:
                self._log("没有有效数据")
                return
                
            # X = np.array(features) # LSTM 接受列表
            y = np.array(labels)
            
            self._log(f"开始训练 LSTM 模型 (样本数: {len(features)})...")
            metrics = self.recognition_system.classifier.train(features, y)
            
            self._log(f"训练完成! 准确率: {metrics['test_acc']:.2f}")
            messagebox.showinfo("成功", f"模型训练完成！\n测试集准确率: {metrics['test_acc']:.2f}")
            
        except Exception as e:
            self._log(f"训练失败: {e}")
            messagebox.showerror("错误", f"训练失败: {e}")
        finally:
            self.root.after(0, lambda: self.btn_train.configure(state=tk.NORMAL, text="开始训练模型"))

    def on_closing(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
