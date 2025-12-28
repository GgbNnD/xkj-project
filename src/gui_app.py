"""
远程声音控制系统 - 可视化控制界面
GUI Application for Remote Voice Control System
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

# 导入系统模块
from main import RemoteVoiceControlSystem
from signal_processing import SignalProcessing

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("远程声音控制系统 (Remote Voice Control System)")
        self.root.geometry("1200x800")
        
        # 初始化系统
        self.system = RemoteVoiceControlSystem()
        self.signal_processor = SignalProcessing()
        
        # 状态变量
        self.is_recording = False
        self.recording_thread = None
        self.processing_queue = queue.Queue()
        
        # 创建界面
        self._create_layout()
        
        # 启动处理线程
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def _create_layout(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板 - 控制和波形
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 右侧面板 - 控制系统可视化和日志
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # --- 左侧内容 ---
        
        # 1. 控制按钮区
        control_frame = ttk.LabelFrame(left_panel, text="语音输入控制", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        self.btn_record = ttk.Button(control_frame, text="按住说话 (Hold to Speak)", command=self._toggle_record)
        self.btn_record.pack(side=tk.LEFT, padx=5)
        self.btn_record.bind('<ButtonPress-1>', self._start_recording)
        self.btn_record.bind('<ButtonRelease-1>', self._stop_recording)
        
        self.lbl_status = ttk.Label(control_frame, text="就绪", foreground="green")
        self.lbl_status.pack(side=tk.LEFT, padx=10)
        
        # 2. 参数设置区
        settings_frame = ttk.LabelFrame(left_panel, text="系统参数", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="信噪比 (SNR dB):").grid(row=0, column=0, padx=5)
        self.scale_snr = ttk.Scale(settings_frame, from_=-10, to=30, orient=tk.HORIZONTAL, length=200)
        self.scale_snr.set(20)
        self.scale_snr.grid(row=0, column=1, padx=5)
        self.lbl_snr_val = ttk.Label(settings_frame, text="20 dB")
        self.lbl_snr_val.grid(row=0, column=2, padx=5)
        self.scale_snr.configure(command=lambda v: self.lbl_snr_val.configure(text=f"{float(v):.1f} dB"))
        
        # 3. 波形显示区
        plot_frame = ttk.LabelFrame(left_panel, text="实时波形", padding="5")
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
        
        # 1. 识别结果区
        result_frame = ttk.LabelFrame(right_panel, text="识别结果", padding="10")
        result_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_command = ttk.Label(result_frame, text="等待指令...", font=("Arial", 24, "bold"), foreground="blue")
        self.lbl_command.pack(pady=10)
        
        self.lbl_confidence = ttk.Label(result_frame, text="置信度: -")
        self.lbl_confidence.pack()
        
        # 2. 控制系统可视化
        vis_frame = ttk.LabelFrame(right_panel, text="控制系统状态 (PID Motor)", padding="10")
        vis_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.vis_canvas = tk.Canvas(vis_frame, bg="white", height=300)
        self.vis_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绘制初始小车
        self.car_x = 150
        self.car_y = 150
        self.car_width = 40
        self.car_height = 60
        self.car_id = self.vis_canvas.create_rectangle(
            self.car_x - self.car_width/2, self.car_y - self.car_height/2,
            self.car_x + self.car_width/2, self.car_y + self.car_height/2,
            fill="blue", outline="black"
        )
        # 绘制方向指示
        self.arrow_id = self.vis_canvas.create_line(
            self.car_x, self.car_y, self.car_x, self.car_y - 40,
            arrow=tk.LAST, width=3, fill="red"
        )
        
        # 3. 日志区
        log_frame = ttk.LabelFrame(right_panel, text="系统日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.txt_log = tk.Text(log_frame, height=10, width=40, font=("Courier", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.txt_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_log.config(yscrollcommand=scrollbar.set)
        
    def _log(self, message):
        self.txt_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.txt_log.see(tk.END)
        
    def _start_recording(self, event=None):
        if self.is_recording:
            return
        self.is_recording = True
        self.lbl_status.configure(text="正在录音...", foreground="red")
        self._log("开始录音...")
        
        # 启动录音线程
        self.recording_thread = threading.Thread(target=self._record_task)
        self.recording_thread.start()
        
    def _stop_recording(self, event=None):
        if not self.is_recording:
            return
        self.is_recording = False
        self.lbl_status.configure(text="处理中...", foreground="orange")
        self._log("停止录音，开始处理...")
        
    def _toggle_record(self):
        # 备用按钮点击处理
        pass
        
    def _record_task(self):
        # 实际录音逻辑
        # 这里我们模拟录音，或者使用 signal_processor.record_audio
        # 由于 record_audio 是阻塞的，我们需要修改它或者分段录制
        # 为了简单，我们这里录制固定时长，或者直到松开按钮
        
        # 注意：由于 sounddevice.rec 是非阻塞的，我们可以利用这一点
        # 但为了配合 "按住说话"，我们需要一个循环读取流的方法
        
        # 简化实现：录制固定时长（例如最长3秒），如果提前松开则截断
        # 但更好的方式是使用 sd.InputStream
        
        import sounddevice as sd
        
        fs = 44100
        max_duration = 5 # 最长5秒
        
        try:
            recording = sd.rec(int(max_duration * fs), samplerate=fs, channels=1, dtype='float64')
            
            # 等待直到 is_recording 变为 False
            start_time = time.time()
            while self.is_recording and (time.time() - start_time < max_duration):
                time.sleep(0.1)
            
            sd.stop()
            
            # 截取实际录制的长度
            actual_duration = time.time() - start_time
            actual_samples = int(actual_duration * fs)
            recorded_signal = recording[:actual_samples].flatten()
            
            # 发送到处理队列
            self.processing_queue.put((recorded_signal, fs))
            
        except Exception as e:
            self._log(f"录音错误: {e}")
            self.lbl_status.configure(text="录音失败", foreground="red")

    def _update_loop(self):
        while self.running:
            try:
                # 检查是否有待处理的音频
                if not self.processing_queue.empty():
                    signal, fs = self.processing_queue.get()
                    self._process_audio(signal, fs)
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Update loop error: {e}")

    def _process_audio(self, signal, fs):
        # 更新波形图
        self.ax.clear()
        self.ax.plot(signal)
        self.ax.set_title("Recorded Waveform")
        self.ax.set_ylim(-1, 1)
        self.canvas.draw()
        
        # 调用系统处理
        snr = self.scale_snr.get()
        self._log(f"正在处理信号 (SNR={snr:.1f}dB)...")
        
        try:
            result = self.system.process_audio_signal(signal, fs, snr_db=snr)
            
            if result['overall_success']:
                summary = result['summary']
                command = summary['recognized_command']
                
                # 更新界面
                self.root.after(0, lambda: self._update_ui_result(command, result))
            else:
                self._log(f"处理失败: {result.get('error')}")
                self.root.after(0, lambda: self.lbl_status.configure(text="处理失败", foreground="red"))
                
        except Exception as e:
            self._log(f"系统错误: {e}")

    def _update_ui_result(self, command, result):
        self.lbl_command.configure(text=command)
        
        # 获取置信度
        confidence = result['stages']['speech_recognition']['confidence']
        self.lbl_confidence.configure(text=f"置信度: {confidence:.2f}")
        
        # 获取BER
        ber = result['stages']['demodulation']['ber']
        self._log(f"识别: {command}, BER: {ber:.6f}")
        
        self.lbl_status.configure(text="就绪", foreground="green")
        
        # 执行可视化动画
        self._animate_control(command)

    def _animate_control(self, command):
        # 简单的动画模拟
        dx, dy = 0, 0
        rotation = 0
        
        if command == "Forward":
            dy = -50
        elif command == "Backward":
            dy = 50
        elif command == "Rotate":
            rotation = 90
        elif command == "Stop":
            pass
            
        # 动画步骤
        steps = 20
        delay = 50 # ms
        
        def step_anim(i):
            if i >= steps:
                return
            
            if rotation:
                # 旋转动画（简化：只改变箭头方向）
                # 实际旋转矩形比较复杂，这里只演示位移
                pass
            else:
                self.vis_canvas.move(self.car_id, dx/steps, dy/steps)
                self.vis_canvas.move(self.arrow_id, dx/steps, dy/steps)
            
            self.root.after(delay, lambda: step_anim(i+1))
            
        step_anim(0)
        
    def on_closing(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
