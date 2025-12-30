"""
远程声音控制系统 - 可视化控制界面
GUI Application for Remote Voice Control System
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import logging
import queue
import sounddevice as sd

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
        self.last_recorded_signal = None
        self.last_decoded_signal = None
        self.last_fs = 44100
        
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
        
        self.btn_file = ttk.Button(control_frame, text="选择文件 (Select File)", command=self._select_file)
        self.btn_file.pack(side=tk.LEFT, padx=5)
        
        self.btn_play = ttk.Button(control_frame, text="回放录音 (Playback)", command=self._play_recording, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.btn_play_decoded = ttk.Button(control_frame, text="播放解码 (Play Decoded)", command=self._play_decoded, state=tk.DISABLED)
        self.btn_play_decoded.pack(side=tk.LEFT, padx=5)

        self.btn_analyze = ttk.Button(control_frame, text="详细分析 (Analysis)", command=self._open_analysis_window, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=5)

        self.btn_step_resp = ttk.Button(control_frame, text="阶跃响应 (Step Resp)", command=self._show_step_response)
        self.btn_step_resp.pack(side=tk.LEFT, padx=5)
        
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
        
        # 识别模式选择
        ttk.Label(settings_frame, text="识别模式:").grid(row=1, column=0, padx=5, pady=5)
        self.combo_mode = ttk.Combobox(settings_frame, values=["本地模型 (Local AI)"], state="readonly")
        self.combo_mode.set("本地模型 (Local AI)")
        self.combo_mode.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # 3. 波形显示区
        plot_frame = ttk.LabelFrame(left_panel, text="信号波形 (Signal Waveforms)", padding="5")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.fig = Figure(figsize=(5, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("Recorded Waveform")
        self.ax1.set_ylim(-1, 1)
        self.ax1.grid(True)
        
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Decoded Waveform (After Transmission)")
        self.ax2.set_ylim(-1, 1)
        self.ax2.grid(True)
        
        self.fig.tight_layout()
        
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
        
    def _play_decoded(self):
        """回放解码后的信号"""
        if self.last_decoded_signal is not None:
            self._log("正在播放解码信号...")
            try:
                sd.play(self.last_decoded_signal, self.last_fs)
            except Exception as e:
                self._log(f"播放失败: {e}")
        else:
            self._log("没有可播放的解码信号")

    def _select_file(self):
        """选择音频文件作为输入"""
        file_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("Audio Files", "*.wav *.flac"), ("All files", "*.*")]
        )
        
        if file_path:
            self._log(f"正在加载文件: {file_path}...")
            # 在新线程中加载，避免卡顿
            threading.Thread(target=self._load_file_task, args=(file_path,)).start()

    def _load_file_task(self, file_path):
        try:
            # 使用信号处理模块读取文件
            signal, fs = self.signal_processor.signal_sampling(file_path)
            
            self._log(f"文件加载成功 (fs={fs}, len={len(signal)})")
            # 发送到处理队列
            self.processing_queue.put((signal, fs))
            
        except Exception as e:
            self._log(f"加载文件失败: {e}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"无法加载文件: {e}"))

    def _log(self, message):
        self.txt_log.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.txt_log.see(tk.END)
        
    def _play_recording(self):
        """回放最后一次录音"""
        if self.last_recorded_signal is not None:
            self._log("正在回放录音...")
            try:
                sd.play(self.last_recorded_signal, self.last_fs)
                # sd.wait() # 不阻塞UI
            except Exception as e:
                self._log(f"回放失败: {e}")
        else:
            self._log("没有可回放的录音")

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
        import sounddevice as sd
        
        fs = 44100
        max_duration = 5 # 最长5秒
        
        try:
            # 强制停止之前的流，防止干扰
            sd.stop()
            time.sleep(0.1)
            
            recording = sd.rec(int(max_duration * fs), samplerate=fs, channels=1, dtype='float64')
            
            # 等待直到 is_recording 变为 False
            start_time = time.time()
            while self.is_recording and (time.time() - start_time < max_duration):
                time.sleep(0.05)
            
            sd.stop()
            
            # 截取实际录制的长度
            actual_duration = time.time() - start_time
            actual_samples = int(actual_duration * fs)
            # 确保不越界
            actual_samples = min(actual_samples, len(recording))
            
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
        # 保存最后一次录音
        self.last_recorded_signal = signal
        self.last_fs = fs
        
        # 启用回放按钮
        self.root.after(0, lambda: self.btn_play.configure(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_analyze.configure(state=tk.NORMAL))
        
        # 更新波形图 (录音)
        self.ax1.clear()
        self.ax1.plot(signal)
        self.ax1.set_title("Recorded Waveform")
        self.ax1.set_ylim(-1, 1)
        self.ax1.grid(True)
        
        # 清空解码波形
        self.ax2.clear()
        self.ax2.set_title("Decoded Waveform (Processing...)")
        self.ax2.grid(True)
        self.canvas.draw()
        
        # 检查信号强度
        max_amp = np.max(np.abs(signal))
        if max_amp < 0.05:
            self._log(f"警告: 录音信号太弱 (最大幅值: {max_amp:.3f})")
            self._log("请靠近麦克风或大声说话")
        
        # 调用系统处理
        snr = self.scale_snr.get()
        mode = self.combo_mode.get()
        self._log(f"正在处理信号 (SNR={snr:.1f}dB, Mode={mode})...")
        
        try:
            # 传递识别模式
            # use_online = "Online" in mode # 移除在线模式判断
            result = self.system.process_audio_signal(signal, fs, snr_db=snr)
            
            if result['overall_success']:
                summary = result['summary']
                command = summary['recognized_command']
                
                # 获取解码后的信号
                if 'reconstructed_signal' in result['stages']['speech_recognition']:
                    decoded_signal = result['stages']['speech_recognition']['reconstructed_signal']
                    self.last_decoded_signal = decoded_signal
                    
                    # 更新解码波形
                    self.ax2.clear()
                    self.ax2.plot(decoded_signal, color='orange')
                    self.ax2.set_title(f"Decoded Waveform (SNR={snr}dB)")
                    self.ax2.set_ylim(-1, 1)
                    self.ax2.grid(True)
                    self.canvas.draw()
                    
                    # 启用播放解码按钮
                    self.root.after(0, lambda: self.btn_play_decoded.configure(state=tk.NORMAL))
                
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
        
        # 支持中文和英文命令
        if command in ["Forward", "前进"]:
            dy = -50
        elif command in ["Backward", "后退"]:
            dy = 50
        elif command in ["Rotate", "旋转"]:
            rotation = 90
        elif command in ["Stop", "停止"]:
            pass
            
        # 动画步骤
        steps = 20
        delay = 50 # ms
        
        def step_anim(i):
            if i >= steps:
                return
            
            if rotation:
                # 旋转动画（改变箭头方向）
                angle_deg = rotation * (i + 1) / steps
                angle_rad = np.radians(angle_deg)
                
                # 箭头起点 (center)
                cx, cy = self.car_x, self.car_y
                # 箭头长度
                length = 40
                
                # 初始角度 (向上，对应 -pi/2)
                start_angle = -np.pi / 2
                
                # 当前角度
                current_angle = start_angle + angle_rad
                
                # 新的终点
                end_x = cx + length * np.cos(current_angle)
                end_y = cy + length * np.sin(current_angle)
                
                self.vis_canvas.coords(self.arrow_id, cx, cy, end_x, end_y)
            else:
                self.vis_canvas.move(self.car_id, dx/steps, dy/steps)
                self.vis_canvas.move(self.arrow_id, dx/steps, dy/steps)
            
            self.root.after(delay, lambda: step_anim(i+1))
            
        step_anim(0)

    def _show_step_response(self):
        """显示控制系统阶跃响应"""
        window = tk.Toplevel(self.root)
        window.title("控制系统阶跃响应 (Step Response)")
        window.geometry("800x600")
        
        # 使用系统中的控制器进行仿真
        from control_system import ControlSystem
        cs = ControlSystem(system_type='motor')
        
        # 执行阶跃信号 (目标值 1.0)
        cs.controller.reset()
        cs.system.reset()
        cs.controller.set_setpoint(1.0)
        
        duration = 2.0
        dt = 0.01
        steps = int(duration / dt)
        
        times = []
        positions = []
        outputs = []
        
        for i in range(steps):
            t = i * dt
            current_pos = cs.system.get_state().position
            output = cs.controller.update(current_pos)
            cs.system.apply_voltage(output, dt)
            
            times.append(t)
            positions.append(current_pos)
            outputs.append(output)
            
        # 绘图
        fig = Figure(figsize=(8, 6), dpi=100)
        ax1 = fig.add_subplot(211)
        ax1.plot(times, positions, label='Response')
        ax1.plot(times, [1.0]*len(times), 'r--', label='Setpoint')
        ax1.set_title('Step Response (Position)')
        ax1.set_ylabel('Position')
        ax1.grid(True)
        ax1.legend()
        
        ax2 = fig.add_subplot(212)
        ax2.plot(times, outputs, color='green')
        ax2.set_title('Control Output (Voltage)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Voltage (V)')
        ax2.grid(True)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        
    def _open_analysis_window(self):
        """打开详细分析窗口"""
        if self.last_recorded_signal is None:
            return
            
        # 创建新窗口
        window = tk.Toplevel(self.root)
        window.title("信号分析 (Signal Analysis)")
        window.geometry("1400x900")
        
        # 显示加载中
        lbl_loading = ttk.Label(window, text="正在分析信号...", font=("Arial", 14))
        lbl_loading.pack(pady=20)
        window.update()
        
        try:
            # 处理信号获取所有阶段数据
            snr = self.scale_snr.get()
            result = self.system.process_audio_signal(
                self.last_recorded_signal, 
                self.last_fs, 
                snr_db=snr, 
                return_signals=True
            )
            
            if not result['overall_success']:
                lbl_loading.configure(text=f"分析失败: {result.get('error')}")
                return
                
            lbl_loading.destroy()
            
            stages = result['stages']
            
            # --- Statistics Section ---
            stats_frame = ttk.LabelFrame(window, text="Statistics Summary", padding="10")
            stats_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Huffman Stats
            huff_stats = stages['source_encoding']
            huff_text = (
                f"Huffman Encoding:\n"
                f"  Input Bits: {huff_stats['original_bits']}\n"
                f"  Output Bits: {huff_stats['encoded_bits']}\n"
                f"  Compression Ratio: {huff_stats['compression_ratio']:.4f}\n"
                f"  Avg Code Length: {huff_stats['avg_code_length']:.4f}"
            )
            ttk.Label(stats_frame, text=huff_text, font=("Courier", 10)).pack(side=tk.LEFT, padx=20)
            
            # Channel Stats (BER)
            demod_stats = stages['demodulation']
            ber_text = (
                f"Channel Transmission:\n"
                f"  SNR: {result['snr_db']} dB\n"
                f"  Bit Errors: {demod_stats['bit_errors']}\n"
                f"  BER: {demod_stats['ber']:.6f}"
            )
            ttk.Label(stats_frame, text=ber_text, font=("Courier", 10)).pack(side=tk.LEFT, padx=20)
            
            # 准备绘图数据
            # (标题, 信号, 采样率, 是否绘制语谱图)
            plot_data = [
                ("Original Signal", stages['sampling']['signal'], self.last_fs, True),
                ("Quantized Signal", stages['quantization']['signal'], self.last_fs, True),
                ("Modulated (BPSK Symbols)", stages['modulation']['signal'], 1, False),
                ("Received (Noisy Symbols)", stages['channel_transmission']['signal'], 1, False),
                ("Reconstructed Signal", stages['speech_recognition']['reconstructed_signal'], self.last_fs, True)
            ]
            
            # 创建画布
            fig = Figure(figsize=(14, 10), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=window)
            
            # 添加滚动条 (如果需要)
            # 这里直接全屏显示
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # 绘图
            rows = 5
            cols = 3
            
            for i, (title, signal, fs, show_spec) in enumerate(plot_data):
                # 1. 时域波形
                ax_time = fig.add_subplot(rows, cols, i*cols + 1)
                ax_time.plot(signal)
                ax_time.set_title(f"{title} - Time Domain")
                ax_time.grid(True)
                # 限制点数以提高性能
                if len(signal) > 10000:
                    ax_time.set_xlim(0, len(signal))
                
                # 2. 语谱图 (Spectrogram)
                ax_spec = fig.add_subplot(rows, cols, i*cols + 2)
                if show_spec and len(signal) > 256:
                    try:
                        ax_spec.specgram(signal, Fs=fs, NFFT=256, noverlap=128)
                        ax_spec.set_title(f"{title} - Spectrogram")
                    except Exception as e:
                        ax_spec.text(0.5, 0.5, f"Error: {e}", ha='center')
                else:
                    ax_spec.text(0.5, 0.5, "N/A", ha='center')
                    ax_spec.set_title(f"{title} - Spectrogram")
                
                # 3. 频谱图 (Spectrum)
                ax_freq = fig.add_subplot(rows, cols, i*cols + 3)
                if len(signal) > 0:
                    try:
                        # 计算FFT
                        n = len(signal)
                        Y = np.fft.fft(signal)
                        freq = np.fft.fftfreq(n, d=1/fs if fs > 1 else 1)
                        
                        # 只取正频率
                        mask = freq >= 0
                        ax_freq.plot(freq[mask], np.abs(Y[mask]))
                        ax_freq.set_title(f"{title} - Spectrum")
                        ax_freq.grid(True)
                        # 对数坐标可能更好，但线性坐标更直观
                        # ax_freq.set_yscale('log')
                    except Exception as e:
                        ax_freq.text(0.5, 0.5, f"Error: {e}", ha='center')
                
            fig.tight_layout()
            canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
            logger.error(f"Analysis failed: {e}", exc_info=True)

    def on_closing(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceControlGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
