"""
远程声音控制系统 - 调制解调模块（BPSK + RRC成型滤波 + 通带传输）
Modulation/Demodulation Module for Remote Voice Control System
Strictly following the MATLAB implementation logic.
"""

import numpy as np
from typing import Tuple
import logging
from scipy.signal import convolve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Modulation:
    """BPSK调制和解调 (仿照MATLAB逻辑)"""
    
    def __init__(self, fs: int = 44100, fc: int = 10000, sps: int = 8, span: int = 8, alpha: float = 0.3):
        """
        初始化调制解调器
        
        Args:
            fs: 采样频率 (Hz)
            fc: 载波频率 (Hz)
            sps: 每个符号的采样数 (Samples Per Symbol) = Fs/Rs
            span: 滤波器跨度 (符号数)
            alpha: 滚降系数
        """
        self.fs = fs
        self.fc = fc
        self.sps = sps
        self.span = span
        self.alpha = alpha
        
        # 预计算滤波器系数 (对应 MATLAB: h_filter = rcosdesign(alpha, span, sps))
        self.h_filter = self._design_rrc_filter(sps, span, alpha)
        
        logger.info(f"Modem initialized: BPSK with RRC(alpha={alpha}, sps={sps})")
        logger.info(f"Carrier: fc={fc}Hz, Fs={fs}Hz")

    def _design_rrc_filter(self, sps, span, alpha):
        """
        生成根升余弦滤波器系数 (Root Raised Cosine Filter Design)
        对应 MATLAB 的 rcosdesign(alpha, span, sps)
        """
        # MATLAB rcosdesign 生成的长度是 span * sps + 1
        # 时间轴范围 [-span/2, span/2] (单位：符号时间)
        t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
        
        h = np.zeros_like(t)
        epsilon = 1e-8
        
        for i, ti in enumerate(t):
            if abs(ti) < epsilon:
                h[i] = 1.0 - alpha + (4 * alpha / np.pi)
            elif abs(abs(ti) - 1 / (4 * alpha)) < epsilon:
                h[i] = (alpha / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            else:
                num = np.cos((1 + alpha) * np.pi * ti) + \
                      (np.sin((1 - alpha) * np.pi * ti) / (4 * alpha * ti))
                denom = 1 - (4 * alpha * ti) ** 2
                h[i] = (4 * alpha / np.pi) * num / denom
        
        # 归一化能量，使得滤波器的能量为1
        h = h / np.sqrt(np.sum(h**2))
        return h
    
    def bpsk_modulate(self, bits: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        """
        BPSK调制：
        1. 比特 -> 双极性码 (0->-1, 1->+1)
        2. 矩形脉冲扩展 (kron)
        3. 成型滤波 (conv)
        4. 载波调制 (.* carrier)
        """
        bits = bits.astype(int).flatten()
        N = len(bits)
        
        # 1. 双极性码: 0 -> -1, 1 -> +1 (注意：MATLAB代码中是 2*data-1，即 0->-1, 1->1)
        # 之前的代码是 1-2*bits (0->1, 1->-1)，这里改为与MATLAB一致
        bipolar_data = 2 * bits - 1
        
        # 2. 矩形脉冲扩展 (对应 MATLAB: baseband = kron(2*data-1, ones(1, sps)))
        # 这里 sps = Fs*Tb
        baseband = np.kron(bipolar_data, np.ones(self.sps))
        
        # 3. 成型滤波 (对应 MATLAB: baseband_filtered = conv(baseband, h_filter, 'same'))
        # 注意：MATLAB 'same' 模式返回中心部分，长度与 baseband 相同
        baseband_filtered = convolve(baseband, self.h_filter, mode='same')
        
        # 4. 载波调制 (对应 MATLAB: bpsk_signal = baseband_filtered .* carrier)
        # 生成时间向量 t = 0:1/Fs:N*Tb-1/Fs
        # 总采样点数应该等于 baseband_filtered 的长度
        total_samples = len(baseband_filtered)
        t = np.arange(total_samples) / self.fs
        
        carrier = np.cos(2 * np.pi * self.fc * t)
        modulated_signal = baseband_filtered * carrier * amplitude
        
        logger.info(f"BPSK modulation: {len(bits)} bits -> {len(modulated_signal)} samples")
        
        return modulated_signal
    
    def channel_transmission(self, modulated_signal: np.ndarray, snr_db: float = 20) -> np.ndarray:
        """
        模拟加性高斯白噪声（AWGN）信道
        对应 MATLAB: rx_signal = awgn(bpsk_signal, snr, 'measured')
        """
        signal_power = np.mean(modulated_signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(len(modulated_signal))
        received_signal = modulated_signal + noise
        return received_signal
    
    def bpsk_demodulate(self, received_signal: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        BPSK解调：
        1. 生成本地载波
        2. 混频 (相乘)
        3. 积分 (滑动求和)
        4. 判决
        """
        total_samples = len(received_signal)
        
        # 1. 生成本地相干载波 (对应 MATLAB: local_carrier = cos(2*pi*fc*t))
        t = np.arange(total_samples) / self.fs
        local_carrier = np.cos(2 * np.pi * self.fc * t)
        
        # 2. 相乘器 (对应 MATLAB: mixed_signal = rx_signal .* local_carrier)
        mixed_signal = received_signal * local_carrier
        
        # 3. 积分 (对应 MATLAB: integrated_samples(k) = sum(mixed_signal(start_idx:end_idx)))
        # 计算符号数 N
        N = total_samples // self.sps
        integrated_samples = np.zeros(N)
        
        for k in range(N):
            start_idx = k * self.sps
            end_idx = (k + 1) * self.sps
            # 注意：Python切片不包含end_idx，MATLAB包含
            integrated_samples[k] = np.sum(mixed_signal[start_idx:end_idx])
            
        # 4. 判决 (对应 MATLAB: detected_bits = integrated_samples > 0)
        # MATLAB中 >0 为 1 (对应原始数据1->1, 0->-1)
        # 所以 >0 判为 1，<=0 判为 0
        demodulated_bits = (integrated_samples > threshold).astype(int)
        
        logger.info(f"BPSK demodulation: {len(demodulated_bits)} bits recovered")
        
        return demodulated_bits

    def get_error_statistics(self, transmitted_bits: np.ndarray, received_bits: np.ndarray) -> dict:
        """计算误码率"""
        t_bits = transmitted_bits.astype(int).flatten()
        r_bits = received_bits.astype(int).flatten()
        
        min_len = min(len(t_bits), len(r_bits))
        bit_errors = np.sum(t_bits[:min_len] != r_bits[:min_len])
        
        if abs(len(t_bits) - len(r_bits)) > 10:
            logger.warning(f"Bit length mismatch: Tx={len(t_bits)}, Rx={len(r_bits)}")
            
        ber = bit_errors / min_len if min_len > 0 else 0
        
        return {
            'total_bits': min_len,
            'bit_errors': bit_errors,
            'ber': ber
        }

if __name__ == "__main__":
    # 简单测试
    modem = Modulation(fs=44100, fc=10000, sps=8)
    bits = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    
    tx = modem.bpsk_modulate(bits)
    rx = modem.channel_transmission(tx, snr_db=10)
    rx_bits = modem.bpsk_demodulate(rx)
    
    print(f"Original: {bits}")
    print(f"Received: {rx_bits}")
    
    stats = modem.get_error_statistics(bits, rx_bits)
    print(f"Test Results: {stats}")
