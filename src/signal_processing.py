"""
远程声音控制系统 - 信号处理模块
Signal Processing Module for Remote Voice Control System
"""

import numpy as np
import soundfile as sf
import sounddevice as sd
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalProcessing:
    """信号采样和量化处理"""
    
    def __init__(self, fs: int = 44100, quantization_bits: int = 8):
        """
        初始化信号处理类
        
        Args:
            fs: 采样频率 (Hz)
            quantization_bits: 量化比特数
        """
        self.fs = fs
        self.quantization_bits = quantization_bits
        self.quantization_levels = 2 ** quantization_bits
        
    def record_audio(self, duration: float = 2.0) -> Tuple[np.ndarray, int]:
        """
        录制音频
        
        Args:
            duration: 录音时长（秒）
            
        Returns:
            sampled_signal: 录制的信号
            fs: 采样频率
        """
        logger.info(f"Recording audio for {duration} seconds...")
        try:
            # 录制音频
            recording = sd.rec(int(duration * self.fs), samplerate=self.fs, channels=1, dtype='float64')
            sd.wait()  # 等待录音结束
            
            # 展平数组
            sampled_signal = recording.flatten()
            
            logger.info("Recording finished")
            return sampled_signal, self.fs
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            # 返回静音信号作为回退
            return np.zeros(int(duration * self.fs)), self.fs

    def signal_sampling(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        读取和采样音频信号
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            sampled_signal: 采样后的音频信号
            fs: 采样频率
        """
        try:
            sampled_signal, fs = sf.read(audio_path)
            
            # 如果是立体声，转换为单声道
            if len(sampled_signal.shape) > 1:
                sampled_signal = np.mean(sampled_signal, axis=1)
            
            logger.info(f"Successfully loaded audio from {audio_path}")
            logger.info(f"Sample rate: {fs} Hz, Signal length: {len(sampled_signal)} samples")
            return sampled_signal, fs
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def signal_quantization(self, signal: np.ndarray) -> np.ndarray:
        """
        对信号进行量化处理
        
        Args:
            signal: 原始信号
            
        Returns:
            quantized_signal: 量化后的信号
        """
        # 找出信号的最小值和最大值
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        
        # 计算量化步长
        delta = (signal_max - signal_min) / self.quantization_levels
        
        # 生成量化电平
        quantization_levels = np.array([signal_min + delta * i 
                                        for i in range(self.quantization_levels + 1)])
        
        # 对每个样本进行量化 - 使用中点量化
        quantized_signal = np.zeros_like(signal)
        for i, sample in enumerate(signal):
            # 找到最接近的量化电平区间
            idx = np.searchsorted(quantization_levels, sample, side='right') - 1
            idx = np.clip(idx, 0, self.quantization_levels - 1)
            # 量化为该区间的中点
            quantized_signal[i] = (quantization_levels[idx] + quantization_levels[idx + 1]) / 2
        
        logger.info(f"Quantized signal with {self.quantization_bits} bits")
        logger.info(f"Quantization levels: {self.quantization_levels}")
        logger.info(f"Quantization error (MSE): {np.mean((signal - quantized_signal) ** 2):.6f}")
        
        return quantized_signal
    
    def quantization_to_bits(self, quantized_signal: np.ndarray) -> np.ndarray:
        """
        将量化后的信号转换为比特序列（用于编码）
        
        Args:
            quantized_signal: 量化后的信号
            
        Returns:
            bit_sequence: 比特序列
        """
        signal_min = np.min(quantized_signal)
        signal_max = np.max(quantized_signal)
        delta = (signal_max - signal_min) / self.quantization_levels
        
        # 将每个样本映射到 [0, 2^bits-1] 范围
        normalized = np.floor((quantized_signal - signal_min) / delta + 0.5)
        normalized = np.clip(normalized, 0, self.quantization_levels - 1).astype(int)
        
        return normalized
    
    def bits_to_quantization(self, bits: np.ndarray, 
                            signal_min: float, signal_max: float) -> np.ndarray:
        """
        将比特序列转换回量化信号
        
        Args:
            bits: 比特序列
            signal_min: 信号最小值
            signal_max: 信号最大值
            
        Returns:
            quantized_signal: 量化后的信号
        """
        delta = (signal_max - signal_min) / self.quantization_levels
        quantization_levels = np.array([signal_min + delta * i 
                                        for i in range(self.quantization_levels + 1)])
        
        quantized_signal = np.zeros_like(bits, dtype=float)
        for i, bit_val in enumerate(bits):
            idx = int(bit_val)
            idx = np.clip(idx, 0, self.quantization_levels - 1)
            quantized_signal[i] = (quantization_levels[idx] + quantization_levels[idx + 1]) / 2
        
        return quantized_signal


# 辅助函数
def estimate_snr(original: np.ndarray, noisy: np.ndarray) -> float:
    """
    估计信噪比
    
    Args:
        original: 原始信号
        noisy: 含噪信号
        
    Returns:
        snr: 信噪比（dB）
    """
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - noisy) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr


if __name__ == "__main__":
    # 测试信号处理
    processor = SignalProcessing(fs=44100, quantization_bits=8)
    
    # 尝试加载音频
    try:
        signal, fs = processor.signal_sampling('../data/command.flac')
        quantized = processor.signal_quantization(signal)
        bit_sequence = processor.quantization_to_bits(quantized)
        
        print(f"Original signal shape: {signal.shape}")
        print(f"Quantized signal shape: {quantized.shape}")
        print(f"Bit sequence shape: {bit_sequence.shape}")
        print(f"Bit sequence (first 100): {bit_sequence[:100]}")
    except Exception as e:
        print(f"Error: {e}")
