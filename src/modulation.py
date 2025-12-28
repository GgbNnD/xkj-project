"""
远程声音控制系统 - 调制解调模块（BPSK）
Modulation/Demodulation Module for Remote Voice Control System
"""

import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Modulation:
    """BPSK调制和解调"""
    
    def __init__(self, samples_per_bit: int = 100):
        """
        初始化调制解调器
        
        Args:
            samples_per_bit: 每个比特的采样数（用于模拟信号）
        """
        self.samples_per_bit = samples_per_bit
    
    def bpsk_modulate(self, bits: np.ndarray, 
                     amplitude: float = 1.0) -> np.ndarray:
        """
        BPSK调制：0->+A, 1->-A
        
        Args:
            bits: 比特序列 (0或1)
            amplitude: 信号幅度
            
        Returns:
            modulated_signal: 调制后的信号
        """
        bits = bits.astype(int).flatten()
        
        # BPSK映射：0 -> +1, 1 -> -1
        modulated_symbols = 1 - 2 * bits  # 将0映射为1，1映射为-1
        modulated_symbols = modulated_symbols * amplitude
        
        logger.info(f"BPSK modulation: {len(bits)} bits modulated")
        
        return modulated_symbols
    
    def channel_transmission(self, modulated_signal: np.ndarray, 
                            snr_db: float = 20) -> np.ndarray:
        """
        模拟加性高斯白噪声（AWGN）信道
        
        Args:
            modulated_signal: 调制后的信号
            snr_db: 信噪比（dB）
            
        Returns:
            received_signal: 含噪信号
        """
        # 计算信号功率
        signal_power = np.mean(modulated_signal ** 2)
        
        # 从SNR(dB)计算噪声功率
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # 生成高斯白噪声
        noise = np.sqrt(noise_power) * np.random.randn(len(modulated_signal))
        
        # 加入噪声
        received_signal = modulated_signal + noise
        
        # 计算实际SNR
        actual_noise = received_signal - modulated_signal
        actual_noise_power = np.mean(actual_noise ** 2)
        actual_snr_db = 10 * np.log10(signal_power / (actual_noise_power + 1e-10))
        
        logger.info(f"Channel transmission: SNR = {snr_db} dB -> Actual SNR = {actual_snr_db:.2f} dB")
        logger.info(f"Signal power: {signal_power:.6f}, Noise power: {actual_noise_power:.6f}")
        
        return received_signal
    
    def bpsk_demodulate(self, received_signal: np.ndarray,
                       threshold: float = 0.0) -> np.ndarray:
        """
        BPSK解调：使用硬判决
        
        Args:
            received_signal: 接收到的信号
            threshold: 判决门限（通常为0）
            
        Returns:
            demodulated_bits: 解调后的比特
        """
        # 硬判决：
        # 如果信号 > threshold，判决为0（映射回1-2*0=1）
        # 如果信号 <= threshold，判决为1（映射回1-2*1=-1）
        
        demodulated_bits = np.zeros_like(received_signal, dtype=int)
        demodulated_bits[received_signal <= threshold] = 1
        demodulated_bits[received_signal > threshold] = 0
        
        logger.info(f"BPSK demodulation: {len(demodulated_bits)} bits demodulated")
        
        return demodulated_bits.astype(int)
    
    def soft_demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        BPSK软解调：返回接收信号本身作为置信度
        
        Args:
            received_signal: 接收到的信号
            
        Returns:
            soft_bits: 软判决值（-1到1之间的实数）
        """
        # 归一化到[-1, 1]
        max_val = np.max(np.abs(received_signal))
        if max_val > 0:
            soft_bits = received_signal / max_val
        else:
            soft_bits = received_signal
        
        logger.info(f"Soft demodulation: {len(soft_bits)} soft bits generated")
        
        return soft_bits
    
    def get_error_statistics(self, transmitted_bits: np.ndarray, 
                            received_bits: np.ndarray) -> dict:
        """
        计算接收错误统计
        
        Args:
            transmitted_bits: 发送的比特
            received_bits: 接收的比特
            
        Returns:
            stats: 错误统计字典
        """
        transmitted_bits = transmitted_bits.astype(int).flatten()
        received_bits = received_bits.astype(int).flatten()
        
        # 确保长度相同
        min_len = min(len(transmitted_bits), len(received_bits))
        transmitted_bits = transmitted_bits[:min_len]
        received_bits = received_bits[:min_len]
        
        # 计算比特错误
        bit_errors = np.sum(transmitted_bits != received_bits)
        total_bits = len(transmitted_bits)
        ber = bit_errors / total_bits if total_bits > 0 else 0
        
        stats = {
            'total_bits': total_bits,
            'bit_errors': bit_errors,
            'ber': ber,  # 比特错误率
        }
        
        logger.info(f"Channel Statistics:")
        logger.info(f"  Total bits: {total_bits}")
        logger.info(f"  Bit errors: {bit_errors}")
        logger.info(f"  BER: {ber:.6f}")
        
        return stats


class ChannelModel:
    """通信信道模型"""
    
    @staticmethod
    def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        添加高斯白噪声（AWGN）
        
        Args:
            signal: 输入信号
            snr_db: 信噪比（dB）
            
        Returns:
            noisy_signal: 含噪信号
        """
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        noisy_signal = signal + noise
        return noisy_signal
    
    @staticmethod
    def fading(signal: np.ndarray, fade_magnitude: float = 0.5) -> np.ndarray:
        """
        模拟衰落信道
        
        Args:
            signal: 输入信号
            fade_magnitude: 衰落大小 (0-1)
            
        Returns:
            faded_signal: 衰落后的信号
        """
        # Rayleigh衰落
        fade_coefficient = np.sqrt(1 - fade_magnitude) + \
                          fade_magnitude * np.random.randn(len(signal))
        faded_signal = signal * fade_coefficient
        return faded_signal
    
    @staticmethod
    def impulse_noise(signal: np.ndarray, 
                     noise_probability: float = 0.01,
                     noise_magnitude: float = 3.0) -> np.ndarray:
        """
        添加脉冲噪声
        
        Args:
            signal: 输入信号
            noise_probability: 脉冲出现概率
            noise_magnitude: 脉冲幅度倍数
            
        Returns:
            noisy_signal: 含脉冲噪声的信号
        """
        noise = np.zeros_like(signal)
        impulse_mask = np.random.rand(len(signal)) < noise_probability
        noise[impulse_mask] = noise_magnitude * np.sign(np.random.randn(np.sum(impulse_mask)))
        noisy_signal = signal + noise
        return noisy_signal


def constellation_diagram(transmitted_symbols: np.ndarray,
                         received_symbols: np.ndarray,
                         title: str = "BPSK Constellation Diagram"):
    """
    绘制星座图（需要matplotlib）
    
    Args:
        transmitted_symbols: 发送的符号
        received_symbols: 接收的符号
        title: 图表标题
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        # 绘制发送的符号
        plt.scatter(transmitted_symbols.real, transmitted_symbols.imag, 
                   c='blue', marker='o', s=50, label='Transmitted', alpha=0.5)
        
        # 绘制接收的符号
        plt.scatter(received_symbols.real, received_symbols.imag, 
                   c='red', marker='x', s=50, label='Received', alpha=0.5)
        
        plt.xlabel('I (In-phase)')
        plt.ylabel('Q (Quadrature)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    except ImportError:
        logger.warning("Matplotlib not available for constellation diagram")


if __name__ == "__main__":
    # 测试BPSK调制解调
    modem = Modulation()
    
    # 创建测试比特
    test_bits = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 10, dtype=int)
    
    # 调制
    modulated = modem.bpsk_modulate(test_bits, amplitude=1.0)
    print(f"Original bits: {test_bits[:20]}")
    print(f"Modulated signal (first 20): {modulated[:20]}")
    
    # 信道传输
    received = modem.channel_transmission(modulated, snr_db=20)
    
    # 解调
    demodulated = modem.bpsk_demodulate(received, threshold=0.0)
    print(f"Demodulated bits (first 20): {demodulated[:20]}")
    
    # 计算错误
    stats = modem.get_error_statistics(test_bits, demodulated)
    print(f"\nError statistics: {stats}")
