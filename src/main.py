"""
远程声音控制系统 - 主程序
Main Program for Remote Voice Control System

完整流程：
1. 语音信号采集 (Signal Sampling)
2. 量化处理 (Signal Quantization)
3. 信源编码 - Huffman (Source Encoding)
4. 信道编码 - BCH (Channel Encoding)
5. 调制 - BPSK (Modulation)
6. 信道传输 - AWGN (Channel Transmission)
7. 解调 (Demodulation)
8. 信道解码 - BCH (Channel Decoding)
9. 信源解码 - Huffman (Source Decoding)
10. 语音识别 (Speech Recognition)
11. 控制系统执行 (Control System)
"""

import numpy as np
import os
import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# 导入各个模块
from signal_processing import SignalProcessing
from source_encoding import SourceEncoding
from channel_encoding import BCHCoder
from modulation import Modulation
from speech_recognition_system import SpeechRecognitionSystem
from control_system import ControlSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('remote_voice_control.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RemoteVoiceControlSystem:
    """远程声音控制系统的完整实现"""
    
    def __init__(self, config: dict = None):
        """
        初始化远程声音控制系统
        
        Args:
            config: 系统配置字典
        """
        # 默认配置
        self.config = {
            'fs': 44100,                    # 采样频率
            'quantization_bits': 8,        # 量化位数
            'channel_code_n': 15,          # BCH码字长
            'channel_code_k': 7,           # BCH信息位
            'snr_db': 20,                  # 信噪比
            'num_commands': 4,             # 命令数量
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 初始化各个模块
        self._initialize_modules()
        
        # 统计信息
        self.stats = {
            'total_commands': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'ber_history': [],
            'snr_history': [],
        }
        
        logger.info("Remote Voice Control System initialized")
        logger.info(f"Configuration: {self.config}")
    
    def _initialize_modules(self):
        """初始化所有模块"""
        self.signal_processor = SignalProcessing(
            fs=self.config['fs'],
            quantization_bits=self.config['quantization_bits']
        )
        
        self.source_encoder = SourceEncoding()
        
        self.channel_coder = BCHCoder(
            n=self.config['channel_code_n'],
            k=self.config['channel_code_k']
        )
        
        self.modem = Modulation()
        
        self.speech_recognizer = SpeechRecognitionSystem(
            fs=self.config['fs'],
            num_commands=self.config['num_commands']
        )
        
        self.control_system = ControlSystem(system_type='motor')
    
    def process_audio_signal(self, sampled_signal: np.ndarray, fs: int, snr_db: float = None) -> dict:
        """
        处理音频信号数据（内存中）
        
        Args:
            sampled_signal: 采样后的音频信号
            fs: 采样频率
            snr_db: 信噪比
            
        Returns:
            results: 处理结果字典
        """
        if snr_db is None:
            snr_db = self.config['snr_db']
            
        results = {
            'timestamp': datetime.now().isoformat(),
            'audio_source': 'memory_stream',
            'snr_db': snr_db,
            'stages': {}
        }
        
        try:
            # 1. 采样信息
            logger.info("Stage 1: Signal Sampling (From Memory)")
            stage_results = {
                'signal_length': len(sampled_signal),
                'sample_rate': fs,
            }
            results['stages']['sampling'] = stage_results
            
            # 2. 量化处理
            logger.info("Stage 2: Signal Quantization")
            quantized_signal = self.signal_processor.signal_quantization(sampled_signal)
            quantized_bits = self.signal_processor.quantization_to_bits(quantized_signal)
            stage_results = {
                'quantized_length': len(quantized_signal),
                'quantized_bits_length': len(quantized_bits),
                'quantization_bits': self.config['quantization_bits'],
            }
            results['stages']['quantization'] = stage_results
            
            # 3. 信源编码 (Huffman)
            logger.info("Stage 3: Source Encoding (Huffman)")
            encoded_bits_str, code_dict, huffman_stats = self.source_encoder.encode(quantized_bits)
            encoded_bits = np.array([int(b) for b in encoded_bits_str])
            stage_results = {
                'original_bits': huffman_stats['original_bits'],
                'encoded_bits': huffman_stats['encoded_bits'],
                'compression_ratio': huffman_stats['compression_ratio'],
                'entropy': huffman_stats['entropy'],
                'avg_code_length': huffman_stats['avg_code_length'],
            }
            results['stages']['source_encoding'] = stage_results
            
            # 4. 信道编码 (BCH)
            logger.info("Stage 4: Channel Encoding (BCH)")
            channel_coded = self.channel_coder.encode(encoded_bits)
            stage_results = {
                'input_bits': len(encoded_bits),
                'output_bits': len(channel_coded),
                'code_rate': len(encoded_bits) / len(channel_coded),
            }
            results['stages']['channel_encoding'] = stage_results
            
            # 5. 调制 (BPSK)
            logger.info("Stage 5: Modulation (BPSK)")
            modulated_signal = self.modem.bpsk_modulate(channel_coded, amplitude=1.0)
            stage_results = {
                'signal_length': len(modulated_signal),
                'signal_power': float(np.mean(modulated_signal ** 2)),
            }
            results['stages']['modulation'] = stage_results
            
            # 6. 信道传输 (AWGN)
            logger.info("Stage 6: Channel Transmission (AWGN)")
            received_signal = self.modem.channel_transmission(modulated_signal, snr_db)
            stage_results = {
                'signal_length': len(received_signal),
                'noise_power': float(np.mean((received_signal - modulated_signal) ** 2)),
            }
            results['stages']['channel_transmission'] = stage_results
            
            # 7. 解调 (BPSK)
            logger.info("Stage 7: Demodulation (BPSK)")
            demodulated_bits = self.modem.bpsk_demodulate(received_signal, threshold=0.0)
            error_stats = self.modem.get_error_statistics(channel_coded, demodulated_bits)
            ber = error_stats['ber']
            self.stats['ber_history'].append(ber)
            self.stats['snr_history'].append(snr_db)
            stage_results = {
                'demodulated_bits': len(demodulated_bits),
                'bit_errors': error_stats['bit_errors'],
                'total_bits': error_stats['total_bits'],
                'ber': ber,
            }
            results['stages']['demodulation'] = stage_results
            
            # 8. 信道解码 (BCH)
            logger.info("Stage 8: Channel Decoding (BCH)")
            channel_decoded = self.channel_coder.decode(demodulated_bits)
            stage_results = {
                'decoded_bits': len(channel_decoded),
            }
            results['stages']['channel_decoding'] = stage_results
            
            # 9. 信源解码 (Huffman)
            logger.info("Stage 9: Source Decoding (Huffman)")
            decoded_bits_str = ''.join([str(int(b)) for b in channel_decoded])
            source_decoded = self.source_encoder.decode(decoded_bits_str)
            stage_results = {
                'decoded_symbols': len(source_decoded),
            }
            results['stages']['source_decoding'] = stage_results
            
            # 10. 语音识别
            logger.info("Stage 10: Speech Recognition")
            # 将解码后的信号重构为语音
            signal_min = np.min(quantized_signal)
            signal_max = np.max(quantized_signal)
            reconstructed_signal = self.signal_processor.bits_to_quantization(
                source_decoded.astype(float), signal_min, signal_max
            )
            
            command, confidence = self.speech_recognizer.recognize_command(reconstructed_signal)
                
            stage_results = {
                'recognized_command': command,
                'confidence': float(confidence),
                'reconstructed_signal_length': len(reconstructed_signal),
                'reconstructed_signal': reconstructed_signal,
            }
            results['stages']['speech_recognition'] = stage_results
            
            # 11. 控制系统执行
            logger.info("Stage 11: Control System Execution")
            success = self.control_system.execute_command(command, duration=1.0, dt=0.01)
            metrics = self.control_system.get_performance_metrics()
            stage_results = {
                'command': command,
                'execution_success': success,
                'performance_metrics': metrics,
            }
            results['stages']['control_execution'] = stage_results
            
            # 更新统计信息
            self.stats['total_commands'] += 1
            if success:
                self.stats['successful_commands'] += 1
            else:
                self.stats['failed_commands'] += 1
            
            results['overall_success'] = True
            results['summary'] = {
                'total_processing_stages': len(results['stages']),
                'recognized_command': command,
                'control_success': success,
                'system_stats': self.get_system_stats(),
            }
            
            logger.info("Processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            results['overall_success'] = False
            results['error'] = str(e)
            self.stats['failed_commands'] += 1
        
        return results

    def process_audio_file(self, audio_path: str, snr_db: float = None) -> dict:
        """
        处理音频文件的完整流程
        
        Args:
            audio_path: 音频文件路径
            snr_db: 信噪比（如果为None，使用配置中的值）
            
        Returns:
            results: 处理结果字典
        """
        if snr_db is None:
            snr_db = self.config['snr_db']
        
        logger.info(f"Processing audio file: {audio_path}")
        logger.info(f"SNR: {snr_db} dB")
        
        try:
            # 1. 语音采样
            sampled_signal, fs = self.signal_processor.signal_sampling(audio_path)
            
            # 调用通用处理函数
            results = self.process_audio_signal(sampled_signal, fs, snr_db)
            results['audio_path'] = audio_path
            return results
            
        except Exception as e:
            logger.error(f"Error during file processing: {e}", exc_info=True)
            return {
                'overall_success': False,
                'error': str(e),
                'audio_path': audio_path
            }
    
    def process_multiple_snr_levels(self, audio_path: str, 
                                   snr_levels: list = None) -> dict:
        """
        在不同信噪比下处理音频
        
        Args:
            audio_path: 音频文件路径
            snr_levels: SNR值列表 (dB)
            
        Returns:
            results: 所有处理结果
        """
        if snr_levels is None:
            snr_levels = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
        
        logger.info(f"Processing audio at multiple SNR levels: {snr_levels}")
        
        results = {
            'audio_path': audio_path,
            'snr_levels': snr_levels,
            'results': {}
        }
        
        for snr in snr_levels:
            logger.info(f"Processing at SNR = {snr} dB")
            result = self.process_audio_file(audio_path, snr_db=snr)
            results['results'][snr] = result
        
        return results
    
    def print_system_summary(self) -> None:
        """打印系统摘要"""
        logger.info("=" * 60)
        logger.info("REMOTE VOICE CONTROL SYSTEM SUMMARY")
        logger.info("=" * 60)
        
        logger.info("System Configuration:")
        for key, value in self.config.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\nSystem Statistics:")
        for key, value in self.stats.items():
            if key == 'ber_history':
                if value:
                    logger.info(f"  {key}: mean={np.mean(value):.6f}, "
                              f"min={np.min(value):.6f}, max={np.max(value):.6f}")
                else:
                    logger.info(f"  {key}: No data")
            elif key == 'snr_history':
                logger.info(f"  {key}: {len(value)} measurements")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("=" * 60)
    
    def get_system_stats(self) -> dict:
        """获取系统统计信息"""
        return {
            'total_commands': self.stats['total_commands'],
            'successful_commands': self.stats['successful_commands'],
            'failed_commands': self.stats['failed_commands'],
            'success_rate': (self.stats['successful_commands'] / 
                           max(self.stats['total_commands'], 1)) * 100,
        }
    
    def save_results(self, results: dict, output_path: str = None) -> None:
        """
        保存处理结果到文件
        
        Args:
            results: 处理结果字典
            output_path: 输出文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'results_{timestamp}.json'
        
        try:
            # 转换numpy类型为Python原生类型
            results_serializable = self._make_serializable(results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_serializable, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    @staticmethod
    def _make_serializable(obj):
        """将numpy类型转换为可序列化的类型"""
        if isinstance(obj, dict):
            return {k: RemoteVoiceControlSystem._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [RemoteVoiceControlSystem._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def main():
    """主函数"""
    # 创建系统实例
    config = {
        'fs': 44100,
        'quantization_bits': 8,
        'channel_code_n': 15,
        'channel_code_k': 7,
        'snr_db': 20,
        'num_commands': 4,
    }
    
    system = RemoteVoiceControlSystem(config=config)
    
    # 音频文件路径
    audio_path = 'data/command.flac'
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        logger.info("Please ensure the audio file is available at: {audio_path}")
        return
    
    logger.info("Starting Remote Voice Control System processing...")
    logger.info("-" * 60)
    
    # 处理音频文件
    result = system.process_audio_file(audio_path, snr_db=20)
    
    # 打印摘要
    logger.info("-" * 60)
    system.print_system_summary()
    
    # 保存结果
    system.save_results(result)
    
    # 在多个SNR下处理
    logger.info("-" * 60)
    logger.info("Processing at multiple SNR levels...")
    multi_snr_results = system.process_multiple_snr_levels(
        audio_path, 
        snr_levels=[0, 5, 10, 15, 20, 25, 30]
    )
    
    system.save_results(multi_snr_results, 'multi_snr_results.json')
    
    logger.info("-" * 60)
    logger.info("All processing completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
