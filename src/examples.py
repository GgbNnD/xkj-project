"""
远程声音控制系统 - 简单使用示例
Simple Usage Examples for Remote Voice Control System
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import RemoteVoiceControlSystem


def example_1_basic_processing():
    """
    示例1：基本处理流程
    """
    print("\n" + "="*70)
    print("示例 1: 基本的音频处理流程")
    print("="*70)
    
    # 创建系统
    system = RemoteVoiceControlSystem()
    
    # 处理音频文件
    audio_file = 'data/command.flac'
    
    if not os.path.exists(audio_file):
        print(f"错误：找不到音频文件 {audio_file}")
        return
    
    result = system.process_audio_file(audio_file, snr_db=20)
    
    # 打印结果摘要
    if result['overall_success']:
        print("\n处理成功！")
        summary = result['summary']
        print(f"识别的命令: {summary['recognized_command']}")
        print(f"控制执行: {'成功' if summary['control_success'] else '失败'}")
        print(f"\n系统统计:")
        for key, value in summary['system_stats'].items():
            print(f"  {key}: {value}")
    else:
        print(f"处理失败: {result.get('error', 'Unknown error')}")
    
    return result


def example_2_multiple_snr_levels():
    """
    示例2：在多个信噪比下处理
    """
    print("\n" + "="*70)
    print("示例 2: 多信噪比水平下的处理")
    print("="*70)
    
    # 创建系统
    system = RemoteVoiceControlSystem()
    
    # 处理音频文件
    audio_file = 'data/command.flac'
    
    if not os.path.exists(audio_file):
        print(f"错误：找不到音频文件 {audio_file}")
        return
    
    # 定义SNR水平
    snr_levels = [0, 5, 10, 15, 20, 25, 30]
    
    print(f"在 {len(snr_levels)} 个不同的SNR水平下处理...")
    results = system.process_multiple_snr_levels(audio_file, snr_levels)
    
    # 分析BER随SNR的变化
    print("\nBER vs SNR 分析:")
    print("-" * 40)
    print("SNR (dB)  |  BER       |  Command")
    print("-" * 40)
    
    for snr in snr_levels:
        result = results['results'][snr]
        if result['overall_success']:
            stages = result['stages']
            demod_stage = stages.get('demodulation', {})
            ber = demod_stage.get('ber', 0)
            command = stages.get('speech_recognition', {}).get('recognized_command', 'Unknown')
            print(f"{snr:3d}       |  {ber:.6f}  |  {command}")
        else:
            print(f"{snr:3d}       |  Failed")
    
    return results


def example_3_single_module_test():
    """
    示例3：单个模块的测试
    """
    print("\n" + "="*70)
    print("示例 3: 单个模块的独立测试")
    print("="*70)
    
    # 测试信号处理模块
    print("\n[1] 信号处理模块")
    print("-" * 40)
    from signal_processing import SignalProcessing
    
    processor = SignalProcessing(fs=44100, quantization_bits=8)
    t = np.linspace(0, 1, 44100, False)
    signal = 0.5 * np.sin(2 * np.pi * 1000 * t)
    quantized = processor.signal_quantization(signal)
    print(f"✓ 信号量化完成，MSE = {np.mean((signal - quantized)**2):.6f}")
    
    # 测试Huffman编码
    print("\n[2] Huffman编码模块")
    print("-" * 40)
    from source_encoding import SourceEncoding
    
    encoder = SourceEncoding()
    test_data = np.array([0, 1, 2, 1, 0, 3, 1, 0, 2, 1] * 50, dtype=int)
    encoded, dict_table, stats = encoder.encode(test_data)
    print(f"✓ Huffman编码完成")
    print(f"  压缩比: {stats['compression_ratio']:.4f}")
    print(f"  信息熵: {stats['entropy']:.6f} bits/symbol")
    print(f"  编码效率: {stats['coding_efficiency']:.6f}")
    
    # 测试BCH编码
    print("\n[3] BCH编码模块")
    print("-" * 40)
    from channel_encoding import BCHCoder
    
    coder = BCHCoder(n=15, k=7)
    test_bits = np.array([1, 0, 1, 1, 0, 1, 0] * 20, dtype=int)
    encoded_bch = coder.encode(test_bits)
    print(f"✓ BCH编码完成")
    print(f"  输入比特: {len(test_bits)}")
    print(f"  输出比特: {len(encoded_bch)}")
    print(f"  码率: {len(test_bits)/len(encoded_bch):.4f}")
    
    # 测试BPSK调制
    print("\n[4] BPSK调制模块")
    print("-" * 40)
    from modulation import Modulation
    
    modem = Modulation()
    test_bits = np.array([0, 1, 0, 1, 1, 0] * 20, dtype=int)
    modulated = modem.bpsk_modulate(test_bits)
    received = modem.channel_transmission(modulated, snr_db=20)
    demodulated = modem.bpsk_demodulate(received)
    stats = modem.get_error_statistics(test_bits, demodulated)
    print(f"✓ BPSK调制/解调完成")
    print(f"  BER: {stats['ber']:.6f}")


def example_4_custom_configuration():
    """
    示例4：使用自定义配置
    """
    print("\n" + "="*70)
    print("示例 4: 使用自定义配置")
    print("="*70)
    
    # 定义自定义配置
    custom_config = {
        'fs': 44100,
        'quantization_bits': 6,           # 降低量化位数
        'channel_code_n': 15,
        'channel_code_k': 7,
        'snr_db': 15,                     # 降低SNR
        'num_commands': 4,
    }
    
    print("自定义配置:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    
    # 创建系统
    system = RemoteVoiceControlSystem(config=custom_config)
    
    # 处理音频
    audio_file = 'data/command.flac'
    
    if not os.path.exists(audio_file):
        print(f"错误：找不到音频文件 {audio_file}")
        return
    
    result = system.process_audio_file(audio_file, snr_db=custom_config['snr_db'])
    
    if result['overall_success']:
        print("\n✓ 处理成功！")
        summary = result['summary']
        print(f"识别的命令: {summary['recognized_command']}")
    
    return result


def example_5_analyze_performance():
    """
    示例5：分析系统性能
    """
    print("\n" + "="*70)
    print("示例 5: 系统性能分析")
    print("="*70)
    
    # 创建系统
    system = RemoteVoiceControlSystem()
    
    # 处理音频文件
    audio_file = 'data/command.flac'
    
    if not os.path.exists(audio_file):
        print(f"错误：找不到音频文件 {audio_file}")
        return
    
    # 在多个SNR下处理
    snr_levels = np.linspace(0, 30, 7).astype(int)
    results = system.process_multiple_snr_levels(audio_file, snr_levels.tolist())
    
    # 分析性能
    print("\n性能分析:")
    print("-" * 50)
    
    ber_values = []
    snr_values = []
    
    for snr in snr_levels:
        result = results['results'][int(snr)]
        if result['overall_success']:
            stages = result['stages']
            demod_stage = stages.get('demodulation', {})
            ber = demod_stage.get('ber', 0)
            ber_values.append(ber)
            snr_values.append(snr)
    
    if ber_values:
        print(f"SNR范围: {min(snr_values):.1f} - {max(snr_values):.1f} dB")
        print(f"平均BER: {np.mean(ber_values):.6f}")
        print(f"最低BER: {np.min(ber_values):.6f} (at SNR={snr_values[np.argmin(ber_values)]:.1f} dB)")
        print(f"最高BER: {np.max(ber_values):.6f} (at SNR={snr_values[np.argmax(ber_values)]:.1f} dB)")
        
        # BER改进比例
        if len(ber_values) > 1:
            improvement = (ber_values[0] - ber_values[-1]) / (ber_values[0] + 1e-10) * 100
            print(f"SNR提高时BER改进: {improvement:.1f}%")
    
    return results


def main():
    """主函数"""
    print("\n")
    print("#" * 70)
    print("# 远程声音控制系统 - 使用示例")
    print("# Remote Voice Control System - Usage Examples")
    print("#" * 70)
    
    examples = [
        ("基本处理流程", example_1_basic_processing),
        ("多信噪比处理", example_2_multiple_snr_levels),
        ("单个模块测试", example_3_single_module_test),
        ("自定义配置", example_4_custom_configuration),
        ("性能分析", example_5_analyze_performance),
    ]
    
    print("\n可用的示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print(f"  0. 运行所有示例")
    print(f"  q. 退出")
    
    while True:
        choice = input("\n请选择要运行的示例 (0-5): ").strip().lower()
        
        if choice == 'q':
            print("退出程序")
            break
        elif choice == '0':
            for name, example_func in examples:
                try:
                    example_func()
                except Exception as e:
                    print(f"错误: {e}")
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            idx = int(choice) - 1
            try:
                examples[idx][1]()
            except Exception as e:
                print(f"错误: {e}")
        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"\n错误: {e}")
