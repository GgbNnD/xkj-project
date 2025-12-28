"""
远程声音控制系统 - 测试脚本
Test Script for Remote Voice Control System
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from signal_processing import SignalProcessing
from source_encoding import SourceEncoding
from channel_encoding import BCHCoder
from modulation import Modulation
from speech_recognition import SpeechRecognitionSystem
from control_system import ControlSystem


def test_signal_processing():
    """测试信号处理模块"""
    print("\n" + "="*60)
    print("TEST 1: Signal Processing Module")
    print("="*60)
    
    try:
        processor = SignalProcessing(fs=44100, quantization_bits=8)
        
        # 生成测试信号
        t = np.linspace(0, 1, 44100, False)
        test_signal = 0.5 * np.sin(2 * np.pi * 1000 * t)
        
        # 量化
        quantized = processor.signal_quantization(test_signal)
        bits = processor.quantization_to_bits(quantized)
        
        print(f"✓ Signal quantization successful")
        print(f"  Original signal length: {len(test_signal)}")
        print(f"  Quantization levels: {processor.quantization_levels}")
        print(f"  Bit sequence length: {len(bits)}")
        print(f"  Quantization MSE: {np.mean((test_signal - quantized)**2):.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Signal processing test failed: {e}")
        return False


def test_source_encoding():
    """测试信源编码模块"""
    print("\n" + "="*60)
    print("TEST 2: Source Encoding Module (Huffman)")
    print("="*60)
    
    try:
        encoder = SourceEncoding()
        
        # 生成测试数据
        test_signal = np.array([0, 1, 2, 1, 0, 3, 1, 0, 2, 1] * 100, dtype=int)
        
        # 编码
        encoded, code_dict, stats = encoder.encode(test_signal)
        
        print(f"✓ Huffman encoding successful")
        print(f"  Original bits: {stats['original_bits']}")
        print(f"  Encoded bits: {stats['encoded_bits']}")
        print(f"  Compression ratio: {stats['compression_ratio']:.4f}")
        print(f"  Entropy: {stats['entropy']:.6f} bits/symbol")
        print(f"  Average code length: {stats['avg_code_length']:.6f}")
        print(f"  Unique symbols: {len(stats['symbol_frequencies'])}")
        
        # 解码
        decoded = encoder.decode(encoded)
        errors = np.sum(decoded != test_signal)
        
        print(f"  Decoding errors: {errors}")
        print(f"✓ Huffman decoding successful")
        
        return True
    except Exception as e:
        print(f"✗ Source encoding test failed: {e}")
        return False


def test_channel_encoding():
    """测试信道编码模块"""
    print("\n" + "="*60)
    print("TEST 3: Channel Encoding Module (BCH)")
    print("="*60)
    
    try:
        coder = BCHCoder(n=15, k=7)
        
        # 生成测试比特
        test_bits = np.array([1, 0, 1, 1, 0, 1, 0] * 100, dtype=int)
        
        # 编码
        encoded = coder.encode(test_bits)
        print(f"✓ BCH encoding successful")
        print(f"  Input bits: {len(test_bits)}")
        print(f"  Encoded bits: {len(encoded)}")
        print(f"  Code rate: {len(test_bits)/len(encoded):.4f}")
        
        # 解码无错
        decoded = coder.decode(encoded)
        errors_no_noise = np.sum(decoded != test_bits)
        print(f"  Decoding errors (no noise): {errors_no_noise}")
        
        # 添加错误进行测试
        encoded_with_error = encoded.copy()
        encoded_with_error[10] = (encoded_with_error[10] + 1) % 2
        encoded_with_error[20] = (encoded_with_error[20] + 1) % 2
        
        decoded_with_error = coder.decode(encoded_with_error)
        errors_with_noise = np.sum(decoded_with_error != test_bits)
        print(f"  Decoding errors (with 2 bit errors): {errors_with_noise}")
        print(f"✓ BCH decoding successful")
        
        return True
    except Exception as e:
        print(f"✗ Channel encoding test failed: {e}")
        return False


def test_modulation():
    """测试调制解调模块"""
    print("\n" + "="*60)
    print("TEST 4: Modulation/Demodulation Module (BPSK)")
    print("="*60)
    
    try:
        modem = Modulation()
        
        # 生成测试比特
        test_bits = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 100, dtype=int)
        
        # 调制
        modulated = modem.bpsk_modulate(test_bits, amplitude=1.0)
        print(f"✓ BPSK modulation successful")
        print(f"  Input bits: {len(test_bits)}")
        print(f"  Modulated signal length: {len(modulated)}")
        print(f"  Signal power: {np.mean(modulated**2):.6f}")
        
        # 信道传输
        received = modem.channel_transmission(modulated, snr_db=20)
        print(f"✓ Channel transmission (AWGN) successful")
        print(f"  SNR: 20 dB")
        
        # 解调
        demodulated = modem.bpsk_demodulate(received, threshold=0.0)
        stats = modem.get_error_statistics(test_bits, demodulated)
        
        print(f"✓ BPSK demodulation successful")
        print(f"  Bit errors: {stats['bit_errors']}")
        print(f"  Total bits: {stats['total_bits']}")
        print(f"  BER: {stats['ber']:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Modulation test failed: {e}")
        return False


def test_speech_recognition():
    """测试语音识别模块"""
    print("\n" + "="*60)
    print("TEST 5: Speech Recognition Module")
    print("="*60)
    
    try:
        sys_rec = SpeechRecognitionSystem(fs=44100, num_commands=4)
        
        # 生成测试信号
        t = np.linspace(0, 1, 44100, False)
        test_signal = 0.3 * np.sin(2 * np.pi * 1000 * t)
        
        # 识别命令
        command, confidence = sys_rec.recognize_command(test_signal)
        
        print(f"✓ Speech recognition successful")
        print(f"  Recognized command: {command}")
        print(f"  Confidence: {confidence:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Speech recognition test failed: {e}")
        return False


def test_control_system():
    """测试控制系统模块"""
    print("\n" + "="*60)
    print("TEST 6: Control System Module (PID)")
    print("="*60)
    
    try:
        control_sys = ControlSystem(system_type='motor')
        
        # 执行命令
        success = control_sys.execute_command('forward', duration=0.5, dt=0.01)
        print(f"✓ Control command execution successful")
        print(f"  Command: forward")
        print(f"  Duration: 0.5 seconds")
        print(f"  Execution success: {success}")
        
        # 获取性能指标
        metrics = control_sys.get_performance_metrics()
        print(f"  Performance metrics:")
        for key, value in metrics.items():
            if value is not None:
                print(f"    {key}: {value:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Control system test failed: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("#" * 60)
    print("# Remote Voice Control System - Test Suite")
    print("#" * 60)
    
    tests = [
        ("Signal Processing", test_signal_processing),
        ("Source Encoding (Huffman)", test_source_encoding),
        ("Channel Encoding (BCH)", test_channel_encoding),
        ("Modulation/Demodulation (BPSK)", test_modulation),
        ("Speech Recognition", test_speech_recognition),
        ("Control System (PID)", test_control_system),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
