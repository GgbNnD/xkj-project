"""
远程声音控制系统 - 模型测试脚本
Model Testing Script for Remote Voice Control System
"""

import os
import numpy as np
import soundfile as sf
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import logging
from speech_recognition_system import SpeechRecognitionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    print("="*60)
    print("模型性能测试")
    print("="*60)
    
    data_dir = "data/training"
    model_path = "data/speech_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先使用训练工具训练模型。")
        return

    # 初始化识别系统
    sys = SpeechRecognitionSystem()
    
    # 检查模型是否加载
    if not sys.classifier.trained:
        print("警告: 模型未能成功加载，系统正在使用启发式规则（非AI模式）。")
    else:
        print(f"成功加载模型: {model_path}")

    commands = ["前进", "后退", "停止", "旋转"]
    y_true = []
    y_pred = []
    
    total_files = 0
    
    print("\n开始评估...")
    
    for cmd_idx, cmd in enumerate(commands):
        cmd_dir = os.path.join(data_dir, cmd)
        if not os.path.exists(cmd_dir):
            continue
            
        files = [f for f in os.listdir(cmd_dir) if f.endswith('.wav')]
        print(f"正在测试 '{cmd}' ({len(files)} 个样本)...")
        
        for f in files:
            filepath = os.path.join(cmd_dir, f)
            try:
                signal, fs = sf.read(filepath)
                
                # 使用识别系统进行预测
                predicted_cmd, confidence = sys.recognize_command(signal)
                
                y_true.append(cmd)
                y_pred.append(predicted_cmd)
                total_files += 1
                
            except Exception as e:
                print(f"处理文件 {f} 出错: {e}")

    if total_files == 0:
        print("\n错误: 没有找到测试数据。请先录制一些样本。")
        return

    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    
    # 计算准确率
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total_files
    print(f"\n总体准确率: {accuracy:.2%}")
    
    print("\n详细报告:")
    print(classification_report(y_true, y_pred, target_names=commands, labels=commands, zero_division=0))
    
    print("\n混淆矩阵 (行=真实值, 列=预测值):")
    cm = confusion_matrix(y_true, y_pred, labels=commands)
    print(cm)
    
    print("\n" + "-"*60)
    print("分析建议:")
    if accuracy < 0.5:
        print("1. 模型准确率很低。可能原因：")
        print("   - 样本太少（建议每个命令至少10个）")
        print("   - 录音环境噪音太大")
        print("   - 录音时说话时机不对（比如只录到了静音）")
    elif accuracy > 0.9:
        print("1. 模型在训练数据上表现很好。")
        print("2. 如果GUI中仍然识别错误，可能是GUI录音的问题（如录到了静音）。")
    else:
        print("1. 模型表现尚可，但有提升空间。建议增加更多样本。")

if __name__ == "__main__":
    test_model()
