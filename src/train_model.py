import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from speech_recognition_system import SpeechRecognitionSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.fs = 44100
        self.duration = 2.0  # 录音时长2秒
        self.commands = ["前进", "后退", "停止", "旋转"]
        self.data_dir = "data/training"
        self.recognition_system = SpeechRecognitionSystem(fs=self.fs)
        
        # 确保数据目录存在
        for cmd in self.commands:
            os.makedirs(os.path.join(self.data_dir, cmd), exist_ok=True)

    def record_sample(self, command_idx: int, sample_idx: int):
        """录制单个样本"""
        command = self.commands[command_idx]
        print(f"\n准备录制 '{command}' (样本 #{sample_idx + 1})")
        print("3秒后开始录音...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        print("开始录音! 请说话...")
        
        # 强制停止之前的流，防止干扰
        sd.stop()
        # 增加一点缓冲时间
        time.sleep(0.2)
        
        # 使用 blocking=True 确保录音完全执行
        recording = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=1, blocking=True)
        
        print("录音结束.")
        
        # 保存文件
        filename = os.path.join(self.data_dir, command, f"{command}_{sample_idx}.wav")
        sf.write(filename, recording, self.fs)
        print(f"已保存: {filename}")

    def collect_data(self, samples_per_command: int = 5):
        """收集训练数据"""
        print("=== 开始数据收集 ===")
        print(f"将为每个命令录制 {samples_per_command} 个样本")
        
        for cmd_idx, cmd in enumerate(self.commands):
            print(f"\n--- 正在录制命令: {cmd} ---")
            for i in range(samples_per_command):
                self.record_sample(cmd_idx, i)
                
                if i < samples_per_command - 1:
                    input("按回车键继续录制下一个样本...")
        
        print("\n数据收集完成!")

    def train_model(self):
        """训练模型"""
        print("\n=== 开始训练模型 ===")
        features = []
        labels = []
        
        for cmd_idx, cmd in enumerate(self.commands):
            cmd_dir = os.path.join(self.data_dir, cmd)
            files = [f for f in os.listdir(cmd_dir) if f.endswith('.wav')]
            
            print(f"处理 '{cmd}' 的样本: {len(files)} 个文件")
            
            for f in files:
                filepath = os.path.join(cmd_dir, f)
                try:
                    signal, _ = sf.read(filepath)
                    # 提取特征
                    mfcc = self.recognition_system.feature_extractor.extract_mfcc_features(signal)
                    # 聚合特征 (取平均)
                    if mfcc.shape[0] > 0:
                        feat = np.mean(mfcc, axis=0)
                        features.append(feat)
                        labels.append(cmd_idx)
                except Exception as e:
                    logger.error(f"Error processing {filepath}: {e}")
        
        if not features:
            print("没有找到训练数据!")
            return
            
        X = np.array(features)
        y = np.array(labels)
        
        print(f"特征矩阵形状: {X.shape}")
        
        # 训练
        metrics = self.recognition_system.classifier.train(X, y)
        print(f"训练完成! 准确率: {metrics['test_acc']:.2f}")

def main():
    trainer = ModelTrainer()
    
    while True:
        print("\n请选择操作:")
        print("1. 录制训练数据")
        print("2. 训练模型")
        print("3. 退出")
        
        choice = input("请输入选项 (1-3): ")
        
        if choice == '1':
            try:
                num = int(input("每个命令录制多少个样本? (建议至少5个): "))
                trainer.collect_data(num)
            except ValueError:
                print("请输入有效的数字")
        elif choice == '2':
            trainer.train_model()
        elif choice == '3':
            break
        else:
            print("无效选项")

if __name__ == "__main__":
    main()
