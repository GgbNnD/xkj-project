"""
远程声音控制系统 - 语音识别模块
Speech Recognition Module for Remote Voice Control System
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMNet(nn.Module):
    """LSTM 神经网络模型"""
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out



class AudioFeatureExtractor:
    """音频特征提取"""
    
    def __init__(self, fs: int = 44100):
        """
        初始化特征提取器
        
        Args:
            fs: 采样频率
        """
        self.fs = fs
    
    def extract_mfcc_features(self, signal: np.ndarray, 
                             n_mfcc: int = 13,
                             frame_length: int = 2048,
                             hop_length: int = 512) -> np.ndarray:
        """
        提取MFCC（梅尔频率倒谱系数）特征
        
        Args:
            signal: 音频信号
            n_mfcc: MFCC系数数量
            frame_length: 帧长
            hop_length: 帧间隔
            
        Returns:
            mfcc_features: MFCC特征矩阵
        """
        # 信号预处理：归一化
        if len(signal) > 0:
            # 去除直流分量
            signal = signal - np.mean(signal)
            # 幅度归一化
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val
                
        # 帧化处理
        n_frames = int(np.ceil((len(signal) - frame_length) / hop_length)) + 1
        frames = np.zeros((n_frames, frame_length))
        
        for i in range(n_frames):
            start = i * hop_length
            end = min(start + frame_length, len(signal))
            frames[i, :end-start] = signal[start:end]
        
        # 应用Hamming窗
        window = np.hamming(frame_length)
        frames = frames * window
        
        # 计算功率谱
        fft_size = 2 * frame_length
        power_spec = np.zeros((n_frames, fft_size // 2 + 1))
        
        for i in range(n_frames):
            fft_result = np.fft.rfft(frames[i], n=fft_size)
            power_spec[i] = np.abs(fft_result) ** 2
        
        # 梅尔滤波组
        mel_fb = self._create_mel_filterbank(fft_size // 2 + 1, n_mfcc)
        mel_spec = np.dot(power_spec, mel_fb.T)
        mel_spec = np.log(mel_spec + 1e-10)
        
        # DCT变换得到MFCC
        mfcc = self._dct(mel_spec)[:, :n_mfcc]
        
        logger.info(f"Extracted MFCC features: {mfcc.shape}")
        
        return mfcc
    
    def extract_spectral_features(self, signal: np.ndarray,
                                 frame_length: int = 2048,
                                 hop_length: int = 512) -> Dict:
        """
        提取频谱特征
        
        Args:
            signal: 音频信号
            frame_length: 帧长
            hop_length: 帧间隔
            
        Returns:
            features: 特征字典
        """
        # 短时傅里叶变换
        stft = self._stft(signal, frame_length, hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 功率谱
        power = magnitude ** 2
        log_power = np.log(power + 1e-10)
        
        # 谱质心
        freqs = np.fft.rfftfreq(frame_length, 1/self.fs)
        spec_centroid = np.sum(freqs * magnitude, axis=1) / (np.sum(magnitude, axis=1) + 1e-10)
        
        # 零交叉率
        zcr = self._zero_crossing_rate(signal, frame_length, hop_length)
        
        # 能量
        energy = np.sum(magnitude, axis=1)
        log_energy = np.log(energy + 1e-10)
        
        features = {
            'stft_magnitude': magnitude,
            'stft_phase': phase,
            'power': power,
            'log_power': log_power,
            'spec_centroid': spec_centroid,
            'zero_crossing_rate': zcr,
            'energy': energy,
            'log_energy': log_energy,
        }
        
        logger.info(f"Extracted spectral features: STFT shape {magnitude.shape}")
        
        return features
    
    def _stft(self, signal: np.ndarray, 
             frame_length: int, hop_length: int) -> np.ndarray:
        """短时傅里叶变换"""
        n_frames = int(np.ceil((len(signal) - frame_length) / hop_length)) + 1
        fft_size = 2 * frame_length
        stft = np.zeros((fft_size // 2 + 1, n_frames), dtype=complex)
        
        window = np.hamming(frame_length)
        
        for i in range(n_frames):
            start = i * hop_length
            end = min(start + frame_length, len(signal))
            frame = signal[start:end]
            
            # 补零
            frame_padded = np.zeros(frame_length)
            frame_padded[:len(frame)] = frame
            
            # 应用窗和FFT
            frame_windowed = frame_padded * window
            stft_frame = np.fft.rfft(frame_windowed, n=fft_size)
            stft[:, i] = stft_frame
        
        return stft
    
    def _zero_crossing_rate(self, signal: np.ndarray,
                           frame_length: int, hop_length: int) -> np.ndarray:
        """计算零交叉率"""
        n_frames = int(np.ceil((len(signal) - frame_length) / hop_length)) + 1
        zcr = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * hop_length
            end = min(start + frame_length, len(signal))
            frame = signal[start:end]
            
            # 计算零交叉次数
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
            zcr[i] = zero_crossings / len(frame) if len(frame) > 0 else 0
        
        return zcr
    
    def _create_mel_filterbank(self, n_fft: int, n_mfcc: int) -> np.ndarray:
        """创建梅尔滤波组"""
        # 简化版本：线性间隔的滤波器
        n_filters = n_mfcc + 2
        mel_fb = np.zeros((n_filters, n_fft))
        
        # 频率中心（线性近似）
        centers = np.linspace(0, n_fft - 1, n_filters).astype(int)
        
        for i in range(n_filters):
            mel_fb[i, centers[i]] = 1.0
        
        return mel_fb
    
    @staticmethod
    def _dct(x: np.ndarray) -> np.ndarray:
        """离散余弦变换"""
        N = x.shape[1]
        dct_matrix = np.zeros((N, N))
        
        for k in range(N):
            for n in range(N):
                dct_matrix[k, n] = np.cos(np.pi * k * (2*n + 1) / (2*N))
        
        return np.dot(x, dct_matrix.T)


class CommandClassifier:
    """命令分类器（基于 PyTorch LSTM）"""
    
    def __init__(self, num_commands: int = 4, model_path: str = 'data/speech_model_lstm.pth'):
        """
        初始化命令分类器
        
        Args:
            num_commands: 命令数量
            model_path: 模型保存路径
        """
        self.num_commands = num_commands
        self.command_names = ["前进", "后退", "停止", "旋转"]
        self.model_path = model_path
        
        # LSTM 参数
        self.input_size = 13  # MFCC 特征数量
        self.hidden_size = 64
        self.num_layers = 2
        self.max_seq_length = 100 # 固定序列长度
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMNet(self.input_size, self.hidden_size, num_commands, self.num_layers).to(self.device)
        self.trained = False
        
        # 尝试加载已有模型
        self.load_model()
    
    def _pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """将序列填充或截断到固定长度"""
        if len(sequence) > self.max_seq_length:
            return sequence[:self.max_seq_length]
        elif len(sequence) < self.max_seq_length:
            pad_width = self.max_seq_length - len(sequence)
            return np.pad(sequence, ((0, pad_width), (0, 0)), mode='constant')
        return sequence

    def train(self, features_list: List[np.ndarray], labels: np.ndarray) -> Dict[str, float]:
        """
        训练分类器
        
        Args:
            features_list: 特征列表，每个元素是 (seq_len, n_mfcc) 的数组
            labels: 标签向量
            
        Returns:
            metrics: 训练指标
        """
        logger.info(f"Training LSTM model on {self.device}...")
        
        # 预处理数据：填充序列
        X_padded = np.array([self._pad_sequence(f) for f in features_list])
        
        # 转换为 Tensor
        X_tensor = torch.FloatTensor(X_padded).to(self.device)
        y_tensor = torch.LongTensor(labels).to(self.device)
        
        # 创建 DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        num_epochs = 50
        
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            if (epoch + 1) % 10 == 0:
                acc = 100 * correct / total
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%')
        
        self.trained = True
        self.save_model()
        
        return {'train_acc': correct / total, 'test_acc': correct / total} # 简化，暂无验证集
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        预测命令
        
        Args:
            features: 特征序列 (seq_len, n_mfcc)
            
        Returns:
            predicted_command: 预测的命令索引
            confidence: 置信度
        """
        if not self.trained:
            logger.warning("Classifier not trained, returning random prediction")
            return np.random.randint(0, self.num_commands), 0.0
        
        self.model.eval()
        with torch.no_grad():
            # 预处理
            features_padded = self._pad_sequence(features)
            # 增加 batch 维度: (1, seq_len, input_size)
            input_tensor = torch.FloatTensor(features_padded).unsqueeze(0).to(self.device)
            
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            confidence, predicted = torch.max(probs, 1)
            
            cmd_id = predicted.item()
            conf_val = confidence.item()
            
            # 详细日志
            probs_list = probs[0].cpu().numpy()
            probs_str = ", ".join([f"{self.command_names[i]}: {p:.2f}" for i, p in enumerate(probs_list)])
            logger.info(f"LSTM Prediction: {probs_str}")
            
            return cmd_id, conf_val
    
    def save_model(self) -> None:
        """保存模型到磁盘"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            logger.info(f"LSTM Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self) -> bool:
        """从磁盘加载模型"""
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.trained = True
                logger.info(f"LSTM Model loaded from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        return False

    def get_command_name(self, command_id: int) -> str:
        """获取命令名称"""
        if 0 <= command_id < len(self.command_names):
            return self.command_names[command_id]
        return "Unknown"


import speech_recognition as sr

class SpeechRecognitionSystem:
    """完整的语音识别系统"""
    
    def __init__(self, fs: int = 44100, num_commands: int = 4):
        """
        初始化语音识别系统
        
        Args:
            fs: 采样频率
            num_commands: 命令数量
        """
        self.fs = fs
        self.feature_extractor = AudioFeatureExtractor(fs)
        self.classifier = CommandClassifier(num_commands)
        # self.recognizer = sr.Recognizer() # 移除在线识别器
        
    # def recognize_google(self, signal: np.ndarray) -> Tuple[str, float]:
    #     """
    #     使用 Google Speech Recognition API 进行识别 (需要联网)
    #     """
    #     pass # 移除在线识别功能
    
    def recognize_command(self, signal: np.ndarray) -> Tuple[str, float]:
        """
        识别音频信号中的命令
        
        Args:
            signal: 音频信号
            
        Returns:
            command: 识别的命令名称
            confidence: 置信度
        """
        try:
            # 提取特征 (返回序列)
            mfcc_features = self.feature_extractor.extract_mfcc_features(signal)
            
            # 不再聚合特征，直接使用序列
            # if mfcc_features.shape[0] > 0:
            #     aggregated_features = np.mean(mfcc_features, axis=0)
            # else:
            #     aggregated_features = np.zeros(13)
            
            if mfcc_features.shape[0] == 0:
                logger.warning("No MFCC features extracted")
                return "Unknown", 0.0

            # 预测命令
            if self.classifier.trained:
                command_id, confidence = self.classifier.predict(mfcc_features)
                command = self.classifier.get_command_name(command_id)
            else:
                # 如果未训练，使用简单的启发式方法
                command = self._heuristic_recognition(signal)
                confidence = 0.5
            
            logger.info(f"Recognized command: {command} (confidence: {confidence:.2f})")
            
            return command, confidence
        except Exception as e:
            logger.error(f"Error in command recognition: {e}")
            return "Unknown Command", 0.0

    # def recognize_google(self, signal: np.ndarray) -> Tuple[str, float]:
    #     """
    #     使用 Google Web Speech API 进行在线语音识别 (支持英文)
    #     """
    #     pass # 移除在线识别功能
    
    def _heuristic_recognition(self, signal: np.ndarray) -> str:
        """
        启发式识别方法（基于简单特征）
        
        Args:
            signal: 音频信号
            
        Returns:
            command: 识别的命令
        """
        # 计算简单特征
        energy = np.sum(signal ** 2)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal)))) / 2
        spectrum = np.abs(np.fft.rfft(signal))
        dominant_freq = np.argmax(spectrum) * self.fs / len(signal)
        
        # 基于简单规则的分类
        commands = ["前进", "后退", "停止", "旋转"]
        
        if dominant_freq < 2000:
            return commands[0]  # "前进"
        elif dominant_freq < 4000:
            return commands[1]  # "后退"
        elif dominant_freq < 6000:
            return commands[2]  # "停止"
        else:
            return commands[3]  # "旋转"


if __name__ == "__main__":
    # 测试语音识别
    fs = 44100
    duration = 1
    
    # 生成测试信号（简单的正弦波）
    t = np.linspace(0, duration, int(fs * duration), False)
    test_signal = 0.3 * np.sin(2 * np.pi * 1000 * t)
    
    # 创建识别系统
    sys = SpeechRecognitionSystem(fs=fs, num_commands=4)
    
    # 识别命令
    command, confidence = sys.recognize_command(test_signal)
    print(f"Recognized: {command} (confidence: {confidence:.4f})")
