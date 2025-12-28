"""
远程声音控制系统 - 语音识别模块
Speech Recognition Module for Remote Voice Control System
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
import os
import joblib
import speech_recognition as sr
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """命令分类器（基于神经网络 MLP）"""
    
    def __init__(self, num_commands: int = 4, model_path: str = 'data/speech_model.pkl'):
        """
        初始化命令分类器
        
        Args:
            num_commands: 命令数量
            model_path: 模型保存路径
        """
        self.num_commands = num_commands
        # 更新为中文命令
        self.command_names = ["前进", "后退", "停止", "旋转"]
        self.model_path = model_path
        self.model = None
        self.trained = False
        
        # 尝试加载已有模型
        self.load_model()
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        训练分类器
        
        Args:
            features: 特征矩阵 (样本数, 特征数)
            labels: 标签向量 (样本数,)
            
        Returns:
            metrics: 训练指标
        """
        # 创建MLP管道：标准化 -> MLP
        # 使用两个隐藏层 (128, 64)，最大迭代次数 1000
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), 
                                max_iter=1000, 
                                activation='relu',
                                solver='adam',
                                random_state=42,
                                early_stopping=True,
                                validation_fraction=0.1))
        ])
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 训练模型
        logger.info(f"Training MLP model with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # 评估模型
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        logger.info(f"Model trained. Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        self.trained = True
        self.save_model()
        
        return {'train_acc': train_acc, 'test_acc': test_acc}
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        预测命令
        
        Args:
            features: 特征向量 (特征数,)
            
        Returns:
            predicted_command: 预测的命令索引
            confidence: 置信度
        """
        if not self.trained or self.model is None:
            logger.warning("Classifier not trained, returning random prediction")
            return np.random.randint(0, self.num_commands), 0.0
        
        # 确保特征是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # 预测
        try:
            probs = self.model.predict_proba(features)[0]
            predicted_command = np.argmax(probs)
            confidence = probs[predicted_command]
            
            # 详细日志
            probs_str = ", ".join([f"{self.command_names[i]}: {p:.2f}" for i, p in enumerate(probs)])
            logger.info(f"Prediction probabilities: {probs_str}")
            
            return predicted_command, confidence
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.0
    
    def save_model(self) -> None:
        """保存模型到磁盘"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self) -> bool:
        """从磁盘加载模型"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.trained = True
                logger.info(f"Model loaded from {self.model_path}")
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
        self.recognizer = sr.Recognizer()
        
    def recognize_google(self, signal: np.ndarray) -> Tuple[str, float]:
        """
        使用 Google Speech Recognition API 进行识别 (需要联网)
        
        Args:
            signal: 音频信号
            
        Returns:
            command: 识别的命令
            confidence: 置信度 (模拟值，API不总是返回)
        """
        try:
            # 将 float32 (-1.0 到 1.0) 转换为 int16 PCM
            signal_int16 = (signal * 32767).astype(np.int16)
            audio_data = sr.AudioData(signal_int16.tobytes(), self.fs, 2)
            
            # 识别英文
            text = self.recognizer.recognize_google(audio_data, language="en-US")
            logger.info(f"Google Speech Recognition result: {text}")
            
            # 映射到系统命令
            text = text.lower()
            if "forward" in text or "go" in text or "move" in text:
                return "前进", 0.95
            elif "back" in text or "backward" in text or "retreat" in text:
                return "后退", 0.95
            elif "stop" in text or "halt" in text or "wait" in text:
                return "停止", 0.95
            elif "rotate" in text or "turn" in text or "spin" in text:
                return "旋转", 0.95
            else:
                return "Unknown", 0.0
                
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
            return "Unknown", 0.0
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return "Error", 0.0
        except Exception as e:
            logger.error(f"Error in google recognition: {e}")
            return "Error", 0.0
    
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
            # 提取特征
            mfcc_features = self.feature_extractor.extract_mfcc_features(signal)
            
            # 聚合特征（取平均）
            if mfcc_features.shape[0] > 0:
                aggregated_features = np.mean(mfcc_features, axis=0)
            else:
                aggregated_features = np.zeros(13)
            
            # 预测命令
            if self.classifier.trained:
                command_id, confidence = self.classifier.predict(aggregated_features)
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

    def recognize_google(self, signal: np.ndarray) -> Tuple[str, float]:
        """
        使用 Google Web Speech API 进行在线语音识别 (支持英文)
        
        Args:
            signal: 音频信号 (numpy array)
            
        Returns:
            command: 识别的命令名称 (映射回中文命令)
            confidence: 置信度
        """
        recognizer = sr.Recognizer()
        
        # 将 numpy float array 转换为 int16 bytes
        # 确保信号在 -1 到 1 之间
        signal = np.clip(signal, -1.0, 1.0)
        signal_int16 = (signal * 32767).astype(np.int16)
        
        # 创建 AudioData 对象
        # sample_width=2 (16-bit), sample_rate=self.fs
        audio_data = sr.AudioData(signal_int16.tobytes(), self.fs, 2)
        
        try:
            logger.info("Sending audio to Google Speech Recognition...")
            # 使用 Google Web Speech API
            text = recognizer.recognize_google(audio_data, language="en-US")
            logger.info(f"Google Speech Recognition result: {text}")
            
            # 简单的关键词映射
            text = text.lower()
            if "forward" in text or "go" in text or "move" in text:
                return "前进", 0.95
            elif "back" in text or "backward" in text or "reverse" in text:
                return "后退", 0.95
            elif "stop" in text or "halt" in text or "wait" in text:
                return "停止", 0.95
            elif "rotate" in text or "turn" in text or "spin" in text:
                return "旋转", 0.95
            else:
                logger.info(f"Unmapped command: {text}")
                return "未知命令", 0.0
                
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
            return "无法识别", 0.0
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return "服务错误", 0.0
    
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
