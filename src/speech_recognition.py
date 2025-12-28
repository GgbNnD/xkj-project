"""
远程声音控制系统 - 语音识别模块
Speech Recognition Module for Remote Voice Control System
"""

import numpy as np
from typing import Tuple, List, Dict
import logging

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
    """命令分类器（简化版本）"""
    
    def __init__(self, num_commands: int = 4):
        """
        初始化命令分类器
        
        Args:
            num_commands: 命令数量
        """
        self.num_commands = num_commands
        self.command_names = [f'Command_{i}' for i in range(num_commands)]
        self.trained = False
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        训练分类器
        
        Args:
            features: 特征矩阵 (样本数, 特征数)
            labels: 标签向量 (样本数,)
        """
        # 计算每个类的特征均值
        self.class_means = np.zeros((self.num_commands, features.shape[1]))
        self.class_covs = np.zeros((self.num_commands, features.shape[1], features.shape[1]))
        
        for i in range(self.num_commands):
            class_features = features[labels == i]
            if len(class_features) > 0:
                self.class_means[i] = np.mean(class_features, axis=0)
                self.class_covs[i] = np.cov(class_features.T)
        
        self.trained = True
        logger.info(f"Classifier trained on {len(features)} samples")
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        预测命令
        
        Args:
            features: 特征向量 (特征数,)
            
        Returns:
            predicted_command: 预测的命令索引
            confidence: 置信度
        """
        if not self.trained:
            logger.warning("Classifier not trained, returning random prediction")
            return np.random.randint(0, self.num_commands), 0.5
        
        # 计算到每个类中心的距离
        distances = np.zeros(self.num_commands)
        for i in range(self.num_commands):
            distances[i] = np.linalg.norm(features - self.class_means[i])
        
        # 选择最近的类
        predicted_command = np.argmin(distances)
        
        # 计算置信度（使用反向距离作为近似）
        min_distance = distances[predicted_command]
        confidence = 1.0 / (1.0 + min_distance)
        
        return predicted_command, confidence
    
    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测
        
        Args:
            features: 特征矩阵 (样本数, 特征数)
            
        Returns:
            predictions: 预测的命令 (样本数,)
            confidences: 置信度 (样本数,)
        """
        predictions = []
        confidences = []
        
        for feature in features:
            pred, conf = self.predict(feature)
            predictions.append(pred)
            confidences.append(conf)
        
        return np.array(predictions), np.array(confidences)
    
    def get_command_name(self, command_id: int) -> str:
        """获取命令名称"""
        if 0 <= command_id < self.num_commands:
            return self.command_names[command_id]
        return "Unknown Command"


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
        commands = ["Forward", "Backward", "Stop", "Rotate"]
        
        if dominant_freq < 2000:
            return commands[0]  # "Forward"
        elif dominant_freq < 4000:
            return commands[1]  # "Backward"
        elif dominant_freq < 6000:
            return commands[2]  # "Stop"
        else:
            return commands[3]  # "Rotate"


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
