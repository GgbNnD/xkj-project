"""
远程声音控制系统 - 信道编码模块（BCH编码）
Channel Encoding Module for Remote Voice Control System
"""

import numpy as np
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BCHCoder:
    """(n,k) BCH编码器/解码器"""
    
    def __init__(self, n: int = 15, k: int = 7):
        """
        初始化BCH编码器
        
        Args:
            n: 码字长度 (例如15)
            k: 信息位长度 (例如7)
        """
        self.n = n  # 码字长度
        self.k = k  # 信息位长度
        self.t = (n - k) // 2  # 可纠正的错误位数
        
        # 对于(15,7) BCH码，生成多项式
        # G(x) = x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1
        # 对应系数: [1,0,0,1,0,1,1,0,1]
        self.generator_poly = self._get_generator_poly()
        
        logger.info(f"BCH({n},{k}) coder initialized")
        logger.info(f"Error correction capability: {self.t} bits")
    
    def _get_generator_poly(self) -> np.ndarray:
        """获取生成多项式系数"""
        if self.n == 15 and self.k == 7:
            # (15,7) BCH码的生成多项式
            # G(x) = x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1
            return np.array([1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=int)
        else:
            # 一般情况：使用简化的多项式
            return np.ones(self.n - self.k + 1, dtype=int)
    
    def _gf_poly_divide(self, dividend: np.ndarray, 
                       divisor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        在GF(2)上进行多项式除法
        
        Args:
            dividend: 被除数
            divisor: 除数
            
        Returns:
            quotient: 商
            remainder: 余数
        """
        dividend = dividend.copy()
        divisor = divisor.copy()
        
        # 移除首位0
        while len(dividend) > 0 and dividend[0] == 0:
            dividend = dividend[1:]
        while len(divisor) > 0 and divisor[0] == 0:
            divisor = divisor[1:]
        
        quotient = []
        
        while len(dividend) >= len(divisor) and len(divisor) > 0:
            quotient.append(1)
            # XOR操作
            for i in range(len(divisor)):
                dividend[i] = (dividend[i] + divisor[i]) % 2
            # 移除首位0
            while len(dividend) > 0 and dividend[0] == 0:
                dividend = dividend[1:]
        
        remainder = dividend if len(dividend) > 0 else np.array([0], dtype=int)
        quotient = np.array(quotient, dtype=int) if quotient else np.array([0], dtype=int)
        
        return quotient, remainder
    
    def encode(self, information_bits: np.ndarray) -> np.ndarray:
        """
        对信息比特进行BCH编码
        
        Args:
            information_bits: 信息比特数组 (长度应为k的倍数)
            
        Returns:
            coded_bits: 编码后的比特数组 (长度为n的倍数)
        """
        # 确保输入是1D数组
        information_bits = information_bits.astype(int).flatten()
        
        # 填充到k的倍数
        if len(information_bits) % self.k != 0:
            pad_length = self.k - (len(information_bits) % self.k)
            information_bits = np.concatenate([information_bits, 
                                             np.zeros(pad_length, dtype=int)])
        
        num_blocks = len(information_bits) // self.k
        coded_bits = []
        
        for block_idx in range(num_blocks):
            # 提取当前块的信息比特
            info_block = information_bits[block_idx * self.k:(block_idx + 1) * self.k]
            
            # 创建多项式：info_block * x^(n-k)
            shifted = np.concatenate([info_block, np.zeros(self.n - self.k, dtype=int)])
            
            # 计算余数（奇偶校验位）
            _, parity = self._gf_poly_divide(shifted, self.generator_poly)
            
            # 填充余数到正确长度
            parity = np.concatenate([np.zeros(self.n - self.k - len(parity), dtype=int), parity])
            
            # 码字 = 信息比特 || 奇偶校验比特
            codeword = np.concatenate([info_block, parity])
            coded_bits.extend(codeword)
        
        coded_bits = np.array(coded_bits, dtype=int)
        
        logger.info(f"BCH Encoding: {len(information_bits)} info bits -> "
                   f"{len(coded_bits)} coded bits")
        logger.info(f"Code rate: {len(information_bits) / len(coded_bits):.4f}")
        
        return coded_bits
    
    def _syndrome_decode(self, received: np.ndarray) -> np.ndarray:
        """
        计算伴随式用于纠错
        
        Args:
            received: 接收到的码字
            
        Returns:
            syndrome: 伴随式
        """
        _, syndrome = self._gf_poly_divide(received, self.generator_poly)
        # 填充到正确长度
        syndrome = np.concatenate([np.zeros(self.n - self.k - len(syndrome), dtype=int), 
                                  syndrome])
        return syndrome[:self.n - self.k]
    
    def _find_error_position(self, syndrome: np.ndarray) -> List[int]:
        """
        使用伴随式找到错误位置（简化版本）
        
        Args:
            syndrome: 伴随式
            
        Returns:
            error_positions: 错误比特的位置列表
        """
        # 对于简单的实现，如果伴随式全为0，则无错误
        if np.all(syndrome == 0):
            return []
        
        # 对于更复杂的纠错需要使用Peterson-Gorenstein-Zierler算法
        # 这里使用简化版本：尝试所有可能的单比特错误
        error_positions = []
        
        for pos in range(self.n):
            test = np.zeros(self.n, dtype=int)
            test[pos] = 1
            test_syndrome = self._syndrome_decode(test)
            if np.array_equal(test_syndrome, syndrome):
                error_positions.append(pos)
                break
        
        return error_positions
    
    def decode(self, coded_bits: np.ndarray) -> np.ndarray:
        """
        对BCH编码的比特进行解码和纠错
        
        Args:
            coded_bits: 编码后的比特数组
            
        Returns:
            decoded_bits: 解码后的信息比特
        """
        coded_bits = coded_bits.astype(int).flatten()
        
        # 确保长度是n的倍数
        if len(coded_bits) % self.n != 0:
            # 填充到n的倍数
            pad_length = self.n - (len(coded_bits) % self.n)
            coded_bits = np.concatenate([coded_bits, np.zeros(pad_length, dtype=int)])
        
        num_blocks = len(coded_bits) // self.n
        decoded_bits = []
        errors_corrected = 0
        
        for block_idx in range(num_blocks):
            # 提取当前块
            received = coded_bits[block_idx * self.n:(block_idx + 1) * self.n]
            
            # 计算伴随式
            syndrome = self._syndrome_decode(received)
            
            # 检测错误
            error_positions = self._find_error_position(syndrome)
            
            # 纠正错误
            if error_positions:
                for pos in error_positions:
                    received[pos] = (received[pos] + 1) % 2
                    errors_corrected += 1
            
            # 提取信息比特
            info_bits = received[:self.k]
            decoded_bits.extend(info_bits)
        
        decoded_bits = np.array(decoded_bits, dtype=int)
        
        logger.info(f"BCH Decoding: {len(coded_bits)} coded bits -> "
                   f"{len(decoded_bits)} info bits")
        logger.info(f"Errors corrected: {errors_corrected}")
        
        return decoded_bits
    
    def get_parity_check_matrix(self) -> np.ndarray:
        """获取奇偶校验矩阵（用于参考）"""
        # 对于(15,7) BCH码
        if self.n == 15 and self.k == 7:
            # H矩阵是 8x15 的
            H = np.array([
                [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            ], dtype=int)
            return H
        else:
            # 返回简化版本
            return np.eye(self.n - self.k, self.n, dtype=int)


if __name__ == "__main__":
    # 测试BCH编码
    coder = BCHCoder(n=15, k=7)
    
    # 创建测试信息比特
    test_info = np.array([1, 0, 1, 1, 0, 1, 0] * 10, dtype=int)
    
    # 编码
    encoded = coder.encode(test_info)
    print(f"Information bits: {len(test_info)}")
    print(f"Encoded bits: {len(encoded)}")
    print(f"First codeword: {encoded[:15]}")
    
    # 添加一个错误
    encoded_with_error = encoded.copy()
    encoded_with_error[5] = (encoded_with_error[5] + 1) % 2
    
    # 解码
    decoded = coder.decode(encoded_with_error)
    print(f"\nDecoded bits: {len(decoded)}")
    print(f"Reconstruction errors: {np.sum(decoded != test_info)}")
