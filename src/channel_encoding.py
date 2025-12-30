"""
远程声音控制系统 - 信道编码模块（Reed-Solomon编码）
Channel Encoding Module for Remote Voice Control System
"""

import numpy as np
from typing import Tuple, List
import logging
try:
    from reedsolo import RSCodec, ReedSolomonError
except ImportError:
    print("Error: reedsolo module not found. Please install it using 'pip install reedsolo'")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RSCoder:
    """Reed-Solomon 编码器/解码器"""
    
    def __init__(self, n: int = 255, k: int = 223):
        """
        初始化 RS 编码器
        
        Args:
            n: 码字长度 (字节)
            k: 信息长度 (字节)
        """
        self.n = n
        self.k = k
        self.nsym = n - k # 纠错字节数
        self.rsc = RSCodec(self.nsym)
        
        logger.info(f"RS({n},{k}) coder initialized")
        logger.info(f"Error correction capability: {self.nsym // 2} bytes")

    def _bits_to_bytes(self, bits: np.ndarray) -> bytearray:
        """将比特数组转换为字节数组"""
        bits = bits.astype(int).flatten()
        # 补齐到8的倍数
        if len(bits) % 8 != 0:
            pad_len = 8 - (len(bits) % 8)
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])
        
        # 转换为字节
        byte_arr = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | bits[i+j]
            byte_arr.append(byte)
        return byte_arr

    def _bytes_to_bits(self, byte_arr: bytearray) -> np.ndarray:
        """将字节数组转换为比特数组"""
        bits = []
        for byte in byte_arr:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return np.array(bits, dtype=int)

    def encode(self, information_bits: np.ndarray) -> np.ndarray:
        """
        对信息比特进行RS编码
        
        Args:
            information_bits: 信息比特数组
            
        Returns:
            coded_bits: 编码后的比特数组
        """
        # 转换为字节
        input_bytes = self._bits_to_bytes(information_bits)
        
        encoded_bytes = bytearray()
        # 分块处理
        chunk_size = self.k
        
        # 计算需要多少块
        num_chunks = (len(input_bytes) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(input_bytes))
            chunk = input_bytes[start:end]
            
            # 填充到 k 字节
            if len(chunk) < chunk_size:
                padding = chunk_size - len(chunk)
                chunk_padded = bytearray(chunk)
                chunk_padded.extend(b'\x00' * padding)
                chunk = chunk_padded
            
            # 编码
            encoded_chunk = self.rsc.encode(chunk)
            encoded_bytes.extend(encoded_chunk)
            
        coded_bits = self._bytes_to_bits(encoded_bytes)
        
        logger.info(f"RS Encoding: {len(information_bits)} bits -> {len(coded_bits)} bits")
        return coded_bits

    def decode(self, coded_bits: np.ndarray) -> np.ndarray:
        """
        对RS编码的比特进行解码
        
        Args:
            coded_bits: 编码后的比特数组
            
        Returns:
            decoded_bits: 解码后的信息比特
        """
        input_bytes = self._bits_to_bytes(coded_bits)
        
        decoded_bytes = bytearray()
        chunk_size = self.n # 编码后的块大小 (k + nsym)
        
        errors_corrected = 0
        
        for i in range(0, len(input_bytes), chunk_size):
            chunk = input_bytes[i : i + chunk_size]
            
            if len(chunk) < chunk_size:
                logger.warning(f"Incomplete block received in RS decode: {len(chunk)} bytes")
                continue
                
            try:
                # decode returns (decoded_msg, decoded_msg_with_ecc, errata_pos)
                decoded_chunk, _, errata_pos = self.rsc.decode(chunk)
                decoded_bytes.extend(decoded_chunk)
                if errata_pos:
                    errors_corrected += len(errata_pos)
            except ReedSolomonError:
                logger.error("RS Decoding failed: too many errors")
                # 尽力而为：直接取数据部分（前k个字节）
                raw_data = chunk[:self.k]
                decoded_bytes.extend(raw_data)
        
        decoded_bits = self._bytes_to_bits(decoded_bytes)
        
        logger.info(f"RS Decoding: {len(coded_bits)} bits -> {len(decoded_bits)} bits")
        logger.info(f"Errors corrected (approx symbols): {errors_corrected}")
        
        return decoded_bits

if __name__ == "__main__":
    # 测试 RS 编码
    coder = RSCoder(n=255, k=223)
    
    # 创建测试信息比特
    test_info = np.random.randint(0, 2, 1784)
    
    # 编码
    encoded = coder.encode(test_info)
    print(f"Information bits: {len(test_info)}")
    print(f"Encoded bits: {len(encoded)}")
    
    # 添加错误
    encoded_with_error = encoded.copy()
    for i in range(10):
        bit_idx = i * 8
        encoded_with_error[bit_idx] = 1 - encoded_with_error[bit_idx]
    
    # 解码
    decoded = coder.decode(encoded_with_error)
    print(f"\nDecoded bits: {len(decoded)}")
    
    # 比较
    decoded_trimmed = decoded[:len(test_info)]
    errors = np.sum(decoded_trimmed != test_info)
    print(f"Reconstruction errors: {errors}")
