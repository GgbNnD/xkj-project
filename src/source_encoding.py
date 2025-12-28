"""
远程声音控制系统 - 信源编码模块（Huffman编码）
Source Encoding Module for Remote Voice Control System
"""

import numpy as np
from collections import defaultdict, Counter
from typing import Dict, Tuple, List
import heapq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuffmanNode:
    """Huffman树节点"""
    
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


class SourceEncoding:
    """信源编码 - Huffman编码"""
    
    def __init__(self):
        self.dictionary = {}
        self.inverse_dict = {}
        self.huffman_tree = None
        
    def build_huffman_tree(self, frequencies: Dict[int, float]) -> HuffmanNode:
        """
        构建Huffman树
        
        Args:
            frequencies: 符号频率字典
            
        Returns:
            huffman_tree: Huffman树根节点
        """
        if not frequencies:
            return None
        
        # 创建优先级队列（最小堆）
        heap = [HuffmanNode(symbol=symbol, freq=freq) 
                for symbol, freq in frequencies.items()]
        heapq.heapify(heap)
        
        # 构建Huffman树
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            parent = HuffmanNode(freq=left.freq + right.freq, 
                                left=left, right=right)
            heapq.heappush(heap, parent)
        
        self.huffman_tree = heap[0]
        return self.huffman_tree
    
    def generate_codes(self, node: HuffmanNode, code: str = '', 
                      codes: Dict[int, str] = None) -> Dict[int, str]:
        """
        从Huffman树生成编码字典
        
        Args:
            node: 当前节点
            code: 当前编码
            codes: 编码字典
            
        Returns:
            codes: 符号到编码的映射
        """
        if codes is None:
            codes = {}
        
        if node is None:
            return codes
        
        # 叶节点 - 保存编码
        if node.symbol is not None:
            codes[node.symbol] = code if code else '0'  # 处理只有一个符号的情况
            return codes
        
        # 递归处理左右子树
        self.generate_codes(node.left, code + '0', codes)
        self.generate_codes(node.right, code + '1', codes)
        
        return codes
    
    def encode(self, signal: np.ndarray) -> Tuple[str, Dict[int, str], Dict]:
        """
        对信号进行Huffman编码
        
        Args:
            signal: 量化后的信号（整数值）
            
        Returns:
            encoded_bits: 编码后的比特串
            code_dict: 编码字典
            statistics: 编码统计信息
        """
        # 统计符号频率
        signal_int = signal.astype(int)
        symbol_counts = Counter(signal_int)
        total_symbols = len(signal_int)
        
        # 计算频率
        frequencies = {symbol: count / total_symbols 
                      for symbol, count in symbol_counts.items()}
        
        # 构建Huffman树并生成编码
        self.build_huffman_tree(frequencies)
        self.dictionary = self.generate_codes(self.huffman_tree)
        
        # 创建反向字典
        self.inverse_dict = {v: k for k, v in self.dictionary.items()}
        
        # 编码信号
        encoded_bits = ''.join([self.dictionary[int(sym)] for sym in signal_int])
        
        # 计算统计信息
        original_bits = len(signal_int) * np.ceil(np.log2(len(frequencies))).astype(int)
        encoded_bits_length = len(encoded_bits)
        compression_ratio = original_bits / encoded_bits_length if encoded_bits_length > 0 else 1
        
        # 计算熵
        entropy = -sum(freq * np.log2(freq) for freq in frequencies.values() 
                      if freq > 0)
        
        # 计算平均码长
        avg_code_length = sum(len(code) * frequencies[symbol] 
                             for symbol, code in self.dictionary.items())
        
        # 计算编码效率
        coding_efficiency = entropy / avg_code_length if avg_code_length > 0 else 0
        
        statistics = {
            'original_bits': original_bits,
            'encoded_bits': encoded_bits_length,
            'compression_ratio': compression_ratio,
            'entropy': entropy,
            'avg_code_length': avg_code_length,
            'coding_efficiency': coding_efficiency,
            'num_symbols': len(frequencies),
            'symbol_frequencies': frequencies
        }
        
        logger.info(f"Huffman Encoding Statistics:")
        logger.info(f"  Original bits: {original_bits}")
        logger.info(f"  Encoded bits: {encoded_bits_length}")
        logger.info(f"  Compression ratio: {compression_ratio:.4f}")
        logger.info(f"  Entropy: {entropy:.6f} bits/symbol")
        logger.info(f"  Average code length: {avg_code_length:.6f} bits/symbol")
        logger.info(f"  Coding efficiency: {coding_efficiency:.6f}")
        logger.info(f"  Unique symbols: {len(frequencies)}")
        
        return encoded_bits, self.dictionary, statistics
    
    def decode(self, encoded_bits: str, code_dict: Dict[str, int] = None) -> np.ndarray:
        """
        对Huffman编码进行解码
        
        Args:
            encoded_bits: 编码后的比特串
            code_dict: 编码字典（如果为None，使用之前保存的）
            
        Returns:
            decoded_signal: 解码后的信号
        """
        if code_dict is None:
            code_dict = self.inverse_dict
        if not code_dict:
            raise ValueError("No code dictionary available")
        
        decoded_signal = []
        current_code = ''
        
        for bit in encoded_bits:
            current_code += bit
            if current_code in code_dict:
                decoded_signal.append(code_dict[current_code])
                current_code = ''
        
        # 处理剩余代码（应该为空）
        if current_code:
            logger.warning(f"Incomplete code at end of bit stream: {current_code}")
        
        logger.info(f"Decoded {len(decoded_signal)} symbols from {len(encoded_bits)} bits")
        
        return np.array(decoded_signal, dtype=int)
    
    def get_code_table(self) -> str:
        """获取编码表的字符串表示"""
        if not self.dictionary:
            return "No encoding table available"
        
        table = "Symbol\tCode\n"
        for symbol in sorted(self.dictionary.keys()):
            table += f"{symbol}\t{self.dictionary[symbol]}\n"
        return table


# 辅助函数
def bits_to_bytes(bit_string: str) -> bytes:
    """将比特串转换为字节序列"""
    # 补齐到8的倍数
    while len(bit_string) % 8 != 0:
        bit_string += '0'
    
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte = int(bit_string[i:i+8], 2)
        byte_array.append(byte)
    
    return bytes(byte_array)


def bytes_to_bits(byte_data: bytes) -> str:
    """将字节序列转换为比特串"""
    bit_string = ''
    for byte in byte_data:
        bit_string += format(byte, '08b')
    return bit_string


if __name__ == "__main__":
    # 测试Huffman编码
    encoder = SourceEncoding()
    
    # 创建测试信号
    test_signal = np.array([0, 1, 2, 1, 0, 3, 1, 0, 2, 1] * 10, dtype=int)
    
    # 编码
    encoded, dict_table, stats = encoder.encode(test_signal)
    print(f"Encoded length: {len(encoded)}")
    print(f"Compression ratio: {stats['compression_ratio']:.4f}")
    print(f"\nCode table:\n{encoder.get_code_table()}")
    
    # 解码
    decoded = encoder.decode(encoded)
    print(f"\nDecoded length: {len(decoded)}")
    print(f"Reconstruction error: {np.sum(decoded != test_signal)} symbols")
