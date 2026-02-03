import numpy as np
from bitarray import bitarray


class RepeatECC():
    def __init__(self, num_bits:int, redundancy:int):
        self.num_bits = num_bits 
        self.redundancy = redundancy

    def encode(self, msg:np.ndarray):
        """
        Takes a binary message of shape (bsz, k) and encodes it into a binary message of shape (bsz, k*redundancy).
        """
        msg = np.tile(msg, (1, self.redundancy))  # b nbits -> b nbits*redundancy
        return msg

    def decode(self, msg:np.ndarray):
        """
        Takes a message of shape (bsz, k) and decodes it into a message of shape (bsz, k//redundancy).
        Note that the message may or may not be binary at this stage.
        """
        msg = msg.reshape(-1, self.redundancy, self.num_bits)  # b nbits*redundancy -> b nbits redundancy
        msg = msg.mean(axis=1)  # b nbits
        return msg


class HammingECC(RepeatECC):
    """Hamming(15, 11)"""

    CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ,1234" + "`"  # 5-bit 编码，共支持 32 个字符。 ` 作为终止符
    def __init__(self):
        self.encode_bits_per_char = 5
        self.total_bits = 256
        self.redundancy = 2
        self.num_bits = self.total_bits // self.redundancy
        super().__init__(num_bits=self.num_bits, redundancy=self.redundancy)

    def __hamming_encode_block(self, block):
        assert len(block) == 11
        c = bitarray(15)
        c.setall(0)
        data_idx = [2,4,5,6,8,9,10,11,12,13,14]
        for i, b in enumerate(block):
            c[data_idx[i]] = b
        # 校验位
        c[0] = c[2]^c[4]^c[6]^c[8]^c[10]^c[12]^c[14]
        c[1] = c[2]^c[5]^c[6]^c[9]^c[10]^c[13]^c[14]
        c[3] = c[4]^c[5]^c[6]^c[11]^c[12]^c[13]^c[14]
        c[7] = c[8]^c[9]^c[10]^c[11]^c[12]^c[13]^c[14]
        return c
    
    def __hamming_decode_block(self, c):
        assert len(c) == 15
        s0 = c[0]^c[2]^c[4]^c[6]^c[8]^c[10]^c[12]^c[14]
        s1 = c[1]^c[2]^c[5]^c[6]^c[9]^c[10]^c[13]^c[14]
        s2 = c[3]^c[4]^c[5]^c[6]^c[11]^c[12]^c[13]^c[14]
        s3 = c[7]^c[8]^c[9]^c[10]^c[11]^c[12]^c[13]^c[14]
        syndrome = (s3<<3)|(s2<<2)|(s1<<1)|s0
        if syndrome != 0:
            print(f"Error detected at position {syndrome}")
        if syndrome != 0 and 1 <= syndrome <= 15:
            c[syndrome-1] = not c[syndrome-1]
        data_idx = [2,4,5,6,8,9,10,11,12,13,14]
        return bitarray([c[i] for i in data_idx])
    
    def __hamming_encode(self, bits):
        encoded = bitarray()
        for i in range(0, len(bits), 11):
            block = bits[i:i+11]
            if len(block)<11:
                block.extend('0'*(11-len(block)))
            encoded.extend(self.__hamming_encode_block(block))
        return encoded

    def __hamming_decode(self, bits):
        decoded = bitarray()
        for i in range(0, len(bits), 15):
            block = bits[i:i+15]
            if len(block)<15:
                block.extend('0'*(15-len(block)))
            decode = self.__hamming_decode_block(block)
            decoded.extend(decode)
        return decoded
    
    def __string_to_nbit_bits(self, s):
        """将字符串用 n-bit 编码."""
        # length prefix + terminator
        s = self.CHARSET[len(s)] + s + "`"
        bits = bitarray()
        for c in s.upper():
            idx = self.CHARSET.find(c)
            if idx == -1:
                raise ValueError(f"Character '{c}' not in '{self.CHARSET}'")
            bits.extend(format(idx,'0{0}b'.format(self.encode_bits_per_char)))
        return bits

    def __bits_to_string(self, bits):
        chars = []
        s_length = 0
        for i in range(0, len(bits), self.encode_bits_per_char):
            chunk = bits[i:i+self.encode_bits_per_char]
            if len(chunk) < self.encode_bits_per_char:
                break
            idx = int(chunk.to01(), 2)
            if idx >= len(self.CHARSET):
                continue
            if i == 0:
                s_length = idx
                continue
            if idx == len(self.CHARSET)-1:  # terminator
                break
            chars.append(self.CHARSET[idx])
        return ''.join(chars).rstrip('\x00'), s_length + 2
    
    def str_to_tensor(self, input_string):
        bits = self.__string_to_nbit_bits(input_string) 
        # Hamming(15,11): 每11 bits -> 15 bits
        n_blocks = (len(bits) + 10) // 11
        encoded_len = n_blocks * 15
        if encoded_len > self.num_bits:
            raise ValueError(f"The input string is too long. The encoded result requires {encoded_len} bits, which exceeds the {self.num_bits} limit.")
        
        tensor_ori_msgs = np.array([[int(b) for b in bits]], dtype=np.uint8)
        # 填充 bits 到 n_blocks*11 bits
        total_data_bits = n_blocks*11
        if len(bits) < total_data_bits:
            bits.extend('0'*(total_data_bits - len(bits)))
        
        encoded_bits = self.__hamming_encode(bits)
        
        if len(encoded_bits) < self.num_bits:
            encoded_bits.extend('0' * (self.num_bits - len(encoded_bits)))
        
        tensor = np.array([[int(b) for b in encoded_bits]], dtype=np.uint8)
        tensor = self.encode(tensor)
        return tensor, tensor_ori_msgs

    def tensor_to_string(self, tensor: np.ndarray):
        tensor = (self.decode(tensor) > 0).astype(np.uint8)
        bits = bitarray([bool(b) for b in tensor.flatten().tolist()])
        decoded_bits = self.__hamming_decode(bits)
        msg_str, msg_str_len = self.__bits_to_string(decoded_bits)
        return msg_str, np.array([[int(b) for b in decoded_bits[:msg_str_len*self.encode_bits_per_char]]], dtype=np.uint8)