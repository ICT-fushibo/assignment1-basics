from typing import Iterable,Iterator
import os
import json

class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]|None=None):
        """Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        
        Args:
            vocab (dict[int,bytes]): vocabulary
            merges (list[tuple[bytes,bytes]]): list of merges
            special_tokens (list[str] | None, optional): list of special tokens. Defaults to None.
        """
        
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens
        
        
    @classmethod
    def from_files(self,vocab_filepath:str,merges_filepath:str,special_tokens:list[str]|None=None):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
            and (optionally) a list of special tokens.

        Args:
            vocab_filepath (str): vocabulary filepath
            merges_filepath (str): list of merges filepath
            special_tokens (list[str] | None, optional): list of special tokens. Defaults to None.
        """
        if not os.path.exists(vocab_filepath) or not os.path.exists(merges_filepath):
            raise FileNotFoundError(f"Missing vocab.json or merges.txt in {vocab_filepath} and {merges_filepath}")

        # -----------------------------------------
        # 1. 解析 Vocab
        # -----------------------------------------
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            # JSON 加载进来是 { "token_str": token_id }
            vocab_str = json.load(f)
        
        # 我们需要将其转换为 { token_id: token_bytes }
        vocab = {}
        for token_str, token_id in vocab_str.items():
            # 尝试还原 bytes。
            # 注意：如果在保存时遇到无法用 utf-8 表示的字节使用了 latin-1，
            # 这里理论上需要知道原本的编码。
            # 对于标准的 BPE 应用，通常假设是 utf-8。
            try:
                token_bytes = token_str.encode("utf-8")
            except UnicodeEncodeError:
                # 兼容 fallback 情况
                token_bytes = token_str.encode("latin-1")
                
            vocab[token_id] = token_bytes

        # -----------------------------------------
        # 2. 解析 Merges
        # -----------------------------------------
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                # 跳过注释行 (通常第一行是 #version)
                if line.startswith("#"):
                    continue
                
                # 去除首尾换行符
                line = line.strip("\n")
                if not line:
                    continue
                    
                # 按照空格分割
                # 注意：标准的 merges.txt 格式假设 token 内部不包含用来分割的空格
                # 如果你的 token 本身就是空格，这种简单的 split 可能会有问题
                # 但对于大多数 BPE 实现（如 GPT-2），空格通常被映射为特殊字符（如 Ġ）
                parts = line.split(" ")
                
                # 简单的分割逻辑：假设每行只有两个 token，中间由一个空格分隔
                if len(parts) == 2:
                    s1, s2 = parts[0], parts[1]
                elif len(parts) > 2:
                    # 处理可能存在的空格 token 情况 (例如 token是 " "，导致分割出空字符串)
                    # 这种情况下通常需要根据具体的保存逻辑来写特定的解析
                    # 这里给出一个通用的容错逻辑：
                    # 假设空格只出现在 token 内容中，且不作为分隔符 (这在简单 TXT 格式中很难完美区分)
                    # 作为一个简单的 workaround，我们取第一个和最后一个非空部分，或者仅支持无空格 token
                    s1, s2 = parts[0], parts[-1] 
                else:
                    # 格式错误或空行
                    continue

                # 同样将字符串转回 bytes
                try:
                    b1 = s1.encode("utf-8")
                    b2 = s2.encode("utf-8")
                except UnicodeEncodeError:
                    b1 = s1.encode("latin-1")
                    b2 = s2.encode("latin-1")
                    
                merges.append((b1, b2))

        return Tokenizer(vocab,merges,special_tokens)
    
    def encode(self,text:str) -> list[int]:
        """Encode an input text into a sequence of token IDs.

        Args:
            text (str): text to be encode

        Returns:
            list[int]: encode text
        """
        pass
    
    def encode_iterable(self,iterable:Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.

        Args:
            iterable (Iterable[str]): _description_

        Yields:
            Iterator[int]: _description_
        """
        pass
    
    def decode(self,ids:list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids (list[int]): encode ids to be decode

        Returns:
            str: decode text
        """
        pass