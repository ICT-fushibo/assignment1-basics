from typing import Iterable,Iterator
import os
import json
import regex as re

class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]|None=None):
        """Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        
        Args:
            vocab (dict[int,bytes]): vocabulary
            merges (list[tuple[bytes,bytes]]): list of merges
            special_tokens (list[str] | None, optional): list of special tokens. Defaults to None.
        """
        
        self.vocab=vocab
        self.bytes2int={v:k for k,v in vocab.items()}
        self.merges=merges
        self.special_tokens=special_tokens
        # sorted_special_tokens=sorted(special_tokens,key=len,reverse=True)
        self.special_tokens_re="|".join(re.escape(t) for t in sorted(special_tokens,key=len,reverse=True)) if special_tokens else None
        self.PAT=r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        
        
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
        # handel special token
        if self.special_tokens:
            chunks=re.split(f"({self.special_tokens_re})",text)
        else:
            chunks=[text]
        
        # pre-tokenize
        pre_token_list=[]
        
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                pre_token_list.append(chunk.encode("utf-8"))
                continue
            for m in re.finditer(self.PAT,chunk):
                token=m.group(0)
                b_token=token.encode("utf-8")
                token_list=tuple(bytes([x])for x in b_token)
                
                    
                pre_token_list.append(token_list)
        
        # apply merges
        for merge in self.merges:
            A,B=merge
            AB=A+B
            i=0
            new_pretoken_list=[]
            
            for pre_token in pre_token_list:
                if A not in pre_token or B not in pre_token or (isinstance(pre_token,bytes) and self.special_tokens and pre_token.decode() in self.special_tokens):
                    new_pretoken_list.append(pre_token)
                    continue
                lst = list(pre_token)
                i=0
                new_token_list=[]
                while i <len(lst):
                    if i+1<len(lst) and lst[i]==A and lst[i+1]==B:
                        new_token_list.append(AB)
                        i+=2
                    else:
                        new_token_list.append(lst[i])
                        i+=1
                new_token_tup=tuple(new_token_list)
                new_pretoken_list.append(new_token_tup)
            
            pre_token_list=new_pretoken_list
        
        # byte 2 int
        encode_int=[]
        for pre_token in pre_token_list:
            if isinstance(pre_token,bytes) and self.special_tokens and pre_token.decode() in self.special_tokens:
                encode_int.append(self.bytes2int[pre_token])
                continue
            for token_bytes in pre_token:
                encode_int.append(self.bytes2int[token_bytes])
        
        return encode_int        
                
                    
                
        
        
    
    def encode_iterable(self,iterable:Iterable[str] ) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.

        Args:
            iterable (Iterable[str]): _description_

        Yields:
            Iterator[int]: _description_
        """
        
        for text_chunk in iterable:
            token_ids= self.encode(text_chunk)
            
            yield from token_ids
        
    
    def decode(self,ids:list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids (list[int]): encode ids to be decode

        Returns:
            str: decode text
        """
        byets_str=b""
        for id in ids:
            byets_str+=self.vocab[id]
        return byets_str.decode("utf-8",errors="replace")
    
if __name__=='__main__':
    
    FIXTURES_PATH="/home/fu/assignment1-basics/tests/fixtures/"
    
    VOCAB_PATH = FIXTURES_PATH + "gpt2_vocab.json"
    MERGES_PATH = FIXTURES_PATH + "gpt2_merges.txt"
    
    def get_tokenizer_from_vocab_merges_path(
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        from tests.common import gpt2_bytes_to_unicode
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return Tokenizer(vocab, merges, special_tokens)
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    print(encoded_ids)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    print(tokenized_string)
    decoded_string = tokenizer.decode(encoded_ids)
    print(decoded_string)
    assert test_string == decoded_string
    