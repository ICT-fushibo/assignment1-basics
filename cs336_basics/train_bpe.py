import os
import regex as re
import time
from typing import BinaryIO, List, Tuple, Dict
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count

def run_train_bpe_back(input_path:str,vocab_size:int,special_tokens:list[str]):
    '''Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    
    
    '''
    
    special_tokens_re="|".join(re.escape(t) for t in special_tokens) if special_tokens else None
    
    PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    pretoken_counts=defaultdict(int)
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        chunks = re.split(f"({special_tokens_re})", text)
    else:
        chunks = [text]

    for chunk in chunks:
        if chunk in special_tokens:
            # b = chunk.encode("utf-8")
            # tup = (b,)
            # pretoken_counts[tup] += 1
            continue
        for m in re.finditer(PAT, chunk):
            tok = m.group(0)
            b = tok.encode("utf-8")
            tup = tuple(bytes([x]) for x in b)
            pretoken_counts[tup] += 1
    
    
    
    vocab={}
    bytes2id={}
    for i in range(256):
        b= bytes([i])
        vocab[i]=b
        bytes2id[b]=i
    
    next_id=256
    
    for st in special_tokens or []:
        b = st.encode("utf-8")
        vocab[next_id]=b
        bytes2id[b]=next_id
        next_id+=1
    
    merges=[]
    
    def comput_pair_counts(pretoken_counts: defaultdict):
        pc = Counter()
        for tup, freq in pretoken_counts.items():
            for i in range(len(tup) - 1):
                pc[(tup[i], tup[i+1])] += freq
        return pc
    
    pair_counts=comput_pair_counts(pretoken_counts)
    
    while next_id < vocab_size and pair_counts:
        max_freq=max(pair_counts.values())
        
        candidates = [p for p,c in pair_counts.items() if c==max_freq]
        
        pair_to_merge= max(candidates)
        A,B=pair_to_merge
        # A=vocab[A]
        # B=vocab[B]
        AB=A+B
        merges.append((A,B))
        
        vocab[next_id]=AB
        bytes2id[AB]=next_id
        next_id+=1
        
        
        # new_pretoken_counts=defaultdict(int)
        # for tup,freq in pretoken_counts.items():
        #     lst=list(tup)
        #     i = 0
        #     new_list=[]
        #     while i<len(lst):
        #         if i+1<len(lst) and lst[i]==A and lst[i+1]==B:
        #             new_list.append(AB)
        #             i+=2
        #         else:
        #             new_list.append(lst[i])
        #             i+=1
        #     new_tup=tuple(new_list)
        #     new_pretoken_counts[new_tup]+=freq
        # pretoken_counts=new_pretoken_counts

        # pair_counts=comput_pair_counts(pretoken_counts)
        
        new_pretoken_counts = defaultdict(int)
        del pair_counts[(A,B)]
        for tup, freq in pretoken_counts.items():
            # 快速检查：如果该 tuple 里根本没有 A，那肯定也无法合并出 AB
            # 这是一个简单的剪枝优化
            if A not in tup or B not in tup:
                new_pretoken_counts[tup] += freq
                continue
                
            lst = list(tup)
            i = 0
            new_list = []
            while i < len(lst):
                if i + 1 < len(lst) and lst[i] == A and lst[i+1] == B:
                    new_list.append(AB)
                    
                    # 更新pair计数
                    # 1.更新A之前和A的
                    if i>0:
                        pair_del=(lst[i-1],lst[i])
                        pair_counts[pair_del]-=freq
                        if pair_counts[pair_del]==0:del pair_counts[pair_del]
                        pair_add=(lst[i-1],AB)
                        pair_counts[pair_add]+=freq
                    # 2.更新B和B之后的
                    if i+2<len(lst):
                        pair_del=(lst[i+1],lst[i+2])
                        pair_counts[pair_del]-=freq
                        if pair_counts[pair_del]==0:del pair_counts[pair_del]
                        pair_add=(AB,lst[i+2])
                        pair_counts[pair_add]+=freq
                    i += 2
                else:
                    new_list.append(lst[i])
                    i += 1
            new_tup = tuple(new_list)
            new_pretoken_counts[new_tup] += freq
        
        pretoken_counts = new_pretoken_counts

    return vocab,merges

from cs336_basics import bpe_ops

def run_train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]):
    file_size = os.path.getsize(input_path)
    num_threads = 4 
    raw_counts = bpe_ops.count_tokens_parallel(str(input_path), num_threads, special_tokens)
    
    pretoken_counts = defaultdict(int)
    for token_bytes, freq in raw_counts.items():
        tup = tuple(token_bytes) 
        pretoken_counts[tup] += freq
    
    vocab={}
    bytes2id={}
    for i in range(256):
        b= bytes([i])
        vocab[i]=b
        bytes2id[b]=i
    
    next_id=256
    
    for st in special_tokens or []:
        b = st.encode("utf-8")
        vocab[next_id]=b
        bytes2id[b]=next_id
        next_id+=1
    
    merges=[]
    
    def comput_pair_counts(pretoken_counts: defaultdict):
        pc = Counter()
        for tup, freq in pretoken_counts.items():
            for i in range(len(tup) - 1):
                pc[(tup[i], tup[i+1])] += freq
        return pc
    
    pair_counts=comput_pair_counts(pretoken_counts)
    
    while next_id < vocab_size and pair_counts:
        max_freq=max(pair_counts.values())
        
        candidates = [p for p,c in pair_counts.items() if c==max_freq]
        
        # =============== 核心修正点 ===============
        # 原始代码比较的是 bytes 的字典序，而不是 ID 的大小。
        # 由于新产生的 Token ID (256+) 通常大于原始字节 ID (0-255)，
        # 直接比较 ID 会导致算法倾向于优先合并新产生的 Token，导致顺序不一致。
        # 修正：使用 key 参数，通过 vocab 将 ID 转回 bytes 进行比较。
        pair_to_merge = max(candidates, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        # ========================================

        A,B=pair_to_merge
        
        # 1. 从 vocab 中获取对应的 bytes
        b_A = vocab[A]
        b_B = vocab[B]
        
        # 2. merges 列表存 (bytes, bytes)
        merges.append((b_A, b_B))
        
        # 3. 更新 vocab
        merged_bytes = b_A + b_B
        vocab[next_id] = merged_bytes
        bytes2id[merged_bytes] = next_id
        
        # 4. 定义新的 ID
        AB = next_id
        next_id+=1
        
        new_pretoken_counts = defaultdict(int)
        # 移除当前 pair，防止干扰
        if (A, B) in pair_counts:
            del pair_counts[(A,B)]
            
        for tup, freq in pretoken_counts.items():
            if A not in tup or B not in tup:
                new_pretoken_counts[tup] += freq
                continue
                
            lst = list(tup)
            i = 0
            new_list = []
            while i < len(lst):
                if i + 1 < len(lst) and lst[i] == A and lst[i+1] == B:
                    new_list.append(AB)
                    
                    # 增量更新 pair_counts
                    if i>0:
                        pair_del=(lst[i-1],lst[i])
                        pair_counts[pair_del]-=freq
                        if pair_counts[pair_del]<=0: del pair_counts[pair_del]
                        
                        pair_add=(lst[i-1],AB)
                        pair_counts[pair_add]+=freq
                    
                    if i+2<len(lst):
                        pair_del=(lst[i+1],lst[i+2])
                        pair_counts[pair_del]-=freq
                        if pair_counts[pair_del]<=0: del pair_counts[pair_del]
                        
                        pair_add=(AB,lst[i+2])
                        pair_counts[pair_add]+=freq
                    i += 2
                else:
                    new_list.append(lst[i])
                    i += 1
            new_tup = tuple(new_list)
            new_pretoken_counts[new_tup] += freq
        
        pretoken_counts = new_pretoken_counts

    return vocab,merges



if __name__=="__main__":
    import time
    import cProfile
    input_path=r"/home/fu/assignment1-basics/data/owt_valid.txt"
    vocab_size=50000
    special_tokens=["<|endoftext|>"]
    t1=time.time()
    run_train_bpe(input_path,vocab_size,special_tokens)
    t2=time.time()
    print(t2-t1)
    t1=time.time()
    run_train_bpe_back(input_path,vocab_size,special_tokens)
    t2=time.time()
    print(t2-t1)
    # cProfile.run('run_train_bpe(input_path,vocab_size,special_tokens)')
    
    
    
    
    
