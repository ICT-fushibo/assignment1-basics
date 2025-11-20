from collections import defaultdict,Counter
import regex as re

def run_train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]):
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
            b = chunk.encode("utf-8")
            tup = (b,)
            pretoken_counts[tup] += 1
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
        AB=A+B
        merges.append((A,B))
        
        vocab[next_id]=AB
        bytes2id[AB]=next_id
        next_id+=1
        
        new_pretoken_counts=defaultdict(int)
        for tup,freq in pretoken_counts.items():
            lst=list(tup)
            i = 0
            new_list=[]
            while i<len(lst):
                if i+1<len(lst) and lst[i]==A and lst[i+1]==B:
                    new_list.append(AB)
                    i+=2
                else:
                    new_list.append(lst[i])
                    i+=1
            new_tup=tuple(new_list)
            new_pretoken_counts[new_tup]+=freq
        pretoken_counts=new_pretoken_counts

        pair_counts=comput_pair_counts(pretoken_counts)

    return vocab,merges
        
                
if __name__=="__main__":
    input_path=r"/home/fu/assignment1-basics/data/simple_test.txt"
    vocab_size=500
    special_tokens=["<|endoftext|>"]
    # run_train_bpe(input_path,vocab_size,special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for m in re.finditer(PAT,"123 \n\n12\n 1233 \n\n"):
        print(repr(m.group(0)))
    
    
    
    
