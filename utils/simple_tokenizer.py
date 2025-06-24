import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

##这是一个装饰器，它将下面的函数进行装饰。@lru_cache()表示使用最近最少使用（LRU）缓存机制来缓存函数的结果，以提高函数的执行效率。LRU缓存会保存最近调用的函数结果，当相同的参数再次传递给函数时，会直接返回缓存中的结果，而不是重新执行函数。
#假设你有一个函数，它根据输入的参数计算结果，而且这个计算可能会比较耗时。如果你多次用同样的参数调用这个函数，那么每次调用都会重新计算结果，这可能会浪费时间。
#LRU缓存就像一个记事本，它记录了最近调用的函数结果。当你再次用相同的参数调用函数时，它会先检查记事本。如果之前已经计算过这个参数对应的结果，它会直接从记事本中拿出来，而不是重新计算。这样可以节省时间，因为不用再执行耗时的计算步骤了。
@lru_cache()
def default_bpe():#这个函数的作用是返回一个BPE词汇表文件的绝对路径，并且使用LRU缓存机制来缓存这个结果，以提高效率。
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/bpe_simple_vocab_16e6.txt.gz")#这一行返回了一个文件路径，该路径指向一个BPE（字节对编码）的词汇表文件
    """
    os.path.abspath("./data/bpe_simple_vocab_16e6.txt.gz") ==> /home/202312150002/my_paper/data/bpe_simple_vocab_16e6.txt.gz

    os.path.dirname(os.path.abspath("./data/bpe_simple_vocab_16e6.txt.gz")) ==> /home/202312150002/my_paper/data

    os.path.join(os.path.dirname(os.path.abspath("./data/bpe_simple_vocab_16e6.txt.gz"))) ==> /home/202312150002/my_paper/data
    
    """


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))#这一行创建了一个包含UTF-8字节的列表，这些字节覆盖了可打印字符和一些特殊字符，确保生成的Unicode字符串不包含控制字符和空白字符。
    cs = bs[:]#复制bs
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        
        vocab.pop(-1) # remove last one in vocab(jekyll) to keep vocab_size unchanged
        vocab.extend(['<|mask|>', '<|startoftext|>', '<|endoftext|>']) # vocab_size 49408
        # vocab.extend(['<|startoftext|>', '<|endoftext|>']) # vocab_size 49408
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|mask|>': '<|mask|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|mask\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()#text cleaning
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens #vector representation of text

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

if __name__ == "__main__":
    caption = 'that is a / anD Man!'
    tokenizer = SimpleTokenizer()
    bpe_tokens = tokenizer.encode(caption)
    print(bpe_tokens)