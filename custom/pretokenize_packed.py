"""Tokenize a jsonl text corpus once into fixed-length packed blocks (uint16).

Why this exists (Phase-1 accounting + throughput):
  * MiniMind's PretrainDataset tokenizes inside __getitem__ and PADS every doc to
    max_seq_len. On this laptop that costs ~30-40% of wall time in the dataloader and
    throws away ~40% of the compute on pad tokens.
  * Packing the corpus into contiguous `seq_len` blocks removes both, and gives an
    EXACT token count straight from the tokenizer (reports must not estimate).

Memory-safe streaming design (16GB RAM hard constraint): documents are read in small
batches, batch-tokenized with the fast tokenizer, and appended to a bounded rolling
numpy buffer that is flushed to disk as soon as it holds whole blocks.

Output (D:/llm-rl-lab/data/tokens/):
  <tag>_blocks.bin    raw little-endian uint16, shape (n_blocks, seq_len)
  <tag>_meta.json     {n_blocks, seq_len, total_tokens_consumed, tokenized_tokens, ...}

Usage:
  python pretokenize_packed.py --src D:/llm-rl-lab/data/pretrain_t2t.jsonl \
      --out D:/llm-rl-lab/data/tokens --seq-len 512 --batch-docs 2000 --tag pretrain
"""
import argparse
import json
import os
import time

import numpy as np

# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

TOK_DIR = os.path.join(PROJECT_ROOT, 'src', 'minimind-master', 'model')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=os.path.join(PROJECT_ROOT, 'data', 'pretrain_t2t.jsonl'))
    p.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'data', 'tokens'))
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--batch-docs', type=int, default=2000)
    p.add_argument('--max-docs', type=int, default=0)
    p.add_argument('--tag', default='pretrain')
    p.add_argument('--progress-every', type=int, default=20)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    eos = tok.eos_token_id
    seq = args.seq_len

    bin_path = os.path.join(args.out, f'{args.tag}_blocks.bin')
    meta_path = os.path.join(args.out, f'{args.tag}_meta.json')

    t0 = time.time()
    tokenized_tokens = 0
    n_docs = 0
    n_blocks = 0
    written_elems = 0

    # bounded rolling buffer: at most ~2 * seq * 1024 elements live in RAM
    cap = seq * 4096
    buf = np.empty(cap, dtype=np.uint16)
    buf_pos = 0
    fout = open(bin_path, 'wb', buffering=1 << 23)

    def push(arr):
        nonlocal buf, buf_pos, n_blocks, written_elems
        i = 0
        n = arr.size
        while i < n:
            space = buf.size - buf_pos
            take = min(space, n - i)
            buf[buf_pos:buf_pos + take] = arr[i:i + take]
            buf_pos += take
            i += take
            if buf_pos == buf.size:
                n_full = buf_pos // seq
                if n_full:
                    fout.write(buf[:n_full * seq].tobytes())
                    written_elems += n_full * seq
                    n_blocks += n_full
                    rest = buf_pos - n_full * seq
                    if rest:
                        buf[:rest] = buf[n_full * seq:buf_pos]
                    buf_pos = rest

    nbatch = 0
    with open(args.src, encoding='utf-8') as f:
        texts = []
        for line in f:
            try:
                t = json.loads(line)['text']
            except Exception:
                continue
            if not isinstance(t, str) or not t:
                continue
            texts.append(t)
            if len(texts) < args.batch_docs:
                continue
            enc = tok._tokenizer.encode_batch(texts, add_special_tokens=False)
            ids_flat = []
            for e in enc:
                ids = e.ids
                if ids:
                    ids_flat.extend(ids)
                    ids_flat.append(eos)
                    tokenized_tokens += len(ids) + 1
                    n_docs += 1
            push(np.asarray(ids_flat, dtype=np.uint16))
            texts = []
            del enc, ids_flat
            nbatch += 1
            if nbatch % args.progress_every == 0:
                el = time.time() - t0
                print(f'[tok] docs={n_docs:,} tokenized={tokenized_tokens:,} '
                      f'blocks={n_blocks:,} {el:.0f}s '
                      f'({tokenized_tokens/max(el,1):,.0f} tok/s)', flush=True)
            if args.max_docs and n_docs >= args.max_docs:
                break
        if texts:
            enc = tok._tokenizer.encode_batch(texts, add_special_tokens=False)
            ids_flat = []
            for e in enc:
                ids = e.ids
                if ids:
                    ids_flat.extend(ids)
                    ids_flat.append(eos)
                    tokenized_tokens += len(ids) + 1
                    n_docs += 1
            push(np.asarray(ids_flat, dtype=np.uint16))

    # flush whatever whole blocks remain
    n_full = buf_pos // seq
    if n_full:
        fout.write(buf[:n_full * seq].tobytes())
        written_elems += n_full * seq
        n_blocks += n_full
        buf_pos -= n_full * seq
    fout.close()

    meta = dict(src=args.src, seq_len=seq, n_blocks=int(n_blocks),
                total_tokens_consumed=int(written_elems),
                tokenized_tokens=int(tokenized_tokens),
                dropped_tail_tokens=int(buf_pos),
                n_docs=int(n_docs), vocab_size=6400, eos_token_id=eos, dtype='uint16',
                bin=bin_path, bytes=os.path.getsize(bin_path),
                wall_sec=round(time.time() - t0, 1))
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('[tok] META ' + json.dumps(meta))


if __name__ == '__main__':
    main()
