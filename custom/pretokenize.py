"""Pre-tokenize a jsonl text corpus into a flat uint16 token stream.

Why: MiniMind's PretrainDataset tokenizes on the fly inside __getitem__, which on this
laptop costs ~30% of step time and truncates documents. Pre-tokenizing once
  (a) removes dataloader CPU cost from the training loop,
  (b) lets us pack full sequences with no truncation waste,
  (c) gives an EXACT token count straight from the tokenizer, which is the required
      accounting basis for reports/pretrain.md (no estimation).

Output (all under D:/llm-rl-lab/data/tokens/):
  pretrain_tokens.bin   uint16 little-endian flat token ids
  pretrain_meta.json    {total_tokens, n_docs, vocab_size, seq_len, doc_ends_sha256, ...}
  pretrain_doc_ends.npy int64 offsets of each document end (for boundary-aware packing)

Usage:
  python pretokenize.py --src D:/llm-rl-lab/data/pretrain_t2t.jsonl \
                        --out D:/llm-rl-lab/data/tokens --workers 12 --max-docs 0
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np

# --- project root, derived from this file's location (portable across machines) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

TOK_DIR = os.path.join(PROJECT_ROOT, 'src', 'minimind-master', 'model')
_worker_tok = None


def _init_worker():
    global _worker_tok
    from transformers import AutoTokenizer
    _worker_tok = AutoTokenizer.from_pretrained(TOK_DIR)
    _worker_tok.add_special_tokens({'pad_token': '<|endoftext|>'})


def _tokenize_chunk(lines):
    """lines: list[str] raw jsonl lines -> (flat token list, doc lengths)"""
    import json as _json
    out, lens = [], []
    for line in lines:
        try:
            text = _json.loads(line)['text']
        except Exception:
            continue
        if not isinstance(text, str) or not text:
            continue
        ids = _worker_tok(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        out.extend(ids)
        out.append(_worker_tok.eos_token_id)  # document separator
        lens.append(len(ids) + 1)
    return out, lens


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=os.path.join(PROJECT_ROOT, 'data', 'pretrain_t2t.jsonl'))
    p.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'data', 'tokens'))
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--chunk-lines', type=int, default=4000)
    p.add_argument('--max-docs', type=int, default=0, help='0 = all')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bin_path = os.path.join(args.out, 'pretrain_tokens.bin')
    ends_path = os.path.join(args.out, 'pretrain_doc_ends.npy')
    meta_path = os.path.join(args.out, 'pretrain_meta.json')

    t0 = time.time()
    total_tokens = 0
    n_docs = 0
    ends = []

    with open(bin_path, 'wb', buffering=1 << 22) as fout:
        buf = []
        with open(args.src, encoding='utf-8') as f:
            chunk = []
            with mp.Pool(args.workers, initializer=_init_worker) as pool:
                def flush(chunk):
                    nonlocal total_tokens, n_docs
                    results = pool.map(_tokenize_chunk, [chunk], chunksize=1)
                    for ids, lens in results:
                        arr = np.asarray(ids, dtype=np.uint16)
                        fout.write(arr.tobytes())
                        total_tokens += arr.size
                        for L in lens:
                            n_docs += 1
                            ends.append(total_tokens)
                    done = total_tokens
                    el = time.time() - t0
                    print(f'[tok] docs={n_docs:,} tokens={done:,} '
                          f'({el:.0f}s, {done/max(el,1):,.0f} tok/s)', flush=True)

                for line in f:
                    chunk.append(line)
                    if len(chunk) >= args.chunk_lines:
                        flush(chunk)
                        chunk = []
                        if args.max_docs and n_docs >= args.max_docs:
                            break
                if chunk:
                    flush(chunk)

    np.save(ends_path, np.asarray(ends, dtype=np.int64))
    meta = dict(src=args.src, total_tokens=int(total_tokens), n_docs=int(n_docs),
                vocab_size=6400, eos_token_id=2, dtype='uint16',
                wall_sec=round(time.time() - t0, 1),
                bytes=os.path.getsize(bin_path),
                src_sha256=None)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('[tok] META ' + json.dumps(meta))


if __name__ == '__main__':
    main()
