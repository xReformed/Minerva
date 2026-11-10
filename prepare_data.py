"""
Turn raw text into token files that train.py reads from disk.

    python prepare_data.py data/names.txt                          # small text file -> character tokens
    python prepare_data.py data/raw/fineweb-edu                    # folder of downloaded .parquet files -> GPT-2 tokens
    python prepare_data.py HuggingFaceFW/fineweb-edu --max-tokens 50M   # stream from Hugging Face (quick test)

Output goes to data/processed/<name>/:
    train.bin, val.bin   every token as a 2-byte number, one after another
    meta.json            which tokenizer was used and how many tokens there are (written last,
                         so a folder without it means preparation didn't finish)
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from data import TOKEN_DTYPE, CharTokenizer, GPT2Tokenizer


def parse_count(s):
    """'50M' -> 50_000_000, '2B' -> 2_000_000_000."""
    s = s.strip().upper()
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}
    if s and s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


def iter_documents(source, subset):
    """Yield one document (a string) at a time, from local .parquet files or a Hugging Face dataset."""
    if os.path.isdir(source):
        import pyarrow.parquet as pq
        files = sorted(glob.glob(os.path.join(source, "**", "*.parquet"), recursive=True))
        if not files:
            raise SystemExit(f"no .parquet files found in {source}")
        print(f"reading {len(files)} parquet files from {source}")
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
                yield from batch.column("text").to_pylist()
    else:
        from datasets import load_dataset
        print(f"streaming {source} ({subset}) from Hugging Face")
        for row in load_dataset(source, name=subset, split="train", streaming=True):
            yield row["text"]


def prepare_text_file(path, tokenizer_type, val_fraction, out_dir):
    """A single .txt file: tokenize it all at once, last part becomes validation."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if tokenizer_type == "char":
        tokenizer = CharTokenizer.from_text(text)
        ids = tokenizer.encode(text)
    else:
        tokenizer = GPT2Tokenizer()
        ids = tokenizer.encode(text) + [tokenizer.eot_id]
    ids = np.array(ids, dtype=TOKEN_DTYPE)
    n_train = int((1 - val_fraction) * len(ids))
    ids[:n_train].tofile(os.path.join(out_dir, "train.bin"))
    ids[n_train:].tofile(os.path.join(out_dir, "val.bin"))
    return tokenizer, n_train, len(ids) - n_train


def prepare_documents(source, subset, val_fraction, max_tokens, out_dir):
    """Many documents (FineWeb-Edu etc.): tokenize in batches and append to the files as we go."""
    tokenizer = GPT2Tokenizer()
    val_every = max(1, round(1 / val_fraction))      # every Nth document goes to validation
    counts = {"train": 0, "val": 0}
    files = {split: open(os.path.join(out_dir, f"{split}.bin"), "wb") for split in counts}
    doc_index, start, last_report = 0, time.time(), 0

    def write_batch(texts):
        nonlocal doc_index
        chunks = {"train": [], "val": []}
        for ids in tokenizer.encode_batch(texts):
            split = "val" if doc_index % val_every == 0 else "train"
            chunks[split].append([tokenizer.eot_id] + ids)   # <|endoftext|> marks where each document starts
            doc_index += 1
        for split, docs in chunks.items():
            if docs:
                arr = np.concatenate([np.array(d, dtype=TOKEN_DTYPE) for d in docs])
                arr.tofile(files[split])
                counts[split] += len(arr)

    batch = []
    try:
        for text in iter_documents(source, subset):
            batch.append(text)
            if len(batch) == 1024:
                write_batch(batch)
                batch = []
                total = counts["train"] + counts["val"]
                if total - last_report >= 10_000_000:
                    last_report = total
                    rate = total / (time.time() - start)
                    print(f"  {total / 1e6:,.0f}M tokens, {doc_index:,} documents ({rate / 1e6:.1f}M tokens/s)")
                if max_tokens and total >= max_tokens:
                    break
        if batch:
            write_batch(batch)
    finally:
        for f in files.values():
            f.close()
    return tokenizer, counts["train"], counts["val"]


def main():
    parser = argparse.ArgumentParser(description="Turn raw text into token files for train.py.")
    parser.add_argument("source", help="a .txt file, a folder of .parquet files, or a Hugging Face dataset name")
    parser.add_argument("--tokenizer", choices=["char", "gpt2"],
                        help="default: char for .txt files, gpt2 for everything else")
    parser.add_argument("--subset", default="sample-10BT",
                        help="Hugging Face dataset subset, when streaming (default: sample-10BT)")
    parser.add_argument("--max-tokens", type=parse_count,
                        help="stop after this many tokens, e.g. 50M (for quick tests)")
    parser.add_argument("--val-fraction", type=float,
                        help="share of data held back for validation (default: 0.1 for .txt, 0.005 otherwise)")
    parser.add_argument("--out", help="output folder (default: data/processed/<name>)")
    args = parser.parse_args()

    is_text_file = args.source.endswith(".txt")
    tokenizer_type = args.tokenizer or ("char" if is_text_file else "gpt2")
    val_fraction = args.val_fraction or (0.1 if is_text_file else 0.005)

    name = os.path.splitext(os.path.basename(os.path.normpath(args.source)))[0]
    if args.max_tokens:
        n = args.max_tokens
        name += f"-{n // 1_000_000_000}B" if n % 1_000_000_000 == 0 else f"-{n // 1_000_000}M"
    out_dir = args.out or os.path.join("data", "processed", name)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)   # mark the folder as unfinished until we're done

    start = time.time()
    if is_text_file:
        tokenizer, n_train, n_val = prepare_text_file(args.source, tokenizer_type, val_fraction, out_dir)
    else:
        if tokenizer_type != "gpt2":
            raise SystemExit("large datasets use the gpt2 tokenizer")
        tokenizer, n_train, n_val = prepare_documents(args.source, args.subset, val_fraction,
                                                      args.max_tokens, out_dir)

    meta = {
        "tokenizer": tokenizer.meta(),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "source": args.source,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"done in {time.time() - start:.0f}s: {n_train:,} train + {n_val:,} val tokens "
          f"({tokenizer_type}, vocab {tokenizer.vocab_size:,}) -> {out_dir}")
    print(f"train with: python train.py {out_dir}")


if __name__ == "__main__":
    main()
