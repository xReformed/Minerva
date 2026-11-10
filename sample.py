"""
Generate text from a model saved by train.py.

    python sample.py                                          # uses checkpoints/names.pt
    python sample.py --prompt "ma"                            # names that start with "ma"
    python sample.py checkpoints/fineweb-edu.pt --prompt "Cognitive dissonance is"
    python sample.py --num 50 --temperature 1.2 --seed 42
"""

import argparse

import torch

from data import load_tokenizer
from model import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model.")
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/names.pt",
                        help="saved model to use (default: checkpoints/names.pt)")
    parser.add_argument("--prompt", default="", help="text every sample starts with")
    parser.add_argument("--num", type=int,
                        help="how many samples (default: 20 for character models, 3 for word-piece models)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="below 1 = safer, more typical; above 1 = more creative, more mistakes (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="only choose from the k most likely next tokens")
    parser.add_argument("--max-len", type=int,
                        help="most tokens to generate per sample (default: 100 for character models, 200 otherwise)")
    parser.add_argument("--seed", type=int, default=None, help="set this to get the same samples every time")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_checkpoint(args.checkpoint, device)
    # Older checkpoints only stored the list of characters
    tokenizer_meta = ckpt.get("tokenizer") or {"type": "char", "chars": ckpt["chars"]}
    tokenizer = load_tokenizer(tokenizer_meta)
    is_char = tokenizer_meta["type"] == "char"
    num = args.num or (20 if is_char else 3)
    max_len = args.max_len or (100 if is_char else 200)
    print(f"model trained on {ckpt.get('dataset', '?')} for {ckpt.get('step', '?')} steps "
          f"(val loss {ckpt['val_loss']:.3f})\n")

    # Each sample starts right after an end marker ("\n" or <|endoftext|>), meaning "a new example
    # begins here", followed by the prompt. It stops at the next end marker.
    eot = tokenizer.eot_id
    try:
        start_ids = ([eot] if eot is not None else []) + tokenizer.encode(args.prompt)
    except KeyError as e:
        raise SystemExit(f"the prompt contains {e}, a character this model never saw in training")
    if not start_ids:
        raise SystemExit("this model has no end-of-line marker; give a --prompt to start from")

    start = torch.tensor([start_ids] * num, dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=max_len, temperature=args.temperature, top_k=args.top_k)

    for row in out[:, len(start_ids):].tolist():
        if eot in row:
            row = row[: row.index(eot)]
        text = args.prompt + tokenizer.decode(row)
        print(text if is_char else f"{text}\n{'-' * 60}")


if __name__ == "__main__":
    main()
