import struct
import sys
import subprocess
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "bin" / "main.exe"


class LLM:
    pass


def _read_u32(proc):
    raw = proc.stdout.read(4)
    if len(raw) != 4:
        raise RuntimeError("backend exited unexpectedly")
    return struct.unpack("<I", raw)[0]


def start_llm(weight_dir, max_ctx=32768, vocab_weight=None, max_new_tokens=128, dump_dir=None, dump_layers=0):
    weight_dir = Path(weight_dir).resolve()

    if vocab_weight is None:
        tokenizer_path = weight_dir / "tokenizer.json"
        head_file = None
        embed_file = None
    else:
        vocab_dir = Path(vocab_weight).resolve()
        tokenizer_path = vocab_dir / "tokenizer.json"
        heads = sorted(weight_dir.glob("lm_head.*.safetensors"))
        embeds = sorted(weight_dir.glob("embed_tokens.*.safetensors"))
        if not heads or not embeds:
            raise RuntimeError("custom vocab weights not found in " + str(weight_dir))
        head_file = heads[-1]
        embed_file = embeds[-1]

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("<|im_end|>")
    if eos is None:
        eos = 0

    cmd = [
        str(BACKEND),
        "--weights", str(weight_dir),
        "--eos", str(eos),
        "--max-ctx", str(max_ctx),
        "--max-new", str(max_new_tokens),
    ]
    if head_file is not None:
        cmd += ["--vocab-head", str(head_file), "--vocab-embed", str(embed_file)]
    if dump_dir is not None:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        cmd += ["--dump", str(dump_dir), "--dump-layers", str(dump_layers)]

    proc = subprocess.Popen(
        cmd,
        cwd=str(BACKEND.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    llm = LLM()
    llm.proc = proc
    llm.weight_dir = weight_dir
    llm.max_ctx = max_ctx
    llm.max_new_tokens = max_new_tokens
    llm.vocab_weight = vocab_weight
    llm.tokenizer = tokenizer
    llm.eos = eos
    return llm


def apply_chat_template(llm, messages, system="", enable_thinking=True):
    bos = "<|im_start|>"
    eos = "<|im_end|>"
    parts = []
    if (system != ""):
        parts.append("<|im_start|>system\n")
        parts.append(system+eos+"\n")
    for msg in messages:
        parts.append(f"{bos}{msg['role']}\n{msg['content']}{eos}\n")
    parts.append(f"{bos}assistant\n")
    parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return llm.tokenizer.encode("".join(parts), add_special_tokens=False).ids


def tokenize(llm, text, thinking):
    return apply_chat_template(llm, [{"role": "user", "content": text}], system="You are a helpful assistant.", enable_thinking=thinking)


def detokenize(llm, token_ids):
    return llm.tokenizer.decode(token_ids)


def generate(llm, token_ids):
    if not token_ids:
        return []
    n = len(token_ids)
    llm.proc.stdin.write(struct.pack("<I", n))
    llm.proc.stdin.write(struct.pack("<%dI" % n, *token_ids))
    llm.proc.stdin.flush()
    m = _read_u32(llm.proc)
    return [_read_u32(llm.proc) for _ in range(m)]


def close(llm):
    try:
        llm.proc.stdin.write(struct.pack("<I", 0))
        llm.proc.stdin.flush()
    except Exception:
        pass
    try:
        llm.proc.stdin.close()
    except Exception:
        pass
    llm.proc.wait(timeout=10)
    llm.proc.terminate()


def _main():
    max_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    vocab = sys.argv[3] if len(sys.argv) > 3 else "model/Qwen3.5-pruned-vocab"
    text = sys.argv[4] if len(sys.argv) > 4 else None
    if text is None: return
    weight_dir = sys.argv[1] if len(sys.argv) > 1 else "model/Qwen3.5-9B-weight"
    thinking = sys.argv[5] if len(sys.argv) > 5 else "none"

    llm = start_llm(weight_dir, max_ctx=max_ctx, vocab_weight=vocab, max_new_tokens=16384)
    ids = tokenize(llm, text, thinking == "thinking")
    out = generate(llm, ids)
    close(llm)
    print(detokenize(llm, out))


if __name__ == "__main__":
    _main()