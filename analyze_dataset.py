# analyze_dataset.py
import argparse
from pathlib import Path
from statistics import mean
from transformers import T5TokenizerFast

def read_lines(p):
    with open(p, "r") as f:
        # 去除空行并strip
        return [ln.strip() for ln in f.readlines() if ln.strip()]

def tokenize_len_and_vocab(lines, tok, max_len=None, add_prefix=None):
    lens = []
    vocab = set()
    for s in lines:
        if add_prefix:
            s = add_prefix + s
        toks = tok.tokenize(s)
        if max_len is not None:
            toks = toks[:max_len]
        lens.append(len(toks))
        vocab.update(toks)
    return mean(lens) if lens else 0.0, len(vocab)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--prefix", default="translate to SQL: ")
    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_tgt_len", type=int, default=128)
    ap.add_argument("--out_md", default="q4_stats.md")
    args = ap.parse_args()

    D = Path(args.data_dir)
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    # 读取数据
    train_nl = read_lines(D/"train.nl")
    dev_nl   = read_lines(D/"dev.nl")
    train_sql= read_lines(D/"train.sql")
    dev_sql  = read_lines(D/"dev.sql")

    # ---------- Table 1: Before preprocessing ----------
    t1 = {}
    t1["Number of examples"] = (len(train_nl), len(dev_nl))
    train_nl_len_b, train_nl_vocab_b = tokenize_len_and_vocab(train_nl, tok)
    dev_nl_len_b,   dev_nl_vocab_b   = tokenize_len_and_vocab(dev_nl, tok)
    train_sql_len_b,train_sql_vocab_b= tokenize_len_and_vocab(train_sql, tok)
    dev_sql_len_b,  dev_sql_vocab_b  = tokenize_len_and_vocab(dev_sql, tok)

    # ---------- Table 2: After preprocessing ----------
    # 输入：加 prefix + 截断到 max_src_len
    train_nl_len_a, train_nl_vocab_a = tokenize_len_and_vocab(
        train_nl, tok, max_len=args.max_src_len, add_prefix=args.prefix
    )
    dev_nl_len_a,   dev_nl_vocab_a   = tokenize_len_and_vocab(
        dev_nl, tok, max_len=args.max_src_len, add_prefix=args.prefix
    )
    # 输出（SQL）：只做截断到 max_tgt_len（不加前缀）
    train_sql_len_a,train_sql_vocab_a= tokenize_len_and_vocab(
        train_sql, tok, max_len=args.max_tgt_len
    )
    dev_sql_len_a,  dev_sql_vocab_a  = tokenize_len_and_vocab(
        dev_sql, tok, max_len=args.max_tgt_len
    )

    # 打印到控制台
    print("\n==== Table 1: Before preprocessing ====")
    print(f"Number of examples          | Train {t1['Number of examples'][0]} | Dev {t1['Number of examples'][1]}")
    print(f"Mean sentence length (NL)   | Train {train_nl_len_b:.2f} | Dev {dev_nl_len_b:.2f}")
    print(f"Mean SQL query length       | Train {train_sql_len_b:.2f} | Dev {dev_sql_len_b:.2f}")
    print(f"Vocabulary size (NL)        | Train {train_nl_vocab_b} | Dev {dev_nl_vocab_b}")
    print(f"Vocabulary size (SQL)       | Train {train_sql_vocab_b} | Dev {dev_sql_vocab_b}")

    print("\n==== Table 2: After preprocessing ====")
    print(f"Model name: T5-small")
    print(f"Mean sentence length (NL)   | Train {train_nl_len_a:.2f} | Dev {dev_nl_len_a:.2f}")
    print(f"Mean SQL query length       | Train {train_sql_len_a:.2f} | Dev {dev_sql_len_a:.2f}")
    print(f"Vocabulary size (NL)        | Train {train_nl_vocab_a} | Dev {dev_nl_vocab_a}")
    print(f"Vocabulary size (SQL)       | Train {train_sql_vocab_a} | Dev {dev_sql_vocab_a}")

    # 同时写到 markdown 文件，方便拷到 report
    md = []
    md.append("# Q4 Data Statistics\n")
    md.append("## Table 1: Before preprocessing\n")
    md.append("| Statistics Name | Train | Dev |\n|---|---:|---:|\n")
    md.append(f"| Number of examples | {t1['Number of examples'][0]} | {t1['Number of examples'][1]} |\n")
    md.append(f"| Mean sentence length (NL) | {train_nl_len_b:.2f} | {dev_nl_len_b:.2f} |\n")
    md.append(f"| Mean SQL query length | {train_sql_len_b:.2f} | {dev_sql_len_b:.2f} |\n")
    md.append(f"| Vocabulary size (NL) | {train_nl_vocab_b} | {dev_nl_vocab_b} |\n")
    md.append(f"| Vocabulary size (SQL) | {train_sql_vocab_b} | {dev_sql_vocab_b} |\n")

    md.append("\n## Table 2: After preprocessing\n")
    md.append("_Model: T5-small; Prefix: `translate to SQL: `; Src max len=256; Tgt max len=128._\n\n")
    md.append("| Statistics Name | Train | Dev |\n|---|---:|---:|\n")
    md.append(f"| Mean sentence length (NL) | {train_nl_len_a:.2f} | {dev_nl_len_a:.2f} |\n")
    md.append(f"| Mean SQL query length | {train_sql_len_a:.2f} | {dev_sql_len_a:.2f} |\n")
    md.append(f"| Vocabulary size (NL) | {train_nl_vocab_a} | {dev_nl_vocab_a} |\n")
    md.append(f"| Vocabulary size (SQL) | {train_sql_vocab_a} | {dev_sql_vocab_a} |\n")

    Path(args.out_md).write_text("".join(md), encoding="utf-8")
    print(f"\nSaved markdown table to: {args.out_md}")

if __name__ == "__main__":
    main()
