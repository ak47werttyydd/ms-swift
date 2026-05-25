#!/usr/bin/env python3
"""
Compare first record of:
  (A) raw GSM8K parquet (HuggingFace openai/gsm8k 'main' split)
  (B) verl-converted parquet (after examples/data_preprocess/gsm8k.py)

Run on the box where both files exist. Override paths via env vars if needed:
  RAW_PARQUET   default: /home/canada_group_account/a84400789/dataset/gsm8k/main/train-00000-of-00001.parquet
  VERL_PARQUET  default: ./data/train.parquet   (verl launches from a dir containing data/train.parquet)
"""
import json
import os
import sys

import pyarrow.parquet as pq

RAW = os.environ.get(
    "RAW_PARQUET",
    "/home/canada_group_account/a84400789/dataset/gsm8k/main/train-00000-of-00001.parquet",
)
VERL = os.environ.get("VERL_PARQUET", "/home/s00525112/verl/data/train.parquet")


def load_first(path: str) -> tuple[list[str], dict]:
    tbl = pq.read_table(path)
    cols = tbl.column_names
    row0 = {c: tbl.column(c)[0].as_py() for c in cols}
    return cols, row0


def pretty(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def main() -> int:
    for label, path in [("RAW", RAW), ("VERL", VERL)]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} parquet not found: {path}", file=sys.stderr)
            return 1

    raw_cols, raw_row = load_first(RAW)
    verl_cols, verl_row = load_first(VERL)

    print("=" * 78)
    print(f"RAW  : {RAW}")
    print(f"     columns = {raw_cols}")
    print("-" * 78)
    print(pretty(raw_row))

    print("=" * 78)
    print(f"VERL : {VERL}")
    print(f"     columns = {verl_cols}")
    print("-" * 78)
    print(pretty(verl_row))

    print("=" * 78)
    print("DIFF SUMMARY")
    print("-" * 78)
    print(f"  columns added  : {sorted(set(verl_cols) - set(raw_cols))}")
    print(f"  columns dropped: {sorted(set(raw_cols) - set(verl_cols))}")
    print(f"  columns kept   : {sorted(set(raw_cols) & set(verl_cols))}")

    raw_q = raw_row.get("question", "")
    verl_prompt = verl_row.get("prompt")
    verl_q_user = ""
    if isinstance(verl_prompt, list) and verl_prompt and isinstance(verl_prompt[0], dict):
        verl_q_user = verl_prompt[0].get("content", "")
    raw_a = raw_row.get("answer", "")
    verl_gt = ""
    rm = verl_row.get("reward_model")
    if isinstance(rm, dict):
        verl_gt = rm.get("ground_truth", "")
    verl_ans_extra = ""
    ei = verl_row.get("extra_info")
    if isinstance(ei, dict):
        verl_ans_extra = ei.get("answer", "")

    print()
    print("  question text:")
    print(f"    raw  ({len(raw_q):>5d} chars): {raw_q[:160]!r}")
    print(f"    verl ({len(verl_q_user):>5d} chars): {verl_q_user[:160]!r}")
    suffix = verl_q_user[len(raw_q):] if verl_q_user.startswith(raw_q) else None
    print(f"    suffix appended by verl: {suffix!r}")

    print()
    print("  answer:")
    print(f"    raw.answer        ({len(raw_a):>5d} chars): {raw_a[:120]!r}")
    print(f"    verl.ground_truth : {verl_gt!r}   (extracted #### number)")
    print(
        "    verl.extra_info.answer == raw.answer ? "
        f"{verl_ans_extra == raw_a}"
    )

    print()
    print("  verl-only fields (top level):")
    for k in sorted(set(verl_cols) - set(raw_cols)):
        v = verl_row[k]
        s = pretty(v)
        if len(s) > 400:
            s = s[:400] + " ... (truncated)"
        print(f"    {k}: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
