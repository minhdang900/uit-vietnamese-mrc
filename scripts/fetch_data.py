"""Tải UIT-ViQuAD 2.0 từ HuggingFace và ghi ra JSON thô dạng SQuAD-2.0.

Ghi ra ``data/raw/viquad2_{split}.json``. Pipeline sau đó tự dedup và split, nên
ở đây KHÔNG được xử lý gì ngoài việc đổi định dạng — mọi biến đổi dữ liệu phải
nằm trong code có test.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DATASET_ID = "taidng/UIT-ViQuAD2.0"


def to_squad_format(rows) -> dict:
    """Gom các dòng phẳng của HF về cấu trúc SQuAD-2.0 lồng nhau."""
    by_title: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_title[r["title"]][r["context"]].append(r)

    data = []
    for title, contexts in by_title.items():
        paragraphs = []
        for context, rows_for_ctx in contexts.items():
            qas = []
            for r in rows_for_ctx:
                ans = r.get("answers") or {}
                qa = {
                    "id": r["id"],
                    "question": r["question"],
                    "is_impossible": bool(r.get("is_impossible", False)),
                    "answers": {
                        "text": list(ans.get("text", [])),
                        "answer_start": [int(x) for x in ans.get("answer_start", [])],
                    },
                }
                plaus = r.get("plausible_answers")
                if plaus and plaus.get("text"):
                    qa["plausible_answers"] = {
                        "text": list(plaus.get("text", [])),
                        "answer_start": [int(x) for x in plaus.get("answer_start", [])],
                    }
                qas.append(qa)
            paragraphs.append({"context": context, "qas": qas})
        data.append({"title": title, "paragraphs": paragraphs})
    return {"version": 2.0, "data": data}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/raw")
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_ID)
    for split in ds:
        payload = to_squad_format(ds[split])
        n_q = sum(len(p["qas"]) for a in payload["data"] for p in a["paragraphs"])
        n_ctx = sum(len(a["paragraphs"]) for a in payload["data"])
        n_imp = sum(
            qa["is_impossible"]
            for a in payload["data"]
            for p in a["paragraphs"]
            for qa in p["qas"]
        )
        n_gold = sum(
            bool(qa["answers"]["text"])
            for a in payload["data"]
            for p in a["paragraphs"]
            for qa in p["qas"]
        )
        path = out_dir / f"viquad2_{split}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(
            f"{split:11s} articles={len(payload['data']):4d} contexts={n_ctx:5d} "
            f"qas={n_q:6d} impossible={n_imp:5d} with_gold={n_gold:6d} -> {path}"
        )


if __name__ == "__main__":
    main()
