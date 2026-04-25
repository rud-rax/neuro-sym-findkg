"""
Convert FinDKG dataset files to the format expected by Ruleformer.

FinDKG format  (train/valid/test.txt):  head_id  rel_id  tail_id  time  _
Ruleformer format (train/valid/test.txt): head_name  rel_name  tail_name

Ruleformer also needs:
  entities.txt  — one entity name per line, no IDs
  relations.txt — one relation name per line (original only; inverses added automatically)
"""

import argparse
import os

import pandas as pd


def _load_id2name(entity2id_path: str, relation2id_path: str):
    """Return (id→entity_name, id→relation_name) dicts from FinDKG mapping files."""
    ent_df = pd.read_table(
        entity2id_path, header=None,
        names=["name", "id", "ntype", "ntype_id"],
    )
    id2ent = dict(zip(ent_df["id"].astype(int), ent_df["name"]))

    rel_df = pd.read_table(
        relation2id_path, header=None,
        names=["name", "id"],
    )
    id2rel = dict(zip(rel_df["id"].astype(int), rel_df["name"]))

    return id2ent, id2rel


def _convert_split(src_path: str, dst_path: str, id2ent: dict, id2rel: dict):
    """Convert one FinDKG split file to Ruleformer tab-separated name format."""
    df = pd.read_table(
        src_path, header=None,
        names=["head", "rel", "tail", "time", "_"],
    )
    lines = []
    for row in df.itertuples(index=False):
        h = id2ent[int(row.head)]
        r = id2rel[int(row.rel)]
        t = id2ent[int(row.tail)]
        lines.append(f"{h}\t{r}\t{t}")
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Converted {os.path.basename(src_path)} ({len(lines):,} triples) → {dst_path}")


def adapt_findkg_for_ruleformer(data_dir: str, dataset: str, ruleformer_root: str) -> str:
    """
    Convert FinDKG dataset to Ruleformer input format.

    Writes to Ruleformer/DATASET/<dataset>/:
      train.txt, valid.txt, test.txt  (name-based, 3-col TSV)
      entities.txt                    (one name per line, ordered by entity ID)
      relations.txt                   (one name per line, ordered by relation ID)

    Returns the output directory path.
    """
    src_dir = os.path.join(data_dir, dataset)
    dst_dir = os.path.join(ruleformer_root, "DATASET", dataset)
    os.makedirs(dst_dir, exist_ok=True)

    id2ent, id2rel = _load_id2name(
        os.path.join(src_dir, "entity2id.txt"),
        os.path.join(src_dir, "relation2id.txt"),
    )

    for split in ("train", "valid", "test"):
        src = os.path.join(src_dir, f"{split}.txt")
        dst = os.path.join(dst_dir, f"{split}.txt")
        if os.path.exists(src):
            _convert_split(src, dst, id2ent, id2rel)

    # entities.txt — names in ascending ID order (Ruleformer assigns IDs by position + 1)
    ent_names = [id2ent[i] for i in sorted(id2ent)]
    with open(os.path.join(dst_dir, "entities.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ent_names) + "\n")
    print(f"  Wrote entities.txt ({len(ent_names):,} entities)")

    # relations.txt — original 15 relations in ascending ID order
    rel_names = [id2rel[i] for i in sorted(id2rel)]
    with open(os.path.join(dst_dir, "relations.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rel_names) + "\n")
    print(f"  Wrote relations.txt ({len(rel_names):,} relations)")

    return dst_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt FinDKG dataset for Ruleformer")
    parser.add_argument("--data_dir", default="FinDKG_dataset")
    parser.add_argument("--dataset", default="FinDKG")
    parser.add_argument("--ruleformer_root", default="Ruleformer")
    args = parser.parse_args()

    out = adapt_findkg_for_ruleformer(args.data_dir, args.dataset, args.ruleformer_root)
    print(f"Done. Ruleformer data written to: {out}")
