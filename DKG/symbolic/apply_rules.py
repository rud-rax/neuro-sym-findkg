"""
Parse rules.txt produced by Ruleformer and apply them to the training KG
to generate new symbolic predictions as sym_triplets.tsv.

Usage:
    python -m DKG.symbolic.apply_rules \
        --rules_file Ruleformer/EXPS/findkg-rule-xxx/rules.txt \
        --ruleformer_train Ruleformer/DATASET/FinDKG/train.txt \
        --data_dir FinDKG_dataset \
        --dataset FinDKG \
        --output sym_triplets.tsv \
        --min_weight 0.1 \
        --min_count 3
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# KG loading
# ---------------------------------------------------------------------------

def load_kg(train_path: str):
    """Load Ruleformer-format training triples into fast lookup dicts.

    Returns:
        hr_t  — {(head_name, rel_name): set of tail_names}
        known — set of (head_name, rel_name, tail_name)  [for dedup]
    """
    hr_t = defaultdict(set)
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            hr_t[(h, r)].add(t)

    known = {(h, r, t) for (h, r), tails in hr_t.items() for t in tails}
    return hr_t, known


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

_RULE_RE = re.compile(r"^\s*(\d+)-([\d.]+)\s+(\S+)\s*<-\s*(.+)$")


def parse_rules(rules_path: str, min_weight: float = 0.1, min_count: int = 3):
    """Parse rules.txt into a list of (weight, count, head_rel, [body_rels]).

    Expected line format:
        {count}-{weight}  {head_rel} <- {body_r1}^{body_r2}^...
    """
    rules = []
    with open(rules_path, encoding="utf-8") as f:
        for line in f:
            m = _RULE_RE.match(line)
            if not m:
                continue
            count = int(m.group(1))
            weight = float(m.group(2))
            if weight < min_weight or count < min_count:
                continue
            head_rel = m.group(3)
            body_rels = [r.strip() for r in m.group(4).split("^") if r.strip()]
            rules.append((weight, count, head_rel, body_rels))

    rules.sort(key=lambda x: -x[0])
    return rules


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------

def _apply_rule(head_rel: str, body_rels: list, hr_t: dict, known: set):
    """Yield new (head_name, head_rel, tail_name) triples by following the rule body."""
    # Collect all unique head entities that have at least one outgoing body_rels[0] edge
    head_entities = {h for (h, r) in hr_t if r == body_rels[0]}

    for h in head_entities:
        frontier = {h}
        for rel in body_rels:
            next_frontier = set()
            for node in frontier:
                next_frontier.update(hr_t.get((node, rel), ()))
            frontier = next_frontier
            if not frontier:
                break

        for t in frontier:
            if (h, head_rel, t) not in known:
                yield h, head_rel, t


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def apply_rules_to_kg(
    rules_path: str,
    ruleformer_train_path: str,
    data_dir: str,
    dataset: str,
    output_path: str,
    min_weight: float = 0.1,
    min_count: int = 3,
) -> int:
    """
    Parse Ruleformer rules, apply them to the training KG, and write sym_triplets.tsv.

    Returns the number of predicted triples written.
    """
    print(f"[apply_rules] Loading training KG: {ruleformer_train_path}")
    hr_t, known = load_kg(ruleformer_train_path)
    print(f"  Training triples: {len(known):,}")

    print(f"[apply_rules] Parsing rules: {rules_path}")
    rules = parse_rules(rules_path, min_weight=min_weight, min_count=min_count)
    print(f"  Rules passing filters (weight≥{min_weight}, count≥{min_count}): {len(rules)}")

    predicted_names: set = set()
    for weight, count, head_rel, body_rels in rules:
        before = len(predicted_names)
        for triple in _apply_rule(head_rel, body_rels, hr_t, known):
            predicted_names.add(triple)
        added = len(predicted_names) - before
        print(f"  [{weight:.3f}|{count}] {head_rel} <- {'·'.join(body_rels)}  +{added:,}")

    print(f"  Total unique new triples (names): {len(predicted_names):,}")

    # Load name → integer ID mappings from original FinDKG files
    src_dir = os.path.join(data_dir, dataset)
    ent_df = pd.read_table(
        os.path.join(src_dir, "entity2id.txt"),
        header=None, names=["name", "id", "ntype", "ntype_id"],
    )
    rel_df = pd.read_table(
        os.path.join(src_dir, "relation2id.txt"),
        header=None, names=["name", "id"],
    )
    ent2id = dict(zip(ent_df["name"], ent_df["id"].astype(int)))
    rel2id = dict(zip(rel_df["name"], rel_df["id"].astype(int)))

    # Convert and write
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    skipped = 0
    for h_name, r_name, t_name in predicted_names:
        h_id = ent2id.get(h_name)
        r_id = rel2id.get(r_name)
        t_id = ent2id.get(t_name)
        if h_id is None or r_id is None or t_id is None:
            skipped += 1
            continue
        rows.append(f"{h_id}\t{r_id}\t{t_id}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    if skipped:
        print(f"  Skipped {skipped} triples with unmapped names")
    print(f"  Written {len(rows):,} triples → {output_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply Ruleformer rules to FinDKG training KG → sym_triplets.tsv"
    )
    parser.add_argument("--rules_file", required=True, help="Path to Ruleformer rules.txt")
    parser.add_argument("--ruleformer_train", default="Ruleformer/DATASET/FinDKG/train.txt",
                        help="Ruleformer-format training file (name-based)")
    parser.add_argument("--data_dir", default="FinDKG_dataset")
    parser.add_argument("--dataset", default="FinDKG")
    parser.add_argument("--output", default="sym_triplets.tsv")
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--min_count", type=int, default=3)
    args = parser.parse_args()

    apply_rules_to_kg(
        rules_path=args.rules_file,
        ruleformer_train_path=args.ruleformer_train,
        data_dir=args.data_dir,
        dataset=args.dataset,
        output_path=args.output,
        min_weight=args.min_weight,
        min_count=args.min_count,
    )
