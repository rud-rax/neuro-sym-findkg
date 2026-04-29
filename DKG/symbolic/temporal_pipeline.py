"""
Temporal Ruleformer pipeline for FinDKG.

Divides the dataset into consecutive time groups (default size 10), trains a
fresh Ruleformer model per group, decodes rules, applies them to the group's
subgraph, and tags each prediction with timestamp g_end + 1.

All predictions across all groups are accumulated into sym_triplets.csv with
4 columns:  h_id  r_id  t_id  timestamp
"""

import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

import pandas as pd

from DKG.symbolic.ruleformer_adapter import (
    _load_id2name,
    filter_triplets_by_time,
    write_ruleformer_split_from_memory,
)
from DKG.symbolic.apply_rules import load_kg_from_triplets, parse_rules, _apply_rule


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_triplets(data_dir: str) -> list:
    """Load all triplets from train.txt, valid.txt, test.txt.

    Returns list of (h_id, r_id, t_id, time) tuples.
    """
    triplets = []
    for split in ("train", "valid", "test"):
        path = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(path):
            continue
        df = pd.read_table(path, header=None, names=["h", "r", "t", "time", "_"])
        for row in df.itertuples(index=False):
            triplets.append((int(row.h), int(row.r), int(row.t), int(row.time)))
    return triplets


def load_id2name(data_dir: str):
    """Load global entity/relation ID→name mappings."""
    return _load_id2name(
        os.path.join(data_dir, "entity2id.txt"),
        os.path.join(data_dir, "relation2id.txt"),
    )


# ---------------------------------------------------------------------------
# Time grouping
# ---------------------------------------------------------------------------

def make_groups(all_triplets: list, group_size: int) -> list:
    """Return list of (g_start, g_end) pairs covering all timestamps.

    Groups are contiguous non-overlapping windows of `group_size` timestamps.
    The last group may be smaller if timestamps don't divide evenly.
    """
    timestamps = sorted({t[3] for t in all_triplets})
    groups = []
    i = 0
    while i < len(timestamps):
        g_start = timestamps[i]
        g_end = timestamps[min(i + group_size - 1, len(timestamps) - 1)]
        groups.append((g_start, g_end))
        i += group_size
    return groups


# ---------------------------------------------------------------------------
# Ruleformer data preparation
# ---------------------------------------------------------------------------

def write_global_vocab(id2ent: dict, id2rel: dict, rf_root: str, dataset: str = "FinDKG"):
    """Write entities.txt and relations.txt (global vocab) to the Ruleformer DATASET dir."""
    dst_dir = os.path.join(rf_root, "DATASET", dataset)
    os.makedirs(dst_dir, exist_ok=True)

    ent_names = [id2ent[i] for i in sorted(id2ent)]
    with open(os.path.join(dst_dir, "entities.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ent_names) + "\n")

    rel_names = [id2rel[i] for i in sorted(id2rel)]
    with open(os.path.join(dst_dir, "relations.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rel_names) + "\n")

    print(f"  Global vocab: {len(ent_names)} entities, {len(rel_names)} relations → {dst_dir}")
    return dst_dir


def _prepare_group_data(
    g_idx: int,
    g_start: int,
    g_end: int,
    all_triplets: list,
    groups: list,
    id2ent: dict,
    id2rel: dict,
    rf_root: str,
    group_size: int,
    dataset: str = "FinDKG",
) -> str:
    """Write train/valid/test split files for group g_idx into the Ruleformer DATASET dir.

    train = triplets in [g_start, g_end]
    valid = triplets in group g_idx + 1
    test  = triplets in group g_idx + 2

    Returns the dataset dir path.
    """
    tag = f"{dataset}_g{g_idx:03d}"
    dst_dir = os.path.join(rf_root, "DATASET", tag)
    os.makedirs(dst_dir, exist_ok=True)

    # Copy global vocab
    src_vocab_dir = os.path.join(rf_root, "DATASET", dataset)
    for fname in ("entities.txt", "relations.txt"):
        src = os.path.join(src_vocab_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            with open(src) as f:
                content = f.read()
            with open(dst, "w") as f:
                f.write(content)

    def _write(split_name, t_start, t_end):
        subset = filter_triplets_by_time(all_triplets, t_start, t_end)
        dst_path = os.path.join(dst_dir, f"{split_name}.txt")
        if subset:
            write_ruleformer_split_from_memory(subset, dst_path, id2ent, id2rel)
        else:
            # Ruleformer requires all three files to exist even if empty
            open(dst_path, "w").close()
            print(f"  Wrote {split_name}.txt (empty — no triplets in range)")

    _write("train", g_start, g_end)

    if g_idx + 1 < len(groups):
        vs, ve = groups[g_idx + 1]
        _write("valid", vs, ve)
    else:
        open(os.path.join(dst_dir, "valid.txt"), "w").close()

    if g_idx + 2 < len(groups):
        ts2, te2 = groups[g_idx + 2]
        _write("test", ts2, te2)
    else:
        open(os.path.join(dst_dir, "test.txt"), "w").close()

    return dst_dir, tag


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, cwd: str = None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")


def _find_ckpt(exps_dir: str, desc: str, epochs: int) -> str:
    """Find the latest checkpoint for a given group description."""
    import glob
    pattern = os.path.join(exps_dir, f"{desc}-j*", f"Translator{epochs}.ckpt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        matches = sorted(glob.glob(os.path.join(exps_dir, f"{desc}-j*", "Translator*.ckpt")))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for desc='{desc}' in {exps_dir}")
    return matches[-1]


def _find_rules_file(exps_dir: str, decode_desc: str) -> str:
    import glob
    matches = sorted(glob.glob(os.path.join(exps_dir, f"{decode_desc}-*", "rules.txt")))
    if not matches:
        raise FileNotFoundError(f"No rules.txt found for desc='{decode_desc}' in {exps_dir}")
    return matches[-1]


# ---------------------------------------------------------------------------
# Per-group pipeline
# ---------------------------------------------------------------------------

def process_group(
    g_idx: int,
    g_start: int,
    g_end: int,
    all_triplets: list,
    groups: list,
    id2ent: dict,
    id2rel: dict,
    ent2id: dict,
    rel2id: dict,
    rf_root: str,
    exps_path: str,
    group_size: int,
    jump: int = 2,
    padding: int = 100,
    maxn: int = 40,
    epochs: int = 10,
    batch_size: int = 16,
    n_head: int = 6,
    d_v: int = 64,
    n_layers: int = 2,
    dropout: float = 0.1,
    savestep: int = 5,
    the_rel: float = 0.3,
    the_rel_min: float = 0.1,
    the_all: float = 0.05,
    min_weight: float = 0.1,
    min_count: int = 1,
    dataset: str = "FinDKG",
    skip_if_exists: bool = True,
) -> list:
    """Run the full Ruleformer pipeline for one time group.

    Returns list of (h_id, r_id, t_id, timestamp) predictions tagged with g_end + 1.
    """
    python = sys.executable
    pred_timestamp = g_end + 1
    train_desc = f"g{g_idx:03d}"
    decode_desc = f"g{g_idx:03d}-rule"

    os.makedirs(exps_path, exist_ok=True)

    # Check if already processed
    import glob
    rules_glob = os.path.join(exps_path, f"{decode_desc}-*", "rules.txt")
    if skip_if_exists and glob.glob(rules_glob):
        rules_path = sorted(glob.glob(rules_glob))[-1]
        print(f"  [skip] rules.txt already exists: {rules_path}")
    else:
        # Step 1: write data files
        dst_dir, tag = _prepare_group_data(
            g_idx, g_start, g_end, all_triplets, groups,
            id2ent, id2rel, rf_root, group_size, dataset,
        )

        # Step 2: preprocess (build ego-subgraphs)
        _run(
            [python, "transformer/dataset.py",
             f"-data={tag}", f"-maxN={maxn}",
             f"-padding={padding}", f"-jump={jump}"],
            cwd=rf_root,
        )

        # Step 3: train
        _run(
            [python, "translate.py",
             f"-data=DATASET/{tag}",
             f"-jump={jump}", f"-padding={padding}",
             f"-batch_size={batch_size}", f"-epoch={epochs}",
             f"-n_head={n_head}", f"-d_v={d_v}", f"-n_layers={n_layers}",
             f"-dropout={dropout}", f"-desc={train_desc}",
             f"-savestep={savestep}", f"-exps={exps_path}/"],
            cwd=rf_root,
        )

        # Step 4: decode rules
        ckpt = _find_ckpt(exps_path, train_desc, epochs)
        _run(
            [python, "translate.py",
             f"-data=DATASET/{tag}",
             f"-jump={jump}", f"-padding={padding}",
             f"-batch_size={batch_size}", f"-epoch={epochs}",
             f"-n_head={n_head}", f"-d_v={d_v}", f"-n_layers={n_layers}",
             f"-desc={decode_desc}", f"-ckpt={ckpt}", "-decode_rule",
             f"-the_rel={the_rel}", f"-the_rel_min={the_rel_min}",
             f"-the_all={the_all}", f"-exps={exps_path}/"],
            cwd=rf_root,
        )

        rules_path = _find_rules_file(exps_path, decode_desc)

    # Step 5: apply rules to group G's training subgraph
    train_triplets = filter_triplets_by_time(all_triplets, g_start, g_end)
    hr_t, known = load_kg_from_triplets(train_triplets, id2ent, id2rel)

    rules = parse_rules(rules_path, min_weight=min_weight, min_count=min_count)
    print(f"  Rules loaded: {len(rules)}")

    predicted_names: set = set()
    for weight, count, head_rel, body_rels in rules:
        for triple in _apply_rule(head_rel, body_rels, hr_t, known):
            predicted_names.add(triple)

    # Step 6: map names → IDs and tag with pred_timestamp
    predictions = []
    skipped = 0
    for h_name, r_name, t_name in predicted_names:
        h_id = ent2id.get(h_name)
        r_id = rel2id.get(r_name)
        t_id = ent2id.get(t_name)
        if h_id is None or r_id is None or t_id is None:
            skipped += 1
            continue
        predictions.append((h_id, r_id, t_id, pred_timestamp))

    if skipped:
        print(f"  Skipped {skipped} predictions with unmapped names")
    print(f"  New predictions for t={pred_timestamp}: {len(predictions)}")
    return predictions


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_sym_triplets(predictions: list, output_path: str):
    """Write 4-column TSV: h_id \\t r_id \\t t_id \\t timestamp."""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    rows = [f"{h}\t{r}\t{t}\t{ts}" for h, r, t, ts in predictions]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Written {len(rows):,} rows → {output_path}")
