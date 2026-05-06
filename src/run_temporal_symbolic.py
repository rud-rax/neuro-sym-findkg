"""
Temporal Ruleformer pipeline — script version of temporal_ruleformer.ipynb.

Divides the FinDKG dataset into consecutive time groups, trains a fresh
Ruleformer model per group, decodes rules, applies them to the group's
subgraph, and writes all predictions to sym_triplets.csv (4 columns:
h_id  r_id  t_id  timestamp).

Usage:
    python run_temporal_symbolic.py \
        --dataset_path FinDKG_dataset/FinDKG \
        --rf_root      ruleformer-findkg \
        --exps_path    EXPS_temporal \
        --output       sym_triplets.csv

Resume after interruption:
    Re-run the same command. Groups whose rules.txt already exists are skipped.

Force rerun a specific group range:
    --skip_if_exists False
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Temporal Ruleformer pipeline for FinDKG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--dataset_path", default="FinDKG_dataset/FinDKG",
                   help="Path to FinDKG dataset dir (contains train.txt, entity2id.txt, …)")
    p.add_argument("--rf_root", default="ruleformer-findkg",
                   help="Path to ruleformer-findkg repo root")
    p.add_argument("--exps_path", default="EXPS_temporal",
                   help="Directory for checkpoints and rules (created if missing)")
    p.add_argument("--output", default="sym_triplets.csv",
                   help="Output file path (4-column TSV: h_id r_id t_id timestamp)")

    # Grouping
    p.add_argument("--group_size", type=int, default=10,
                   help="Number of timestamps per group")
    p.add_argument("--groups_limit", type=int, default=None,
                   help="Process only the first N groups (useful for testing)")

    # Ruleformer hyperparameters
    p.add_argument("--jump",       type=int,   default=2)
    p.add_argument("--padding",    type=int,   default=100)
    p.add_argument("--maxn",       type=int,   default=40)
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--n_head",     type=int,   default=6)
    p.add_argument("--d_v",        type=int,   default=64)
    p.add_argument("--n_layers",   type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--savestep",   type=int,   default=5)

    # Rule decoding thresholds
    p.add_argument("--the_rel",     type=float, default=0.3)
    p.add_argument("--the_rel_min", type=float, default=0.1)
    p.add_argument("--the_all",     type=float, default=0.05)
    p.add_argument("--min_weight",  type=float, default=0.1)
    p.add_argument("--min_count",   type=int,   default=1)

    # Misc
    p.add_argument("--skip_if_exists", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip groups whose rules.txt already exists")
    p.add_argument("--dataset", default="FinDKG",
                   help="Dataset tag used for Ruleformer DATASET subdirectory names")

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(script_dir, p)

    dataset_path = _abs(args.dataset_path)
    rf_root      = _abs(args.rf_root)
    exps_path    = _abs(args.exps_path)
    output       = _abs(args.output)

    # Make sure FinDKG modules are importable
    sys.path.insert(0, script_dir)

    from DKG.symbolic.temporal_pipeline import (
        load_all_triplets,
        load_id2name,
        write_global_vocab,
        make_groups,
        process_group,
        write_sym_triplets,
    )
    from tqdm.auto import tqdm

    os.makedirs(exps_path, exist_ok=True)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print("\n[1/4] Loading triplets...")
    all_triplets = load_all_triplets(dataset_path)
    print(f"  Total triplets: {len(all_triplets):,}")

    id2ent, id2rel = load_id2name(dataset_path)
    ent2id = {v: k for k, v in id2ent.items()}
    rel2id = {v: k for k, v in id2rel.items()}
    print(f"  Entities: {len(id2ent):,}  |  Relations: {len(id2rel):,}")

    # ── Step 2: Write global vocab ───────────────────────────────────────────
    print("\n[2/4] Writing global vocabulary to Ruleformer DATASET dir...")
    write_global_vocab(id2ent, id2rel, rf_root, dataset=args.dataset)

    # ── Step 3: Build groups ─────────────────────────────────────────────────
    print("\n[3/4] Building time groups...")
    groups = make_groups(all_triplets, args.group_size)
    if args.groups_limit:
        groups = groups[:args.groups_limit]
        print(f"  (limited to first {args.groups_limit} groups via --groups_limit)")

    print(f"  Total groups: {len(groups)}")
    for i, (g_start, g_end) in enumerate(groups):
        count = sum(1 for t in all_triplets if g_start <= t[3] <= g_end)
        print(f"    Group {i:3d}: t={g_start:3d}–{g_end:3d}  ({count:,} triplets)")

    # ── Step 4: Process each group ───────────────────────────────────────────
    print("\n[4/4] Processing groups...\n")
    all_predictions = []

    for g_idx, (g_start, g_end) in enumerate(tqdm(groups, desc="All groups")):
        print(f"\n{'='*60}")
        print(f"GROUP {g_idx}: timestamps {g_start}–{g_end}")
        print(f"{'='*60}")

        preds = process_group(
            g_idx=g_idx,
            g_start=g_start,
            g_end=g_end,
            all_triplets=all_triplets,
            groups=groups,
            id2ent=id2ent,
            id2rel=id2rel,
            ent2id=ent2id,
            rel2id=rel2id,
            rf_root=rf_root,
            exps_path=exps_path,
            group_size=args.group_size,
            jump=args.jump,
            padding=args.padding,
            maxn=args.maxn,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_head=args.n_head,
            d_v=args.d_v,
            n_layers=args.n_layers,
            dropout=args.dropout,
            savestep=args.savestep,
            the_rel=args.the_rel,
            the_rel_min=args.the_rel_min,
            the_all=args.the_all,
            min_weight=args.min_weight,
            min_count=args.min_count,
            dataset=args.dataset,
            skip_if_exists=args.skip_if_exists,
        )

        print(f"  → {len(preds):,} predictions for t={g_end + 1}")
        all_predictions.extend(preds)

    # ── Write output ─────────────────────────────────────────────────────────
    print(f"\nTotal predictions across all groups: {len(all_predictions):,}")
    write_sym_triplets(all_predictions, output)

    # Summary by timestamp
    from collections import Counter
    ts_counts = Counter(p[3] for p in all_predictions)
    print("\nPredictions per timestamp:")
    for ts in sorted(ts_counts):
        print(f"  t={ts}: {ts_counts[ts]:,}")


if __name__ == "__main__":
    main()
