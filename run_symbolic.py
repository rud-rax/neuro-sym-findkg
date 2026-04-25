"""
run_symbolic.py — Ruleformer symbolic rule mining pipeline for FinDKG.

Steps:
  1. Adapt FinDKG data to Ruleformer name-based format
  2. Preprocess: build ego-centered subgraphs  (Ruleformer/transformer/dataset.py)
  3. Train Ruleformer                           (Ruleformer/translate.py)
  4. Decode mined rules                         (Ruleformer/translate.py -decode_rule)
  5. Apply rules to training KG → sym_triplets.tsv

Usage:
    python run_symbolic.py \\
        --data_dir FinDKG_dataset \\
        --dataset FinDKG \\
        --output sym_triplets.tsv \\
        --jump 2 --padding 200 --maxN 100 \\
        --epochs 50

Skip flags (useful to resume a partial run):
    --skip_adapt --skip_preprocess --skip_train --skip_decode

Provide explicit paths to skip auto-discovery:
    --ckpt  EXPS/findkg-j2-.../Translator50.ckpt
    --rules_file  EXPS/findkg-rule-.../rules.txt
"""

import argparse
import glob
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_RULEFORMER_ROOT = os.path.join(_REPO_ROOT, "Ruleformer")

# Import our helper modules (repo root must be on path)
sys.path.insert(0, _REPO_ROOT)
from DKG.symbolic.ruleformer_adapter import adapt_findkg_for_ruleformer
from DKG.symbolic.apply_rules import apply_rules_to_kg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, cwd: str = None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"Command failed (exit {result.returncode})")


def _find_latest_ckpt(exps_dir: str, desc: str, epoch: int) -> str:
    pattern = os.path.join(exps_dir, f"{desc}-j*", f"Translator{epoch}.ckpt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        # fall back: any epoch, any timestamp
        matches = sorted(glob.glob(os.path.join(exps_dir, f"{desc}-j*", "Translator*.ckpt")))
    if not matches:
        sys.exit(
            f"No checkpoint found matching '{desc}-j*' in {exps_dir}.\n"
            "Pass --ckpt explicitly or check that training completed."
        )
    return matches[-1]


def _find_rules_file(exps_dir: str, decode_desc: str) -> str:
    matches = sorted(glob.glob(os.path.join(exps_dir, f"{decode_desc}-*", "rules.txt")))
    if not matches:
        sys.exit(
            f"No rules.txt found matching '{decode_desc}-*' in {exps_dir}.\n"
            "Pass --rules_file explicitly or check that decoding completed."
        )
    return matches[-1]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Full Ruleformer symbolic pipeline for FinDKG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--data_dir", default="FinDKG_dataset")
    p.add_argument("--dataset", default="FinDKG")
    p.add_argument("--output", default="sym_triplets.tsv")

    # Ruleformer hyperparameters
    p.add_argument("--jump", type=int, default=2, help="Rule length / subgraph hops")
    p.add_argument("--padding", type=int, default=200, help="Max subgraph size")
    p.add_argument("--maxN", type=int, default=100, help="Max neighbours per relation")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--d_v", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--savestep", type=int, default=10)

    # Rule decoding thresholds
    p.add_argument("--the_rel", type=float, default=0.5)
    p.add_argument("--the_rel_min", type=float, default=0.2)
    p.add_argument("--the_all", type=float, default=0.05)

    # Rule application filters
    p.add_argument("--min_rule_weight", type=float, default=0.1)
    p.add_argument("--min_rule_count", type=int, default=3)

    # Skip flags
    p.add_argument("--skip_adapt", action="store_true")
    p.add_argument("--skip_preprocess", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_decode", action="store_true")

    # Explicit overrides
    p.add_argument("--ckpt", default=None, help="Checkpoint path (skips auto-discovery)")
    p.add_argument("--rules_file", default=None, help="rules.txt path (skips auto-discovery)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    python = sys.executable
    ruleformer_data = os.path.join(_RULEFORMER_ROOT, "DATASET", args.dataset)
    exps_dir = os.path.join(_RULEFORMER_ROOT, "EXPS")
    train_desc = "findkg"
    decode_desc = "findkg-rule"

    # ── Step 1: Adapt data ──────────────────────────────────────────────────
    if not args.skip_adapt:
        print("\n[1/5] Adapting FinDKG data to Ruleformer format...")
        adapt_findkg_for_ruleformer(args.data_dir, args.dataset, _RULEFORMER_ROOT)
    else:
        print("\n[1/5] Skipping data adaptation.")

    # ── Step 2: Preprocess subgraphs ────────────────────────────────────────
    if not args.skip_preprocess:
        print("\n[2/5] Preprocessing: building ego-centred subgraphs...")
        _run(
            [python, "transformer/dataset.py",
             f"-data=DATASET/{args.dataset}",
             f"-maxN={args.maxN}",
             f"-padding={args.padding}",
             f"-jump={args.jump}"],
            cwd=_RULEFORMER_ROOT,
        )
    else:
        print("\n[2/5] Skipping preprocessing.")

    # ── Step 3: Train Ruleformer ────────────────────────────────────────────
    if not args.skip_train:
        print("\n[3/5] Training Ruleformer...")
        _run(
            [python, "translate.py",
             f"-data=DATASET/{args.dataset}",
             f"-jump={args.jump}",
             f"-padding={args.padding}",
             f"-batch_size={args.batch_size}",
             f"-epoch={args.epochs}",
             f"-n_head={args.n_head}",
             f"-d_v={args.d_v}",
             f"-n_layers={args.n_layers}",
             f"-dropout={args.dropout}",
             f"-desc={train_desc}",
             f"-savestep={args.savestep}"],
            cwd=_RULEFORMER_ROOT,
        )
    else:
        print("\n[3/5] Skipping training.")

    # ── Step 4: Decode rules ────────────────────────────────────────────────
    if not args.skip_decode:
        ckpt = args.ckpt or _find_latest_ckpt(exps_dir, train_desc, args.epochs)
        print(f"\n[4/5] Decoding rules from: {ckpt}")
        _run(
            [python, "translate.py",
             f"-data=DATASET/{args.dataset}",
             f"-jump={args.jump}",
             f"-padding={args.padding}",
             f"-batch_size={args.batch_size}",
             f"-epoch={args.epochs}",
             f"-n_head={args.n_head}",
             f"-d_v={args.d_v}",
             f"-n_layers={args.n_layers}",
             f"-desc={decode_desc}",
             f"-ckpt={ckpt}",
             "-decode_rule",
             f"-the_rel={args.the_rel}",
             f"-the_rel_min={args.the_rel_min}",
             f"-the_all={args.the_all}"],
            cwd=_RULEFORMER_ROOT,
        )
    else:
        print("\n[4/5] Skipping rule decoding.")

    # ── Step 5: Apply rules → sym_triplets.tsv ──────────────────────────────
    rules_file = args.rules_file or _find_rules_file(exps_dir, decode_desc)
    print(f"\n[5/5] Applying rules: {rules_file} → {args.output}")
    n = apply_rules_to_kg(
        rules_path=rules_file,
        ruleformer_train_path=os.path.join(ruleformer_data, "train.txt"),
        data_dir=args.data_dir,
        dataset=args.dataset,
        output_path=args.output,
        min_weight=args.min_rule_weight,
        min_count=args.min_rule_count,
    )

    print(f"\n{'='*55}")
    print(f"  Symbolic predictions written: {n:,}")
    print(f"  Output: {args.output}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
