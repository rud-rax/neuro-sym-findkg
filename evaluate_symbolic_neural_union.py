"""
Evaluate coverage of symbolic and neural future-quadruple predictions against the gold split.

Usage:
    python evaluate_symbolic_neural_union.py \
        --checkpoint path/to/checkpoint.pt \
        --dataset FinDKG \
        --data_dir FinDKG_dataset \
        --split valid \
        --top_k 10 \
        --symbolic_predictions path/to/sym_triplets.tsv \
        --output_dir outputs/union_eval_valid

Symbolic predictions file format (tab-separated, no header):
    head_id<TAB>rel_id<TAB>tail_id<TAB>time

Coverage metrics:
    sym_coverage  = |sym ∩ gold| / |gold|
    neu_coverage  = |neu ∩ gold| / |gold|   (= Recall@top_k of neural model)
    union_coverage= |(sym ∪ neu) ∩ gold| / |gold|
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from functools import partial

import dgl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from DKG import data as dkg_data
from DKG import settings
import DKG.utils as dkg_utils
from DKG.model import (
    DynamicGraphModel,
    EmbeddingUpdater,
    Combiner,
    EdgeModel,
    InterEventTimeModel,
    MultiAspectEmbedding,
)
from DKG.model.time_interval_transform import TimeIntervalTransform
from DKG.utils.model_utils import get_embedding
from DKG.utils.train_utils import setup_cuda


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate symbolic + neural union coverage against a gold KG split."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--dataset", default="FinDKG",
                        choices=settings.ALL_GRAPHS,
                        help="Dataset / graph name (default: FinDKG)")
    parser.add_argument("--data_dir", required=True,
                        help="Data root directory (parent of <dataset>/ folder)")
    parser.add_argument("--split", choices=["valid", "test"], default="valid",
                        help="Target evaluation split (default: valid)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top-k neural candidates per query (default: 10)")
    parser.add_argument("--symbolic_predictions", required=True,
                        help="TSV file of symbolic predictions: head_id<TAB>rel_id<TAB>tail_id<TAB>time")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save output files")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU id (-1 for CPU, default: -1)")
    parser.add_argument("--model_type", default=None,
                        help="Override model type used in EmbeddingUpdater: "
                             "'KGT+RNN', 'KGT+GRU', 'RGCN+RNN', 'RGCN+GRU'. "
                             "Inferred from checkpoint args if not specified.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(ckpt_args, G, device, model_type_override=None):
    """Reconstruct DynamicGraphModel from checkpoint args and graph metadata."""
    num_nodes = G.number_of_nodes()
    num_rels = G.num_relations

    node_latest_event_time = torch.zeros(
        num_nodes, num_nodes + 1, 2, dtype=settings.INTER_EVENT_TIME_DTYPE
    )

    log_transform = getattr(ckpt_args, "time_interval_log_transform", True)
    time_interval_transform = TimeIntervalTransform(log_transform=log_transform)

    # Determine graph conv type for EmbeddingUpdater
    if model_type_override is not None:
        structural_gconv = model_type_override
        temporal_gconv = model_type_override
    else:
        # Prefer explicit attributes; fall back to KGT+RNN (FinDKG default)
        structural_gconv = getattr(
            ckpt_args, "embedding_updater_structural_gconv",
            getattr(ckpt_args, "model_type", "KGT+RNN")
        )
        temporal_gconv = getattr(
            ckpt_args, "embedding_updater_temporal_gconv",
            getattr(ckpt_args, "model_type", "KGT+RNN")
        )
        # DKG_DEFAULT_CONFIG stores 'RGCN+RNN' but train_DKG_run.py trains KGT+RNN by default.
        # If the stored value is the placeholder default and we're on FinDKG, prefer KGT+RNN.
        if structural_gconv == "RGCN+RNN" and getattr(ckpt_args, "graph", "") in ("FinDKG", "FinDKG-live"):
            print("[build_model] NOTE: checkpoint args show 'RGCN+RNN' but FinDKG typically "
                  "uses 'KGT+RNN'. Pass --model_type to override if reconstruction fails.")

    num_node_types = getattr(ckpt_args, "num_node_types", G.num_node_types)
    num_attn_heads = getattr(ckpt_args, "num_attn_heads", 8)

    embedding_updater = EmbeddingUpdater(
        num_nodes,
        ckpt_args.static_entity_embed_dim,
        ckpt_args.structural_dynamic_entity_embed_dim,
        ckpt_args.temporal_dynamic_entity_embed_dim,
        node_latest_event_time,
        num_rels,
        ckpt_args.rel_embed_dim,
        graph_structural_conv=structural_gconv,
        graph_temporal_conv=temporal_gconv,
        num_node_types=num_node_types,
        num_heads=num_attn_heads,
        num_gconv_layers=ckpt_args.num_gconv_layers,
        num_rnn_layers=ckpt_args.num_rnn_layers,
        time_interval_transform=time_interval_transform,
        dropout=ckpt_args.dropout,
        activation=ckpt_args.embedding_updater_activation,
        graph_name=ckpt_args.graph,
    ).to(device)

    combiner = Combiner(
        ckpt_args.static_entity_embed_dim,
        ckpt_args.structural_dynamic_entity_embed_dim,
        ckpt_args.static_dynamic_combine_mode,
        ckpt_args.combiner_gconv,
        num_rels,
        ckpt_args.dropout,
        ckpt_args.combiner_activation,
    ).to(device)

    edge_model = EdgeModel(
        num_nodes,
        num_rels,
        ckpt_args.rel_embed_dim,
        combiner,
        dropout=ckpt_args.dropout,
    ).to(device)

    inter_event_time_model = InterEventTimeModel(
        dynamic_entity_embed_dim=ckpt_args.temporal_dynamic_entity_embed_dim,
        static_entity_embed_dim=ckpt_args.static_entity_embed_dim,
        num_rels=num_rels,
        rel_embed_dim=ckpt_args.rel_embed_dim,
        num_mix_components=ckpt_args.num_mix_components,
        time_interval_transform=time_interval_transform,
        inter_event_time_mode=ckpt_args.inter_event_time_mode,
        dropout=ckpt_args.dropout,
    )

    model = DynamicGraphModel(
        embedding_updater, combiner, edge_model, inter_event_time_model, node_latest_event_time
    ).to(device)

    return model, node_latest_event_time


# ---------------------------------------------------------------------------
# Neural candidate generation
# ---------------------------------------------------------------------------

def generate_neural_candidates(
    model, data_loader, G,
    static_emb, init_entity_emb, init_rel_emb,
    device, top_k,
):
    """
    For each gold quadruple (s, r, o, t) in the target split, score all entities
    for query (s, r, ?, t) and collect the top-k as candidate quadruples.

    Returns a set of (head_id, rel_id, tail_id, time) integer tuples.

    Temporal leakage is avoided because:
    - Embeddings start from the pre-split state (train-end for valid, post-valid for test).
    - For each timestamp batch, predictions are made BEFORE updating embeddings with that batch.
    """
    model.eval()
    dynamic_entity_emb = init_entity_emb
    dynamic_relation_emb = init_rel_emb
    neu_triplets = set()

    with torch.no_grad():
        batch_tqdm = tqdm(data_loader, desc="Generating neural candidates")
        for prior_G, batch_G, cumul_G, batch_times in batch_tqdm:
            eval_time_from = batch_times[0]
            eval_time_to = batch_times[-1]

            # Move cumul_G to device first so eval_eid is on the same device
            # (EdgeModel.forward indexes edge_head/tail with eid, requiring same device)
            cumul_G_dev = cumul_G.to(device)

            eval_eid = torch.nonzero(
                (eval_time_from <= cumul_G_dev.edata["time"]) &
                (cumul_G_dev.edata["time"] <= eval_time_to),
                as_tuple=False,
            ).squeeze().view(-1)

            if len(eval_eid) > 0:
                # Gather embeddings for nodes present in cumul_G
                cumul_nids = cumul_G_dev.ndata[dgl.NID].long()

                struct_static = static_emb.structural[cumul_nids].to(device)
                struct_dynamic = dynamic_entity_emb.structural[cumul_nids][:, -1, :].to(device)
                combined_emb = model.combiner(struct_static, struct_dynamic, cumul_G_dev)
                rel_emb = dynamic_relation_emb.structural[:, -1, :, :].to(device)

                cumul_G_dev.ndata["emb"] = combined_emb

                # edge_model returns (edge_LL, head_pred, rel_pred, tail_pred)
                # tail_pred shape: (len(eval_eid), num_all_entities) — global entity space
                _, _, _, tail_pred = model.edge_model(
                    cumul_G_dev,
                    combined_emb,
                    struct_static,
                    struct_dynamic,
                    rel_emb,
                    eid=eval_eid,
                    return_pred=True,
                )

                # top-k entity indices are already in global entity ID space
                actual_k = min(top_k, tail_pred.size(1))
                top_k_ids = torch.topk(tail_pred, actual_k, dim=1).indices.cpu()  # (eval_edges, k)

                # Recover global src IDs, relation types, timestamps using CPU tensors
                eval_eid_cpu = eval_eid.cpu()
                cumul_src, _ = cumul_G.edges()
                eval_src_local = cumul_src[eval_eid_cpu.long()]
                eval_src_global = cumul_G.ndata[dgl.NID][eval_src_local].long()
                eval_rel = cumul_G.edata["rel_type"][eval_eid_cpu].long()
                eval_time = cumul_G.edata["time"][eval_eid_cpu]

                for s, r, t, candidates in zip(
                    eval_src_global.tolist(),
                    eval_rel.tolist(),
                    eval_time.tolist(),
                    top_k_ids.tolist(),
                ):
                    t_int = int(t)
                    for o_hat in candidates:
                        neu_triplets.add((int(s), int(r), int(o_hat), t_int))

            # Update embeddings AFTER making predictions for this batch (no leakage).
            # Pass CPU cumul_G — embedding_updater asserts embeddings are on CPU.
            dynamic_entity_emb, dynamic_relation_emb = model.embedding_updater.forward(
                prior_G, batch_G, cumul_G, static_emb,
                dynamic_entity_emb, dynamic_relation_emb, device,
            )

    return neu_triplets


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_symbolic_predictions(path):
    """Load symbolic predictions from TSV: head_id<TAB>rel_id<TAB>tail_id<TAB>time."""
    df = pd.read_table(path, sep="\t", header=None, names=["head", "rel", "tail", "time"])
    return {(int(r.head), int(r.rel), int(r.tail), int(r.time)) for r in df.itertuples(index=False)}


def extract_gold_quads(G, split):
    """Extract gold quadruples from the DGL graph for the given split."""
    mask = G.edata["val_mask"] if split == "valid" else G.edata["test_mask"]
    eids = torch.nonzero(mask, as_tuple=False).squeeze().view(-1)
    src, dst = G.edges()
    gold = set()
    for e in eids.tolist():
        s = int(src[e])
        r = int(G.edata["rel_type"][e])
        o = int(dst[e])
        t = int(G.edata["time"][e])
        gold.add((s, r, o, t))
    return gold


def load_id2entity(data_dir, dataset):
    """Return {entity_id: entity_name} dict from entity2id.txt."""
    path = os.path.join(data_dir, dataset, "entity2id.txt")
    df = pd.read_table(path, sep="\t", header=None, names=["name", "id", "ntype", "ntype_id"])
    return dict(zip(df["id"].astype(int), df["name"]))


def load_id2relation(data_dir, dataset):
    """Return {relation_id: relation_name} dict from relation2id.txt."""
    path = os.path.join(data_dir, dataset, "relation2id.txt")
    df = pd.read_table(path, sep="\t", header=None, names=["name", "id"])
    return dict(zip(df["id"].astype(int), df["name"]))


# ---------------------------------------------------------------------------
# Coverage metrics
# ---------------------------------------------------------------------------

def compute_coverage(gold, sym, neu):
    n = len(gold)
    if n == 0:
        return {"error": "gold set is empty"}
    sym_hit = gold & sym
    neu_hit = gold & neu
    union_hit = gold & (sym | neu)
    return {
        "num_gold": n,
        "sym_count": len(sym),
        "neu_count": len(neu),
        "sym_hit": len(sym_hit),
        "neu_hit": len(neu_hit),
        "union_hit": len(union_hit),
        "sym_coverage": len(sym_hit) / n,
        "neu_coverage": len(neu_hit) / n,
        "union_coverage": len(union_hit) / n,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def quads_to_df(quads, id2entity, id2relation):
    """Convert a set of (s, r, o, t) int tuples to a DataFrame with name columns."""
    rows = [
        {
            "head_id": s, "rel_id": r, "tail_id": o, "time": t,
            "head": id2entity.get(s, str(s)),
            "relation": id2relation.get(r, str(r)),
            "tail": id2entity.get(o, str(o)),
        }
        for s, r, o, t in sorted(quads)
    ]
    return pd.DataFrame(rows, columns=["head_id", "rel_id", "tail_id", "time", "head", "relation", "tail"])


def save_tsv(df, path):
    df.to_csv(path, sep="\t", index=False)
    print(f"  Saved {len(df):,} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]
    loss_weights = checkpoint.get("loss_weights", (1.0, 0.0))

    # Device
    ckpt_args.gpu = args.gpu
    ckpt_args = setup_cuda(ckpt_args)
    device = ckpt_args.device
    print(f"      Device: {device}")

    # ------------------------------------------------------------------
    # 2. Load graph
    # ------------------------------------------------------------------
    print(f"\n[2/6] Loading graph '{args.dataset}' from {args.data_dir}")
    G = dkg_data.load_temporal_knowledge_graph(args.dataset, data_root=args.data_dir)
    print(f"      Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}, Relations={G.num_relations}")

    collate = partial(dkg_utils.collate_fn, G=G)
    if args.split == "valid":
        target_loader = DataLoader(G.val_times, shuffle=False, collate_fn=collate)
    else:
        target_loader = DataLoader(G.test_times, shuffle=False, collate_fn=collate)
    print(f"      Split '{args.split}': {len(G.val_times if args.split == 'valid' else G.test_times)} timestamps")

    # ------------------------------------------------------------------
    # 3. Build and load model
    # ------------------------------------------------------------------
    print(f"\n[3/6] Reconstructing model")
    model, node_latest_event_time = build_model(ckpt_args, G, device, args.model_type)

    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys in state_dict (architecture mismatch?).")
        print(f"    First few: {missing[:5]}")
        print("    Try --model_type KGT+RNN or --model_type RGCN+RNN to resolve.")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys in state_dict.")
    if not missing and not unexpected:
        print("  Model loaded successfully (exact match).")

    model.eval()

    # ------------------------------------------------------------------
    # 4. Select embedding state based on split
    # ------------------------------------------------------------------
    print(f"\n[4/6] Selecting embedding state for split='{args.split}'")
    static_emb = checkpoint["static_entity_emb"]

    if args.split == "valid":
        # Use train-end embeddings; temporal state starts at zero (no leakage into valid)
        init_entity_emb = checkpoint["init_dynamic_entity_emb"]
        init_rel_emb = checkpoint["init_dynamic_relation_emb"]
        model.node_latest_event_time.zero_()
        node_latest_event_time.zero_()
        print("  Using init_dynamic_entity_emb (train-end state); node_latest_event_time reset to zero.")
    else:
        # Use post-validation embeddings (mirrors existing test evaluation in train_DKG_run.py)
        init_entity_emb = checkpoint["val_dynamic_entity_emb"]
        init_rel_emb = checkpoint["val_dynamic_relation_emb"]
        saved_nlet = checkpoint["node_latest_event_time"]
        if saved_nlet.is_sparse:
            saved_nlet = saved_nlet.to_dense()
        model.node_latest_event_time.copy_(saved_nlet)
        node_latest_event_time.copy_(saved_nlet)
        print("  Using val_dynamic_entity_emb (post-validation state); node_latest_event_time restored.")

    # ------------------------------------------------------------------
    # 5. Generate neural candidates
    # ------------------------------------------------------------------
    print(f"\n[5/6] Generating neural top-{args.top_k} candidates for split='{args.split}'")
    neu_triplets = generate_neural_candidates(
        model, target_loader, G,
        static_emb, init_entity_emb, init_rel_emb,
        device, args.top_k,
    )
    print(f"  Neural candidates generated: {len(neu_triplets):,} unique quadruples")

    # ------------------------------------------------------------------
    # 6. Load symbolic predictions, gold quads, compute coverage
    # ------------------------------------------------------------------
    print(f"\n[6/6] Computing coverage metrics")

    sym_triplets = load_symbolic_predictions(args.symbolic_predictions)
    print(f"  Symbolic predictions loaded: {len(sym_triplets):,} quadruples")

    gold_quads = extract_gold_quads(G, args.split)
    print(f"  Gold quadruples ({args.split}): {len(gold_quads):,}")

    metrics = compute_coverage(gold_quads, sym_triplets, neu_triplets)

    print(f"\n{'='*55}")
    print(f"  Split:          {args.split}")
    print(f"  Top-k:          {args.top_k}")
    print(f"  Gold facts:     {metrics['num_gold']:,}")
    print(f"  Sym coverage:   {metrics['sym_coverage']:.4f}  ({metrics['sym_hit']:,}/{metrics['num_gold']:,})")
    print(f"  Neu coverage:   {metrics['neu_coverage']:.4f}  ({metrics['neu_hit']:,}/{metrics['num_gold']:,})")
    print(f"  Union coverage: {metrics['union_coverage']:.4f}  ({metrics['union_hit']:,}/{metrics['num_gold']:,})")
    print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    id2entity = load_id2entity(args.data_dir, args.dataset)
    id2relation = load_id2relation(args.data_dir, args.dataset)

    # metrics.json
    metrics_out = {
        **metrics,
        "split": args.split,
        "top_k": args.top_k,
        "checkpoint": os.path.abspath(args.checkpoint),
        "symbolic_predictions": os.path.abspath(args.symbolic_predictions),
        "dataset": args.dataset,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Saved metrics → {metrics_path}")

    # TSV files with ID + name columns
    save_tsv(quads_to_df(gold_quads, id2entity, id2relation),
             os.path.join(args.output_dir, "gold_quads.tsv"))

    save_tsv(quads_to_df(neu_triplets, id2entity, id2relation),
             os.path.join(args.output_dir, "neural_candidates.tsv"))

    save_tsv(quads_to_df(sym_triplets, id2entity, id2relation),
             os.path.join(args.output_dir, "symbolic_predictions_used.tsv"))

    save_tsv(quads_to_df(sym_triplets | neu_triplets, id2entity, id2relation),
             os.path.join(args.output_dir, "union_candidates.tsv"))

    print(f"\nAll outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
