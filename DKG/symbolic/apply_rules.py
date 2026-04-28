"""
Parse rules.txt produced by Ruleformer and apply them to a FinDKG knowledge graph.

Two modes:

1. Static (original): apply rules to the full training graph → sym_triplets.tsv
       python -m DKG.symbolic.apply_rules \
           --rules_file rules.txt \
           --ruleformer_train Ruleformer/DATASET/FinDKG/train.txt \
           --data_dir FinDKG_dataset --dataset FinDKG \
           --output sym_triplets.tsv

2. Temporal (new): answer (s, r, ?, t) queries using only prior_G (edges before t)
       python -m DKG.symbolic.apply_rules \
           --temporal \
           --rules_file rules.txt \
           --query_file FinDKG_dataset/FinDKG/valid.txt \
           --background_files FinDKG_dataset/FinDKG/train.txt \
           --data_dir FinDKG_dataset --dataset FinDKG \
           --output sym_triplets.tsv \
           --top_k 10 --scoring confidence_sum \
           [--eval]
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# KG loading (static mode)
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
# Rule application (static mode)
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
# Static pipeline
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
# Temporal inference helpers
# ---------------------------------------------------------------------------

def _load_id2name(data_dir: str, dataset: str):
    """Return (id2ent, id2rel, ent2id, rel2id) from FinDKG mapping files."""
    src_dir = os.path.join(data_dir, dataset)
    ent_df = pd.read_table(
        os.path.join(src_dir, "entity2id.txt"),
        header=None, names=["name", "id", "ntype", "ntype_id"],
    )
    rel_df = pd.read_table(
        os.path.join(src_dir, "relation2id.txt"),
        header=None, names=["name", "id"],
    )
    id2ent = dict(zip(ent_df["id"].astype(int), ent_df["name"]))
    id2rel = dict(zip(rel_df["id"].astype(int), rel_df["name"]))
    ent2id = dict(zip(ent_df["name"], ent_df["id"].astype(int)))
    rel2id = dict(zip(rel_df["name"], rel_df["id"].astype(int)))
    return id2ent, id2rel, ent2id, rel2id


def build_temporal_hr_t(background_paths: list, id2ent: dict, id2rel: dict) -> dict:
    """Build temporal edge lookup from one or more FinDKG split files.

    Args:
        background_paths: FinDKG-format files (head_id, rel_id, tail_id, time, _)
        id2ent: id → entity name
        id2rel: id → relation name

    Returns:
        {(head_name, rel_name): [(tail_name, edge_time), ...]}  sorted by edge_time asc
    """
    temporal_hr_t: dict = defaultdict(list)
    for path in background_paths:
        df = pd.read_table(path, header=None, names=["head", "rel", "tail", "time", "_"])
        for row in df.itertuples(index=False):
            h = id2ent.get(int(row.head))
            r = id2rel.get(int(row.rel))
            t = id2ent.get(int(row.tail))
            if h and r and t:
                temporal_hr_t[(h, r)].append((t, float(row.time)))
    for key in temporal_hr_t:
        temporal_hr_t[key].sort(key=lambda x: x[1])
    return dict(temporal_hr_t)


def _apply_rule_from_entity(
    s_name: str,
    body_rels: list,
    temporal_hr_t: dict,
    t_cutoff: float,
) -> set:
    """Chain-follow body_rels from s_name using only edges with edge_time < t_cutoff (prior_G).

    Returns the set of reachable tail entity names.
    """
    frontier = {s_name}
    for rel in body_rels:
        next_frontier = set()
        for node in frontier:
            for (tail, edge_time) in temporal_hr_t.get((node, rel), []):
                if edge_time < t_cutoff:
                    next_frontier.add(tail)
        frontier = next_frontier
        if not frontier:
            break
    return frontier


def score_candidates(rule_hits: dict, method: str) -> dict:
    """Aggregate per-candidate rule confidences into a single score.

    Args:
        rule_hits: {candidate_name: [confidence, ...]}
        method: 'max_confidence' | 'hit_count' | 'confidence_sum'

    Returns:
        {candidate_name: score}
    """
    scores = {}
    for entity, confs in rule_hits.items():
        if method == 'max_confidence':
            scores[entity] = max(confs)
        elif method == 'hit_count':
            scores[entity] = len(confs)
        elif method == 'confidence_sum':
            scores[entity] = sum(confs)
        else:
            raise ValueError(f"Unknown scoring method: {method!r}")
    return scores


# ---------------------------------------------------------------------------
# Temporal pipeline
# ---------------------------------------------------------------------------

def generate_temporal_sym_triplets(
    rules_path: str,
    query_path: str,
    background_paths: list,
    data_dir: str,
    dataset: str,
    output_path: str,
    top_k: int = 10,
    scoring: str = 'max_confidence',
    min_weight: float = 0.1,
    min_count: int = 3,
) -> int:
    """Apply Ruleformer rules to (s, r, ?, t) queries using prior_G.

    For each query in query_path:
      - prior_G = edges from background_paths with edge_time < query_time
      - chain-follow rules from the query subject
      - rank top-k candidates by scoring method
      - write (s_id, r_id, o_id, time) to output_path

    Background graph:
      - validation queries: background_paths = [train.txt]
      - test queries:       background_paths = [train.txt, valid.txt]

    Returns the number of rows written.
    """
    print(f"[temporal_apply] Loading ID mappings …")
    id2ent, id2rel, ent2id, _rel2id = _load_id2name(data_dir, dataset)

    print(f"[temporal_apply] Building temporal prior graph from: {background_paths}")
    temporal_hr_t = build_temporal_hr_t(background_paths, id2ent, id2rel)
    print(f"  Unique (head, rel) pairs in background: {len(temporal_hr_t):,}")

    print(f"[temporal_apply] Parsing rules: {rules_path}")
    rules = parse_rules(rules_path, min_weight=min_weight, min_count=min_count)
    print(f"  Rules after filtering: {len(rules)}")

    rules_by_head: dict = defaultdict(list)
    for weight, _count, head_rel, body_rels in rules:
        rules_by_head[head_rel].append((weight, body_rels))

    print(f"[temporal_apply] Loading queries: {query_path}")
    qdf = pd.read_table(query_path, header=None, names=["head", "rel", "tail", "time", "_"])
    print(f"  Total queries: {len(qdf):,}")

    rows = []
    num_answered = 0

    for row in qdf.itertuples(index=False):
        s_id   = int(row.head)
        r_id   = int(row.rel)
        t      = float(row.time)
        s_name = id2ent.get(s_id)
        r_name = id2rel.get(r_id)
        if not s_name or not r_name:
            continue
        if r_name not in rules_by_head:
            continue

        rule_hits: dict = defaultdict(list)
        for (weight, body_rels) in rules_by_head[r_name]:
            for cand in _apply_rule_from_entity(s_name, body_rels, temporal_hr_t, t_cutoff=t):
                rule_hits[cand].append(weight)

        if not rule_hits:
            continue

        scores = score_candidates(rule_hits, method=scoring)
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        num_answered += 1

        for (o_name, _score) in top:
            o_id = ent2id.get(o_name)
            if o_id is not None:
                rows.append(f"{s_id}\t{r_id}\t{o_id}\t{int(t)}")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"  Queries answered: {num_answered:,} / {len(qdf):,}")
    print(f"  Total prediction rows: {len(rows):,} → {output_path}")
    return len(rows)


def evaluate_temporal_symbolic_predictions(
    rules_path: str,
    query_path: str,
    background_paths: list,
    data_dir: str,
    dataset: str,
    scoring: str = 'max_confidence',
    min_weight: float = 0.1,
    min_count: int = 3,
) -> dict:
    """Evaluate symbolic predictions using MRR and Hits@{1,3,10}.

    For each query (s, r, o_true, t): apply rules on prior_G, rank all candidates,
    find the rank of o_true. Aggregates over all queries.

    Returns:
        {'MRR': float, 'Hits@1': float, 'Hits@3': float, 'Hits@10': float,
         'num_queries': int, 'num_answered': int}
    """
    id2ent, id2rel, _ent2id, _rel2id = _load_id2name(data_dir, dataset)
    temporal_hr_t = build_temporal_hr_t(background_paths, id2ent, id2rel)
    rules = parse_rules(rules_path, min_weight=min_weight, min_count=min_count)

    rules_by_head: dict = defaultdict(list)
    for weight, _count, head_rel, body_rels in rules:
        rules_by_head[head_rel].append((weight, body_rels))

    qdf = pd.read_table(query_path, header=None, names=["head", "rel", "tail", "time", "_"])
    num_queries = len(qdf)

    mrr_total = 0.0
    hits = {1: 0, 3: 0, 10: 0}
    num_answered = 0

    for row in qdf.itertuples(index=False):
        s_id        = int(row.head)
        r_id        = int(row.rel)
        o_true_id   = int(row.tail)
        t           = float(row.time)
        s_name      = id2ent.get(s_id)
        r_name      = id2rel.get(r_id)
        o_true_name = id2ent.get(o_true_id)
        if not s_name or not r_name or not o_true_name:
            continue
        if r_name not in rules_by_head:
            continue

        rule_hits: dict = defaultdict(list)
        for (weight, body_rels) in rules_by_head[r_name]:
            for cand in _apply_rule_from_entity(s_name, body_rels, temporal_hr_t, t_cutoff=t):
                rule_hits[cand].append(weight)

        if not rule_hits:
            continue

        scores = score_candidates(rule_hits, method=scoring)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        entity_rank = {name: i + 1 for i, (name, _) in enumerate(ranked)}

        rank = entity_rank.get(o_true_name)
        if rank is None:
            continue

        num_answered += 1
        mrr_total += 1.0 / rank
        for k in (1, 3, 10):
            if rank <= k:
                hits[k] += 1

    denom = num_queries if num_queries else 1
    return {
        'MRR':          mrr_total / denom,
        'Hits@1':       hits[1]   / denom,
        'Hits@3':       hits[3]   / denom,
        'Hits@10':      hits[10]  / denom,
        'num_queries':  num_queries,
        'num_answered': num_answered,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply Ruleformer rules to FinDKG — static or temporal mode"
    )

    # Mode
    parser.add_argument("--temporal", action="store_true", default=False,
                        help="use temporal mode: answer (s,r,?,t) queries using prior_G")

    # Shared
    parser.add_argument("--rules_file", required=True, help="Path to Ruleformer rules.txt")
    parser.add_argument("--data_dir", default="FinDKG_dataset")
    parser.add_argument("--dataset", default="FinDKG")
    parser.add_argument("--output", default="sym_triplets.tsv")
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--min_count", type=int, default=3)

    # Static mode
    parser.add_argument("--ruleformer_train", default="Ruleformer/DATASET/FinDKG/train.txt",
                        help="[static] Ruleformer name-based training file")

    # Temporal mode
    parser.add_argument("--query_file", default=None,
                        help="[temporal] FinDKG split to use as queries (valid.txt or test.txt)")
    parser.add_argument("--background_files", nargs="+", default=None,
                        help="[temporal] FinDKG splits for prior_G (e.g. train.txt for val; "
                             "train.txt valid.txt for test)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="[temporal] top-k predictions per query")
    parser.add_argument("--scoring", default="max_confidence",
                        choices=["max_confidence", "hit_count", "confidence_sum"],
                        help="[temporal] candidate scoring method")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="[temporal] also compute MRR/Hits@N against true answers")

    args = parser.parse_args()

    if args.temporal:
        if not args.query_file or not args.background_files:
            parser.error("--temporal requires --query_file and --background_files")

        generate_temporal_sym_triplets(
            rules_path=args.rules_file,
            query_path=args.query_file,
            background_paths=args.background_files,
            data_dir=args.data_dir,
            dataset=args.dataset,
            output_path=args.output,
            top_k=args.top_k,
            scoring=args.scoring,
            min_weight=args.min_weight,
            min_count=args.min_count,
        )

        if args.eval:
            print(f"\n[eval] Scoring method: {args.scoring}")
            metrics = evaluate_temporal_symbolic_predictions(
                rules_path=args.rules_file,
                query_path=args.query_file,
                background_paths=args.background_files,
                data_dir=args.data_dir,
                dataset=args.dataset,
                scoring=args.scoring,
                min_weight=args.min_weight,
                min_count=args.min_count,
            )
            print(f"  MRR:    {metrics['MRR']:.4f}")
            print(f"  Hits@1: {metrics['Hits@1']:.4f}")
            print(f"  Hits@3: {metrics['Hits@3']:.4f}")
            print(f"  Hits@10:{metrics['Hits@10']:.4f}")
            print(f"  Answered: {metrics['num_answered']:,} / {metrics['num_queries']:,} queries")
    else:
        apply_rules_to_kg(
            rules_path=args.rules_file,
            ruleformer_train_path=args.ruleformer_train,
            data_dir=args.data_dir,
            dataset=args.dataset,
            output_path=args.output,
            min_weight=args.min_weight,
            min_count=args.min_count,
        )
