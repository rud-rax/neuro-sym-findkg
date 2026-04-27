"""
Evaluation utilities for neuro-symbolic link prediction.
Provides metrics and analysis tools to measure improvements in MRR, Hits@N, etc.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class NeuroSymbolicMetrics:
    """Container for neuro-symbolic evaluation metrics"""
    mrr: float  # Mean Reciprocal Rank
    mrr_neural: float  # MRR from neural only
    mrr_improvement: float  # Improvement percentage
    hits_1: float
    hits_1_neural: float
    hits_1_improvement: float
    hits_3: float
    hits_3_neural: float
    hits_3_improvement: float
    hits_10: float
    hits_10_neural: float
    hits_10_improvement: float
    ndcg_10: float
    ndcg_10_neural: float
    ndcg_10_improvement: float
    constraint_violations: int
    rules_applied: int
    avg_symbolic_score: float

    def __str__(self):
        return f"""
        ===== Neuro-Symbolic Link Prediction Metrics =====
        MRR:
          Neuro-Symbolic: {self.mrr:.4f}
          Neural Only:    {self.mrr_neural:.4f}
          Improvement:    {self.mrr_improvement:+.2f}%

        Hits@1:
          Neuro-Symbolic: {self.hits_1:.4f}
          Neural Only:    {self.hits_1_neural:.4f}
          Improvement:    {self.hits_1_improvement:+.2f}%

        Hits@3:
          Neuro-Symbolic: {self.hits_3:.4f}
          Neural Only:    {self.hits_3_neural:.4f}
          Improvement:    {self.hits_3_improvement:+.2f}%

        Hits@10:
          Neuro-Symbolic: {self.hits_10:.4f}
          Neural Only:    {self.hits_10_neural:.4f}
          Improvement:    {self.hits_10_improvement:+.2f}%

        NDCG@10:
          Neuro-Symbolic: {self.ndcg_10:.4f}
          Neural Only:    {self.ndcg_10_neural:.4f}
          Improvement:    {self.ndcg_10_improvement:+.2f}%

        Statistics:
          Constraint Violations: {self.constraint_violations}
          Rules Applied: {self.rules_applied}
          Avg Symbolic Score: {self.avg_symbolic_score:.4f}
        """


class NeuroSymbolicEvaluator:
    """
    Evaluates link prediction performance with neural and symbolic components.
    """

    def __init__(self):
        self.neural_ranks = []
        self.neuro_symbolic_ranks = []
        self.constraint_violations = 0
        self.rules_applied = 0
        self.symbolic_scores = []

    def add_prediction(self, neural_rank: int, neuro_symbolic_rank: int,
                       symbolic_score: float = 0.5, constraint_violated: bool = False,
                       rule_applied: bool = False):
        """
        Add a prediction evaluation result.

        Args:
            neural_rank: Rank from neural-only model (1-based)
            neuro_symbolic_rank: Rank from neuro-symbolic model (1-based)
            symbolic_score: Score from symbolic component [0, 1]
            constraint_violated: Whether a constraint was violated
            rule_applied: Whether a symbolic rule was applied
        """
        self.neural_ranks.append(neural_rank)
        self.neuro_symbolic_ranks.append(neuro_symbolic_rank)
        self.symbolic_scores.append(symbolic_score)
        if constraint_violated:
            self.constraint_violations += 1
        if rule_applied:
            self.rules_applied += 1

    def evaluate(self) -> NeuroSymbolicMetrics:
        """
        Compute comprehensive evaluation metrics.

        Returns:
            NeuroSymbolicMetrics object with all metrics
        """
        if not self.neural_ranks:
            raise ValueError("No predictions added")

        # Compute MRR
        mrr_neural = self._compute_mrr(self.neural_ranks)
        mrr_ns = self._compute_mrr(self.neuro_symbolic_ranks)

        # Compute Hits@K
        hits_1_neural = self._compute_hits(self.neural_ranks, k=1)
        hits_1_ns = self._compute_hits(self.neuro_symbolic_ranks, k=1)

        hits_3_neural = self._compute_hits(self.neural_ranks, k=3)
        hits_3_ns = self._compute_hits(self.neuro_symbolic_ranks, k=3)

        hits_10_neural = self._compute_hits(self.neural_ranks, k=10)
        hits_10_ns = self._compute_hits(self.neuro_symbolic_ranks, k=10)

        # Compute NDCG@10
        ndcg_10_neural = self._compute_ndcg(self.neural_ranks, k=10)
        ndcg_10_ns     = self._compute_ndcg(self.neuro_symbolic_ranks, k=10)

        # Compute improvements
        mrr_improvement = ((mrr_ns - mrr_neural) / mrr_neural * 100) if mrr_neural > 0 else 0
        hits_1_improvement = ((hits_1_ns - hits_1_neural) / max(hits_1_neural, 0.001) * 100)
        hits_3_improvement = ((hits_3_ns - hits_3_neural) / max(hits_3_neural, 0.001) * 100)
        hits_10_improvement = ((hits_10_ns - hits_10_neural) / max(hits_10_neural, 0.001) * 100)
        ndcg_10_improvement = ((ndcg_10_ns - ndcg_10_neural) / max(ndcg_10_neural, 1e-6)) * 100

        return NeuroSymbolicMetrics(
            mrr=mrr_ns,
            mrr_neural=mrr_neural,
            mrr_improvement=mrr_improvement,
            hits_1=hits_1_ns,
            hits_1_neural=hits_1_neural,
            hits_1_improvement=hits_1_improvement,
            hits_3=hits_3_ns,
            hits_3_neural=hits_3_neural,
            hits_3_improvement=hits_3_improvement,
            hits_10=hits_10_ns,
            hits_10_neural=hits_10_neural,
            hits_10_improvement=hits_10_improvement,
            ndcg_10=ndcg_10_ns,
            ndcg_10_neural=ndcg_10_neural,
            ndcg_10_improvement=ndcg_10_improvement,
            constraint_violations=self.constraint_violations,
            rules_applied=self.rules_applied,
            avg_symbolic_score=np.mean(self.symbolic_scores) if self.symbolic_scores else 0.0
        )

    @staticmethod
    def _compute_mrr(ranks: List[int]) -> float:
        """Compute Mean Reciprocal Rank"""
        return np.mean([1.0 / r for r in ranks]) if ranks else 0.0

    @staticmethod
    def _compute_hits(ranks: List[int], k: int = 10) -> float:
        """Compute Hits@K"""
        return sum(1 for r in ranks if r <= k) / len(ranks) if ranks else 0.0

    @staticmethod
    def _compute_ndcg(ranks: List[int], k: int = 10) -> float:
        """Compute NDCG@K with binary relevance (single correct answer per query)"""
        scores = [1.0 / np.log2(r + 1) if r <= k else 0.0 for r in ranks]
        return np.mean(scores) if ranks else 0.0

    def reset(self):
        """Reset all accumulated data"""
        self.neural_ranks = []
        self.neuro_symbolic_ranks = []
        self.constraint_violations = 0
        self.rules_applied = 0
        self.symbolic_scores = []


class RankingComparison:
    """
    Detailed comparison between neural and neuro-symbolic rankings.
    """

    def __init__(self):
        self.comparisons: List[Dict] = []

    def add_comparison(self, head: int, relation: int, tail: int,
                      neural_rank: int, ns_rank: int,
                      neural_score: float, ns_score: float,
                      improvement_type: str = "none"):
        """
        Add a ranking comparison result.

        Args:
            head: Head entity ID
            relation: Relation ID
            tail: Tail entity ID (true target)
            neural_rank: Rank from neural model
            ns_rank: Rank from neuro-symbolic model
            neural_score: Neural model score
            ns_score: Neuro-symbolic score
            improvement_type: 'improved', 'degraded', or 'unchanged'
        """
        self.comparisons.append({
            'head': head,
            'relation': relation,
            'tail': tail,
            'neural_rank': neural_rank,
            'ns_rank': ns_rank,
            'rank_change': neural_rank - ns_rank,
            'neural_score': neural_score,
            'ns_score': ns_score,
            'score_change': ns_score - neural_score,
            'improvement_type': improvement_type,
        })

    def get_statistics(self) -> Dict:
        """Get comparison statistics"""
        if not self.comparisons:
            return {}

        improvements = [c for c in self.comparisons if c['improvement_type'] == 'improved']
        degradations = [c for c in self.comparisons if c['improvement_type'] == 'degraded']

        improved_ranks = np.mean([c['rank_change'] for c in improvements]) if improvements else 0
        degraded_ranks = np.mean([c['rank_change'] for c in degradations]) if degradations else 0

        return {
            'total_predictions': len(self.comparisons),
            'improved': len(improvements),
            'degraded': len(degradations),
            'unchanged': len(self.comparisons) - len(improvements) - len(degradations),
            'improvement_rate': len(improvements) / len(self.comparisons) if self.comparisons else 0,
            'avg_rank_improvement': improved_ranks,
            'avg_rank_degradation': degraded_ranks,
            'net_rank_change': np.mean([c['rank_change'] for c in self.comparisons]),
            'avg_score_improvement': np.mean([c['score_change'] for c in self.comparisons]),
        }

    def get_detailed_report(self) -> str:
        """Get a detailed text report"""
        stats = self.get_statistics()
        if not stats:
            return "No comparisons available"

        report = f"""
        ===== Ranking Comparison Report =====
        Total Predictions: {stats['total_predictions']}
        
        Results:
          Improved: {stats['improved']} ({stats['improvement_rate']*100:.1f}%)
          Degraded: {stats['degraded']}
          Unchanged: {stats['unchanged']}
        
        Average Changes:
          Rank Improvement (improved cases): {stats['avg_rank_improvement']:.2f}
          Rank Degradation (degraded cases): {stats['avg_rank_degradation']:.2f}
          Net Rank Change: {stats['net_rank_change']:.2f}
          Average Score Change: {stats['avg_score_improvement']:+.4f}
        """
        return report


def compare_rankings(neural_scores: np.ndarray, symbolic_scores: np.ndarray,
                    true_tail_indices: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare neural and neuro-symbolic rankings.

    Args:
        neural_scores: Neural model scores [batch_size, num_candidates]
        symbolic_scores: Symbolic scores [batch_size, num_candidates]
        true_tail_indices: Indices of true tail entities [batch_size]
        k: Number of top predictions to consider

    Returns:
        Tuple of (neural_ranks, ns_ranks) both [batch_size]
    """
    neural_ranks = []
    ns_ranks = []

    for i in range(neural_scores.shape[0]):
        true_idx = true_tail_indices[i]

        # Get neural ranking
        neural_sorted = np.argsort(-neural_scores[i])  # Descending order
        neural_rank = np.where(neural_sorted == true_idx)[0][0] + 1

        # Fuse scores and get neuro-symbolic ranking
        fused_scores = 0.7 * neural_scores[i] + 0.3 * symbolic_scores[i]
        ns_sorted = np.argsort(-fused_scores)
        ns_rank = np.where(ns_sorted == true_idx)[0][0] + 1

        neural_ranks.append(neural_rank)
        ns_ranks.append(ns_rank)

    return np.array(neural_ranks), np.array(ns_ranks)


def analyze_constraint_impact(predictions: List[Dict], constraint_violations: int,
                             total_predictions: int) -> Dict:
    """
    Analyze the impact of constraint violations on ranking performance.

    Args:
        predictions: List of prediction dictionaries with 'rank_change' key
        constraint_violations: Number of constraint violations
        total_predictions: Total number of predictions

    Returns:
        Dictionary with constraint impact analysis
    """
    violated_preds = [p for p in predictions if 'constraint_violated' in p and p['constraint_violated']]
    satisfied_preds = [p for p in predictions if 'constraint_violated' not in p or not p['constraint_violated']]

    violated_rank_change = np.mean([p['rank_change'] for p in violated_preds]) if violated_preds else 0
    satisfied_rank_change = np.mean([p['rank_change'] for p in satisfied_preds]) if satisfied_preds else 0

    return {
        'constraint_violation_rate': constraint_violations / total_predictions if total_predictions > 0 else 0,
        'violated_preds_rank_change': violated_rank_change,
        'satisfied_preds_rank_change': satisfied_rank_change,
        'rank_change_difference': abs(violated_rank_change - satisfied_rank_change),
    }
