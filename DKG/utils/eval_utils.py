import numpy as np


class RankingMetric:
    @classmethod
    def mean_reciprocal_rank(cls, true_ranks):
        return np.mean([1.0 / r for r in true_ranks])

    @classmethod
    def recall(cls, true_ranks, k=10):
        return sum(np.array(true_ranks) <= k) * 1.0 / len(true_ranks)

    @classmethod
    def ndcg(cls, true_ranks, k=10):
        # IDCG = 1/log2(2) = 1.0 for single-answer case (ideal rank = 1)
        scores = [1.0 / np.log2(r + 1) if r <= k else 0.0 for r in true_ranks]
        return np.mean(scores)


class RegressionMetric:
    @classmethod
    def mean_absolute_error(cls, diffs):
        return np.mean([abs(diff) for diff in diffs])

    @classmethod
    def mean_squared_error(cls, diffs):
        return np.mean([diff ** 2 for diff in diffs])

    @classmethod
    def root_mean_squared_error(cls, diffs):
        return np.sqrt(cls.mean_squared_error(diffs))
