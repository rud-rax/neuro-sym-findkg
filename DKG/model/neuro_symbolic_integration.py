"""
Integration module for applying neuro-symbolic reasoning to EdgeModel predictions.
This module provides utilities to enhance the existing edge model with symbolic reasoning.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import numpy as np
from DKG.model.neuro_symbolic import (
    NeuroSymbolicLinkPredictor, SymbolicReasoner, SymbolicRule,
    create_financial_kg_rules, create_financial_kg_constraints
)


class NeuroSymbolicEdgeModel(nn.Module):
    """
    Wrapper around the original EdgeModel that adds neuro-symbolic reasoning.
    
    This module:
    1. Gets neural scores from the existing EdgeModel
    2. Applies symbolic reasoning and constraints
    3. Fuses the scores for improved ranking
    """

    def __init__(self, base_edge_model: nn.Module, num_entities: int, num_relations: int,
                 neural_weight: float = 0.7, symbolic_weight: float = 0.3,
                 fusion_method: str = 'weighted_sum', device: torch.device = None):
        """
        Args:
            base_edge_model: The original EdgeModel from the DKG library
            num_entities: Number of entities
            num_relations: Number of relations
            neural_weight: Weight for neural scores
            symbolic_weight: Weight for symbolic scores
            fusion_method: Score fusion method
            device: Computation device
        """
        super().__init__()
        self.base_edge_model = base_edge_model
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.device = device or torch.device('cpu')

        # Create neuro-symbolic components
        self.ns_predictor = NeuroSymbolicLinkPredictor(
            num_entities, num_relations,
            neural_weight=neural_weight,
            symbolic_weight=symbolic_weight,
            fusion_method=fusion_method,
            device=device
        )

    def add_financial_rules(self):
        """Add pre-defined financial domain rules"""
        rules = create_financial_kg_rules(self.num_relations)
        for rule in rules:
            self.ns_predictor.reasoner.add_rule(rule)

    def add_rule(self, rule: SymbolicRule):
        """Add a custom symbolic rule"""
        self.ns_predictor.reasoner.add_rule(rule)

    def add_type_constraint(self, entity_id: int, entity_type: int, confidence: float = 1.0):
        """Add type constraint for an entity"""
        self.ns_predictor.reasoner.add_type_constraint(entity_id, entity_type, confidence)

    def add_relation_constraint(self, relation_id: int, domain_types: set, range_types: set):
        """Add domain/range constraints for a relation"""
        self.ns_predictor.reasoner.add_relation_constraint(relation_id, domain_types, range_types)

    def mark_transitive_relation(self, relation_id: int):
        """Mark a relation as transitive"""
        self.ns_predictor.reasoner.add_transitive_relation(relation_id)

    def mark_symmetric_relation(self, relation_id: int):
        """Mark a relation as symmetric"""
        self.ns_predictor.reasoner.add_symmetric_relation(relation_id)

    def set_inverse_relations(self, relation_id: int, inverse_relation_id: int):
        """Define inverse relations"""
        self.ns_predictor.reasoner.add_inverse_relation(relation_id, inverse_relation_id)

    def forward(self, *args, **kwargs):
        """
        Forward pass combining neural and symbolic scoring.
        
        Expected call signature (same as base EdgeModel):
        forward(static_entity_emb, dynamic_entity_emb, static_relation_emb, 
                dynamic_relation_emb, head, relation, tail, batch_size)
        """
        # Get neural scores from the base model
        neural_scores = self.base_edge_model(*args, **kwargs)

        # For now, return neural scores as-is
        # In practice, you would extract head/tail/relation info and compute symbolic scores
        # The symbolic reasoning is applied during ranking (see score_candidates method)
        return neural_scores

    def score_candidates(self, static_entity_emb, dynamic_entity_emb,
                        dynamic_relation_emb, head: int, relation: int,
                        candidate_tails: torch.Tensor,
                        entity_types: Optional[torch.Tensor] = None,
                        triple_scores_dict: Optional[Dict[Tuple[int, int, int], float]] = None) -> torch.Tensor:
        """
        Score candidate tail entities for a given (head, relation) pair.
        Applies neuro-symbolic reasoning to improve ranking.

        Args:
            static_entity_emb: Static entity embeddings
            dynamic_entity_emb: Dynamic entity embeddings
            dynamic_relation_emb: Dynamic relation embeddings
            head: Head entity ID
            relation: Relation ID
            candidate_tails: Tensor of candidate tail entities [num_candidates]
            entity_types: Optional entity type information
            triple_scores_dict: Optional pre-computed triple scores for rule reasoning

        Returns:
            Neuro-symbolic scores for candidates [num_candidates]
        """
        num_candidates = candidate_tails.shape[0]

        # Get neural scores from base model for each candidate
        neural_scores = []
        heads = torch.full((num_candidates,), head, dtype=torch.long, device=self.device)
        relations = torch.full((num_candidates,), relation, dtype=torch.long, device=self.device)

        # Score each candidate using the base model
        for i, tail in enumerate(candidate_tails):
            tail_id = tail.item() if isinstance(tail, torch.Tensor) else tail
            h_emb = static_entity_emb[head:head+1] if hasattr(static_entity_emb, '__getitem__') else static_entity_emb
            t_emb = static_entity_emb[tail_id:tail_id+1] if hasattr(static_entity_emb, '__getitem__') else static_entity_emb
            # This is a simplified scoring; adjust based on your actual EdgeModel signature
            score = torch.tensor(0.5, device=self.device)  # Placeholder
            neural_scores.append(score)

        neural_scores_tensor = torch.stack(neural_scores) if neural_scores else torch.ones(num_candidates, device=self.device)

        # Apply neuro-symbolic scoring
        fused_scores = self.ns_predictor(
            neural_scores_tensor, heads, relations, candidate_tails,
            entity_types, triple_scores_dict
        )

        return fused_scores


class SymbolicRankingBooster:
    """
    Post-processes neural predictions with symbolic ranking adjustments.
    Useful for improving rankings without modifying the base model.
    """

    def __init__(self, reasoner: SymbolicReasoner):
        self.reasoner = reasoner
        self.triple_cache: Dict[Tuple[int, int, int], float] = {}

    def boost_ranking(self, neural_scores: np.ndarray,
                     heads: np.ndarray, relations: np.ndarray, tails: np.ndarray,
                     entity_types: Optional[np.ndarray] = None,
                     boost_factor: float = 0.3) -> np.ndarray:
        """
        Boost rankings based on symbolic constraints and rules.

        Args:
            neural_scores: Original neural scores [batch_size] or [batch_size, num_candidates]
            heads: Head entity IDs
            relations: Relation IDs
            tails: Tail entity IDs
            entity_types: Optional entity types
            boost_factor: How much to boost valid triples (default 0.3 = 30% boost)

        Returns:
            Adjusted scores with same shape as neural_scores
        """
        adjusted_scores = neural_scores.copy().astype(np.float32)

        # Handle both 1D and 2D inputs
        if len(neural_scores.shape) == 1:
            for i in range(len(heads)):
                head = heads[i].item() if hasattr(heads[i], 'item') else int(heads[i])
                rel = relations[i].item() if hasattr(relations[i], 'item') else int(relations[i])
                tail = tails[i].item() if hasattr(tails[i], 'item') else int(tails[i])

                sym_score = self.reasoner.get_symbolic_score(
                    head, rel, tail,
                    entity_types[head:head+1] if entity_types is not None else None,
                    self.triple_cache
                )
                adjusted_scores[i] = neural_scores[i] * (1.0 + (sym_score - 0.5) * boost_factor)
        else:
            # 2D case: [batch_size, num_candidates]
            for i in range(heads.shape[0]):
                head = heads[i].item() if hasattr(heads[i], 'item') else int(heads[i])
                rel = relations[i].item() if hasattr(relations[i], 'item') else int(relations[i])

                for j in range(adjusted_scores.shape[1]):
                    tail = tails[i, j].item() if hasattr(tails[i, j], 'item') else int(tails[i, j])
                    sym_score = self.reasoner.get_symbolic_score(
                        head, rel, tail,
                        entity_types[head:head+1] if entity_types is not None else None,
                        self.triple_cache
                    )
                    adjusted_scores[i, j] = neural_scores[i, j] * (1.0 + (sym_score - 0.5) * boost_factor)

        return adjusted_scores

    def cache_triple_score(self, head: int, relation: int, tail: int, score: float):
        """Cache a triple score for rule application"""
        self.triple_cache[(head, relation, tail)] = score

    def clear_cache(self):
        """Clear the triple score cache"""
        self.triple_cache.clear()


class FinancialKGConfiguration:
    """
    Pre-configured neuro-symbolic setup for financial knowledge graphs.
    """

    # Entity type IDs (customize based on your data)
    ENTITY_TYPES = {
        'company': 1,
        'person': 2,
        'country': 3,
        'industry': 4,
        'financial_instrument': 5,
    }

    # Relation type IDs (customize based on your data)
    RELATIONS = {
        'owns': 0,
        'controls': 1,
        'employed_by': 2,
        'located_in': 3,
        'trades': 4,
    }

    @classmethod
    def configure_edge_model(cls, edge_model: nn.Module, num_entities: int,
                            num_relations: int, device: torch.device = None) -> NeuroSymbolicEdgeModel:
        """
        Create and configure a NeuroSymbolicEdgeModel for financial KGs.

        Args:
            edge_model: The base EdgeModel
            num_entities: Number of entities
            num_relations: Number of relations
            device: Computation device

        Returns:
            Configured NeuroSymbolicEdgeModel
        """
        ns_model = NeuroSymbolicEdgeModel(
            edge_model, num_entities, num_relations,
            neural_weight=0.7, symbolic_weight=0.3,
            fusion_method='weighted_sum', device=device
        )

        # Add financial domain rules
        ns_model.add_financial_rules()

        # Add type constraints
        # Example: companies can own other companies
        ns_model.add_relation_constraint(
            cls.RELATIONS['owns'],
            domain_types={cls.ENTITY_TYPES['company']},
            range_types={cls.ENTITY_TYPES['company']}
        )

        # Example: persons are employed by companies
        ns_model.add_relation_constraint(
            cls.RELATIONS['employed_by'],
            domain_types={cls.ENTITY_TYPES['person']},
            range_types={cls.ENTITY_TYPES['company']}
        )

        # Mark transitive relations
        ns_model.mark_transitive_relation(cls.RELATIONS['owns'])
        ns_model.mark_transitive_relation(cls.RELATIONS['controls'])

        return ns_model
