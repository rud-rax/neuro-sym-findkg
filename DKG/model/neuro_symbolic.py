"""
Neuro-Symbolic Module for Temporal Knowledge Graphs
=====================================================

This module combines neural embeddings with symbolic reasoning to improve
link prediction performance on Temporal Knowledge Graphs (TKGs).

The neuro-symbolic approach leverages:
1. Neural components: Graph embeddings and neural link predictors
2. Symbolic components: Domain rules, logical constraints, and reasoning

This hybrid approach helps improve MRR and Hits@N metrics by:
- Enforcing type constraints and logical rules
- Applying domain-specific reasoning patterns
- Ranking candidates based on combined neural and symbolic scores
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict


class ConstraintType(Enum):
    """Types of constraints in the knowledge graph"""
    TYPE_CONSTRAINT = "type_constraint"  # Entity must be of specific type
    CARDINALITY = "cardinality"  # Relation has min/max cardinality
    TEMPORAL = "temporal"  # Temporal constraints
    DOMAIN_RANGE = "domain_range"  # Domain/range constraints for relations
    TRANSITIVITY = "transitivity"  # Transitive relations
    SYMMETRY = "symmetry"  # Symmetric relations
    INVERSE = "inverse"  # Inverse relations


@dataclass
class SymbolicRule:
    """Represents a symbolic reasoning rule"""
    name: str
    description: str
    head_relation: int  # Target relation (what we're predicting)
    body_relations: List[int]  # Relations in the rule body
    confidence: float = 1.0  # Confidence of the rule [0, 1]
    temporal_constraint: Optional[str] = None  # "ordered", "immediate", etc.

    def __hash__(self):
        return hash((self.name, tuple(self.body_relations)))


@dataclass
class TypeConstraint:
    """Entity type constraints"""
    entity_id: int
    entity_type: int
    confidence: float = 1.0


class SymbolicReasoner:
    """
    Performs symbolic reasoning over knowledge graphs using rules and constraints.
    """

    def __init__(self, num_entities: int, num_relations: int, device: torch.device = None):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.device = device or torch.device('cpu')

        # Storage for rules and constraints
        self.rules: List[SymbolicRule] = []
        self.type_constraints: Dict[int, Set[int]] = defaultdict(set)  # entity_id -> {valid_types}
        self.relation_constraints: Dict[int, Tuple[Set[int], Set[int]]] = {}  # relation -> (domain_types, range_types)
        self.transitive_relations: Set[int] = set()
        self.symmetric_relations: Set[int] = set()
        self.inverse_relations: Dict[int, int] = {}  # relation -> inverse_relation

    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a reasoning rule"""
        self.rules.append(rule)

    def add_type_constraint(self, entity_id: int, entity_type: int, confidence: float = 1.0) -> None:
        """Add type constraint for an entity"""
        self.type_constraints[entity_id].add(entity_type)

    def add_relation_constraint(self, relation_id: int, domain_types: Set[int], 
                                range_types: Set[int]) -> None:
        """Add domain/range constraints for a relation"""
        self.relation_constraints[relation_id] = (domain_types.copy(), range_types.copy())

    def add_transitive_relation(self, relation_id: int) -> None:
        """Mark a relation as transitive"""
        self.transitive_relations.add(relation_id)

    def add_symmetric_relation(self, relation_id: int) -> None:
        """Mark a relation as symmetric"""
        self.symmetric_relations.add(relation_id)

    def add_inverse_relation(self, relation_id: int, inverse_relation_id: int) -> None:
        """Define inverse relations"""
        self.inverse_relations[relation_id] = inverse_relation_id
        self.inverse_relations[inverse_relation_id] = relation_id

    def check_type_constraints(self, head: int, relation: int, tail: int,
                               entity_types: Optional[torch.Tensor] = None) -> float:
        """
        Check if triple satisfies type constraints.
        Returns a confidence score [0, 1]
        """
        if relation not in self.relation_constraints:
            return 1.0  # No constraints, fully valid

        domain_types, range_types = self.relation_constraints[relation]

        # If no entity types provided, assume constraints are satisfied
        if entity_types is None:
            return 1.0

        head_type = entity_types[head].item() if len(entity_types.shape) > 0 else entity_types
        tail_type = entity_types[tail].item() if len(entity_types.shape) > 0 else entity_types

        confidence = 1.0
        if domain_types and head_type not in domain_types:
            confidence *= 0.5

        if range_types and tail_type not in range_types:
            confidence *= 0.5

        return confidence

    def apply_rules(self, head: int, relation: int, tail: int,
                    triple_scores: Dict[Tuple[int, int, int], float]) -> float:
        """
        Apply symbolic rules to boost or penalize scores.
        Returns a multiplicative factor to apply to neural scores.
        """
        boost_factor = 1.0

        for rule in self.rules:
            if rule.head_relation != relation:
                continue

            # Check if rule applies to this triple
            # For simplicity, we check if intermediate triples exist with high confidence
            rule_applies = self._check_rule_body(head, tail, rule, triple_scores)

            if rule_applies:
                boost_factor *= (1.0 + rule.confidence)

        return boost_factor

    def _check_rule_body(self, head: int, tail: int, rule: SymbolicRule,
                         triple_scores: Dict[Tuple[int, int, int], float]) -> bool:
        """Check if rule body is satisfied"""
        # For chain rules: head --r1--> X --r2--> tail
        if len(rule.body_relations) == 2:
            r1, r2 = rule.body_relations
            for intermediate in range(self.num_entities):
                score1 = triple_scores.get((head, r1, intermediate), 0.0)
                score2 = triple_scores.get((intermediate, r2, tail), 0.0)
                if score1 > 0.5 and score2 > 0.5:
                    return True
        return False

    def get_symbolic_score(self, head: int, relation: int, tail: int,
                          entity_types: Optional[torch.Tensor] = None,
                          triple_scores: Optional[Dict[Tuple[int, int, int], float]] = None) -> float:
        """
        Compute composite symbolic score combining constraints and rules.
        Returns a score in [0, 1].
        """
        score = 1.0

        # Apply type constraints
        constraint_score = self.check_type_constraints(head, relation, tail, entity_types)
        score *= constraint_score

        # Apply rules
        if triple_scores is not None:
            rule_boost = self.apply_rules(head, relation, tail, triple_scores)
            score *= rule_boost

        # Apply symmetry/inverse constraints
        if relation in self.symmetric_relations:
            score *= 1.1  # Boost symmetric relation predictions
        if relation in self.inverse_relations:
            score *= 1.05  # Slight boost for relations with defined inverses

        return min(score, 1.0)


class HybridScorer(nn.Module):
    """
    Combines neural and symbolic scores for improved link prediction.
    """

    def __init__(self, neural_weight: float = 0.7, symbolic_weight: float = 0.3,
                 fusion_method: str = 'weighted_sum'):
        """
        Args:
            neural_weight: Weight for neural scores [0, 1]
            symbolic_weight: Weight for symbolic scores [0, 1]
            fusion_method: 'weighted_sum', 'product', or 'mlp'
        """
        super().__init__()
        self.neural_weight = neural_weight
        self.symbolic_weight = symbolic_weight
        self.fusion_method = fusion_method

        if fusion_method == 'mlp':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def forward(self, neural_scores: torch.Tensor, symbolic_scores: torch.Tensor) -> torch.Tensor:
        """
        Fuse neural and symbolic scores.

        Args:
            neural_scores: Neural model scores [batch_size] or [batch_size, num_candidates]
            symbolic_scores: Symbolic scores [batch_size] or [batch_size, num_candidates]

        Returns:
            Fused scores with same shape as inputs
        """
        if self.fusion_method == 'weighted_sum':
            return self.neural_weight * neural_scores + self.symbolic_weight * symbolic_scores

        elif self.fusion_method == 'product':
            return neural_scores * symbolic_scores

        elif self.fusion_method == 'mlp':
            # Reshape for MLP
            original_shape = neural_scores.shape
            neural_flat = neural_scores.reshape(-1)
            symbolic_flat = symbolic_scores.reshape(-1)
            combined = torch.stack([neural_flat, symbolic_flat], dim=1)
            fused = self.fusion_mlp(combined).squeeze(-1)
            return fused.reshape(original_shape)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def set_weights(self, neural_weight: float, symbolic_weight: float):
        """Dynamically adjust fusion weights"""
        total = neural_weight + symbolic_weight
        self.neural_weight = neural_weight / total
        self.symbolic_weight = symbolic_weight / total


class NeuroSymbolicLinkPredictor(nn.Module):
    """
    Enhanced link predictor that combines neural embeddings with symbolic reasoning.
    """

    def __init__(self, num_entities: int, num_relations: int,
                 neural_weight: float = 0.7, symbolic_weight: float = 0.3,
                 fusion_method: str = 'weighted_sum', device: torch.device = None):
        """
        Args:
            num_entities: Number of entities in the graph
            num_relations: Number of relations in the graph
            neural_weight: Weight for neural component
            symbolic_weight: Weight for symbolic component
            fusion_method: How to fuse scores ('weighted_sum', 'product', 'mlp')
            device: Device for computation
        """
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.device = device or torch.device('cpu')

        # Symbolic components
        self.reasoner = SymbolicReasoner(num_entities, num_relations, device)
        self.hybrid_scorer = HybridScorer(neural_weight, symbolic_weight, fusion_method)

    def forward(self, neural_scores: torch.Tensor,
                heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor,
                entity_types: Optional[torch.Tensor] = None,
                triple_scores_dict: Optional[Dict[Tuple[int, int, int], float]] = None) -> torch.Tensor:
        """
        Compute neuro-symbolic scores for link prediction.

        Args:
            neural_scores: Neural model scores for (head, relation, tail) triples
            heads: Entity IDs for head entities
            relations: Relation IDs
            tails: Entity IDs for tail entities
            entity_types: Optional entity type tensor
            triple_scores_dict: Optional dict of existing triple scores for rule application

        Returns:
            Fused neuro-symbolic scores
        """
        batch_size = heads.shape[0]
        symbolic_scores = torch.ones_like(neural_scores)

        # Compute symbolic score for each triple
        for i in range(batch_size):
            head = heads[i].item() if isinstance(heads[i], torch.Tensor) else heads[i]
            relation = relations[i].item() if isinstance(relations[i], torch.Tensor) else relations[i]
            tail = tails[i].item() if isinstance(tails[i], torch.Tensor) else tails[i]

            sym_score = self.reasoner.get_symbolic_score(
                head, relation, tail,
                entity_types[head:head+1] if entity_types is not None else None,
                triple_scores_dict
            )
            symbolic_scores[i] = torch.tensor(sym_score, dtype=neural_scores.dtype, device=neural_scores.device)

        # Fuse neural and symbolic scores
        fused_scores = self.hybrid_scorer(neural_scores, symbolic_scores)

        return fused_scores


def create_financial_kg_rules(num_relations: int) -> List[SymbolicRule]:
    """
    Create common reasoning rules for financial knowledge graphs.
    
    Args:
        num_relations: Total number of relations in the KG
        
    Returns:
        List of symbolic rules
    """
    rules = []

    # Note: These are example rules. Adjust relation indices based on your actual schema.
    # Assuming relation indices:
    # 0: owns, 1: controls, 2: invested_in, 3: subsidiary_of, etc.

    if num_relations >= 4:
        # Transitivity rule: owns(A, B) ∧ owns(B, C) ⟹ owns(A, C)
        rules.append(SymbolicRule(
            name="ownership_transitivity",
            description="Transitive ownership: if A owns B and B owns C, then A owns C",
            head_relation=0,  # owns
            body_relations=[0, 0],  # owns, owns
            confidence=0.8
        ))

        # Control chain rule: controls(A, B) ∧ controls(B, C) ⟹ controls(A, C)
        rules.append(SymbolicRule(
            name="control_chain",
            description="Control chain: if A controls B and B controls C, then A controls C",
            head_relation=1,  # controls
            body_relations=[1, 1],  # controls, controls
            confidence=0.7
        ))

    return rules


def create_financial_kg_constraints(entity_types: Optional[torch.Tensor] = None,
                                    num_relations: int = None) -> Dict:
    """
    Create type and domain constraints for financial knowledge graphs.
    
    Args:
        entity_types: Tensor of entity types
        num_relations: Number of relations
        
    Returns:
        Dictionary of constraints
    """
    constraints = {
        'domain_range': {},
        'type_constraints': {},
    }

    if num_relations is None:
        return constraints

    # Example domain/range constraints for financial relations
    # Relation 0 (owns): Company -> Company
    # Relation 1 (controls): Company -> Company/Person
    # Relation 2 (employed_by): Person -> Company
    # etc.

    constraints['domain_range'] = {
        0: (set([1, 2]), set([1, 2])),  # owns: Company -> Company
        1: (set([1, 2]), set([1, 2, 3])),  # controls: Company/Person -> Company/Person/Country
        2: (set([3]), set([1, 2])),  # employed_by: Person -> Company
    }

    return constraints
