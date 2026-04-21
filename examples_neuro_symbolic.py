"""
Example script demonstrating how to use the neuro-symbolic module
with the FinDKG KGTransformer model.

This script shows:
1. Setting up the neuro-symbolic module
2. Adding financial domain rules and constraints
3. Evaluating improvements in MRR and Hits@N metrics
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple

# Import neuro-symbolic components
from DKG.model.neuro_symbolic import (
    NeuroSymbolicLinkPredictor,
    SymbolicReasoner,
    SymbolicRule,
    create_financial_kg_rules,
)
from DKG.model.neuro_symbolic_integration import (
    NeuroSymbolicEdgeModel,
    SymbolicRankingBooster,
    FinancialKGConfiguration,
)
from DKG.utils.neuro_symbolic_eval import (
    NeuroSymbolicEvaluator,
    RankingComparison,
)


def example_1_basic_setup():
    """
    Example 1: Basic setup of neuro-symbolic module.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Neuro-Symbolic Setup")
    print("="*60)

    num_entities = 1000
    num_relations = 50
    device = torch.device('cpu')

    # Create neuro-symbolic link predictor
    ns_predictor = NeuroSymbolicLinkPredictor(
        num_entities=num_entities,
        num_relations=num_relations,
        neural_weight=0.7,
        symbolic_weight=0.3,
        fusion_method='weighted_sum',
        device=device
    )

    print(f"Created NeuroSymbolicLinkPredictor:")
    print(f"  Entities: {num_entities}")
    print(f"  Relations: {num_relations}")
    print(f"  Neural weight: 0.7, Symbolic weight: 0.3")
    print(f"  Fusion method: weighted_sum")

    return ns_predictor


def example_2_add_rules(ns_predictor):
    """
    Example 2: Add symbolic rules for reasoning.
    """
    print("\n" + "="*60)
    print("Example 2: Adding Symbolic Rules")
    print("="*60)

    # Get financial rules
    rules = create_financial_kg_rules(ns_predictor.num_relations)
    
    for rule in rules:
        ns_predictor.reasoner.add_rule(rule)
        print(f"Added rule: {rule.name}")
        print(f"  Description: {rule.description}")
        print(f"  Head relation: {rule.head_relation}")
        print(f"  Body relations: {rule.body_relations}")
        print(f"  Confidence: {rule.confidence}")

    # Add custom rule
    custom_rule = SymbolicRule(
        name="subsidiary_control",
        description="If A controls subsidiary B and B controls C, then A indirectly controls C",
        head_relation=1,  # controls relation
        body_relations=[1, 1],
        confidence=0.75,
        temporal_constraint="ordered"
    )
    ns_predictor.reasoner.add_rule(custom_rule)
    print(f"\nAdded custom rule: {custom_rule.name}")

    return ns_predictor


def example_3_add_constraints(ns_predictor):
    """
    Example 3: Add type and domain/range constraints.
    """
    print("\n" + "="*60)
    print("Example 3: Adding Constraints")
    print("="*60)

    # Define entity types
    ENTITY_TYPES = {
        'company': 0,
        'person': 1,
        'country': 2,
    }

    # Add type constraints for specific entities
    ns_predictor.reasoner.add_type_constraint(
        entity_id=0, entity_type=ENTITY_TYPES['company'], confidence=1.0
    )
    ns_predictor.reasoner.add_type_constraint(
        entity_id=1, entity_type=ENTITY_TYPES['person'], confidence=1.0
    )
    print("Added entity type constraints")

    # Add domain/range constraints for relations
    # Relation 0: "owns" - only companies own entities
    ns_predictor.reasoner.add_relation_constraint(
        relation_id=0,
        domain_types={ENTITY_TYPES['company']},
        range_types={ENTITY_TYPES['company']}
    )
    print("Added domain/range constraint for 'owns' relation")

    # Relation 1: "employed_by" - persons work for companies
    ns_predictor.reasoner.add_relation_constraint(
        relation_id=1,
        domain_types={ENTITY_TYPES['person']},
        range_types={ENTITY_TYPES['company']}
    )
    print("Added domain/range constraint for 'employed_by' relation")

    # Mark transitive relations
    ns_predictor.reasoner.add_transitive_relation(0)  # owns is transitive
    ns_predictor.reasoner.add_symmetric_relation(2)  # partner is symmetric
    print("Marked relations as transitive/symmetric")

    # Define inverse relations
    ns_predictor.reasoner.add_inverse_relation(1, 3)  # employed_by <-> employs
    print("Defined inverse relation pair")

    return ns_predictor


def example_4_score_predictions(ns_predictor):
    """
    Example 4: Score predictions using neuro-symbolic module.
    """
    print("\n" + "="*60)
    print("Example 4: Scoring Predictions")
    print("="*60)

    # Simulate neural scores
    batch_size = 5
    neural_scores = torch.rand(batch_size)
    
    # Create sample predictions
    heads = torch.tensor([0, 1, 2, 3, 4])
    relations = torch.tensor([0, 1, 0, 1, 2])
    tails = torch.tensor([10, 20, 30, 40, 50])
    
    print(f"Scoring {batch_size} predictions:")
    print(f"Neural scores: {neural_scores}")

    # Get neuro-symbolic scores
    ns_scores = ns_predictor(neural_scores, heads, relations, tails)
    
    print(f"Neuro-symbolic scores: {ns_scores}")
    print(f"\nScore changes:")
    for i in range(batch_size):
        change = ns_scores[i].item() - neural_scores[i].item()
        print(f"  Prediction {i}: {neural_scores[i]:.4f} -> {ns_scores[i]:.4f} (Δ {change:+.4f})")


def example_5_ranking_evaluation():
    """
    Example 5: Evaluate ranking improvements.
    """
    print("\n" + "="*60)
    print("Example 5: Ranking Evaluation")
    print("="*60)

    # Create evaluator
    evaluator = NeuroSymbolicEvaluator()

    # Simulate predictions (neural rank, neuro-symbolic rank, symbolic score)
    test_cases = [
        (5, 2, 0.8, False, True),   # Improved from rank 5 to 2
        (10, 8, 0.7, True, False),  # Slightly improved, constraint violated
        (1, 1, 0.9, False, True),   # No change, already perfect
        (20, 15, 0.75, False, True),  # Significant improvement
        (3, 4, 0.5, False, False),  # Slightly degraded
    ]

    for neural_rank, ns_rank, sym_score, constraint_viol, rule_applied in test_cases:
        evaluator.add_prediction(
            neural_rank, ns_rank, sym_score, constraint_viol, rule_applied
        )

    # Compute metrics
    metrics = evaluator.evaluate()
    print(metrics)


def example_6_integration_with_edge_model():
    """
    Example 6: Integration with the existing EdgeModel.
    This shows how to wrap an EdgeModel with neuro-symbolic capabilities.
    """
    print("\n" + "="*60)
    print("Example 6: Integration with EdgeModel")
    print("="*60)

    num_entities = 1000
    num_relations = 50
    device = torch.device('cpu')

    # Create a dummy edge model (in practice, use your actual EdgeModel)
    class DummyEdgeModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.rand(kwargs.get('batch_size', 1))

    base_edge_model = DummyEdgeModel()

    # Wrap with neuro-symbolic capabilities
    ns_edge_model = NeuroSymbolicEdgeModel(
        base_edge_model,
        num_entities=num_entities,
        num_relations=num_relations,
        neural_weight=0.7,
        symbolic_weight=0.3,
        device=device
    )

    # Add financial configuration
    ns_edge_model.add_financial_rules()
    
    print("Created NeuroSymbolicEdgeModel wrapping base EdgeModel")
    print("Added financial domain rules")

    # Show available configuration methods
    print("\nAvailable configuration methods:")
    print("  - add_financial_rules()")
    print("  - add_rule(rule)")
    print("  - add_type_constraint(entity_id, entity_type)")
    print("  - add_relation_constraint(relation_id, domain_types, range_types)")
    print("  - mark_transitive_relation(relation_id)")
    print("  - mark_symmetric_relation(relation_id)")
    print("  - set_inverse_relations(relation_id, inverse_relation_id)")

    return ns_edge_model


def example_7_ranking_booster():
    """
    Example 7: Using SymbolicRankingBooster for post-processing.
    """
    print("\n" + "="*60)
    print("Example 7: Symbolic Ranking Booster (Post-processing)")
    print("="*60)

    # Create a symbolic reasoner
    reasoner = SymbolicReasoner(num_entities=100, num_relations=10)
    
    # Add some rules and constraints
    rule = SymbolicRule(
        name="test_rule",
        description="Test rule",
        head_relation=0,
        body_relations=[0, 0],
        confidence=0.8
    )
    reasoner.add_rule(rule)

    # Create booster
    booster = SymbolicRankingBooster(reasoner)

    # Simulate neural predictions
    neural_scores = np.array([0.1, 0.5, 0.8, 0.3, 0.6])
    heads = np.array([0, 1, 2, 3, 4])
    relations = np.array([0, 0, 0, 0, 0])
    tails = np.array([10, 20, 30, 40, 50])

    # Boost rankings
    boosted_scores = booster.boost_ranking(
        neural_scores, heads, relations, tails,
        boost_factor=0.3
    )

    print(f"Original scores: {neural_scores}")
    print(f"Boosted scores:  {boosted_scores}")
    print(f"Score changes:   {boosted_scores - neural_scores}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("NEURO-SYMBOLIC MODULE EXAMPLES")
    print("="*60)

    # Run examples
    ns_predictor = example_1_basic_setup()
    ns_predictor = example_2_add_rules(ns_predictor)
    ns_predictor = example_3_add_constraints(ns_predictor)
    example_4_score_predictions(ns_predictor)
    example_5_ranking_evaluation()
    example_6_integration_with_edge_model()
    example_7_ranking_booster()

    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("1. Integrate NeuroSymbolicEdgeModel into your training pipeline")
    print("2. Configure rules and constraints specific to your financial domain")
    print("3. Evaluate performance improvements on validation and test sets")
    print("4. Adjust neural/symbolic weights for optimal performance")
    print("5. Export metrics using NeuroSymbolicEvaluator")


if __name__ == "__main__":
    main()
