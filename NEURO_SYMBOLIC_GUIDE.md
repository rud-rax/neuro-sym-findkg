# Neuro-Symbolic Module for FinDKG

## Overview

The neuro-symbolic module enhances the FinDKG Temporal Knowledge Graph system by combining neural embeddings with symbolic reasoning. This hybrid approach leverages both the pattern recognition capabilities of deep learning and the interpretability and constraint enforcement of symbolic systems to improve link prediction performance.

### Key Benefits

- **Improved MRR and Hits@N metrics** through constraint enforcement and rule-based reasoning
- **Domain knowledge integration** via explicit rules and type constraints
- **Interpretable predictions** with symbolic explanations
- **Flexible configuration** for different financial domain scenarios
- **Minimal integration effort** with existing KGTransformer model

## Components

### 1. SymbolicReasoner
Performs symbolic reasoning over the knowledge graph using rules and constraints.

**Key methods:**
- `add_rule(rule)`: Add reasoning rules (e.g., transitivity)
- `add_type_constraint(entity_id, entity_type, confidence)`: Enforce entity types
- `add_relation_constraint(relation_id, domain_types, range_types)`: Define domain/range constraints
- `add_transitive_relation(relation_id)`: Mark relations as transitive
- `add_symmetric_relation(relation_id)`: Mark relations as symmetric
- `get_symbolic_score()`: Compute symbolic score for a triple

### 2. HybridScorer
Fuses neural and symbolic scores using multiple fusion methods.

**Fusion methods:**
- `'weighted_sum'`: Linear combination with configurable weights
- `'product'`: Element-wise multiplication
- `'mlp'`: Learned fusion via a small neural network

### 3. NeuroSymbolicLinkPredictor
Main component combining SymbolicReasoner and HybridScorer for improved link prediction.

### 4. NeuroSymbolicEdgeModel
Wrapper around the existing EdgeModel that adds neuro-symbolic capabilities.

### 5. SymbolicRankingBooster
Post-processes predictions to boost rankings based on symbolic reasoning without modifying the base model.

## Installation & Setup

### Step 1: Import the Module

```python
from DKG.model import (
    NeuroSymbolicLinkPredictor,
    NeuroSymbolicEdgeModel,
    SymbolicRule,
    create_financial_kg_rules,
)
from DKG.utils import NeuroSymbolicEvaluator
from DKG.model.neuro_symbolic_integration import FinancialKGConfiguration
```

### Step 2: Create Neuro-Symbolic Predictor

```python
ns_predictor = NeuroSymbolicLinkPredictor(
    num_entities=G.number_of_nodes(),
    num_relations=G.num_relations,
    neural_weight=0.7,      # 70% weight to neural scores
    symbolic_weight=0.3,    # 30% weight to symbolic scores
    fusion_method='weighted_sum',
    device=args.device
)
```

### Step 3: Add Domain Rules

```python
# Add pre-defined financial rules
rules = create_financial_kg_rules(G.num_relations)
for rule in rules:
    ns_predictor.reasoner.add_rule(rule)

# Or add custom rules
custom_rule = SymbolicRule(
    name="investment_chain",
    description="Transitive investment relationships",
    head_relation=2,  # invested_in relation
    body_relations=[2, 2],
    confidence=0.85
)
ns_predictor.reasoner.add_rule(custom_rule)
```

### Step 4: Add Constraints

```python
# Define entity types
COMPANY = 0
PERSON = 1
COUNTRY = 2

# Add type constraints for specific entities
for entity_id in company_entities:
    ns_predictor.reasoner.add_type_constraint(entity_id, COMPANY, confidence=1.0)

# Add domain/range constraints
ns_predictor.reasoner.add_relation_constraint(
    relation_id=0,  # "owns" relation
    domain_types={COMPANY},
    range_types={COMPANY}
)

# Mark transitive relations
ns_predictor.reasoner.add_transitive_relation(0)  # owns
ns_predictor.reasoner.add_transitive_relation(1)  # controls
```

## Usage Examples

### Example 1: Basic Link Prediction

```python
import torch

# Prepare neural scores from your model
neural_scores = torch.tensor([0.8, 0.6, 0.9, 0.5])

# Entity and relation info
heads = torch.tensor([0, 1, 2, 3])
relations = torch.tensor([0, 1, 0, 2])
tails = torch.tensor([10, 20, 30, 40])

# Get neuro-symbolic scores
fused_scores = ns_predictor(neural_scores, heads, relations, tails)
print(fused_scores)  # Improved ranking scores
```

### Example 2: Integrate with Training

```python
# In your training loop, wrap the edge model
from DKG.model.neuro_symbolic_integration import NeuroSymbolicEdgeModel

ns_edge_model = NeuroSymbolicEdgeModel(
    base_edge_model,
    num_entities=G.number_of_nodes(),
    num_relations=G.num_relations,
    neural_weight=0.7,
    symbolic_weight=0.3,
    device=args.device
)

# Add financial rules
ns_edge_model.add_financial_rules()

# Use in place of the original edge model
# Note: You'll need to adapt the evaluation function
```

### Example 3: Post-processing Evaluation

```python
from DKG.model.neuro_symbolic_integration import SymbolicRankingBooster
from DKG.utils import NeuroSymbolicEvaluator

# Create booster for post-processing
booster = SymbolicRankingBooster(ns_predictor.reasoner)

# In evaluation loop:
for predictions in eval_batches:
    neural_scores = evaluate_neural_model(predictions)
    
    # Boost with symbolic reasoning
    boosted_scores = booster.boost_ranking(
        neural_scores,
        heads=predictions['heads'],
        relations=predictions['relations'],
        tails=predictions['tails'],
        boost_factor=0.3  # 30% boost for valid triples
    )
    
    # Evaluate metrics
    evaluator = NeuroSymbolicEvaluator()
    # ... add predictions and evaluate
```

### Example 4: Evaluate Performance

```python
from DKG.utils import NeuroSymbolicEvaluator

evaluator = NeuroSymbolicEvaluator()

# Evaluate on test set
for batch in test_loader:
    neural_rank = get_neural_rank(batch)
    ns_rank = get_ns_rank(batch)
    
    evaluator.add_prediction(
        neural_rank, ns_rank,
        symbolic_score=compute_symbolic_score(batch),
        constraint_violated=check_constraint(batch)
    )

# Get results
metrics = evaluator.evaluate()
print(metrics)
```

## Configuration for Financial Domains

### Pre-configured Setup

```python
from DKG.model.neuro_symbolic_integration import FinancialKGConfiguration

ns_model = FinancialKGConfiguration.configure_edge_model(
    edge_model,
    num_entities=G.number_of_nodes(),
    num_relations=G.num_relations,
    device=args.device
)
```

This automatically configures:
- Financial domain rules (ownership chains, control relations)
- Entity type constraints (companies, persons, countries, industries)
- Domain/range constraints for financial relations
- Transitivity for ownership and control relations

### Financial Domain Rules

Pre-defined rules included:

1. **Ownership Transitivity**: If A owns B and B owns C, then A owns C
2. **Control Chain**: If A controls B and B controls C, then A controls C
3. **Investment Transitivity**: If A invests in B and B invests in C, then A invests in C

### Entity Types for Finance

```python
COMPANY = 1
PERSON = 2
COUNTRY = 3
INDUSTRY = 4
FINANCIAL_INSTRUMENT = 5
```

## Performance Metrics

The NeuroSymbolicEvaluator provides comprehensive metrics:

```
MRR (Mean Reciprocal Rank)
  - Neuro-Symbolic score
  - Neural-only baseline
  - Improvement percentage

Hits@K (K = 1, 3, 10)
  - Same metrics as MRR

Statistics:
  - Constraint violations
  - Number of rules applied
  - Average symbolic score
```

## Hyperparameter Tuning

### Neural/Symbolic Weight Balance

```python
# Conservative (more weight to neural scores)
ns_predictor.hybrid_scorer.set_weights(0.8, 0.2)

# Balanced
ns_predictor.hybrid_scorer.set_weights(0.7, 0.3)

# Aggressive (more weight to symbolic scores)
ns_predictor.hybrid_scorer.set_weights(0.5, 0.5)
```

### Fusion Methods

Try different fusion methods based on your data:

```python
# Linear combination (recommended for basic use)
fusion_method='weighted_sum'

# Element-wise multiplication (good for filtering)
fusion_method='product'

# Learned fusion (requires more training data)
fusion_method='mlp'
```

## Best Practices

### 1. Rule Quality
- Focus on high-confidence domain rules
- Start with a few rules and gradually add more
- Monitor rule application frequency during evaluation

### 2. Constraint Accuracy
- Ensure type constraints are correct
- Validate domain/range constraints against your data
- Use confidence scores to indicate uncertainty

### 3. Weight Balance
- Start with 0.7/0.3 neural/symbolic split
- Adjust based on improvement in validation metrics
- Monitor both MRR and Hits@N

### 4. Evaluation Strategy
- Always compare against neural-only baseline
- Track improvements at different k values (Hits@1, @3, @10)
- Monitor constraint violation rates

## Troubleshooting

### Issue: No improvement in metrics
- Check if rules and constraints are correctly defined
- Verify entity type assignments
- Try increasing symbolic weight
- Check rule application frequency

### Issue: Degraded performance
- Reduce symbolic weight back to neural model
- Review and validate rules/constraints
- Check for constraint violations

### Issue: Memory issues
- Use `SymbolicRankingBooster` for post-processing instead of in-training
- Reduce number of rules or constraints
- Use simpler fusion method (weighted_sum)

## Advanced Usage

### Custom Rule Implementation

```python
from DKG.model import SymbolicRule

# Create a specialized rule for financial relationships
financial_rule = SymbolicRule(
    name="acquisition_chain",
    description="Company A acquired B, B acquired C => A controls C",
    head_relation=1,  # controls
    body_relations=[3, 3],  # acquired, acquired
    confidence=0.9,
    temporal_constraint="ordered"  # Must maintain temporal order
)

ns_predictor.reasoner.add_rule(financial_rule)
```

### Constraint Customization

```python
# Add industry-specific constraints
ns_predictor.reasoner.add_relation_constraint(
    relation_id=4,  # "trades_in"
    domain_types={COMPANY, FINANCIAL_ENTITY},
    range_types={FINANCIAL_INSTRUMENT}
)
```

## Integration with Training Pipeline

See `examples_neuro_symbolic.py` for a complete training integration example.

Key integration points:
1. Create NS model wrapper around EdgeModel
2. Add rules and constraints in initialization
3. Use boosted scores in evaluation loop
4. Track metrics with NeuroSymbolicEvaluator

## References

- **Neural-Symbolic Integration**: Combines neural networks with symbolic reasoning
- **Knowledge Graph Reasoning**: Uses rules and constraints for valid predictions
- **Hybrid Scoring**: Fuses multiple scoring mechanisms for improved ranking

## Citation

If you use this neuro-symbolic module, please cite:

```
@software{findkg_neuro_symbolic_2024,
  title={Neuro-Symbolic Module for FinDKG},
  author={Your Name},
  year={2024},
  url={https://github.com/xiaohui-victor-li/FinDKG}
}
```
