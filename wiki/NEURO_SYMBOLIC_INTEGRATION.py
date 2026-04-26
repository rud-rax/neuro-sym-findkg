"""
Integration guide and practical examples for adding neuro-symbolic 
reasoning to the FinDKG training and evaluation pipeline.

This file shows how to modify train.py and eval.py to use the neuro-symbolic module.
"""

# ============================================================================
# EXAMPLE 1: Modifying eval.py to use Neuro-Symbolic Evaluation
# ============================================================================

"""
In DKG/eval.py, modify the evaluate() function:

    OLD CODE:
    if phase in ["Test", "Validation"]:
        batch_eval_dict = eval_link_prediction(...)
        
    NEW CODE:
    if phase in ["Test", "Validation"]:
        # Standard evaluation
        batch_eval_dict = eval_link_prediction(...)
        
        # Neuro-Symbolic evaluation
        if args.use_neuro_symbolic:
            ns_eval_dict = eval_link_prediction_neuro_symbolic(
                model, batch_G, static_entity_emb,
                dynamic_entity_emb, dynamic_relation_emb,
                args, ns_edge_model
            )
            batch_eval_dict.update(ns_eval_dict)
"""

def eval_link_prediction_neuro_symbolic_example(
        model, batch_G, static_entity_emb, dynamic_entity_emb, 
        dynamic_relation_emb, args, ns_edge_model):
    """
    Example function to evaluate with neuro-symbolic scoring.
    Add this to DKG/eval.py
    """
    from DKG.utils.neuro_symbolic_eval import NeuroSymbolicEvaluator
    import torch
    
    evaluator = NeuroSymbolicEvaluator()
    
    # Iterate through test triples
    for batch_edges in batch_G.edges():
        head, tail = batch_edges
        relation = batch_G[head][tail]['relation_type']
        
        # Get neural scores
        neural_score = model.edge_model.forward(...)
        
        # Get neuro-symbolic score
        ns_score = ns_edge_model.score_candidates(
            static_entity_emb,
            dynamic_entity_emb,
            dynamic_relation_emb,
            head=head,
            relation=relation,
            candidate_tails=torch.tensor([tail]),
        )
        
        # Evaluate ranking
        # (This is simplified; adapt based on your actual evaluation code)
        evaluator.add_prediction(
            neural_rank=compute_rank(neural_score),
            neuro_symbolic_rank=compute_rank(ns_score),
            symbolic_score=float(ns_score[0])
        )
    
    return {
        'neuro_symbolic_metrics': evaluator.evaluate()
    }


# ============================================================================
# EXAMPLE 2: Modifying train.py to create and use NeuroSymbolicEdgeModel
# ============================================================================

"""
In DKG/train.py, after creating the edge_model, add:

    # Create neuro-symbolic wrapper
    if args.use_neuro_symbolic:
        from DKG.model import NeuroSymbolicEdgeModel
        from DKG.model.neuro_symbolic_integration import FinancialKGConfiguration
        
        ns_edge_model = FinancialKGConfiguration.configure_edge_model(
            edge_model,
            num_entities=G.number_of_nodes(),
            num_relations=G.num_relations,
            device=args.device
        )
    else:
        ns_edge_model = None
        
    # Pass ns_edge_model to evaluation function
"""

def setup_neuro_symbolic_model_example(edge_model, G, args):
    """
    Example setup function for neuro-symbolic model.
    Add this to DKG/train.py
    """
    from DKG.model import NeuroSymbolicEdgeModel
    from DKG.model.neuro_symbolic_integration import FinancialKGConfiguration
    
    # Create neuro-symbolic edge model
    ns_edge_model = FinancialKGConfiguration.configure_edge_model(
        edge_model,
        num_entities=G.number_of_nodes(),
        num_relations=G.num_relations,
        device=args.device
    )
    
    # Optionally adjust weights
    if hasattr(args, 'neural_weight'):
        ns_edge_model.ns_predictor.hybrid_scorer.set_weights(
            args.neural_weight,
            1.0 - args.neural_weight
        )
    
    return ns_edge_model


# ============================================================================
# EXAMPLE 3: Configuration Args
# ============================================================================

"""
Add these arguments to your argument parser:

parser.add_argument('--use-neuro-symbolic', type=bool, default=False,
                    help='Enable neuro-symbolic reasoning')
parser.add_argument('--neural-weight', type=float, default=0.7,
                    help='Weight for neural component [0, 1]')
parser.add_argument('--symbolic-weight', type=float, default=0.3,
                    help='Weight for symbolic component [0, 1]')
parser.add_argument('--fusion-method', type=str, default='weighted_sum',
                    choices=['weighted_sum', 'product', 'mlp'],
                    help='Score fusion method')
parser.add_argument('--financial-rules', type=bool, default=True,
                    help='Use pre-configured financial domain rules')
"""


# ============================================================================
# EXAMPLE 4: Complete Training Loop Modification
# ============================================================================

TRAINING_LOOP_MODIFICATION = """
# In DKG/train.py, modify the main training loop:

def main(args):
    # ... existing code ...
    
    # Create edge model
    edge_model = EdgeModel(...)
    
    # Create neuro-symbolic wrapper if enabled
    ns_edge_model = None
    if args.use_neuro_symbolic:
        ns_edge_model = FinancialKGConfiguration.configure_edge_model(
            edge_model,
            num_entities=G.number_of_nodes(),
            num_relations=G.num_relations,
            device=args.device
        )
    
    # Training loop (modified for NS evaluation)
    for epoch in range(args.epochs):
        # ... existing training code ...
        
        # Evaluation with neuro-symbolic scores
        if epoch % args.eval_every == 0:
            val_metrics = evaluate(
                model, val_data_loader, entire_G,
                static_entity_embs, init_dynamic_entity_embeds,
                init_dynamic_relation_embeds, G.num_relations,
                args, "Validation", full_link_pred_validation,
                args.time_pred_eval, epoch=epoch,
                ns_edge_model=ns_edge_model  # Pass NS model
            )
            
            # Log neuro-symbolic metrics if available
            if args.use_neuro_symbolic and 'neuro_symbolic_metrics' in val_metrics:
                ns_metrics = val_metrics['neuro_symbolic_metrics']
                logger.info(f"NS Metrics: MRR={ns_metrics.mrr:.4f}, "
                           f"Hits@10={ns_metrics.hits_10:.4f}")
"""


# ============================================================================
# EXAMPLE 5: Custom Rule Addition
# ============================================================================

def add_custom_financial_rules_example(ns_edge_model):
    """
    Example of adding custom rules specific to your financial data.
    """
    from DKG.model import SymbolicRule
    
    # Example 1: Subsidiary control rule
    subsidiary_rule = SymbolicRule(
        name="subsidiary_control_chain",
        description="Parent controls subsidiary, subsidiary controls entity => Parent controls entity",
        head_relation=1,  # controls
        body_relations=[4, 1],  # subsidiary_of, controls
        confidence=0.85
    )
    ns_edge_model.add_rule(subsidiary_rule)
    
    # Example 2: Investment partnership
    partnership_rule = SymbolicRule(
        name="joint_investment",
        description="A invests with B, B invests in C => A indirectly invests in C",
        head_relation=2,  # invested_in
        body_relations=[5, 2],  # partners_with, invested_in
        confidence=0.75
    )
    ns_edge_model.add_rule(partnership_rule)
    
    # Example 3: Executive movement
    executive_rule = SymbolicRule(
        name="executive_knowledge_transfer",
        description="Executive left company A, now at company B => Knowledge transfer",
        head_relation=6,  # worked_for (inverse)
        body_relations=[7, 8],  # left_company, joined_company
        confidence=0.6
    )
    ns_edge_model.add_rule(executive_rule)
    
    print(f"Added {3} custom financial rules")


# ============================================================================
# EXAMPLE 6: Custom Constraint Configuration
# ============================================================================

def add_financial_constraints_example(ns_edge_model):
    """
    Example of adding financial domain constraints.
    """
    # Entity type definitions (customize based on your data)
    COMPANY = 0
    PERSON = 1
    COUNTRY = 2
    INDUSTRY = 3
    EXCHANGE = 4
    
    # Example 1: Ownership constraints
    # Only companies can own other companies
    ns_edge_model.add_relation_constraint(
        relation_id=0,  # "owns"
        domain_types={COMPANY},
        range_types={COMPANY}
    )
    
    # Example 2: Employment constraints
    # Only persons can be employed by companies
    ns_edge_model.add_relation_constraint(
        relation_id=1,  # "employed_by"
        domain_types={PERSON},
        range_types={COMPANY}
    )
    
    # Example 3: Trading constraints
    # Companies trade in exchanges
    ns_edge_model.add_relation_constraint(
        relation_id=3,  # "trades_on"
        domain_types={COMPANY},
        range_types={EXCHANGE}
    )
    
    # Example 4: Geographic constraints
    # Companies are located in countries
    ns_edge_model.add_relation_constraint(
        relation_id=4,  # "located_in"
        domain_types={COMPANY},
        range_types={COUNTRY}
    )
    
    print(f"Added {4} constraint rules")


# ============================================================================
# EXAMPLE 7: Performance Monitoring
# ============================================================================

def setup_ns_monitoring_example():
    """
    Example of setting up monitoring for neuro-symbolic improvements.
    """
    from DKG.utils.neuro_symbolic_eval import NeuroSymbolicEvaluator, RankingComparison
    
    # Create evaluators
    ns_evaluator = NeuroSymbolicEvaluator()
    ranking_comparison = RankingComparison()
    
    return {
        'ns_evaluator': ns_evaluator,
        'ranking_comparison': ranking_comparison,
    }


# ============================================================================
# EXAMPLE 8: Results Reporting
# ============================================================================

def report_neuro_symbolic_results_example(metrics, comparison_stats):
    """
    Example of reporting neuro-symbolic evaluation results.
    """
    print("\n" + "="*70)
    print("NEURO-SYMBOLIC EVALUATION RESULTS")
    print("="*70)
    
    # Print metrics
    print(metrics)
    
    # Print comparison statistics
    print("\nRanking Comparison Statistics:")
    print(f"  Improved: {comparison_stats['improved']} "
          f"({comparison_stats['improvement_rate']*100:.1f}%)")
    print(f"  Degraded: {comparison_stats['degraded']}")
    print(f"  Unchanged: {comparison_stats['unchanged']}")
    print(f"\nAverage Rank Change: {comparison_stats['net_rank_change']:.2f}")
    print(f"Average Score Change: {comparison_stats['avg_score_improvement']:+.4f}")


# ============================================================================
# SUMMARY: Step-by-Step Integration Guide
# ============================================================================

INTEGRATION_STEPS = """
Step 1: Add Command-Line Arguments
   - Add --use-neuro-symbolic flag
   - Add --neural-weight and --symbolic-weight parameters
   - Add --fusion-method option

Step 2: Import Neuro-Symbolic Components
   from DKG.model import NeuroSymbolicEdgeModel
   from DKG.model.neuro_symbolic_integration import FinancialKGConfiguration
   from DKG.utils.neuro_symbolic_eval import NeuroSymbolicEvaluator

Step 3: Create NS Model (in train.py)
   if args.use_neuro_symbolic:
       ns_edge_model = FinancialKGConfiguration.configure_edge_model(...)
   
   # Optionally add custom rules and constraints
   add_custom_financial_rules_example(ns_edge_model)
   add_financial_constraints_example(ns_edge_model)

Step 4: Modify Evaluation Function (in eval.py)
   - Accept ns_edge_model as parameter
   - Use it to score predictions
   - Track metrics with NeuroSymbolicEvaluator

Step 5: Monitor Performance
   - Compare neural vs neuro-symbolic metrics
   - Adjust weights if needed
   - Track rule applications

Step 6: Hyperparameter Tuning
   - Adjust neural/symbolic weights
   - Try different fusion methods
   - Add/remove rules based on impact
"""

if __name__ == "__main__":
    print(INTEGRATION_STEPS)
