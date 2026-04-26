# Train DKG Run - Neuro-Symbolic Integration (Approach 1)

## Summary of Changes

The `train_DKG_run.py` file has been updated to support **Approach 1: Post-Processing** for neuro-symbolic reasoning. This approach applies symbolic reasoning **after** neural evaluation with **minimal changes** to the existing code.

## What Changed

### 1. Configuration Section (Lines 73-77)
```python
# Neuro-Symbolic Configuration (Approach 1: Post-Processing)
use_neuro_symbolic = True           # Enable/disable neuro-symbolic post-processing
neural_weight = 0.7                 # Weight for neural scores [0, 1]
symbolic_weight = 0.3               # Weight for symbolic scores [0, 1]
use_financial_rules = True          # Use pre-configured financial domain rules
neuro_symbolic_boost_factor = 0.3   # How much to boost valid predictions
```

**Change these to control the neuro-symbolic behavior:**
- `use_neuro_symbolic`: Toggle on/off with `True`/`False`
- `neural_weight`: Higher = trust neural more (default 0.7)
- `symbolic_weight`: Higher = trust rules more (default 0.3)
- `use_financial_rules`: Enable financial domain rules (default `True`)
- `neuro_symbolic_boost_factor`: Strength of boost (default 0.3 = 30%)

### 2. New Imports (After line 51)
```python
from DKG.model.neuro_symbolic import SymbolicReasoner, create_financial_kg_rules
from DKG.model.neuro_symbolic_integration import SymbolicRankingBooster
from DKG.utils.neuro_symbolic_eval import NeuroSymbolicEvaluator
```

### 3. Neuro-Symbolic Initialization (After model creation)
The script now creates a `SymbolicRankingBooster` with:
- SymbolicReasoner with your graph configuration
- Pre-configured financial rules (if enabled)
- Logging of configuration details

### 4. Validation & Test Evaluation
- Validation loop now includes neuro-symbolic evaluation tracking
- Test evaluation includes detailed neuro-symbolic configuration summary

## How to Use

### Quick Start (Default)
Just run the script as before - neuro-symbolic is enabled by default:

```bash
python train_DKG_run.py
```

### Disable Neuro-Symbolic
To run without neuro-symbolic reasoning:

```python
use_neuro_symbolic = False  # Line 73
```

### Adjust Weights
Modify the weights based on your needs:

```python
# More trust in neural predictions
neural_weight = 0.8
symbolic_weight = 0.2

# More trust in symbolic rules
neural_weight = 0.6
symbolic_weight = 0.4

# Balanced
neural_weight = 0.7
symbolic_weight = 0.3
```

### Disable Financial Rules
To use symbolic framework without financial domain rules:

```python
use_financial_rules = False  # Line 76
```

You can still add custom rules programmatically.

## Output

The script now logs neuro-symbolic information:

### During Initialization
```
================================================================================
INITIALIZING NEURO-SYMBOLIC POST-PROCESSING (Approach 1)
================================================================================
Added 3 pre-configured financial domain rules
  - ownership_transitivity: Transitive ownership...
  - control_chain: Control chain...
  - ...
Neuro-Symbolic Configuration:
  - Neural weight: 0.70
  - Symbolic weight: 0.30
  - Boost factor: 0.30
  - Financial rules enabled: True
================================================================================
```

### During Validation
```
[Neuro-Symbolic] Collecting validation predictions for post-processing...
```

### At Test Time
```
================================================================================
NEURO-SYMBOLIC TEST SET EVALUATION
================================================================================
Note: Approach 1 (Post-Processing) - Symbolic reasoning applied after neural evaluation
...
================================================================================
```

## Architecture (Approach 1: Post-Processing)

```
Training Loop (UNCHANGED)
    ↓
    ├─ Standard neural training continues
    └─ No modifications to loss computation

Evaluation (MINIMAL CHANGES)
    ↓
    ├─ Neural evaluation (standard)
    ├─ Neuro-symbolic logging (new)
    └─ Results collected and displayed
```

## Next Steps for Full Integration

To apply neuro-symbolic post-processing to **actual predictions**, you have two options:

### Option A: Modify eval.py (Recommended)
Pass `ns_booster` to the evaluate function and apply post-processing to link prediction scores:

```python
# In eval.py
if ns_booster is not None:
    boosted_scores = ns_booster.boost_ranking(
        neural_scores, heads, relations, tails
    )
```

### Option B: Wrap evaluate() function
Create a wrapper function in train_DKG_run.py that:
1. Calls evaluate()
2. Collects prediction scores
3. Applies ns_booster.boost_ranking()
4. Recalculates metrics

### Option C: Modify EdgeModel Usage
Update how EdgeModel scores are used to include neuro-symbolic reasoning.

See **NEURO_SYMBOLIC_INTEGRATION.py** for detailed examples of each approach.

## Configuration Examples

### Example 1: Conservative (Mostly Neural)
```python
use_neuro_symbolic = True
neural_weight = 0.85
symbolic_weight = 0.15
use_financial_rules = True
neuro_symbolic_boost_factor = 0.2
```
**When to use:** If you're uncertain about rule quality

### Example 2: Balanced (Recommended)
```python
use_neuro_symbolic = True
neural_weight = 0.7
symbolic_weight = 0.3
use_financial_rules = True
neuro_symbolic_boost_factor = 0.3
```
**When to use:** Default, good starting point

### Example 3: Aggressive (Mostly Symbolic)
```python
use_neuro_symbolic = True
neural_weight = 0.5
symbolic_weight = 0.5
use_financial_rules = True
neuro_symbolic_boost_factor = 0.5
```
**When to use:** If you have high-confidence domain rules

### Example 4: Disabled
```python
use_neuro_symbolic = False
```
**When to use:** Benchmark against neural-only baseline

## Troubleshooting

### Issue: Not seeing neuro-symbolic output
- Check `use_neuro_symbolic = True` (line 73)
- Ensure logs are at INFO level
- Check terminal output/log file

### Issue: Want to disable financial rules
- Set `use_financial_rules = False` (line 76)

### Issue: Want to add custom rules
- Modify the initialization section after line 178
- Use `reasoner.add_rule(custom_rule)` to add your own rules

## Files Modified

- `train_DKG_run.py`: Main training script with Approach 1 integration
- No other files were modified (backward compatible)

## Files Referenced

- `DKG/model/neuro_symbolic.py`: Core symbolic reasoning
- `DKG/model/neuro_symbolic_integration.py`: SymbolicRankingBooster
- `DKG/utils/neuro_symbolic_eval.py`: Evaluation metrics
- `NEURO_SYMBOLIC_GUIDE.md`: Complete user manual
- `NEURO_SYMBOLIC_INTEGRATION.py`: Advanced integration examples

## Important Notes

1. **Approach 1 (Post-Processing) Status**:
   - ✅ Initialization and configuration complete
   - ✅ Symbolic reasoner created and rules added
   - ⏳ Actual score boosting requires eval.py modification (see examples)

2. **To Enable Full Post-Processing**:
   - Modify the validation section to collect actual neural predictions
   - Use `ns_booster.boost_ranking()` to apply post-processing
   - Or follow examples in NEURO_SYMBOLIC_INTEGRATION.py

3. **No Training Changes**:
   - Training loop is completely unchanged
   - No impact on training speed or memory
   - Can be easily disabled by setting `use_neuro_symbolic = False`

## Running with Neuro-Symbolic

```bash
# With neuro-symbolic (default)
python train_DKG_run.py

# Without neuro-symbolic (set use_neuro_symbolic = False)
python train_DKG_run.py
```

Both will run identically except for neuro-symbolic logging and potential post-processing.

## Questions?

Refer to:
1. **NEURO_SYMBOLIC_README.md** - Overview
2. **NEURO_SYMBOLIC_GUIDE.md** - Complete guide
3. **NEURO_SYMBOLIC_INTEGRATION.py** - Integration examples
4. **examples_neuro_symbolic.py** - Working code examples
