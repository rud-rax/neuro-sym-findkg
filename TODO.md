

# Colab Env stored in Google Drive (not working)
- create colab env and store it in google drive

```python

import sys
import os
from google.colab import drive

#### 1. Mount your Drive
drive.mount('/content/drive')

#### 2. Define the path where you saved your environment
env_path = "/content/drive/MyDrive/my_conda_env"

#### 3. Add the site-packages folder to your Python path
#### IMPORTANT: Check if your env uses 'python3.10' or 'python3.9' folder
site_packages = f"{env_path}/lib/python3.10/site-packages"

if os.path.exists(site_packages):
    sys.path.insert(0, site_packages)
    print("Environment successfully linked!")
else:
    print("Error: Path not found. Check your python version or folder path.")

#### 4. (Optional) Set environment variables for CUDA if using GPU
os.environ['PATH'] = f"{env_path}/bin:" + os.environ['PATH']
```


—--

# Symbolic Module to output triplets (symbolic reasoning)
- what method / algorithm to use ? 
  - https://github.com/zjukg/Ruleformer.git
- [ ] draw architecture diagram ; [Open File](./symbolic_architecture.txt)

### Static Ruleformer
- [x] Translator10 (stored in google drive)
- [x] Rules mined : [rules.txt](https://drive.google.com/file/d/17yGwDjk4b6roR6rBssduqBRjplvHKR4B/view?usp=drive_link)
- [x] Extract Triplets [Y](./FinDKG_dataset/FinDKG/sym_triplets.tsv)

### Dynamic RuleFormer
- [ ] change architecture to include temporal dimension
- [ ] evaluate only symbolic module


```bash
python -m DKG.train --graph FinDKG --use-temporal-rules \
    --rules-base-dir temporal_rules \
    --ruleformer-root ruleformer-findkg
```




```bash
python3 run_symbolic.py \
    --data_dir FinDKG_dataset --dataset FinDKG \
    --output sym_triplets.tsv \
    --jump 2 --padding 200 --maxN 100 --epochs 50
```
- triplet prediction -> outputs Y (sym_triplets)
- [x] save model on google drive 


---


# KGT + RNN module to output triplets (neural reasoning)
- outputs X (neural_triplets) , KGT + RNN triplet prediction

```bash
# Verify the valid split first (checks all 3 changes):
!python evaluate_symbolic_neural_union.py \
  --checkpoint /content/neuro-sym-findkg/result/FinDKG_KGTransformer_overall_best_checkpoint_opt_edge.pt \
  --dataset FinDKG \
  --data_dir /content/neuro-sym-findkg/data \
  --split valid \
  --top_k 10 \
  --symbolic_predictions /content/neuro-sym-findkg/data/sample_sym_triplets.tsv \
  --output_dir /content/neuro-sym-findkg/outputs/union_eval_valid_v2 \
  --gpu 0 \
  --model_type KGT+RNN

  
#Then run test split:
!python evaluate_symbolic_neural_union.py \
  --checkpoint /content/neuro-sym-findkg/result/FinDKG_KGTransformer_overall_best_checkpoint_opt_edge.pt \
  --dataset FinDKG \
  --data_dir /content/neuro-sym-findkg/data \
  --split test \
  --top_k 10 \
  --symbolic_predictions /content/neuro-sym-findkg/data/sample_sym_triplets.tsv \
  --output_dir /content/neuro-sym-findkg/outputs/union_eval_test \
  --gpu 0 \
  --model_type KGT+RNN

```

- [ ] draw architecture diagram refer [Open File](./neural_architecture.txt)
[Go To Section](#save-the-model)
- [ ] save the model in google drive for quick loading of ckpts
- [ ] evaluate only neural module

---

# Fusion Module to combine triplets
- X (neu_triplets) + Y (sym_triplets)
- [ ] fix device bug cpu / gpu in .ipynb 
- [ ] test / evaluate the fusion model using trained KGT+RNN (neural) and Ruleformer (symbolic)


---

# Structure the Repo
- change output dirs with proper logging
- store .pt files in output dirs along with configs
- [ ] configs dir to modify the models
- [x] make a wiki directory

---

# Challenges :
- more sym triplets may have more confidence then neu triplets that may hamper the predicted triplets
- sym_triplets may add noise to the full_triplets 
- save this model
 

---

# Save the Model 
- on google drive
- implemented for symbolic reasoning


---

# Add metric NDCG
- add NDCG metric [more info](https://www.marqo.ai/blog/what-is-normalized-discounted-cumulative-gain-ndcg) for contrastive learning
- [x] test metric on findkg after symbolic reasoning


---

# Report 

- [ ] add more lit survey about symbolic reasoning and types of neuro-symbolic reasoning
- [ ] add diagrams
- [ ] add results
- [ ] add conclusion and future work
- 


