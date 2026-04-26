

# Colab Env stored in Google Drive (not working)
- create colab env and store it in google drive

"""
python

import sys
import os
from google.colab import drive

# 1. Mount your Drive
drive.mount('/content/drive')

# 2. Define the path where you saved your environment
env_path = "/content/drive/MyDrive/my_conda_env"

# 3. Add the site-packages folder to your Python path
# IMPORTANT: Check if your env uses 'python3.10' or 'python3.9' folder
site_packages = f"{env_path}/lib/python3.10/site-packages"

if os.path.exists(site_packages):
    sys.path.insert(0, site_packages)
    print("Environment successfully linked!")
else:
    print("Error: Path not found. Check your python version or folder path.")

# 4. (Optional) Set environment variables for CUDA if using GPU
os.environ['PATH'] = f"{env_path}/bin:" + os.environ['PATH']
"""


# Symbolic Module to output triplets (symbolic reasoning)
- what method / algorithm to use ? 
  - https://github.com/zjukg/Ruleformer.git

# Create env inside the project directory
conda create --prefix ./ruleformer_env python=3.8 -y

# Activate it
conda activate ./ruleformer_env

# Install PyTorch 1.10 (CPU — swap for CUDA build if you have a GPU)
pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

## Replace cu113 with your CUDA version
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy pandas

## Install remaining dependencies
pip install numpy pandas


```
bash
python3 run_symbolic.py \
    --data_dir FinDKG_dataset --dataset FinDKG \
    --output sym_triplets.tsv \
    --jump 2 --padding 200 --maxN 100 --epochs 50
```
- triplet prediction -> outputs Y (sym_triplets)


---


# KGT + RNN module to output triplets (neural reasoning)
- outputs X (neural_triplets)
- KGT + RNN triplet prediction



# Fusion Module to combine triplets
- X (neu_triplets) + Y (sym_triplets)



# Structure the Repo
- change output dirs with proper logging
- store .pt files in output dirs along with configs
- configs dir to modify the models



# Challenges :
- more sym triplets may have more confidence then neu triplets that may hamper the predicted triplets
- sym_triplets may add noise to the full_triplets 
- save this model
 



# Save the Model 
- on google drive
- 


