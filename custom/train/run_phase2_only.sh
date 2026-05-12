#!/bin/bash
source /data/litengmo/anaconda3/etc/profile.d/conda.sh
conda activate gvhmr
python custom/train/train_frame_codebook.py --config custom/configs/run_codebook_dim128_phase2_mia.yaml
