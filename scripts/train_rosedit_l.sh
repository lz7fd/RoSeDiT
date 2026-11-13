#!/bin/bash
# RoSeDiT-L/2, 400K steps, one GPU, batch 128 with 2 accumulation steps = 256 per update (FID 15.34).
# Usage: DATA=/path/to/preprocessed/imagenet bash scripts/train_rosedit_l.sh [resume step]
# A second argument resumes from exps/$EXP/checkpoints/<step>.pt; a resumed launch draws a new seed.
set -e
DATA=${DATA:?set DATA to the preprocessed ImageNet folder}
EXP=${EXP:-rosedit-l}
STEP=${1:-0}
if [ "$STEP" -gt 0 ]; then SEED=${SEED:-$(( (RANDOM * 32768 + RANDOM) % 900000 + 100000 ))}; else SEED=${SEED:-0}; fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

accelerate launch --num_processes 1 train.py \
  --report-to=wandb --allow-tf32 --mixed-precision=fp16 --seed=$SEED \
  --path-type=linear --prediction=v --weighting=uniform \
  --model="SiT-BR-B1W-DG-L/2" --enc-type=None --proj-coeff=0.0 --encoder-depth=8 --compile \
  --data-dir=$DATA --batch-size=128 --gradient-accumulation-steps=2 --resume-step=$STEP \
  --max-train-steps=400000 --output-dir=exps --exp-name=$EXP
