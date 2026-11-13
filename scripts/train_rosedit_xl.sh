#!/bin/bash
# RoSeDiT-XL/2, four GPUs, global batch 256 (4 x 64), --qk-norm; FID 12.63 at 400K, 7.98 at 1.4M and 7.61 at 2.4M steps.
# Usage: DATA=/path/to/preprocessed/imagenet bash scripts/train_rosedit_xl.sh [resume step]
# A second argument resumes from exps/$EXP/checkpoints/<step>.pt; a resumed launch draws a new seed.
set -e
DATA=${DATA:?set DATA to the preprocessed ImageNet folder}
EXP=${EXP:-rosedit-xl}
STEP=${1:-0}
if [ "$STEP" -gt 0 ]; then SEED=${SEED:-$(( (RANDOM * 32768 + RANDOM) % 900000 + 100000 ))}; else SEED=${SEED:-0}; fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py \
  --report-to=wandb --allow-tf32 --mixed-precision=fp16 --seed=$SEED \
  --path-type=linear --prediction=v --weighting=uniform \
  --model="SiT-BR-B1W-DG-XL/2" --qk-norm --enc-type=None --proj-coeff=0.0 --encoder-depth=8 --compile \
  --data-dir=$DATA --batch-size=256 --gradient-accumulation-steps=1 --learning-rate=1e-4 \
  --resume-step=$STEP \
  --max-train-steps=2400000 --checkpointing-steps=10000 --output-dir=exps --exp-name=$EXP
