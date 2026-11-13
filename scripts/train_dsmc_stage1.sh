#!/bin/bash
# DSMC Stage 1 for RoSeDiT-XL/2: 1.4M to 2.2M steps. Each batch of 256 = 128 fresh triples + 128 triples from a fixed set of 12.8M (FID 6.75 at 2.2M).
# Usage: DATA=/path/to/preprocessed/imagenet bash scripts/train_dsmc_stage1.sh [resume step]
# The first launch starts from the checkpoint SRC of the previous run; a second argument resumes this run from
# exps/$EXP/checkpoints/<step>.pt and restores the positions of the fixed triples. Every launch draws a new seed.
set -e
DATA=${DATA:?set DATA to the preprocessed ImageNet folder}
EXP=${EXP:-rosedit-xl-dsmc-stage1}
SRC=${SRC:-exps/rosedit-xl/checkpoints/1400000.pt}
STEP=${1:-0}
if [ "$STEP" -gt 0 ]; then RESUME="--resume-step=$STEP"; else RESUME="--resume-step=1400000 --resume-ckpt=$SRC"; fi
SEED=${SEED:-$(( (RANDOM * 32768 + RANDOM) % 900000 + 100000 ))}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py \
  --report-to=wandb --allow-tf32 --mixed-precision=fp16 --seed=$SEED \
  --path-type=linear --prediction=v --weighting=uniform \
  --model="SiT-BR-B1W-DG-XL/2" --qk-norm --enc-type=None --proj-coeff=0.0 --encoder-depth=8 --compile \
  --data-dir=$DATA --batch-size=256 --gradient-accumulation-steps=1 --learning-rate=1e-4 \
  $RESUME \
  --fixed-set-seed=20261024 --fixed-set-steps=50000 --fixed-frac=0.5 \
  --max-train-steps=2200000 --checkpointing-steps=10000 --output-dir=exps --exp-name=$EXP
