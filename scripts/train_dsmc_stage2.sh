#!/bin/bash
# DSMC Stage 2 for RoSeDiT-XL/2: 2.2M to 2.4M steps. Each batch of 256 = 128 triples from the fixed set of Stage 1 + eight groups of 16 from rotating pools of 400K triples (FID 5.84 at 2.4M).
# Usage: DATA=/path/to/preprocessed/imagenet bash scripts/train_dsmc_stage2.sh [resume step]
# The first launch starts from the checkpoint SRC of the previous run; a second argument resumes this run from
# exps/$EXP/checkpoints/<step>.pt and restores the positions of the fixed triples. Every launch draws a new seed.
set -e
DATA=${DATA:?set DATA to the preprocessed ImageNet folder}
EXP=${EXP:-rosedit-xl-dsmc-stage2}
SRC=${SRC:-exps/rosedit-xl-dsmc-stage1/checkpoints/2200000.pt}
STEP=${1:-0}
if [ "$STEP" -gt 0 ]; then RESUME="--resume-step=$STEP"; else RESUME="--resume-step=2200000 --resume-ckpt=$SRC"; fi
SEED=${SEED:-$(( (RANDOM * 32768 + RANDOM) % 900000 + 100000 ))}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py \
  --report-to=wandb --allow-tf32 --mixed-precision=fp16 --seed=$SEED \
  --path-type=linear --prediction=v --weighting=uniform \
  --model="SiT-BR-B1W-DG-XL/2" --qk-norm --enc-type=None --proj-coeff=0.0 --encoder-depth=8 --compile \
  --data-dir=$DATA --batch-size=256 --gradient-accumulation-steps=1 --learning-rate=1e-4 \
  $RESUME \
  --fixed-set-seed=20261027 --fixed-set-steps=3125 --rotate-sets=8 --rotate-window=25000 --rotate-start=2200000 \
  --persist-sections=8 --persist-set-seed=20261024 --persist-set-steps=50000 \
  --max-train-steps=2400000 --checkpointing-steps=10000 --output-dir=exps --exp-name=$EXP
