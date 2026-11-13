#!/bin/bash
# Sample 50,000 images from a released checkpoint with the setting of the reported numbers
# (EMA weights, 250-step Euler-Maruyama SDE sampler, no guidance, seed 0, one process, batch 64).
# Usage: bash scripts/sample_50k.sh checkpoints/RoSeDiT-XL-2_DSMC_2400K.pt
set -e
CKPT=${1:?usage: bash scripts/sample_50k.sh <checkpoint file>}
OUT=${OUT:-samples}
case "$(basename "$CKPT")" in
  RoSeDiT-S-2_*)       ARGS=(--model "SiT-BR-B1W-DG-S/2"  --encoder-depth=4 --projector-embed-dims=None) ;;
  RoSeDiT-B-2_REPA_*)  ARGS=(--model "SiT-BR-B1W-DG-B/2"  --encoder-depth=4 --projector-embed-dims=768) ;;
  RoSeDiT-B-2_*)       ARGS=(--model "SiT-BR-B1W-DG-B/2"  --encoder-depth=4 --projector-embed-dims=None) ;;
  RoSeDiT-L-2_*)       ARGS=(--model "SiT-BR-B1W-DG-L/2"  --encoder-depth=8 --projector-embed-dims=None) ;;
  RoSeDiT-XL-2_*)      ARGS=(--model "SiT-BR-B1W-DG-XL/2" --encoder-depth=8 --projector-embed-dims=None --qk-norm) ;;
  *) echo "unknown checkpoint name: $CKPT"; exit 1 ;;
esac
torchrun --nnodes=1 --nproc_per_node=1 generate.py "${ARGS[@]}" \
  --num-fid-samples 50000 --ckpt "$CKPT" \
  --path-type=linear --per-proc-batch-size=64 --mode=sde --num-steps=250 \
  --cfg-scale=1.0 --global-seed=0 --sample-dir="$OUT"
