import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

# --- optional parameter group for --entry-lr: the timestep and class embedders, the controller tokens and every gate vector g_*.
# With --entry-lr > 0 these parameters (about 0.6% of the weights) get their own learning rate. Off by default.
ENTRY_PREFIXES = ("t_embedder.", "y_embedder.")
ENTRY_NAMES = {"class_tokens", "time_tokens"}
def is_entry_param(name):
    n = name.replace("_orig_mod.", "").replace("module.", "")
    return n.startswith(ENTRY_PREFIXES) or n in ENTRY_NAMES or n.rsplit(".", 1)[-1].startswith("g_")

def build_param_groups(model, body_lr, entry_lr):
    names = [n for n, _ in model.named_parameters()]
    body = [p for n, p in model.named_parameters() if not is_entry_param(n)]
    entry = [p for n, p in model.named_parameters() if is_entry_param(n)]
    return [{"params": body, "lr": body_lr}, {"params": entry, "lr": entry_lr}], names

def remap_single_group_state(sd, names, body_lr, entry_lr):
    """Map a one-group optimizer state (indices in model.parameters() order) onto the [body, entry] groups."""
    old_idx = {n: i for i, n in enumerate(names)}
    body_names = [n for n in names if not is_entry_param(n)]
    entry_names = [n for n in names if is_entry_param(n)]
    new_order = body_names + entry_names
    state = {i: sd["state"][old_idx[n]] for i, n in enumerate(new_order) if old_idx[n] in sd["state"]}
    pg = {k: v for k, v in sd["param_groups"][0].items() if k != "params"}
    nb = len(body_names)
    return {"state": state, "param_groups": [{**pg, "params": list(range(nb)), "lr": body_lr},
                                            {**pg, "params": list(range(nb, len(new_order))), "lr": entry_lr}]}

class FixedTripleSet:
    """DSMC fixed set: a fixed set of (image, noise level, noise) triples, `steps` global batches in size. Every entry (the VAE posterior
    sample, the noise level t and the noise) is regenerated from `seed` and its index, so nothing is stored. The set is revisited in
    shuffled passes; each process draws `per_step` entries per step from its own shard of the current pass. Mixed into every batch
    with fresh triples (DSMC Stage 1)."""
    def __init__(self, dataset, seed, steps, global_batch, rank, world, per_step, device, latents_scale, latents_bias):
        self.ds, self.seed, self.rank, self.world, self.per_step, self.device = dataset, int(seed), rank, world, per_step, device
        self.n = int(steps) * int(global_batch)
        rs = np.random.RandomState(self.seed)
        reps = int(math.ceil(self.n / len(dataset)))
        self.order = np.concatenate([rs.permutation(len(dataset)) for _ in range(reps)])[:self.n].astype(np.int64)
        self.ls = latents_scale.detach().float().cpu().view(-1, 1, 1); self.lb = latents_bias.detach().float().cpu().view(-1, 1, 1)
        self.pass_no, self.cursor = 0, 0; self._new_pass()
    def _new_pass(self):
        rs = np.random.RandomState((self.seed * 1000003 + self.pass_no + 1) % (2**31 - 1))
        self.pass_order = rs.permutation(self.n)[self.rank::self.world]; self.cursor = 0
    def state(self): return {"pass_no": self.pass_no, "cursor": self.cursor, "seed": self.seed, "n": self.n}
    def load_state(self, st):
        if st is None or int(st.get("seed", self.seed)) != self.seed or int(st.get("n", self.n)) != self.n: return False
        self.pass_no = int(st["pass_no"]); self._new_pass(); self.cursor = int(st["cursor"]); return True
    def entry(self, k):
        idx = int(self.order[k])
        g = torch.Generator().manual_seed((self.seed * 1000003 + int(k)) % (2**63 - 1))
        m = torch.from_numpy(np.load(os.path.join(self.ds.features_dir, self.ds.feature_fnames[idx]))).float()
        m = m.reshape(m.shape[-3], m.shape[-2], m.shape[-1])
        mean, std = torch.chunk(m, 2, dim=0)
        z = (mean + std * torch.randn(mean.shape, generator=g)) * self.ls + self.lb
        t = torch.rand((1,), generator=g); e = torch.randn(z.shape, generator=g)
        return z, int(self.ds.labels[idx]), t, e
    def signature(self):
        z, y, t, e = self.entry(0); return f"fixed-set seed {self.seed} n {self.n} entry0 img {int(self.order[0])} y {y} t {t.item():.6f} z-sum {z.sum().item():.4f} e-sum {e.sum().item():.4f}"
    def next(self):
        if self.cursor + self.per_step > len(self.pass_order): self.pass_no += 1; self._new_pass()
        ks = self.pass_order[self.cursor:self.cursor + self.per_step]; self.cursor += self.per_step
        zs, ys, ts, es = zip(*(self.entry(int(k)) for k in ks))
        return (torch.stack(zs).to(self.device), torch.tensor(ys, device=self.device), torch.stack(ts).view(-1, 1, 1, 1).to(self.device), torch.stack(es).to(self.device))

class RotatingTripleSets:
    """DSMC rotating pools: K equal groups of rows; group j holds a pool of fixed triples on its j-th pass (group 1 = a new pool, so its
    triples are fresh). Every `window` steps (= exactly one pass of a pool at its per-step share) the pools rotate: a new pool (seed =
    base + 1000*g) enters group 1, every pool moves one group down, and the pool that completed K passes is retired. Every triple is
    seen exactly K times. Rows come back grouped (group 1 first). The pool size follows from the window: rows per step x window."""
    def __init__(self, dataset, base_seed, steps, global_batch, rank, world, device, latents_scale, latents_bias, K, window, start_step):
        assert int(global_batch) % (int(K) * int(world)) == 0, "the global batch must split into K sections across the ranks"
        self.K, self.window, self.start, self.base_seed, self.rank = int(K), int(window), int(start_step), int(base_seed), int(rank)
        self.per_sec = int(global_batch) // (self.K * int(world))
        self.n = self.per_sec * int(world) * self.window   # set size in triples = rows per step x window: one pass per window by construction (need not be a multiple of the batch)
        if int(steps) * int(global_batch) != self.n and rank == 0: print(f"[rotation] set size derived from the window: {self.n} triples (--fixed-set-steps x batch = {int(steps) * int(global_batch)} ignored)", flush=True)
        self.mk = lambda g: FixedTripleSet(dataset, self.base_seed + 1000 * int(g), self.n, 1, rank, world, self.per_sec, device, latents_scale, latents_bias)
        self.w = 0; self.gens = [self.K - 1 - j for j in range(self.K)]   # section j (0-based) holds generation K-1-j; all K sets are new at the start
        self.sets = [self.mk(g) for g in self.gens]
    def rotate(self):
        self.w += 1; self.gens = [self.gens[0] + 1] + self.gens[:-1]; self.sets = [self.mk(self.gens[0])] + self.sets[:-1]
        if self.rank == 0: print(f"[rotation] window {self.w} begins at step {self.start + self.w * self.window}: section generations {self.gens} (seeds {[s.seed for s in self.sets]}); generation {self.gens[0] - self.K} retired after {self.K} passes", flush=True)
    def state(self): return {"rotation": True, "w": self.w, "gens": list(self.gens), "sets": [s.state() for s in self.sets], "base_seed": self.base_seed, "K": self.K, "window": self.window, "start": self.start}
    def load_state(self, st):
        if not st or not st.get("rotation") or int(st.get("base_seed", -1)) != self.base_seed or int(st.get("K", -1)) != self.K \
           or int(st.get("window", -1)) != self.window or int(st.get("start", -1)) != self.start: return False
        self.w = int(st["w"]); self.gens = [int(g) for g in st["gens"]]; self.sets = [self.mk(g) for g in self.gens]
        return all(s.load_state(ss) for s, ss in zip(self.sets, st["sets"]))
    def signature(self): return f"rotation K {self.K} window {self.window} set {self.n} triples start {self.start} w {self.w} gens {self.gens} per-section/rank {self.per_sec} | " + " | ".join(s.signature() for s in self.sets)
    def next(self, step):
        target_w = max(0, (int(step) - self.start) // self.window)
        while self.w < target_w: self.rotate()
        parts = [s.next() for s in self.sets]
        return tuple(torch.cat([p[i] for p in parts]) for i in range(4))

class CompositeReplay:
    """DSMC Stage 2: the fixed set of Stage 1 stays in the batch (same cadence, never retired; its pass cursor resumes from the Stage 1
    checkpoint) and a RotatingTripleSets fills the remaining rows. Rows: the fixed-set block first, then the rotating groups."""
    def __init__(self, persist, rot):
        self.persist, self.rot = persist, rot
    def state(self): return {"composite": True, "persist": self.persist.state(), "rotation": self.rot.state()}
    def load_state(self, st):
        if not st: return False
        if st.get("composite"): return self.persist.load_state(st.get("persist")) and self.rot.load_state(st.get("rotation"))
        return self.persist.load_state(st)   # a Stage 1 checkpoint: restore the fixed set's cursor, the rotation starts fresh
    def signature(self): return f"persistent {self.persist.signature()} || {self.rot.signature()}"
    def next(self, step):
        a = self.persist.next(); b = self.rot.next(step)
        return tuple(torch.cat([a[i], b[i]]) for i in range(4))

def rotation_ticks(res):
    """Rotating pools supply every row, so the loader only feeds the preview panel; the loop is driven by empty rows."""
    while True: yield torch.zeros(0, 3, res, res), torch.zeros(0, 1, 8, res // 8, res // 8), torch.zeros(0, dtype=torch.long)

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != 'None':
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
    else:  # no encoder: training without representation alignment
        encoders, encoder_types, architectures = [], [], []
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else []
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    if args.compile:  # in-place compile: state_dict keys stay clean; EMA copied above stays eager
        model.compile()
        if args.enc_type != 'None':
            for encoder in encoders:
                encoder.compile()
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    param_names = None
    if args.entry_lr > 0:
        param_groups, param_names = build_param_groups(model, args.learning_rate, args.entry_lr)
        accelerator.print(f"entry group: {sum(p.numel() for p in param_groups[1]['params']):,} params at lr {args.entry_lr}; "
                          f"body: {sum(p.numel() for p in param_groups[0]['params']):,} params at lr {args.learning_rate}")   # rank-safe (logger exists on rank 0 only)
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data:
    train_dataset = CustomDataset(args.data_dir)
    fixed_on = args.fixed_set_seed >= 0   # DSMC: part of every batch comes from a fixed set of triples
    rotate_on = fixed_on and args.rotate_sets > 0   # rotating pools: every row comes from one of K rotating pools
    fixed_per_proc = (int(args.batch_size // accelerator.num_processes) if rotate_on else (int(round(args.batch_size * args.fixed_frac)) // accelerator.num_processes)) if fixed_on else 0
    local_batch_size = int(args.batch_size // accelerator.num_processes) - fixed_per_proc
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size if local_batch_size > 0 else 64,   # rotation: the loader only feeds the 64-image preview panel
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    persist_on = rotate_on and args.persist_sections > 0
    persist_per_proc = 0
    if persist_on:   # DSMC Stage 2: P groups hold the fixed set of Stage 1, beside K rotating groups of equal width
        sec_rows = args.batch_size // (args.persist_sections + args.rotate_sets)
        assert sec_rows * (args.persist_sections + args.rotate_sets) == args.batch_size and (sec_rows * args.persist_sections) % accelerator.num_processes == 0, "sections must tile the batch across ranks"
        persist_per_proc = sec_rows * args.persist_sections // accelerator.num_processes
        persist = FixedTripleSet(train_dataset, args.persist_set_seed, args.persist_set_steps, args.batch_size, accelerator.process_index, accelerator.num_processes, persist_per_proc, device, latents_scale, latents_bias)
        rot = RotatingTripleSets(train_dataset, args.fixed_set_seed, args.fixed_set_steps, args.batch_size - sec_rows * args.persist_sections, accelerator.process_index, accelerator.num_processes,
                                 device, latents_scale, latents_bias, args.rotate_sets, args.rotate_window, args.rotate_start if args.rotate_start >= 0 else args.resume_step)
        fixed_set = CompositeReplay(persist, rot)
        accelerator.print(f"DSMC Stage 2: {fixed_set.signature()}; per process {persist_per_proc} persistent + {args.rotate_sets} x {rot.per_sec} rotating rows; loader idle (preview only)")
    elif rotate_on:
        fixed_set = RotatingTripleSets(train_dataset, args.fixed_set_seed, args.fixed_set_steps, args.batch_size, accelerator.process_index, accelerator.num_processes,
                                       device, latents_scale, latents_bias, args.rotate_sets, args.rotate_window, args.rotate_start if args.rotate_start >= 0 else args.resume_step)
        accelerator.print(f"DSMC rotating pools: {fixed_set.signature()}; per process {fixed_per_proc} rows = {args.rotate_sets} sections x {fixed_set.per_sec}; loader idle (preview only)")
    else:
        fixed_set = FixedTripleSet(train_dataset, args.fixed_set_seed, args.fixed_set_steps, args.batch_size, accelerator.process_index, accelerator.num_processes, fixed_per_proc, device, latents_scale, latents_bias) if fixed_on else None
    if fixed_set is not None and not rotate_on:
        accelerator.print(f"DSMC Stage 1: {fixed_set.signature()}; per process {fixed_per_proc} fixed + {local_batch_size} fresh per step; frac {args.fixed_frac}")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    skipped_steps = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt_path = args.resume_ckpt or f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}'   # --resume-ckpt: explicit file (e.g. the checkpoint of another experiment)
        ckpt = torch.load(
            ckpt_path,
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        opt_sd = ckpt['opt']
        if args.entry_lr > 0 and len(opt_sd['param_groups']) == 1:   # checkpoint from the single-group recipe: split its state
            opt_sd = remap_single_group_state(opt_sd, param_names, args.learning_rate, args.entry_lr)
        optimizer.load_state_dict(opt_sd)
        group_lrs = [args.learning_rate, args.entry_lr] if args.entry_lr > 0 else [args.learning_rate]
        for g, lr in zip(optimizer.param_groups, group_lrs):   # the restored state carries the old lr/betas; the flags win on resume (no-op when equal)
            g['lr'] = lr
            g['betas'] = (args.adam_beta1, args.adam_beta2)
        if fixed_set is not None:
            ok = fixed_set.load_state(ckpt.get('fixed_set'))
            accelerator.print(f"fixed-set state {'restored' if ok else 'started fresh'}: {fixed_set.state()}")
        global_step = args.resume_step if args.resume_ckpt else ckpt['steps']   # with --resume-ckpt the counter is the flag
        accelerator.print(f"resumed from step {global_step} (checkpoint steps {ckpt['steps']}): lrs {[g['lr'] for g in optimizer.param_groups]} (groups {[len(g['params']) for g in optimizer.param_groups]}), betas {optimizer.param_groups[0]['betas']}, skip-grad-norm {args.skip_grad_norm}, max-grad-norm {args.max_grad_norm}, warmup {args.warmup_steps}")

    group_target_lrs = [g['lr'] for g in optimizer.param_groups]   # targets for --warmup-steps (linear ramp from warmup_start)
    warmup_start = args.warmup_start if args.warmup_start >= 0 else args.resume_step   # --warmup-start: the step the ramp began (default: this resume)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64   # 64 previews per process: rank 0 keeps the single-process labels/noise; extra ranks get their own panel
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
        
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in (rotation_ticks(args.resolution) if rotate_on else train_dataloader):
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice: 
                # once in train.py and once in sit.py. 
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        zs.append(z)

            with accelerator.accumulate(model):
                n_fresh = x.shape[0]
                if fixed_set is not None:   # DSMC: append the fixed triples; fresh rows get explicit draws so one call scores both halves
                    t_in = torch.rand((n_fresh, 1, 1, 1), device=device, dtype=x.dtype); e_in = torch.randn_like(x)
                    xf, yf, tf, ef = fixed_set.next(global_step) if rotate_on else fixed_set.next()
                    x = torch.cat([x, xf.to(x.dtype)]); labels = torch.cat([labels, yf]); t_in = torch.cat([t_in, tf.to(x.dtype)]); e_in = torch.cat([e_in, ef.to(x.dtype)])
                else:
                    t_in = e_in = None
                model_kwargs = dict(y=labels)
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs, time_input=t_in, noises=e_in)
                loss_mean = loss.mean()
                if rotate_on:   # rows are grouped: [fixed-set block] group 1 = first pass (fresh triples) ... group K = K-th pass
                    rep = loss[n_fresh:].detach(); loss_persist = rep[:persist_per_proc].mean() if persist_per_proc else None
                    loss_sec = rep[persist_per_proc:].view(args.rotate_sets, -1).mean(dim=1); loss_fresh = loss_sec[0]
                    loss_fixed = torch.cat([rep[:persist_per_proc], rep[persist_per_proc:].view(args.rotate_sets, -1)[1:].flatten()]).mean() if persist_per_proc else loss_sec[1:].mean()
                else:
                    loss_fresh = loss[:n_fresh].mean().detach(); loss_fixed = loss[n_fresh:].mean().detach() if fixed_set is not None else loss_fresh
                proj_loss_mean = proj_loss.mean()
                loss = loss_mean * args.denoise_coeff + proj_loss_mean * args.proj_coeff   # denoise_coeff=0 => alignment-only
                    
                ## optimization
                accelerator.backward(loss)
                skip_step = False
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    # optional (--skip-grad-norm > 0): discard a step whose pre-clip gradient norm is finite but above the threshold;
                    # non-finite norms are left to the fp16 scaler, which skips them and halves its scale. Off when the flag is 0.
                    in_warmup = args.warmup_steps > 0 and (global_step - warmup_start) < args.warmup_steps   # no skip during a warmup
                    if args.skip_grad_norm > 0 and not in_warmup and torch.isfinite(grad_norm) and grad_norm > args.skip_grad_norm:
                        skip_step = True; skipped_steps += 1
                        if accelerator.is_main_process:
                            unclip = float(grad_norm) / args.max_grad_norm if float(grad_norm) > args.max_grad_norm else 1.0
                            tops = sorted(((float(p.grad.norm()) * unclip, n) for n, p in model.named_parameters() if p.grad is not None), reverse=True)[:10]
                            with open(os.path.join(args.output_dir, args.exp_name, "grad_spikes.log"), "a") as fh:
                                fh.write(f"step {global_step + 1} grad_norm {float(grad_norm):.4g} loss {loss_mean.item():.4f} top: "
                                         + "; ".join(f"{n} {g:.3g}" for g, n in tops) + "\n")
                if args.warmup_steps > 0 and (global_step - warmup_start) < args.warmup_steps:   # linear lr warmup anchored at warmup_start (survives a mid-warmup resume)
                    ramp = (global_step - warmup_start + 1) / args.warmup_steps
                    for g, lr in zip(optimizer.param_groups, group_target_lrs):
                        g['lr'] = lr * ramp
                if skip_step:
                    optimizer.zero_grad(set_to_none=True)
                    if accelerator.scaler is not None:
                        accelerator.scaler.update()   # clip_grad_norm_ already unscaled; reset the scaler's per-step state without stepping
                else:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if accelerator.sync_gradients:
                        update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0 and accelerator.sync_gradients:   # sync guard: with grad accumulation >1 the same global_step spans several micro-steps
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                        "fixed_set": fixed_set.state() if fixed_set is not None else None,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if accelerator.sync_gradients and (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, 
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                img_logs = {"samples": wandb.Image(array2grid(out_samples[:64])),
                            "gt_samples": wandb.Image(array2grid(gt_samples[:64]))}
                for r in range(1, accelerator.num_processes):   # one 8x8 panel per extra process
                    img_logs[f"samples_rank{r}"] = wandb.Image(array2grid(out_samples[64*r:64*(r+1)]))
                accelerator.log(img_logs)
                logging.info("Generating EMA samples done.")

            if accelerator.sync_gradients:   # grad_norm only exists on sync micro-steps (official code assumed accumulation=1); identical behaviour at accumulation=1
                logs = {"loss_fresh": loss_fresh.item(), "loss_fixed": loss_fixed.item(), 
                    **({f"loss_s{j + 1}": loss_sec[j].item() for j in range(args.rotate_sets)} if rotate_on else {}),
                    **({"loss_persist": loss_persist.item()} if rotate_on and persist_per_proc else {}),
                    "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                    "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                    "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                    "skipped": skipped_steps,
                    "lr": optimizer.param_groups[0]['lr']
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile", action="store_true")

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--skip-grad-norm", type=float, default=0.0)   # >0: discard steps whose pre-clip gradient norm exceeds it
    parser.add_argument("--entry-lr", type=float, default=0.0)   # >0: separate learning rate for the embedders, the controller tokens and the gates
    parser.add_argument("--warmup-steps", type=int, default=0)   # >0: linear lr warmup over this many steps after the resume step
    parser.add_argument("--fixed-set-seed", type=int, default=-1)    # DSMC: >=0 enables; seed of the fixed set of triples
    parser.add_argument("--fixed-set-steps", type=int, default=50000) # size of the fixed set in global batches (50k x 256 = 12.8M triples)
    parser.add_argument("--fixed-frac", type=float, default=0.5)     # share of every global batch drawn from the fixed set
    parser.add_argument("--rotate-sets", type=int, default=0)        # rotating pools: K>0 = K rotating pools (seed = --fixed-set-seed + 1000*g)
    parser.add_argument("--rotate-window", type=int, default=100000) # steps per rotation = one pass of a set at its per-step share
    parser.add_argument("--rotate-start", type=int, default=-1)      # step the windows count from (default -1 = the resume step); pass it on every relaunch
    parser.add_argument("--persist-sections", type=int, default=0)   # DSMC Stage 2: P groups hold a persistent fixed set (never rotated) beside the K rotating groups
    parser.add_argument("--persist-set-seed", type=int, default=20261024)  # seed of the persistent set (the fixed set of Stage 1)
    parser.add_argument("--persist-set-steps", type=int, default=50000)    # its size in global batches (50k x 256 = 12.8M)
    parser.add_argument("--warmup-start", type=int, default=-1)   # step at which the warmup ramp began (default -1 = the resume step); pass it when resuming inside a warmup
    parser.add_argument("--resume-ckpt", type=str, default=None)   # explicit checkpoint file to resume from (with --resume-step for the counter)

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--denoise-coeff", type=float, default=1.0, help="weight of the denoising loss; 0 = alignment-only run")
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
