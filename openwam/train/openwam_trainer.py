"""OpenWAM joint video-action trainer (self-contained).

Call order — core skeleton only:

  __init__():  seed -> build_architecture -> freeze -> init_schedulers
               -> [latent setup] -> log param counts
  train():     build optimizer/dataloader/scheduler -> setup output dir
               -> accelerate prepare -> loop{ compute_loss -> backward/clip/step
               -> reduce metrics -> log step -> maybe save ckpt }
               -> finish_training (final ckpt, drop resume state, close wandb)

Methods below are ordered by call sequence: __init__, train, then the
helpers in the order train() reaches them (sub-helpers follow their caller).

Stateless helpers live in ``openwam.train.utils`` (config / param report / LR /
wandb / metric reduction / debug-CSV / checkpoint mgmt / optimizer groups /
seeding).

Usage:
    trainer = OpenWAMTrainer(cfg, accelerator, dataset)
    trainer.train()
"""

import itertools
import logging
import math
import os

import torch
from omegaconf import DictConfig

from openwam.train.utils.checkpointing import (
    compute_resume_position,
    finalize_keep_weights_only,
    find_latest_accel_state,
    load_full_state,
    manage_checkpoints,
    save_config,
    save_full_state,
    save_normalization_stats,
    save_weights,
    verify_resume_normalization_stats,
)
from openwam.train.utils.optimizer_groups import build_trainable_parameters
from openwam.train.utils.seeding import per_step_seed, seed_process, wire_sampler_seed
from openwam.train.utils.training_utils import (
    build_cosine_scheduler,
    cfg_get,
    init_wandb,
    log_parameter_counts,
    reduce_step_metrics,
    write_debug_loss_row,
)

logger = logging.getLogger(__name__)


class OpenWAMTrainer:
    """Joint video-action trainer for OpenWAM. See module docstring for call order.

    Args:
        cfg: Hydra DictConfig with model, training, data, project sections.
        accelerator: HuggingFace Accelerator instance.
        dataset: Training dataset (used for action stats loading).
    """

    # (1) Constructor — seed -> build architecture -> freeze -> init schedulers -> param report.
    def __init__(self, cfg: DictConfig, accelerator=None, dataset=None):
        self.cfg = cfg
        self.dataset = dataset
        self.accelerator = accelerator

        # ---- Reproducible seed (FastWAM-style, yaml-driven) ----
        # Seed before build_architecture so DiT/ActionDiT weight init is
        # deterministic. seed_process uses the same RANK_OFFSET rank stride as
        # per_step_seed and the launcher's seed_everything, so a process's init
        # and per-step seeds share one rank window (cudnn left to the launcher).
        project_cfg = getattr(cfg, "project", None)
        yaml_seed = getattr(project_cfg, "seed", None) if project_cfg is not None else None
        self._rank = int(os.environ.get("RANK", 0))
        self._run_seed = int(yaml_seed) if yaml_seed is not None else None
        if self._run_seed is not None:
            seed_process(self._run_seed, rank=self._rank)
            if self._rank == 0:
                logger.info("Reproducible mode: cfg.project.seed=%d (rank=%d)", self._run_seed, self._rank)

        t = cfg.training
        m = cfg.model

        # Build architecture (creates video_backbone internally from config).
        from openwam.model import build_architecture, resolve_architecture_config

        # Finetune/resume: build from the self-contained checkpoint dir
        # (skeletons from its config.yaml component specs, weights from its
        # safetensors) so model.video_backbone.model_path need not exist on
        # this host. Weights land here, BEFORE the freeze below. The ckpt config
        # is the reconstruction BASE only — on finetune this run's cfg.model is
        # layered on top (see ckpt_model_loader.merge_ckpt_model_cfg).
        finetune_path = cfg_get(t, "finetune_ckpt_path", None) or None
        resume_path = cfg_get(t, "resume_ckpt_path", None) or None
        if finetune_path and resume_path:
            raise ValueError("finetune_ckpt_path and resume_ckpt_path are mutually exclusive; set at most one.")
        self._ckpt_source_dir = finetune_path or resume_path
        use_t5_cache = bool(getattr(getattr(cfg, "dataloader", None), "use_t5_cache", False))
        if self._ckpt_source_dir is not None:
            from openwam.train.utils.ckpt_model_loader import (
                build_architecture_from_ckpt_dir,
                propagate_component_specs,
                warn_live_model_cfg_ignored,
            )

            # Finetune starts a NEW run: the ckpt config is only the
            # reconstruction base (it alone carries the component/tokenizer
            # specs), and this run's cfg.model overrides it — merged in place, so
            # the architecture built here and the config save_config() writes to
            # the new run dir are the same one. Resume continues ONE run whose
            # config.yaml is already on disk and is reused untouched, so there the
            # ckpt config stays authoritative and divergence is only reported.
            # use_t5_cache → skip UMT5 construction (disk embeds); not written into
            # saved config.yaml so deploy can still build T5 from component specs.
            resolved_arch, self.architecture, ckpt_cfg = build_architecture_from_ckpt_dir(
                self._ckpt_source_dir,
                weights_required=finetune_path is not None,
                override_cfg=cfg if finetune_path is not None else None,
                skip_text_encoder=use_t5_cache,
            )
            if finetune_path is not None:
                # The merge replaced the cfg.model node, so `m` (bound above)
                # still points at the pre-merge one — rebind it before `freeze`
                # and the architecture logging below read from it.
                m = cfg.model
            else:
                warn_live_model_cfg_ignored(ckpt_cfg, cfg, self._ckpt_source_dir)
                propagate_component_specs(ckpt_cfg, cfg)
        else:
            resolved_arch = resolve_architecture_config(m)
            self.architecture = build_architecture(resolved_arch.registry_name, resolved_arch.params)
        logger.info(
            "Architecture: %s (framework=%s variant=%s)",
            resolved_arch.registry_name,
            resolved_arch.canonical.framework,
            resolved_arch.canonical.variant,
        )

        # Device placement: skip .to(device) when initialize_model_on_cpu and a real
        # training Accelerator is present — DeepSpeed's prepare() then handles the move.
        _init_on_cpu = bool(t.get("initialize_model_on_cpu", False))
        if not (_init_on_cpu and self.accelerator is not None):
            self.architecture.set_dtype_device(self.architecture.dtype, self.architecture.device)

        # --- Freeze: declared per-architecture in the model yaml (freeze:);
        # freeze_modules silently skips paths absent on a given architecture.
        freeze_list = list(getattr(m, "freeze", []))
        # Single extra knob: freeze video DiT (default false). In FPD mode this
        # only applies to the student; the teacher is fully frozen separately.
        if bool(getattr(m, "freeze_video_dit", False)):
            freeze_list.append("video_backbone.dit")
        for name in self.architecture.freeze_modules(freeze_list):
            logger.info("Frozen: %s", name)

        # Initialize all schedulers (video + action) inside architecture
        self.architecture.init_training_schedulers(1000)

        # Loss weights from the training config
        self.lambda_video = float(t.lambda_video)
        self.lambda_action = float(t.lambda_action)
        self.lambda_distill = float(getattr(t, "lambda_distill", 0.0) or 0.0)
        self.use_distill_gate = bool(getattr(t, "use_distill_gate", False))
        # ``clean`` = FastWAM-VLA-style teacher pretrain: GT video at σ≈0 endpoint,
        # action FM only (pair with lambda_video=0). Default ``train`` = noisy video.
        self.video_mode = str(getattr(t, "video_mode", "train") or "train")
        if self.video_mode not in ("train", "clean"):
            raise ValueError(f"training.video_mode must be 'train' or 'clean', got {self.video_mode!r}")

        # Optional FPD: build a frozen teacher (not registered on self.architecture).
        self.fpd = None
        fpd_cfg = getattr(cfg, "fpd", None)
        self._fpd_enabled = bool(getattr(fpd_cfg, "enabled", False)) if fpd_cfg is not None else False
        if self._fpd_enabled:
            if self.video_mode == "clean":
                raise ValueError(
                    "fpd.enabled=true is incompatible with training.video_mode=clean "
                    "(clean mode is for teacher pretrain only; FPD student uses noisy video)."
                )
            self._setup_fpd(fpd_cfg)
        elif self.video_mode == "clean":
            logger.info(
                "Teacher-pretrain mode: video_mode=clean (σ≈0 GT video), "
                "lambda_video=%.4g lambda_action=%.4g",
                self.lambda_video,
                self.lambda_action,
            )

        # Push forward-time training flags onto the architecture so prepare_inputs
        # is self-contained.
        self.architecture.set_training_runtime(
            use_gradient_checkpointing=bool(t.use_gradient_checkpointing),
            use_gradient_checkpointing_offload=bool(t.use_gradient_checkpointing_offload),
            max_timestep_boundary=float(t.max_timestep_boundary),
            min_timestep_boundary=float(t.min_timestep_boundary),
        )

        self.model = self  # self-reference some external callers expect

        is_main = self.accelerator is None or self.accelerator.is_main_process
        if is_main:
            mot_ckpt = None
            driver = getattr(self.architecture, "_mot_driver", None)
            if driver is not None:
                mot_ckpt = getattr(driver, "mot_checkpoint_mixed_attn", None)
            logger.info(
                "[mem-cfg] use_gradient_checkpointing=%s mot_checkpoint_mixed_attn=%s "
                "zero_stage=%s (FastWAM robotwin: both ckpt flags false)",
                bool(t.use_gradient_checkpointing),
                mot_ckpt,
                getattr(t, "zero_stage", None),
            )
        log_parameter_counts(self.architecture, is_main=is_main)

    # (2) Driver — build optimizer/dataloader/scheduler -> setup dir -> accelerate prepare
    #     -> (resume) -> epoch/step loop{compute_loss -> log_step -> save} -> finish_training.
    def train(self, num_epochs: int = None, max_steps: int = None):
        """Run the training loop (HuggingFace Accelerate distributed).

        Three entry modes (train.yaml finetune/resume fields): fresh, finetune warm-start
        (weights loaded at construction), or resume (load full state after prepare).
        """
        t = self.cfg.training
        num_epochs = num_epochs or cfg_get(t, "num_epochs", None)
        num_epochs = int(num_epochs) if num_epochs is not None else None
        max_steps = max_steps or getattr(t, "max_steps", None)
        batch_size = int(t.batch_size)
        grad_accum = int(t.gradient_accumulation_steps)

        debug = bool(getattr(t, "debug", False))
        if debug:
            if max_steps is None:
                max_steps = 20
            else:
                max_steps = min(int(max_steps), 20)
            save_steps_override = min(10, int(max_steps))
            logger.info("DEBUG mode: max_steps=%s, save@%s, constant LR", max_steps, save_steps_override)

        # num_epochs=null means step-only training: max_steps is the sole stop condition.
        if num_epochs is None and not max_steps:
            raise ValueError(
                "training.num_epochs and training.max_steps are both unset; "
                "set num_epochs, or set max_steps for step-only training."
            )

        # finetune/resume validation + architecture construction happen in
        # __init__ (self-contained ckpt-dir path); only resume needs a path here.
        resume_path = cfg_get(t, "resume_ckpt_path", None) or None

        optimizer = self.build_optimizer()
        dataloader = self.build_dataloader(batch_size)
        max_grad_norm = float(t.max_grad_norm) if getattr(t, "max_grad_norm", None) else None

        n_proc = self.accelerator.num_processes
        steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
        # max_steps counts micro-steps (global_step); convert to optimizer steps for the LR horizon.
        max_opt_steps = math.ceil(max_steps / grad_accum) if max_steps else None
        if num_epochs is not None:
            # prepare() shards the dataloader ~1/n_proc; fold that in so total_opt_steps is the
            # per-process optimizer steps the loop actually runs (same unit as max_opt_steps).
            steps_per_epoch = math.ceil(steps_per_epoch / n_proc)
            total_opt_steps = steps_per_epoch * num_epochs
            if max_opt_steps:
                total_opt_steps = min(total_opt_steps, max_opt_steps)
        else:
            total_opt_steps = max_opt_steps
        scheduler = self.build_lr_scheduler(optimizer, total_opt_steps, debug=debug)

        if debug:
            save_steps = save_steps_override
        else:
            save_steps = getattr(t, "save_steps", None)
            if save_steps is not None:
                save_steps = int(save_steps)
        keep_last_k = int(getattr(t, "keep_last_k_ckpts", 3))
        save_full_states_for_resume = bool(getattr(t, "save_full_states_for_resume", False))

        # Finetune warm-start weights were already loaded at architecture
        # construction (__init__, self-contained ckpt-dir path). Step stays 0.

        output_path, resume_state_dir = self.setup_output_dir(debug, resume_path)
        optimizer, dataloader, scheduler = self.prepare_accelerate(optimizer, dataloader, scheduler)

        if self._run_seed is not None:
            from openwam.dataloader.mixture import MixtureDataset

            if isinstance(self.dataset, MixtureDataset):
                logger.info(
                    "Skipping DataLoader sampler seed wiring for MixtureDataset; "
                    "its index-map shuffle is seeded by the dataset itself."
                )
            else:
                wire_sampler_seed(dataloader, int(self._run_seed), rank=self._rank)

        all_params = [p for group in optimizer.param_groups for p in group["params"]]

        is_main = self.accelerator is None or self.accelerator.is_main_process
        wandb_run = None if (debug or not is_main) else init_wandb(self.cfg)

        from tqdm import tqdm

        if num_epochs is not None:
            total_steps = len(dataloader) * num_epochs
            if max_steps:
                total_steps = min(total_steps, max_steps)
        else:
            total_steps = max_steps

        import time as _time

        opt_step = 0
        global_step = 0
        start_epoch = 0
        skip_first = 0

        # Resume: restore full state AFTER prepare, then map global_step -> (epoch, skip).
        if resume_state_dir is not None:
            global_step, opt_step, start_epoch, skip_first = self.resume_if_configured(
                resume_state_dir, dataloader, grad_accum
            )
            already_done = (num_epochs is not None and start_epoch >= num_epochs) or (
                max_steps and global_step >= max_steps
            )
            if already_done:
                logger.info("[resume] global_step=%d already complete; finishing.", global_step)
                self.finish_training(output_path, global_step, save_steps, keep_last_k, is_main, wandb_run)
                return
            if skip_first > 0 and self._run_seed is None:
                logger.warning(
                    "[resume] mid-epoch resume (skip_first=%d) without project.seed: DataLoader "
                    "shuffle is non-reproducible, so the resumed epoch's batch order differs from "
                    "the original run — samples may be silently re-fed or skipped. Set project.seed "
                    "for faithful mid-epoch resume.",
                    skip_first,
                )

        _step_t0 = _time.monotonic()
        pbar = tqdm(total=total_steps, desc="Training", unit="step", initial=min(global_step, total_steps))

        assert self.accelerator is not None, "OpenWAMTrainer requires an Accelerator"

        # Per-step manual_seed makes timestep + noise sampling in compute_loss
        # reproducible across runs and ZeRO stages: same (rank, step) -> same RNG,
        # different ranks at the same step keep in-batch timestep diversity.
        # Gated on _run_seed so unseeded production runs stay fully stochastic.
        epochs = itertools.count(start_epoch) if num_epochs is None else range(start_epoch, num_epochs)
        for epoch in epochs:
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(epoch)
            # On the resumed epoch, skip the batches already consumed before the checkpoint.
            if epoch == start_epoch and skip_first > 0:
                from accelerate import skip_first_batches

                epoch_iter = skip_first_batches(dataloader, skip_first)
            else:
                epoch_iter = dataloader
            for batch in epoch_iter:
                if self._run_seed is not None:
                    step_seed = per_step_seed(self._run_seed, rank=self._rank, step=global_step)
                    torch.manual_seed(step_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(step_seed)
                if debug and global_step == 0 and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    was_training = self.architecture.training
                    self.architecture.eval()
                    with torch.no_grad():
                        self.compute_loss(batch)
                    logger.info(
                        "[mem] infer_forward allocated=%.2fGiB reserved=%.2fGiB peak=%.2fGiB",
                        torch.cuda.memory_allocated() / (1024**3),
                        torch.cuda.memory_reserved() / (1024**3),
                        torch.cuda.max_memory_allocated() / (1024**3),
                    )
                    if was_training:
                        self.architecture.train()
                    torch.cuda.reset_peak_memory_stats()
                with self.accelerator.accumulate(self.architecture):
                    losses = self.compute_loss(batch)
                    loss = losses["total"]
                    if not torch.isfinite(loss).all():
                        # Distinguish "bad batch / forward" vs "weights already poisoned".
                        bad_params = 0
                        bad_names: list[str] = []
                        try:
                            arch = self.accelerator.unwrap_model(self.architecture)
                            for name, p in arch.named_parameters():
                                if p.requires_grad and p.numel() and not torch.isfinite(p).all():
                                    bad_params += 1
                                    if len(bad_names) < 8:
                                        bad_names.append(name)
                        except Exception:
                            bad_params = -1
                        raise RuntimeError(
                            f"non-finite loss at step={global_step} epoch={epoch}: "
                            f"total={loss.detach()} video={losses.get('video')} "
                            f"action={losses.get('action')} bad_trainable_params={bad_params} "
                            f"bad_names={bad_names}"
                        )
                    self.accelerator.backward(loss)

                    grad_norm = torch.tensor(0.0, device=loss.device)
                    if self.accelerator.sync_gradients:
                        if max_grad_norm is not None:
                            grad_norm_val = self.accelerator.clip_grad_norm_(all_params, max_grad_norm)
                            grad_norm = torch.tensor(float(grad_norm_val), device=loss.device)
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        opt_step += 1

                global_step += 1

                if is_main and torch.cuda.is_available() and (debug or global_step % 10 == 0):
                    alloc = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    peak = torch.cuda.max_memory_allocated() / (1024**3)
                    avg_gib = 0.0
                    ipg_gib = 0.0
                    try:
                        wrapped = getattr(self.accelerator, "deepspeed_engine_wrapped", None)
                        ds_opt = getattr(getattr(wrapped, "engine", None), "optimizer", None)
                        if ds_opt is not None:
                            for lst in getattr(ds_opt, "averaged_gradients", {}).values() or ():
                                if not lst:
                                    continue
                                for t_ in lst:
                                    if t_ is not None and hasattr(t_, "numel"):
                                        avg_gib += t_.numel() * t_.element_size()
                            avg_gib /= 1024**3
                            for bucket in getattr(ds_opt, "ipg_buckets", {}).values() or ():
                                for g in getattr(bucket, "grads", []) or []:
                                    if g is not None and hasattr(g, "numel"):
                                        ipg_gib += g.numel() * g.element_size()
                            ipg_gib /= 1024**3
                    except Exception:
                        pass
                    logger.info(
                        "[mem] step=%d allocated=%.2fGiB reserved=%.2fGiB peak=%.2fGiB "
                        "ds_avg_grads=%.2fGiB ds_ipg=%.2fGiB",
                        global_step,
                        alloc,
                        reserved,
                        peak,
                        avg_gib,
                        ipg_gib,
                    )

                metrics = reduce_step_metrics(self.accelerator, losses, grad_norm)

                current_lr = optimizer.param_groups[0]["lr"]
                _now = _time.monotonic()
                steps_per_sec = 1.0 / max(_now - _step_t0, 1e-9)
                _step_t0 = _now

                self.log_step(
                    metrics=metrics,
                    global_step=global_step,
                    opt_step=opt_step,
                    epoch=epoch,
                    lr=current_lr,
                    steps_per_sec=steps_per_sec,
                    batch_size=batch_size,
                    pbar=pbar,
                    wandb_run=wandb_run,
                    debug=debug,
                    output_path=output_path,
                )

                # save_steps: write the weights line (+ the resumable full state when
                # save_full_states_for_resume=true), then prune in lockstep.
                if save_steps and global_step > 0 and global_step % save_steps == 0:
                    save_weights(self.accelerator, self.architecture, output_path, global_step, final=False)
                    if save_full_states_for_resume:
                        save_full_state(self.accelerator, output_path, global_step, opt_step, epoch)
                    if is_main:
                        manage_checkpoints(output_path, keep_last_k)

                if max_steps and global_step >= max_steps:
                    pbar.close()
                    self.finish_training(output_path, global_step, save_steps, keep_last_k, is_main, wandb_run)
                    return

        pbar.close()
        self.finish_training(output_path, global_step, save_steps, keep_last_k, is_main, wandb_run)

    # (3) Called by train() first — AdamW over the per-module (action/video) LR param groups.
    def build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW over the per-module (action/video) LR parameter groups."""
        t = self.cfg.training
        params = build_trainable_parameters(
            self,
            action_lr=float(t.action_lr) if getattr(t, "action_lr", None) else None,
            video_lr=float(t.video_lr) if getattr(t, "video_lr", None) else None,
        )
        betas = tuple(getattr(t, "adam_betas", [0.9, 0.95]))
        return torch.optim.AdamW(params, lr=float(t.learning_rate), weight_decay=float(t.weight_decay), betas=betas)

    # (4) Called by train() — build the training DataLoader (seeded generator when reproducible).
    def build_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Build the training DataLoader.

        ``MixtureDataset`` already owns a shuffled virtual index map and
        reshuffles it in ``set_epoch``.  Do not wrap it in PyTorch's
        ``RandomSampler``: on a large mixture that would build a second full-size
        permutation, convert it to Python integers, and duplicate it in every
        rank. Other datasets keep the standard DataLoader shuffle.

        When ``cfg.project.seed`` is set, wires a per-rank ``generator`` and a
        ``worker_init_fn`` so dataset-side randomness is reproducible across
        runs while keeping per-epoch / per-worker variation.
        """
        from openwam.dataloader.mixture import MixtureDataset

        t = self.cfg.training
        shuffle = not isinstance(self.dataset, MixtureDataset)
        kwargs: dict = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(t.dataset_num_workers),
            collate_fn=list,
            pin_memory=True,
        )
        if not shuffle:
            logger.info(
                "DataLoader shuffle disabled for MixtureDataset; its shuffled index map "
                "is reshuffled by dataset.set_epoch(epoch)."
            )
        if self._run_seed is not None:
            from openwam.train.utils.seeding import dataloader_worker_init_fn, make_dataloader_generator

            kwargs["generator"] = make_dataloader_generator(self._run_seed, rank=self._rank)
            kwargs["worker_init_fn"] = dataloader_worker_init_fn
        return torch.utils.data.DataLoader(self.dataset, **kwargs)

    # (5) Called by train() — linear-warmup + cosine schedule, or None (constant LR / debug).
    def build_lr_scheduler(self, optimizer, total_opt_steps: int, debug: bool = False):
        """Linear-warmup + cosine schedule, or None (constant LR / debug)."""
        t = self.cfg.training
        if debug:
            return None
        if getattr(t, "lr_scheduler", None) == "cosine":
            return build_cosine_scheduler(
                optimizer, total_opt_steps=total_opt_steps, cfg=self.cfg, num_processes=self.accelerator.num_processes
            )
        return None

    # (6) Called by train() — locate/create the run dir and resolve the resume state dir.
    def setup_output_dir(self, debug: bool, resume_path: str | None) -> tuple[str, str | None]:
        """Locate/create the run dir and resolve the resume state dir.

        With a usable resume state the run dir is REUSED (assets/config/norm already
        present); otherwise rank-0 creates a fresh timestamped dir and broadcasts it.
        All ranks resolve ``resume_state_dir`` independently (shared FS, deterministic),
        so a missing-state error raises on every rank without deadlocking the broadcast.
        Returns ``(output_path, resume_state_dir)``.
        """
        base_output_path = getattr(self.cfg.training, "output_path", "./models")
        is_main = self.accelerator.is_main_process

        resume_state_dir = find_latest_accel_state(resume_path) if resume_path else None
        if resume_path and resume_state_dir is None:
            raise FileNotFoundError(
                f"resume_ckpt_path={resume_path} has no usable accel_state_step_*; full states "
                f"are only written when training.save_full_states_for_resume=true (and a finished run drops "
                f"them) — use finetune_ckpt_path to warm-start from weights instead."
            )
        if resume_state_dir is not None:
            output_path = os.path.dirname(resume_state_dir)
            logger.info("[resume] reusing run dir %s (state=%s)", output_path, os.path.basename(resume_state_dir))
            # Keep the checkpoint-dir deploy artifact; refuse resume when it
            # diverges from the dataset transform (legacy or regenerated stats).
            if self.dataset is not None:
                verify_resume_normalization_stats(output_path, self.dataset)
            return output_path, resume_state_dir

        if is_main:
            from datetime import datetime

            run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if debug:
                run_dir_name += "_debug"
            output_path = os.path.join(base_output_path, run_dir_name)
            os.makedirs(output_path, exist_ok=True)
            # Self-contained deploy: backbones save assets, then config + action stats.
            # BEFORE save_config so config.yaml carries the merged reconstruction specs.
            self.architecture.save_assets_for_deployment(output_path, self.cfg)
            if self._ckpt_source_dir is not None:
                # Self-contained finetune/resume: model_path may be unreachable,
                # so relay the tokenizer files from the source ckpt dir instead.
                from openwam.train.utils.ckpt_model_loader import copy_ckpt_artifacts

                copy_ckpt_artifacts(self._ckpt_source_dir, output_path)
            save_config(output_path, self.cfg)
            if self.dataset is not None:
                save_normalization_stats(output_path, self.dataset)
        else:
            output_path = None
        import torch.distributed as dist

        path_list = [output_path] if is_main else [None]
        dist.broadcast_object_list(path_list, src=0)
        output_path = path_list[0]
        logger.info("Checkpoints will be saved to %s", output_path)
        return output_path, None

    # (7) Called by train() — wrap architecture/optimizer/dataloader with the DeepSpeed Accelerator.
    def prepare_accelerate(self, optimizer, dataloader, scheduler):
        """Wrap architecture/optimizer/dataloader with the DeepSpeed Accelerator."""
        prepare_args = [self.architecture, optimizer, dataloader]
        if scheduler is not None:
            prepare_args.append(scheduler)
            self.architecture, optimizer, dataloader, scheduler = self.accelerator.prepare(*prepare_args)
        else:
            self.architecture, optimizer, dataloader = self.accelerator.prepare(*prepare_args)
        # After prepare, self.architecture is the wrapped handle used for the loop's
        # forward/backward. Architecture-level helpers must run on the UNDERLYING
        # module, not the wrapper: DeepSpeedEngine.__getattr__ forwards unknown attrs
        # to the inner module, but DistributedDataParallel does NOT — so calling
        # set_dtype_device/move_frozen_to_device on the wrapper would raise
        # AttributeError under a hypothetical plain-DDP accelerator (world_size>1).
        # unwrap_model returns the inner module for both backends (no-op single-GPU,
        # where Accelerate adds no wrapper). NB: this only covers these setup-time
        # helpers — the training loop's self.architecture.prepare_inputs/compute_loss
        # calls would hit the same non-forwarding wrapper if a non-DeepSpeed multi-GPU
        # path is ever reintroduced.
        arch = self.accelerator.unwrap_model(self.architecture)
        # Propagate device down through architecture; frozen modules (T5/VAE) idempotent move.
        arch.set_dtype_device(arch.dtype, self.accelerator.device)
        # When RoboTwin T5 cache is on, prepare_inputs never calls text_encoder.
        # Prefer skip_text_encoder at build time (no UMT5); if still present, park on CPU.
        use_t5_cache = bool(getattr(getattr(self.cfg, "dataloader", None), "use_t5_cache", False))
        if use_t5_cache:
            arch.move_frozen_to_device(self.accelerator.device, names=("vae",))
            _te = None
            try:
                _te = arch.get_submodule("text_encoder")
            except (AttributeError, KeyError):
                _te = None
            if _te is None:
                for bb in getattr(arch, "backbones", {}).values():
                    try:
                        _te = bb.get_submodule("text_encoder")
                    except (AttributeError, KeyError):
                        _te = None
                    if _te is not None:
                        break
            if _te is not None:
                _te.to(device="cpu")
                logger.info("use_t5_cache=true: text_encoder kept on CPU (disk embeds used)")
            else:
                logger.info("use_t5_cache=true: text_encoder not present (skip_text_encoder)")
        else:
            arch.move_frozen_to_device(self.accelerator.device, names=("text_encoder", "vae"))
        if self.fpd is not None:
            # Teacher is not accelerator-prepared; move DiT/action only.
            # Frozen encoders/VAE already shared with student (see _setup_fpd).
            teacher = self.fpd.teacher
            self._share_fpd_frozen_modules(arch, teacher)
            teacher.set_dtype_device(arch.dtype, self.accelerator.device)
            if use_t5_cache:
                # Keep teacher's T5 on CPU too if present (usually shared pointer).
                try:
                    t_te = teacher.video_backbone.get_submodule("text_encoder")
                    if t_te is not None:
                        t_te.to(device="cpu")
                except Exception:
                    pass
        logger.info(
            "architecture wrapped (%s), device=%s",
            type(self.architecture).__name__,
            self.accelerator.device,
        )
        return optimizer, dataloader, scheduler

    # (8) Called by train() on the resume path — restore full state, map step -> (start_epoch, skip).
    def resume_if_configured(self, resume_state_dir: str, dataloader, grad_accum: int) -> tuple[int, int, int, int]:
        """Load full state (after prepare) and map global_step -> (start_epoch, skip_first_batches).

        Returns (global_step, opt_step, start_epoch, skip).
        """
        is_main = self.accelerator.is_main_process
        if is_main:
            logger.info("[resume] loading Accelerate state from %s", resume_state_dir)
        meta = load_full_state(self.accelerator, resume_state_dir)
        global_step = int(meta.get("global_step", 0))
        # Align global_step to the grad_accum boundary skip was floored to, then derive
        # opt_step from it — otherwise floored-off batches re-train and per-step seeds
        # (keyed on global_step) drift. No-op at grad_accum=1.
        start_epoch, skip, global_step = compute_resume_position(global_step, len(dataloader), grad_accum)
        opt_step = global_step // grad_accum
        if is_main:
            logger.info(
                "[resume] resumed at global_step=%d opt_step=%d epoch=%d skip_first=%d",
                global_step,
                opt_step,
                start_epoch,
                skip,
            )
        return global_step, opt_step, start_epoch, skip

    # (9) Called each step in train()'s loop — joint video-action loss dict (total/video/action).

    @staticmethod
    def _share_fpd_frozen_modules(student, teacher, names: tuple[str, ...] = ("text_encoder", "vae")) -> None:
        """Point teacher's frozen encoder/VAE attrs at the student's modules.

        FPD never runs ``teacher.prepare_inputs``; only DiT/ActionDiT need a
        separate GPU copy. Sharing T5/VAE frees tens of GiB on H20-class cards.
        """
        import gc

        import torch

        def _find_owner(root, attr: str):
            if hasattr(root, attr) and getattr(root, attr) is not None:
                return root
            for bb in getattr(root, "backbones", {}).values():
                if hasattr(bb, attr) and getattr(bb, attr) is not None:
                    return bb
            return None

        freed = []
        for name in names:
            s_owner = _find_owner(student, name)
            t_owner = _find_owner(teacher, name)
            if s_owner is None or t_owner is None:
                continue
            s_mod = getattr(s_owner, name)
            t_mod = getattr(t_owner, name)
            if t_mod is s_mod:
                continue
            setattr(t_owner, name, s_mod)
            del t_mod
            freed.append(name)
        if freed:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("FPD: shared student frozen modules with teacher: %s", ", ".join(freed))

    def _setup_fpd(self, fpd_cfg) -> None:
        """Build a frozen teacher and wrap student+teacher as OpenWAMFPD."""
        from openwam.model.fpd import OpenWAMFPD
        from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir

        teacher_ckpt = getattr(fpd_cfg, "teacher_ckpt_path", None)
        if not teacher_ckpt:
            raise ValueError("fpd.enabled=true requires fpd.teacher_ckpt_path")

        logger.info("FPD enabled: building frozen teacher from %s", teacher_ckpt)
        # Build teacher from ckpt config only (no live override) so it matches the
        # released OpenWAM checkpoint; student already applied live overrides.
        # Pass live cfg so video_backbone.model_path (and other host paths)
        # override the placeholder paths in the released checkpoint config.yaml.
        _, teacher, _ = build_architecture_from_ckpt_dir(
            teacher_ckpt,
            weights_required=True,
            override_cfg=self.cfg,
            skip_text_encoder=bool(getattr(getattr(self.cfg, "dataloader", None), "use_t5_cache", False)),
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # FPD only needs teacher DiT/ActionDiT; share frozen VAE/T5 with student
        # so we do not keep a second copy on GPU (otherwise bs=1 OOMs on H20 96G).
        self._share_fpd_frozen_modules(self.architecture, teacher)

        # Teacher uses the same training schedule buffers as student.
        teacher.init_training_schedulers(1000)
        t = self.cfg.training
        teacher.set_training_runtime(
            use_gradient_checkpointing=bool(t.use_gradient_checkpointing),
            use_gradient_checkpointing_offload=bool(t.use_gradient_checkpointing_offload),
            max_timestep_boundary=float(t.max_timestep_boundary),
            min_timestep_boundary=float(t.min_timestep_boundary),
        )

        self.fpd = OpenWAMFPD(
            student=self.architecture,
            teacher=teacher,
            lambda_action=self.lambda_action,
            lambda_video=self.lambda_video,
            lambda_distill=self.lambda_distill if self.lambda_distill > 0 else 1.0,
            use_distill_gate=self.use_distill_gate,
        )
        logger.info(
            "FPD ready: lambda_a=%.4g lambda_v=%.4g lambda_d=%.4g gate=%s",
            self.fpd.lambda_action,
            self.fpd.lambda_video,
            self.fpd.lambda_distill,
            self.fpd.use_distill_gate,
        )

    def compute_loss(self, batch) -> dict:
        """Compute joint video-action loss. Returns dict: total/video/action[+distill]."""
        if not isinstance(batch, list):
            batch = [batch]

        if self.fpd is not None:
            result = self.fpd.compute_loss(batch)
            out = {
                "total": result["loss"],
                "video": result.get("loss_video", torch.tensor(0.0)),
                "action": result.get("loss_action", torch.tensor(0.0)),
                "distill": result.get("loss_distill", torch.tensor(0.0)),
                "distill_gate_frac": result.get("distill_gate_frac", 0.0),
                "student_minus_teacher": result.get("student_minus_teacher", 0.0),
            }
            return out

        inputs = self.architecture.prepare_inputs(batch)
        if self.lambda_action > 0 and inputs.get("actions") is None:
            raise ValueError("lambda_action > 0 but no action in data.")

        result = self.architecture.compute_loss(
            **inputs,
            lambda_video=self.lambda_video,
            lambda_action=self.lambda_action,
            video_mode=self.video_mode,
        )

        return {
            "total": result["loss"],
            "video": result.get("loss_video", torch.tensor(0.0)),
            "action": result.get("loss_action", torch.tensor(0.0)),
        }

    # (10) Called each step in train()'s loop — progress bar, wandb log, debug loss-history CSV.
    def log_step(
        self,
        *,
        metrics,
        global_step,
        opt_step,
        epoch,
        lr,
        steps_per_sec,
        batch_size,
        pbar,
        wandb_run,
        debug,
        output_path,
    ) -> None:
        """Update progress bar, log to wandb, and (debug) write the loss-history CSV row."""
        labels = [("action", "loss_action")]
        if "loss_distill" in metrics:
            labels.append(("distill", "loss_distill"))
        loss_total = metrics["loss_total"]
        loss_video = metrics["loss_video"]
        grad_norm = metrics["grad_norm"]

        if pbar is not None:
            postfix = {"loss": f"{loss_total:.4f}", "video": f"{loss_video:.4f}"}
            for name, key in labels:
                postfix[name] = f"{metrics[key]:.4f}"
            postfix["lr"] = f"{lr:.2e}"
            postfix["epoch"] = epoch
            pbar.set_postfix(postfix)
            pbar.update(1)

        if wandb_run is not None:
            num_procs = self.accelerator.num_processes if self.accelerator is not None else 1
            log_dict = {
                "train/loss": loss_total,
                "train/loss_video": loss_video,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "performance/steps_per_sec": steps_per_sec,
                "performance/samples_per_sec": steps_per_sec * batch_size * num_procs,
            }
            for name, key in labels:
                log_dict[f"train/loss_{name}"] = metrics[key]
            if "distill_gate_frac" in metrics:
                log_dict["train/distill_gate_frac"] = metrics["distill_gate_frac"]
            if "student_minus_teacher" in metrics:
                log_dict["train/student_minus_teacher"] = metrics["student_minus_teacher"]
            wandb_run.log(log_dict, step=global_step)

        if not debug:
            return
        is_main = self.accelerator is None or self.accelerator.is_main_process
        if not is_main:
            return
        loss_parts = " ".join(f"{name}={metrics[key]:.6f}" for name, key in labels)
        msg = (
            f"[debug][step {global_step:04d} opt {opt_step:04d}] "
            f"loss={loss_total:.6f} video={loss_video:.6f} {loss_parts} "
            f"grad_norm={grad_norm:.6f} lr={lr:.3e} epoch={epoch} "
            f"steps_per_sec={steps_per_sec:.3f}"
        )
        logger.info(msg)
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg, flush=True)

        if output_path:
            write_debug_loss_row(
                output_path,
                labels=labels,
                metrics=metrics,
                global_step=global_step,
                opt_step=opt_step,
                epoch=epoch,
                lr=lr,
                steps_per_sec=steps_per_sec,
            )

    # (11) Called on every train() exit — final weights, drop resume state, close wandb.
    def finish_training(
        self,
        output_path: str,
        global_step: int,
        save_steps,
        keep_last_k: int,
        is_main: bool,
        wandb_run,
    ) -> None:
        """Unified teardown for every exit path: final weights, drop resume state, close wandb.

        Falsy ``save_steps`` = a profiling/no-write run, so no final artifact (matches the
        periodic-save gating). After the final weights land, ``finalize_keep_weights_only``
        removes every ``accel_state_step_*`` and retains the configured number of recent
        weight checkpoints (rank-0, post-barrier).
        """
        if save_steps:
            save_weights(self.accelerator, self.architecture, output_path, global_step, final=True)
        self.accelerator.wait_for_everyone()
        if is_main:
            finalize_keep_weights_only(output_path, keep_last_k)
        if wandb_run is not None:
            wandb_run.finish()
