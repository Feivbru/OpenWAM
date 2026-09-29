"""Hydra entry point for OpenWAM training.

Launch with torchrun (DeepSpeed via HuggingFace Accelerate):
    torchrun --nproc_per_node=4 scripts/train.py

World size comes from torchrun; everything else (mixed precision, ZeRO stage,
gradient accumulation/clipping, optimizer offload) from cfg.training
(e.g. select the ZeRO stage via ``training.zero_stage=1``).
"""

import faulthandler
import logging
import os
import sys
import traceback
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Force tracebacks to flush even when an exception fires inside DataLoader
# workers / Hydra's own try-except wrapper. Without this, a silent failure
# on rank 0 leaves the other ranks deadlocked at FSDP all-gather with no
# clue what went wrong (observed during a mixture smoke run).
faulthandler.enable(file=sys.stderr, all_threads=True)


def _force_flush_excepthook(exc_type, exc_value, exc_tb):
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    sys.stderr.write(f"\n===== UNHANDLED EXCEPTION ON RANK {rank} =====\n")
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()


sys.excepthook = _force_flush_excepthook

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _StartupNoiseFilter(logging.Filter):
    # Hide optional compiler probes while keeping warnings and errors visible.
    _HIDDEN_PREFIXES = ("gcc -pthread ", "NCCL version ")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (record.name == "root" and message.startswith(self._HIDDEN_PREFIXES))


def _install_startup_noise_filter() -> None:
    # Suppress verbose optional-op probes emitted by DeepSpeed/distutils.
    noise_filter = _StartupNoiseFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(noise_filter)


logger = logging.getLogger(__name__)


def _build_accelerator(cfg: DictConfig):
    """Build a DeepSpeed Accelerator for the torchrun launch path.

    torchrun has already initialised torch.distributed (``RANK``, ``WORLD_SIZE``,
    ``LOCAL_RANK`` are set); we construct the ``DeepSpeedPlugin`` from cfg.training
    so DeepSpeed is activated without ``accelerate launch``.

    ZeRO-2 defaults to FastWAM's ``scripts/ds_configs/ds_zero2_config.json``
    (``contiguous_gradients: false`` + 200MB buckets). Accelerate's bare
    ``zero_stage=2`` otherwise inherits DeepSpeed's ``contiguous_gradients:
    true``, which flattens multi-GiB FP32 grad partitions every step and
    fragments VRAM until OOM.
    """
    import accelerate

    t = cfg.training
    grad_accum = int(t.gradient_accumulation_steps)
    max_grad_norm = getattr(t, "max_grad_norm", None)
    mixed_precision = str(t.mixed_precision)
    zero_stage = int(t.zero_stage)
    offload = str(getattr(t, "offload_optimizer_device", "none") or "none")
    if os.environ.get("LOCAL_RANK", "0") == "0":
        logger.info("mixed_precision = %s (from cfg.training.mixed_precision)", mixed_precision)

    ds_config = getattr(t, "deepspeed_config", None)
    if ds_config is None and zero_stage == 2:
        default_zero2 = PROJECT_ROOT / "scripts" / "ds_configs" / "ds_zero2_config.json"
        if default_zero2.is_file():
            ds_config = str(default_zero2)

    plugin_kwargs = {
        "gradient_accumulation_steps": grad_accum,
        # DeepSpeed clips internally with this value (the loop's clip_grad_norm_ only reads
        # back the norm under DeepSpeed); single source = training.max_grad_norm, 0.0 = off.
        "gradient_clipping": float(max_grad_norm) if max_grad_norm else 0.0,
        "offload_optimizer_device": offload,
    }
    if ds_config:
        plugin_kwargs["hf_ds_config"] = str(ds_config)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            logger.info("DeepSpeed config = %s", ds_config)
    else:
        plugin_kwargs["zero_stage"] = zero_stage

    plugin = accelerate.DeepSpeedPlugin(**plugin_kwargs)

    return accelerate.Accelerator(
        gradient_accumulation_steps=grad_accum,
        deepspeed_plugin=plugin,
        mixed_precision=mixed_precision,
    )


def _inject_project_seed(cfg: DictConfig) -> None:
    """Propagate ``cfg.project.seed`` down to ``cfg.dataloader.seed``.

    Dataloader yamls no longer carry their own ``seed`` field; the
    authoritative source is ``project.seed`` in train.yaml. When
    ``project.seed`` is null (production stochastic runs), the dataset
    ctor's ``seed=42`` default kicks in.
    """
    project_seed = OmegaConf.select(cfg, "project.seed", default=None)
    if project_seed is None:
        return
    dl = cfg.get("dataloader", None)
    if dl is None:
        return
    OmegaConf.update(dl, "seed", int(project_seed), force_add=True)


def _train(cfg: DictConfig) -> None:
    """Package-native training path."""
    _inject_project_seed(cfg)
    _train_openwam(cfg)


def _train_openwam(cfg: DictConfig) -> None:
    """Original OpenWAM training path."""
    from openwam.dataloader.registry import build_dataset
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.seeding import seed_everything

    # Seed Python random / numpy / torch BEFORE dataset construction so that
    # any reader-time randomness (e.g. MixtureDataset index_map shuffle when
    # seed isn't explicitly set, lerobot splits, etc.) is reproducible.
    # OpenWAMTrainer.__init__ re-seeds via seed_process for model init using the
    # same RANK_OFFSET rank stride (cudnn stays as configured here). Null
    # cfg.project.seed = production stochastic run, so we skip seeding here.
    project_seed = OmegaConf.select(cfg, "project.seed", default=None)
    if project_seed is not None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        seed_everything(int(project_seed), rank=rank)

    accelerator = _build_accelerator(cfg)

    # Build dataset via registry
    dataset = build_dataset(cfg.dataloader, split="train")

    # Build trainer and run
    trainer = OpenWAMTrainer(cfg, accelerator=accelerator, dataset=dataset)
    trainer.train()


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
    _install_startup_noise_filter()

    # Only print on rank 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("=" * 60)
        print("OpenWAM Training")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)
    else:
        # Silence stray ``print(...)`` calls in vendored loaders (e.g. diffsynth's
        # ``model_loader.py``) on non-main ranks — they otherwise print "Loading
        # models from: ..." / "Loaded model: { ... }" once per rank, doubling
        # the startup log. Logging-based output is unaffected.
        import builtins

        builtins.print = lambda *a, **kw: None

    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        _train(cfg)
    except BaseException:
        # destroy_process_group below is a collective. If only this rank
        # raised, the other ranks are still in mid-training collectives
        # (e.g. accelerate's RNG-state broadcast inside dataloader.__iter__),
        # and destroy will block forever waiting for them — masking the
        # actual rank-0 exception. Dump traceback to a file FIRST so the
        # real error survives even if stderr/tee is lost or NCCL hangs.
        import traceback as _tb
        from datetime import datetime as _dt
        from pathlib import Path as _Path

        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
        tb_text = _tb.format_exc()
        banner = f"===== RANK {rank} EXCEPTION (pre-destroy) =====\n"
        sys.stderr.write("\n" + banner)
        sys.stderr.write(tb_text)
        sys.stderr.flush()
        sys.stdout.flush()

        # Also dump CUDA memory + traceback to a durable crash file.
        mem_lines = []
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                for i in range(_torch.cuda.device_count()):
                    alloc = _torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = _torch.cuda.memory_reserved(i) / (1024**3)
                    peak = _torch.cuda.max_memory_allocated(i) / (1024**3)
                    mem_lines.append(
                        f"cuda:{i} allocated={alloc:.2f}GiB reserved={reserved:.2f}GiB peak_alloc={peak:.2f}GiB"
                    )
        except Exception as _mem_e:
            mem_lines.append(f"cuda_mem_query_failed: {_mem_e}")

        crash_dir = _Path("logs")
        try:
            crash_dir.mkdir(parents=True, exist_ok=True)
            stamp = _dt.now().strftime("%Y%m%d_%H%M%S")
            crash_path = crash_dir / f"crash_rank{rank}_{stamp}.txt"
            crash_path.write_text(
                banner
                + "\n".join(mem_lines)
                + "\n"
                + tb_text
                + "\n"
                + f"cwd={os.getcwd()}\n"
                + f"argv={sys.argv!r}\n"
            )
            sys.stderr.write(f"[crash] wrote {crash_path}\n")
            sys.stderr.flush()
        except Exception as _dump_e:
            sys.stderr.write(f"[crash] failed to write crash file: {_dump_e}\n")
            sys.stderr.flush()

        # Hard-exit: calling destroy_process_group() here deadlocks when peer
        # ranks are still inside NCCL collectives, leaving zombie GPU holders.
        os._exit(1)
    finally:
        # Only reached on clean success (exception path os._exit above).
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
