"""Train a CIFAR-sized ResNet-50 from scratch on the local CIFAR-10 pickles."""

import argparse
from datetime import datetime
import json
import logging
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv_corruption.data.cifar10 import CIFAR10Pickles, class_names
from cv_corruption.models.resnet import MEAN, STD, make_model
from cv_corruption.config.loader import parse_config, save_config
from cv_corruption.visualization.training_curves import write_bundle
from cv_corruption.visualization.entropy_curves import render_entropy_bundle
from cv_corruption.evaluation.metrics import batch_metrics, weighted_average


SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def _success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


logging.Logger.success = _success


class CompactFormatter(logging.Formatter):
    """Use aligned levels and a clear field separator."""

    def __init__(self):
        super().__init__("[%(levelname)s] %(message)s")

# PyTorch 2.9 warns once per DDP rank about an API transition that is not
# actionable for this training recipe. It is retained in the source code's
# behavior and removed from the user-facing training terminal.
warnings.filterwarnings("ignore", message=r"`broadcast_buffers` is deprecated.*")

CV_ROOT = PROJECT_ROOT / "CV_Corruption"
DATA_DIR = CV_ROOT / "data/raw/cifar10/cifar-10-batches-py"
RUNS_DIR = CV_ROOT / "runs"
RECIPE = {
    "batch_size": 128,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "label_smoothing": 0.1,
    "seed": 42,
    "warmup_epochs": 5,
}


def timestamped_run_dir(base: Path) -> Path:
    """Return a fresh run directory with a sortable local-time suffix."""
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    candidate = Path(base).with_name(f"{Path(base).name}_{stamp}")
    counter = 2
    while candidate.exists():
        candidate = Path(base).with_name(f"{Path(base).name}_{stamp}_{counter}")
        counter += 1
    return candidate


class ExactDistributedEvalSampler(Sampler[int]):
    """Shard evaluation without DistributedSampler's repeated padding samples."""

    def __init__(self, size: int, rank: int, world_size: int):
        self.indices = range(rank, size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def rng_state():
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() and torch.cuda.is_initialized() else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device=device)


def save_checkpoint(path: Path, checkpoint: dict):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def run_epoch(model, loader, criterion, device, max_batches, optimizer=None, scaler=None, amp=False,
              ece_bins=10, collect_batch_metrics=False):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    batch_records = []
    for batch_id, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_id >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
        before = batch_metrics(logits.detach().cpu(), targets.cpu(), ece_bins=ece_bins) if collect_batch_metrics else None
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at batch {batch_id}: {loss.item()}")
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if collect_batch_metrics:
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
                    after_logits = model(inputs)
                after = batch_metrics(after_logits.cpu(), targets.cpu(), ece_bins=ece_bins)
                batch_records.append({"batch": batch_id, "before": before, "after": after,
                                      "delta": {key: after[key] - before[key] for key in
                                                 ("entropy", "normalized_entropy", "confidence", "accuracy", "loss", "ece")}})
        totals += torch.tensor(
            (loss.item() * targets.numel(), (logits.argmax(1) == targets).sum().item(), targets.numel()),
            dtype=torch.float64, device=device,
        )
    if dist.is_initialized():
        dist.all_reduce(totals)
    if totals[2].item() == 0:
        raise ValueError("No batches were processed; increase the batch limit")
    result = {"loss": totals[0].item() / totals[2].item(), "accuracy": totals[1].item() / totals[2].item(),
              "samples": int(totals[2].item())}
    if collect_batch_metrics:
        result["batch_metrics"] = batch_records
    return result


def evaluate_entropy(model, loader, device, ece_bins=10, max_batches=None):
    """Evaluate deterministic loader and aggregate metrics by sample count."""
    model.eval()
    records = []
    with torch.inference_mode():
        for batch_id, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_id >= max_batches:
                break
            logits = model(inputs.to(device, non_blocking=True))
            records.append(batch_metrics(logits.cpu(), targets, ece_bins=ece_bins))
    return weighted_average(records)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float, help="Learning rate at global batch size 128")
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--log-every-epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0], help="Physical GPU indices; DDP requires torchrun")
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable automatic mixed precision")
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False,
                        help="One epoch, two train and two test batches by default")
    parser.add_argument("--train-batches", type=int)
    parser.add_argument("--test-batches", type=int)
    parser.add_argument("--ece-bins", type=int, default=10)
    args = parse_config(parser, argv, path_fields=("data_dir", "output_dir", "resume"))
    if args.smoke and not args.resume:
        args.epochs = 1
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.cpu and world_size != 1:
        raise ValueError("CPU DDP is not supported; run CPU smoke with plain python")
    if len(set(args.gpu_ids)) != len(args.gpu_ids) or any(i < 0 for i in args.gpu_ids):
        raise ValueError("--gpu-ids must list distinct non-negative physical GPU indices")
    if not args.cpu and world_size != len(args.gpu_ids):
        raise ValueError("Use torchrun --nproc_per_node=N with exactly N --gpu-ids, or use one GPU")
    if world_size > 1 and (int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != world_size or local_rank != rank):
        raise ValueError("Only single-node DDP is supported")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not args.cpu:
        if visible:
            try:
                exposed = [int(part.strip()) for part in visible.split(",")]
                device_index = exposed.index(args.gpu_ids[local_rank])
            except ValueError as exc:
                raise ValueError("Requested physical GPU is not in numeric CUDA_VISIBLE_DEVICES") from exc
        else:
            device_index = args.gpu_ids[local_rank]
        if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
            raise RuntimeError("Requested CUDA GPU is unavailable; use --cpu for a CPU smoke test")
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)

    try:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
        old_config = checkpoint["config"] if checkpoint else {}
        for name, default in RECIPE.items():
            requested = getattr(args, name)
            previous = old_config.get(name, default)
            if checkpoint and requested is not None and requested != previous:
                raise ValueError(f"--{name.replace('_', '-')} differs from checkpoint ({previous})")
            setattr(args, name, previous if requested is None else requested)
        if checkpoint and old_config["world_size"] != world_size:
            raise ValueError("Resume requires the original GPU count to preserve optimizer and RNG state")
        if checkpoint and args.smoke != old_config["smoke"]:
            if args.smoke:
                raise ValueError("Cannot change a full run into a smoke run on resume")
            args.smoke = old_config["smoke"]
        args.epochs = args.epochs if args.epochs is not None else (old_config["epochs"] if checkpoint else (1 if args.smoke else 200))
        if args.workers is None:
            args.workers = old_config.get("workers", 4) if checkpoint else 4
        if args.amp is None:
            args.amp = old_config.get("amp", True)
        if checkpoint and (args.amp and not args.cpu) != old_config.get("amp", True):
            raise ValueError("AMP setting differs from checkpoint configuration")
        for name in ("train_batches", "test_batches"):
            requested = getattr(args, name)
            previous = old_config.get(name, 2 if args.smoke else None)
            if checkpoint and requested is not None and requested != previous:
                raise ValueError(f"--{name.replace('_', '-')} differs from checkpoint ({previous})")
            setattr(args, name, previous if requested is None else requested)
        if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0
                or args.warmup_epochs < 0 or args.log_every_epochs < 1):
            raise ValueError("Epochs and batch size must be positive; workers must be non-negative")
        if any(value is not None and value < 1 for value in (args.train_batches, args.test_batches)):
            raise ValueError("Batch limits must be positive")
        if args.ece_bins < 1:
            raise ValueError("ECE bins must be positive")
        if args.lr <= 0 or not 0 <= args.label_smoothing < 1 or args.weight_decay < 0 or args.momentum < 0:
            raise ValueError("Invalid optimizer or label smoothing settings")
        if args.seed < 0 or args.seed + world_size - 1 >= 2**32:
            raise ValueError("Rank seeds must be in 0..2**32-1")
        if checkpoint:
            output_dir = args.resume.parent.parent
        else:
            base_dir = args.output_dir or RUNS_DIR / (
                "cifar10_resnet50_smoke" if args.smoke else "cifar10_resnet50"
            )
            selected_dir = timestamped_run_dir(base_dir) if rank == 0 else None
            if world_size > 1:
                payload = [str(selected_dir) if selected_dir else ""]
                dist.broadcast_object_list(payload, src=0)
                output_dir = Path(payload[0])
            else:
                output_dir = selected_dir
        if not checkpoint and (output_dir / "checkpoints/last.pt").exists():
            raise FileExistsError(f"Existing checkpoint in {output_dir}; use --resume or another --output-dir")
        if checkpoint and args.epochs <= checkpoint["epoch"]:
            raise ValueError("--epochs must exceed the completed checkpoint epoch")
        if checkpoint and Path(old_config["data_dir"]).resolve() != args.data_dir.resolve():
            if "data_dir" not in args._provided:
                args.data_dir = Path(old_config["data_dir"])
            else:
                raise ValueError("--data-dir differs from checkpoint")
        if rank == 0:
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            formatter = CompactFormatter()
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logging.basicConfig(level=logging.INFO, force=True, handlers=[console, file_handler])
            logging.info("RUN     started=%s | host=%s | pid=%d",
                         datetime.now().astimezone().isoformat(timespec="seconds"),
                         socket.gethostname(), os.getpid())
            logging.info("PROCESS rank=%d/%d | local-rank=%d | launcher=%s",
                         rank, world_size, local_rank, "torchrun" if world_size > 1 else "python")
        if world_size > 1:
            dist.barrier()
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.backends.cudnn.benchmark = True
        names = class_names(args.data_dir)
        train_data = CIFAR10Pickles(args.data_dir, train=True)
        train_eval_data = CIFAR10Pickles(args.data_dir, train=True, augment=False)
        test_data = CIFAR10Pickles(args.data_dir, train=False)
        if len(train_data) != 50000 or len(test_data) != 10000 or set(train_data.labels) != set(range(10)) or set(test_data.labels) != set(range(10)):
            raise ValueError("CIFAR-10 sample counts or class coverage are incorrect")
        if checkpoint and (checkpoint["class_names"] != names or checkpoint["normalization"] != {"mean": MEAN, "std": STD}):
            raise ValueError("Checkpoint classes or normalization differ from this dataset/model")
        train_sampler = DistributedSampler(train_data, world_size, rank, shuffle=True, seed=args.seed) if world_size > 1 else None
        eval_sampler = ExactDistributedEvalSampler(len(test_data), rank, world_size) if world_size > 1 else None
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=train_sampler is None,
                                  sampler=train_sampler, num_workers=args.workers, pin_memory=device.type == "cuda")
        train_eval_loader = DataLoader(train_eval_data, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.workers, pin_memory=device.type == "cuda")
        test_loader = DataLoader(test_data, batch_size=args.batch_size, sampler=eval_sampler,
                                 num_workers=args.workers, pin_memory=device.type == "cuda")
        full_train_eval_loader = DataLoader(train_eval_data, batch_size=args.batch_size, shuffle=False,
                                            num_workers=args.workers, pin_memory=device.type == "cuda")
        full_test_eval_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.workers, pin_memory=device.type == "cuda")
        model = make_model().to(device)
        parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * args.batch_size * world_size / 128,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        warmup_epochs = min(args.warmup_epochs, args.epochs)
        if warmup_epochs:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer, start_factor=1.0 / warmup_epochs,
                        end_factor=1.0, total_iters=warmup_epochs,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(1, args.epochs - warmup_epochs),
                    ),
                ],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
        start_epoch, best_accuracy = 1, -1.0
        if checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if args.epochs != old_config["epochs"]:
                if hasattr(scheduler, "T_max"):
                    scheduler.T_max = args.epochs
                elif hasattr(scheduler, "_schedulers"):
                    scheduler._schedulers[-1].T_max = max(1, args.epochs - warmup_epochs)
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch, best_accuracy = checkpoint["epoch"] + 1, checkpoint["best_accuracy"]
            restore_rng(checkpoint["rng_states"][rank], device)
        config = {
            "architecture": "resnet50_cifar10", "pretrained": False, "num_classes": 10,
            "data_dir": str(args.data_dir.resolve()), "output_dir": str(output_dir.resolve()),
            "epochs": args.epochs, "batch_size": args.batch_size, "global_batch_size": args.batch_size * world_size,
            "lr": args.lr, "effective_lr": args.lr * args.batch_size * world_size / 128,
            "momentum": args.momentum, "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "label_smoothing": args.label_smoothing, "seed": args.seed, "workers": args.workers,
            "gpu_ids": args.gpu_ids if not args.cpu else [], "rank_to_physical_gpu": args.gpu_ids if not args.cpu else [],
            "rank_seeds": [args.seed + r for r in range(world_size)], "world_size": world_size,
            "amp": scaler.is_enabled(), "smoke": args.smoke,
            "cpu": args.cpu, "resume": str(args.resume) if args.resume else None,
            "source_config": str(args.config) if args.config else None,
            "train_batches": args.train_batches, "test_batches": args.test_batches,
            "class_names": names, "normalization": {"mean": MEAN, "std": STD},
        }
        if rank == 0:
            args.output_dir = output_dir.resolve()
            save_config(output_dir / "resolved_config.json", args)
            with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
                json.dump(config, stream, indent=2, ensure_ascii=False)
            if checkpoint and output_dir.resolve() != args.resume.parent.parent.resolve():
                source_best = args.resume.parent / "best.pt"
                destination_best = output_dir / "checkpoints/best.pt"
                if source_best.exists() and not destination_best.exists():
                    shutil.copy2(source_best, destination_best)
            logging.info("MODEL   ResNet-50 CIFAR-10 | init=from-scratch | params=%s | classes=%d",
                         f"{parameter_count:,}", len(names))
            logging.info("DEVICE  %s | gpus=%s | ddp-ranks=%d | seed=%d",
                         device, config["gpu_ids"] or "cpu", world_size, args.seed)
            logging.info("DATA    train=%d | test=%d | start-epoch=%d | run=%s",
                         len(train_data), len(test_data), start_epoch, output_dir)
            progress = tqdm(
                total=args.epochs - start_epoch + 1, desc="TRAIN", unit="epoch",
                dynamic_ncols=True, leave=True,
            )
        else:
            progress = None
        train_model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False) if world_size > 1 else model
        train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        test_criterion = nn.CrossEntropyLoss()
        if rank == 0:
            metrics_dir = output_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            epoch_metrics_path = metrics_dir / "epoch_metrics.jsonl"
            if not checkpoint:
                eval_limit = args.test_batches if args.smoke else None
                epoch_zero = {"epoch": 0, "train": evaluate_entropy(model, full_train_eval_loader, device, max_batches=eval_limit),
                              "test": evaluate_entropy(model, full_test_eval_loader, device, max_batches=eval_limit)}
                epoch_metrics_path.write_text(json.dumps(epoch_zero, ensure_ascii=False) + "\n", encoding="utf-8")
        if world_size > 1:
            dist.barrier()
        for epoch in range(start_epoch, args.epochs + 1):
            begin = time.monotonic()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            lr = optimizer.param_groups[0]["lr"]
            train = run_epoch(train_model, train_loader, train_criterion, device, args.train_batches,
                              optimizer=optimizer, scaler=scaler, amp=scaler.is_enabled(),
                              ece_bins=args.ece_bins, collect_batch_metrics=rank == 0)
            # Evaluation shards can have unequal sizes; synchronize BN buffers before using the bare model.
            if world_size > 1:
                for buffer in model.buffers():
                    dist.broadcast(buffer, src=0)
            test = run_epoch(model, test_loader, test_criterion, device, args.test_batches, amp=scaler.is_enabled())
            if rank == 0:
                with (output_dir / "metrics/batch_metrics.jsonl").open("a", encoding="utf-8") as stream:
                    for batch_record in train.get("batch_metrics", []):
                        stream.write(json.dumps({"epoch": epoch, **batch_record}, ensure_ascii=False) + "\n")
                entropy_record = {"epoch": epoch,
                                  "train": evaluate_entropy(model, full_train_eval_loader, device, max_batches=args.test_batches if args.smoke else None),
                                  "test": evaluate_entropy(model, full_test_eval_loader, device, max_batches=args.test_batches if args.smoke else None)}
                with (output_dir / "metrics/epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entropy_record, ensure_ascii=False) + "\n")
            if world_size > 1:
                dist.barrier()
            # GradScaler.step delegates to optimizer.step but does not set this scheduler hint.
            optimizer._opt_called = True
            scheduler.step()
            stats = torch.tensor([time.monotonic() - begin,
                                  torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0], device=device)
            if world_size > 1:
                dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            is_best = test["accuracy"] > best_accuracy
            best_accuracy = max(best_accuracy, test["accuracy"])
            states = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(states, rng_state())
            else:
                states[0] = rng_state()
            if rank == 0:
                record = {
                    "epoch": epoch,
                    "train": train,
                    "test": test,
                    "lr": lr,
                    "train_loss": train["loss"],
                    "val_loss": test["loss"],
                    "train_accuracy": train["accuracy"],
                    "val_accuracy": test["accuracy"],
                    "learning_rate": lr,
                    "epoch_seconds": stats[0].item(),
                    "peak_gpu_memory_mib": stats[1].item(),
                    "best_test_accuracy": best_accuracy,
                }
                saved = {"epoch": epoch, "best_accuracy": best_accuracy, "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "scaler": scaler.state_dict(), "rng_states": states, "config": config,
                         "class_names": names, "normalization": {"mean": MEAN, "std": STD}}
                save_checkpoint(output_dir / "checkpoints/last.pt", saved)
                if is_best:
                    save_checkpoint(output_dir / "checkpoints/best.pt", saved)
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                if epoch % args.log_every_epochs == 0 or epoch == args.epochs:
                    logging.info("EPOCH %03d/%03d | lr=%-9.6g | train loss=%.4f acc=%6.2f%% | val loss=%.4f acc=%6.2f%% | best=%6.2f%% | %4.1fs | peak=%4.0f MiB",
                                 epoch, args.epochs, lr, train["loss"], train["accuracy"] * 100,
                                 test["loss"], test["accuracy"] * 100, best_accuracy * 100, *stats.tolist())
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(loss=f"{test['loss']:.3f}",
                                         acc=f"{test['accuracy'] * 100:.1f}%",
                                         best=f"{best_accuracy * 100:.1f}%")
        if rank == 0:
            if progress is not None:
                progress.close()
            try:
                write_bundle(output_dir, render=True)
                render_entropy_bundle(output_dir)
                logging.getLogger(__name__).success("DONE    training complete | curves=%s", output_dir)
            except (FileNotFoundError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
                logging.getLogger(__name__).warning("Training completed, but curve rendering failed: %s", exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
