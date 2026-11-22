# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Can Samudra overfit a handful of samples? CPU-only harness.

Builds the real Samudra model, loss and data pipeline through ``Trainer`` (so
architecture, normalization and masking match a real training run), caches a
fixed set of training batches, and repeatedly trains on just those.

Every ``--eval-every`` optimizer steps it reports, on the memorized batches
(and on a held-out batch), the masked per-channel RMSE in normalized units for:

* the model,
* persistence (the latest input state repeated for every output step).

The headline number is ``model / persistence`` per channel: below 1 means the
model beats doing nothing. A model that can overfit should drive this to ~0 on
the memorized samples.

Usage::

    PYTHONUNBUFFERED=1 uv run python experiments/overfit/overfit.py \
        experiments/overfit/configs/overfit_onedeg.yaml \
        --n-batches 1 --steps 2000 --out .LOCAL/overfit/run1
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import torch

from samudra.config import DynamicLossConfig, TrainConfig
from samudra.stepper import train_batch
from samudra.train import Trainer
from samudra.utils.logging import handle_logging

logger = logging.getLogger("overfit")

VARS = ["uo", "vo", "thetao", "so"]
DEPTHS = [0, 6, 10, 14, 18]  # level indices summarized in the log line


def channel_names(n_prog: int) -> list[str]:
    names = [f"{v}_{j}" for v in VARS for j in range(19)] + ["zos"]
    assert len(names) == n_prog, (len(names), n_prog)
    return names


@torch.no_grad()
def score(model, batches, n_prog: int) -> dict[str, torch.Tensor]:
    """Masked per-channel MSE (normalized units) for model and persistence.

    Returns sums over batches/output-steps of squared error and wet-cell counts,
    per prognostic channel, so callers can form RMSEs.
    """
    model.eval()
    se_model = torch.zeros(n_prog, dtype=torch.float64)
    se_pers = torch.zeros(n_prog, dtype=torch.float64)
    se_res_base = torch.zeros(n_prog, dtype=torch.float64)
    count = torch.zeros(n_prog, dtype=torch.float64)
    for batch in batches:
        prog, _ = batch.get_initial_input()
        preds = model(batch)
        for step, pred in enumerate(preds):
            label = batch.get_label(step)
            wet = batch.ctx.label_mask.to(label.device)
            out_steps = label.shape[1] // n_prog
            last = prog[:, -n_prog:]
            # The residual base used by pred_residuals (same-slot assembly).
            res_base = prog[:, -label.shape[1] :]
            for k in range(out_steps):
                sl = slice(k * n_prog, (k + 1) * n_prog)
                w = wet[sl].unsqueeze(0).to(torch.float64)
                y = label[:, sl].double()
                se_model += (((pred[:, sl].double() - y) ** 2) * w).sum((0, 2, 3))
                se_pers += (((last.double() - y) ** 2) * w).sum((0, 2, 3))
                se_res_base += (((res_base[:, sl].double() - y) ** 2) * w).sum(
                    (0, 2, 3)
                )
                count += w.sum((0, 2, 3)) * y.shape[0]
            # Only score the first rollout step when steps > 1.
            break
    model.train()
    return {
        "rmse_model": (se_model / count).sqrt(),
        "rmse_persistence": (se_pers / count).sqrt(),
        "rmse_residual_base": (se_res_base / count).sqrt(),
    }


def summarize(tag: str, s: dict[str, torch.Tensor], names: list[str]) -> dict:
    ratio = s["rmse_model"] / s["rmse_persistence"]
    out = {
        "median_ratio": ratio.median().item(),
        "max_ratio": ratio.max().item(),
        "frac_beat_persistence": (ratio < 1).double().mean().item(),
        "mean_rmse_model": s["rmse_model"].mean().item(),
    }
    parts = []
    for v_i, v in enumerate(VARS):
        r = [ratio[v_i * 19 + d].item() for d in DEPTHS]
        parts.append(f"{v}[" + " ".join(f"{x:.3g}" for x in r) + "]")
    parts.append(f"zos[{ratio[-1].item():.3g}]")
    logger.info(
        f"{tag}: model/persistence median={out['median_ratio']:.3g} "
        f"max={out['max_ratio']:.3g} beat={out['frac_beat_persistence']:.0%} "
        f"| " + " ".join(parts)
    )
    return out


def parse_limit(value: str) -> float | None:
    if value.lower() == "none":
        return None
    return float(value)


def zero_init_decoder(model: torch.nn.Module) -> None:
    """Zero the final output conv in place (keeps optimizer/EMA references valid)."""
    decoder = getattr(model, "decoder", None)
    if not isinstance(decoder, torch.nn.Conv2d):
        raise TypeError(
            f"--zero-init-decoder needs model.decoder to be an nn.Conv2d, got "
            f"{type(decoder).__name__} on {type(model).__name__}"
        )
    with torch.no_grad():
        decoder.weight.zero_()
        if decoder.bias is not None:
            decoder.bias.zero_()
    logger.info("zero-init-decoder applied: decoder weight and bias set to 0")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--n-batches", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000, help="optimizer steps")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--pred-residuals", choices=["true", "false"], default=None)
    p.add_argument("--loss", choices=["mse", "dynamic"], default=None)
    p.add_argument(
        "--delta-weight",
        action="store_true",
        help="Weight each output channel's MSE by 1/(persistence MSE on the "
        "memorized batches): the loss then measures skill over persistence, "
        "equivalent to normalizing targets by the spread of the change.",
    )
    p.add_argument(
        "--dynamic-limit",
        type=parse_limit,
        default=20.0,
        help="Max/min channel-weight ratio for --loss dynamic: a float >= 1 or "
        "'none' for no limit (default 20).",
    )
    p.add_argument(
        "--zero-init-decoder",
        action="store_true",
        help="Zero the final output conv (weights and bias) so the initial "
        "model output equals the residual base exactly.",
    )
    p.add_argument("--fp32", action="store_true")
    p.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Stop training once wall time exceeds this (a final eval, history "
        "write and model save still happen at the stopping step).",
    )
    args = p.parse_args()
    if args.n_batches < 1:
        raise ValueError(f"--n-batches must be >= 1, got {args.n_batches}")
    if args.eval_every < 1:
        raise ValueError(f"--eval-every must be >= 1, got {args.eval_every}")
    if args.steps < 1:
        raise ValueError(f"--steps must be >= 1, got {args.steps}")
    if args.dynamic_limit != 20.0 and args.loss != "dynamic":
        raise ValueError("--dynamic-limit requires --loss dynamic")

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig.from_yaml(args.config)
    cfg.experiment.base_output_dir = out_dir
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.pred_residuals is not None:
        cfg.model.pred_residuals = args.pred_residuals == "true"
    if args.loss == "mse":
        cfg.loss = "mse"
    elif args.loss == "dynamic":
        cfg.loss = DynamicLossConfig(
            type="dynamic", metric="mse", limit=args.dynamic_limit
        )
    if args.fp32:
        cfg.model.use_bfloat16 = False
    if cfg.scheduler is not None:
        raise ValueError("overfit.py does not step a scheduler; set scheduler: null")
    if cfg.gradient_accumulation_steps != 1:
        raise ValueError(
            "overfit.py steps the optimizer every batch; got "
            f"gradient_accumulation_steps={cfg.gradient_accumulation_steps}"
        )
    if args.delta_weight and cfg.loss != "mse":
        raise ValueError(
            f"--delta-weight requires the plain 'mse' loss, got cfg.loss={cfg.loss!r}"
        )
    cfg.prepare_output_dirs()
    handle_logging(False, cfg.experiment.output_dir)
    logging.getLogger("overfit").setLevel(logging.INFO)

    trainer = Trainer(cfg)
    if args.zero_init_decoder:
        zero_init_decoder(trainer.model)
    trainer.init_data_loaders(cfg.steps[0])
    model, loss_fn, opt = trainer.model, trainer.loss_fn, trainer.optimizer
    n_prog = trainer.N_prog
    names = channel_names(n_prog)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"model params: {n_params / 1e6:.2f}M, pred_residuals="
        f"{model.pred_residuals}, lr={cfg.learning_rate}"
    )

    # Take only what we need: each 1° batch is ~200 MB.
    train_batches = list(itertools.islice(trainer.train_loader, args.n_batches))
    heldout = list(itertools.islice(trainer.val_loader, 1))
    if not heldout:
        raise ValueError("held-out loader produced no batches")
    if len(train_batches) != args.n_batches:
        raise ValueError(
            f"requested {args.n_batches} train batches, loader gave "
            f"{len(train_batches)}"
        )
    logger.info(
        f"memorizing {len(train_batches)} batch(es) of size {cfg.batch_size} "
        f"(of {len(trainer.train_loader)} available); held-out batches: "
        f"{len(heldout)}"
    )

    s0 = score(model, train_batches, n_prog)
    if args.delta_weight:
        base_loss = loss_fn
        # Per output channel (time-major), persistence MSE in the same
        # land-zeroed, all-cells-averaged units the loss uses.
        mse_p = torch.zeros(trainer.num_out, dtype=torch.float64)
        for b in train_batches:
            prog, _ = b.get_initial_input()
            label = b.get_label(0)
            pers = prog[:, -n_prog:].repeat(1, label.shape[1] // n_prog, 1, 1)
            mse_p += base_loss(pers, label, b.ctx).double()
        if not (torch.isfinite(mse_p) & (mse_p > 0)).all():
            raise ValueError(f"persistence MSE must be finite and > 0, got {mse_p}")
        weight = (len(train_batches) / mse_p).float()
        logger.info(f"delta weights: min={weight.min():.3g} max={weight.max():.3g}")

        def loss_fn(pred, target, ctx):  # noqa: F811
            return base_loss(pred, target, ctx) * weight

    logger.info(f"arm: {vars(args)}; loss={cfg.loss}; bf16={cfg.model.use_bfloat16}")
    logger.info(
        "residual-base/persistence median ratio: "
        f"{(s0['rmse_residual_base'] / s0['rmse_persistence']).median().item():.3g}"
    )

    history = []
    t0 = time.time()
    train_s = 0.0  # optimizer-step time only (excludes eval)
    for step in range(args.steps + 1):
        out_of_time = (
            args.max_hours is not None and time.time() - t0 > args.max_hours * 3600
        )
        last = step == args.steps or out_of_time
        if step % args.eval_every == 0 or last:
            s_tr = score(model, train_batches, n_prog)
            s_ho = score(model, heldout, n_prog)
            rec = {
                "step": step,
                "elapsed_s": time.time() - t0,
                "train_s": train_s,
                "train": summarize(f"step {step:5d} train  ", s_tr, names),
                "heldout": summarize(f"step {step:5d} heldout", s_ho, names),
                "train_ratio_per_channel": (
                    s_tr["rmse_model"] / s_tr["rmse_persistence"]
                ).tolist(),
                "train_rmse_persistence": s_tr["rmse_persistence"].tolist(),
                "train_rmse_model": s_tr["rmse_model"].tolist(),
            }
            history.append(rec)
            (out_dir / "history.json").write_text(
                json.dumps({"channels": names, "history": history}, indent=1)
            )
            torch.save(model.state_dict(), out_dir / "model.pt")
        if last:
            break
        t_step = time.time()
        batch = train_batches[step % len(train_batches)]
        opt.zero_grad()
        out = train_batch(model, batch, loss_fn)
        out.loss.backward()
        if args.no_clip:
            if not torch.isfinite(out.loss).all():
                raise RuntimeError(f"non-finite loss at step {step}: {out.loss}")
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
        opt.step()
        trainer._maybe_update_loss(out, batch)
        train_s += time.time() - t_step
        if step % 10 == 0:
            logger.info(
                f"step {step:5d} loss={out.loss.item():.4e} "
                f"({train_s / (step + 1):.1f}s/step)"
            )


if __name__ == "__main__":
    main()
