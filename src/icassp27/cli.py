from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config, model_config


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="icassp27")
    root.add_argument("--config", default="configs/experiment.yaml")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ["preflight", "data-smoke", "prepare-data", "cache-baseline", "replacement-row", "consolidate",
                 "group", "teacher", "geometry", "recover", "select-hparams", "evaluate", "efficiency",
                 "plot-replacement", "feasibility-audit"]:
        sub = commands.add_parser(name)
        sub.add_argument("--backbone", required=True)
        if name in {"cache-baseline", "replacement-row", "consolidate", "group", "geometry", "recover", "select-hparams", "evaluate", "efficiency", "plot-replacement", "feasibility-audit"}:
            sub.add_argument("--valid-tokens", type=int)
        if name in {"group", "geometry", "recover", "select-hparams", "evaluate", "efficiency", "plot-replacement"}:
            sub.add_argument("--k", type=int)
        if name in {"geometry", "recover", "select-hparams", "evaluate", "efficiency", "plot-replacement"}:
            sub.add_argument("--policy", default="full")
        if name == "replacement-row":
            sub.add_argument("--target", type=int)
        if name == "recover":
            sub.add_argument("--variant", required=True)
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--rank", type=int, default=128)
            sub.add_argument("--lambda-align", type=float, default=0.3)
            sub.add_argument("--supervised", choices=["final", "all_shared"], default="final")
            sub.add_argument("--selection-only", action="store_true")
        if name in {"evaluate", "efficiency"}:
            sub.add_argument("--role", choices=["reference", "teacher", "step0", "student"], required=True)
            sub.add_argument("--checkpoint")
            sub.add_argument("--output", required=True)
    commands.add_parser("report")
    commands.add_parser("smoke")
    robustness = commands.add_parser("robustness")
    robustness.add_argument("--replacement-dirs", nargs="+", required=True)
    robustness.add_argument("--group-manifests", nargs="+", required=True)
    robustness.add_argument("--output", required=True)
    return root


def _required(value, name: str):
    if value is None:
        raise SystemExit(f"{name} is required for this command")
    return value


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "preflight":
        from transformers import AutoConfig, AutoTokenizer
        mcfg = model_config(cfg, args.backbone)
        source = mcfg.get("model_path", mcfg["model_id"])
        local = bool(mcfg.get("model_path"))
        config = AutoConfig.from_pretrained(source, revision=mcfg["revision"], local_files_only=local)
        tokenizer = AutoTokenizer.from_pretrained(
            source, revision=mcfg["tokenizer_revision"], use_fast=True, local_files_only=local
        )
        print({"ok": True, "backbone": args.backbone, "model_type": config.model_type,
               "layers": getattr(config, "num_hidden_layers", None), "tokenizer": tokenizer.__class__.__name__,
               "source": source, "revision": mcfg["revision"], "local_files_only": local})
    elif args.command == "prepare-data":
        from .data import prepare_data
        prepare_data(cfg, args.backbone)
    elif args.command == "data-smoke":
        from .data import smoke_data
        smoke_data(cfg, args.backbone)
    elif args.command == "cache-baseline":
        from .replacement import cache_baseline
        cache_baseline(cfg, args.backbone, _required(args.valid_tokens, "--valid-tokens"))
    elif args.command == "replacement-row":
        from .replacement import replacement_row
        target = args.target if args.target is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        replacement_row(cfg, args.backbone, _required(args.valid_tokens, "--valid-tokens"), target)
    elif args.command == "consolidate":
        from .replacement import consolidate_replacement
        consolidate_replacement(cfg, args.backbone, _required(args.valid_tokens, "--valid-tokens"))
    elif args.command == "group":
        from .grouping import build_groups
        build_groups(cfg, args.backbone, _required(args.valid_tokens, "--valid-tokens"), _required(args.k, "--k"))
    elif args.command == "teacher":
        from .teacher import train_teacher
        train_teacher(cfg, args.backbone)
    elif args.command == "geometry":
        from .recovery import estimate_geometry
        estimate_geometry(cfg, args.backbone, _required(args.valid_tokens, "--valid-tokens"),
                          _required(args.k, "--k"), args.policy)
    elif args.command == "recover":
        from .recovery import train_student
        train_student(cfg, args.backbone, _required(args.valid_tokens, "--valid-tokens"), _required(args.k, "--k"),
                      args.policy, args.variant, args.seed, args.rank, args.lambda_align, args.supervised,
                      args.selection_only)
    elif args.command == "select-hparams":
        from .recovery import select_hyperparameters
        select_hyperparameters(cfg, args.backbone, _required(args.valid_tokens, "--valid-tokens"),
                               _required(args.k, "--k"), args.policy)
    elif args.command == "evaluate":
        from .runner import evaluate_role
        evaluate_role(cfg, args.backbone, args.role, args.output, args.valid_tokens, args.k, args.policy, args.checkpoint)
    elif args.command == "efficiency":
        from .runner import efficiency_audit
        efficiency_audit(cfg, args.backbone, args.role, args.output, args.valid_tokens, args.k, args.policy, args.checkpoint)
    elif args.command == "plot-replacement":
        from .report import replacement_plots
        replacement_plots(cfg, args.backbone, _required(args.valid_tokens, "--valid-tokens"), _required(args.k, "--k"), args.policy)
    elif args.command == "feasibility-audit":
        from .grouping import feasibility_audit
        feasibility_audit(cfg, args.backbone, _required(args.valid_tokens, "--valid-tokens"))
    elif args.command == "report":
        from .report import aggregate_report
        aggregate_report(cfg)
    elif args.command == "smoke":
        from .smoke import run_smoke
        run_smoke(cfg)
    elif args.command == "robustness":
        from .statistics import calibration_robustness
        calibration_robustness(args.replacement_dirs, args.group_manifests, args.output)


if __name__ == "__main__":
    main()
