#!/usr/bin/env python3
"""Synthetic VERPO loss smoke; this is not pretrained-model TRL/GRPO training.

The CLI intentionally does not adapt Parquet SDPO records.  Native veRL owns
the formal Section 3 protocols; this path is for small text fixtures and
portable VERPO experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .math_text_protocol import load_jsonl, validate_math_text_row
from .verpo_launch_config import VERPOConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--toy-vocab-size", type=int, default=32)
    parser.add_argument("--toy-hidden-size", type=int, default=16)
    return parser


def _load_config(path: Path | None) -> VERPOConfig:
    if path is None:
        return VERPOConfig()
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = raw.get("config", raw)
    method = values.get("method", values)
    teacher = values.get("teacher", {})
    zpd = values.get("zpd", {})
    return VERPOConfig(
        divergence=str(method.get("divergence", "forward_kl")),
        displacement_mode=str(method.get("displacement", "evidence_vs_none")),
        lambda_ref=float(method.get("lambda_ref", 0.1)),
        lambda_evi=float(method.get("lambda_evi", 0.5)),
        teacher_mode=str(teacher.get("mode", "fixed_initial")),
        teacher_sync_interval=int(teacher.get("sync_interval", 10)),
        teacher_ema_decay=float(teacher.get("ema_decay", 0.95)),
        temperature=float(method.get("temperature", 1.0)),
        vocab_mode=str(method.get("vocab_mode", "full")),
        top_k=int(method.get("top_k", 128)),
        vocab_chunk_size=int(method.get("vocab_chunk_size", 4096)),
        cost_alpha=float(method.get("cost_alpha", 0.0025)),
        cost_epsilon=float(method.get("cost_epsilon", 2.5e-5)),
        group_zpd_enabled=bool(zpd.get("group_enabled", False)),
        group_zpd_mode=str(zpd.get("group_mode", "reward_ranked")),
        sibling_selection_mode=str(zpd.get("sibling_selection", "correctness")),
        evidence_rollout_scope=str(zpd.get("evidence_scope", "all")),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = load_jsonl(args.train_file, limit=args.limit or None)
    for row in rows:
        validate_math_text_row(row)
    if not rows:
        raise ValueError("training JSONL is empty")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("TRL VERPO smoke requires the CPU dependency group: uv sync --extra cpu") from exc

    from .verpo_trainer import VERPOTrainer

    torch.manual_seed(0)
    vocab = max(4, int(args.toy_vocab_size))
    hidden = max(2, int(args.toy_hidden_size))
    model = torch.nn.Sequential(torch.nn.Embedding(vocab, hidden), torch.nn.Linear(hidden, vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = VERPOTrainer(model, config=_load_config(args.config), optimizer=optimizer)
    # A deterministic tiny fixture keeps the smoke independent of a tokenizer.
    token_ids = torch.tensor([[index % vocab, (index + 1) % vocab] for index in range(len(rows))], dtype=torch.long)
    for _ in range(max(1, int(args.max_steps))):
        logits = model(token_ids)
        direction = torch.linspace(-0.3, 0.3, vocab, device=logits.device)
        teacher = logits.detach() + direction
        weights = torch.ones(logits.shape[:-1])
        loss, metrics = trainer.compute_verpo_loss(
            logits,
            reference_logits=teacher,
            base_teacher_logits=teacher,
            evidence_teacher_logits=teacher + direction.flip(0) * 0.2,
            positive_teacher_logits=teacher + direction.flip(0) * 0.2,
            negative_teacher_logits=teacher - direction * 0.4,
            no_evidence_teacher_logits=teacher,
            sampled_token_ids=token_ids,
            evidence_weights=weights,
            nuisance_projection=torch.zeros_like(weights),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = sum(parameter.grad.detach().square().sum() for parameter in model.parameters() if parameter.grad is not None).sqrt()
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError("synthetic smoke requires finite loss and nonzero finite gradient")
        metrics["gradient_norm"] = gradient_norm
        optimizer.step()
        trainer.global_step += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.output_dir / "checkpoint-final")
    (args.output_dir / "metrics.json").write_text(json.dumps({key: float(value) for key, value in metrics.items()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "global_step": trainer.global_step, "loss": float(loss.detach())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
