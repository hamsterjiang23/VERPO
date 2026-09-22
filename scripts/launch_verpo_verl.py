#!/usr/bin/env python3
"""Public native veRL launcher for the registered SDPO Section 3 protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_aware_opsd.verpo_launch_config import (
    ResolvedVerpoLaunch,
    VerpoLaunchConfigError,
    beta_engine_environment,
    engine_command,
    load_launch_matrix_spec,
    native_verl_projection,
    resolve_verpo_launch,
    trl_verpo_projection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--finetuning", default="full")
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--teacher")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--arm")
    selection.add_argument("--matrix")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--print-config", action="store_true")
    output.add_argument("--print-command", action="store_true")
    return parser


def _default_run_id(resolved: ResolvedVerpoLaunch) -> str:
    identity = resolved.config["identity"]
    return (
        "_".join(
            str(identity[key])
            for key in ("model", "finetuning", "protocol", "teacher", "arm", "hardware")
        )
        + "_"
        + resolved.config_hash[:12]
    )


def _engine_command(resolved: ResolvedVerpoLaunch) -> tuple[list[str], dict[str, str]]:
    return engine_command(resolved), beta_engine_environment(resolved)


def _runtime_environment(
    resolved: ResolvedVerpoLaunch,
    *,
    matrix: bool,
    matrix_id: str | None,
    cell_id: str | None,
) -> tuple[dict[str, str], Path]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VERPO_", "PAPER_BASELINE_", "QWEN3_"))
    }
    generated = beta_engine_environment(resolved)
    environment.update(generated)
    run_id = os.environ.get("RUN_ID", _default_run_id(resolved))
    if matrix:
        matrix_root = Path(
            os.environ.get(
                "MATRIX_OUTPUT_ROOT",
                str(
                    ROOT
                    / "outputs"
                    / "sdpo_section3"
                    / str(matrix_id)
                    / str(resolved.config["identity"]["model"])
                    / str(resolved.config["identity"]["finetuning"])
                    / str(resolved.config["identity"]["hardware"])
                ),
            )
        )
        output_root = matrix_root / str(cell_id)
    else:
        output_root = Path(
            os.environ.get(
                "OUTPUT_ROOT", str(ROOT / "outputs" / "sdpo_section3" / run_id)
            )
        )
    protocol = resolved.config["protocol"]
    data_root = Path(
        os.environ.get("SDPO_DATA_DIR", str(ROOT / "data" / "SDPO" / "verl"))
    )

    def resolve_file(value: str, key: str) -> Path:
        if os.environ.get(key):
            return Path(os.environ[key]).expanduser().resolve()
        path = Path(value)
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    environment["TRAIN_FILE"] = str(
        resolve_file(str(protocol["train_file"]), "TRAIN_FILE")
    )
    environment["VAL_FILE"] = str(
        resolve_file(str(protocol["validation_file"]), "VAL_FILE")
    )
    if str(protocol["train_file"]).startswith("data/SDPO/"):
        if not os.environ.get("TRAIN_FILE"):
            environment["TRAIN_FILE"] = str(
                (
                    data_root
                    / Path(str(protocol["train_file"])).relative_to("data/SDPO/verl")
                ).resolve()
            )
        if not os.environ.get("VAL_FILE"):
            environment["VAL_FILE"] = str(
                (
                    data_root
                    / Path(str(protocol["validation_file"])).relative_to(
                        "data/SDPO/verl"
                    )
                ).resolve()
            )
    environment["SDPO_DATA_DIR"] = str(data_root.resolve())
    if os.environ.get("TRAIN_FILE") or os.environ.get("VAL_FILE"):
        environment["SDPO_AUTO_DOWNLOAD"] = "false"
    environment["OUTPUT_ROOT"] = str(output_root.resolve())
    environment["RUN_ID"] = run_id
    return environment, output_root.resolve()


def _write_manifest(
    resolved: ResolvedVerpoLaunch,
    output_root: Path,
    environment: dict[str, str] | None = None,
) -> None:
    if environment is None:
        environment, _ = _runtime_environment(
            resolved, matrix=False, matrix_id=None, cell_id=None
        )
    provenance = output_root / "formal" / "provenance"
    manifest_path = provenance / "semantic_config.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not manifest_path.is_file():
            raise VerpoLaunchConfigError(
                "Refusing to reuse nonempty output without semantic provenance"
            )
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != resolved.config_hash:
            raise VerpoLaunchConfigError(
                "Output belongs to another config/evidence protocol; use a new output root"
            )
    actual_projection = native_verl_projection(resolved, environment)
    previous_projection_path = provenance / "native_verl_projection.json"
    if previous_projection_path.exists():
        old_projection = json.loads(
            previous_projection_path.read_text(encoding="utf-8")
        )
        for section in ("data",):
            if old_projection.get(section) != actual_projection.get(section):
                raise VerpoLaunchConfigError(
                    "Effective data inputs changed; use a new output root"
                )
        if (
            old_projection["actor_rollout_ref"]["model"]["path"]
            != actual_projection["actor_rollout_ref"]["model"]["path"]
        ):
            raise VerpoLaunchConfigError(
                "Effective model input changed; use a new output root"
            )
    provenance.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sorted(set(paths))
        if name
        and (ROOT / name).is_file()
        and Path(name).suffix in {".py", ".sh", ".yaml", ".json"}
    }
    (provenance / "code_identity.json").write_text(
        json.dumps(
            {
                "commit": revision,
                "dirty": bool(
                    subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
                ),
                "source_sha256": source_hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (provenance / "semantic_config.json").write_text(
        json.dumps(resolved.manifest(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (provenance / "semantic_config.yaml").write_text(
        yaml.safe_dump(resolved.config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (provenance / "native_verl_projection.json").write_text(
        json.dumps(
            native_verl_projection(resolved, environment), indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    (provenance / "trl_verpo_projection.json").write_text(
        json.dumps(trl_verpo_projection(resolved), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _print_resolved(
    resolved: ResolvedVerpoLaunch,
    *,
    command: bool,
    environment: dict[str, str] | None = None,
) -> None:
    identity = resolved.config["identity"]
    print(
        f"# model={identity['model']} protocol={identity['protocol']} teacher={identity['teacher']} arm={identity['arm']} hash={resolved.config_hash}"
    )
    print(yaml.safe_dump(resolved.config, sort_keys=False, allow_unicode=True).rstrip())
    if command:
        engine, env = _engine_command(resolved)
        print("# native_verl_projection")
        print(
            yaml.safe_dump(
                native_verl_projection(resolved, environment), sort_keys=True
            ).rstrip()
        )
        print("# trl_verpo_projection")
        print(yaml.safe_dump(trl_verpo_projection(resolved), sort_keys=True).rstrip())
        print("# engine_environment")
        for key in sorted(env):
            print(f"{key}={env[key]}")
        print("# engine_command")
        print(shlex.join(engine))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.arm:
        if not args.protocol or not args.teacher:
            raise VerpoLaunchConfigError("--arm requires --protocol and --teacher")
        cells = ((args.arm, args.protocol, args.teacher, args.arm),)
        matrix_overrides: tuple[str, ...] = ()
        cell_overrides: dict[str, tuple[str, ...]] = {}
    else:
        spec = load_launch_matrix_spec(args.matrix)
        if any(cell.protocol is not None for cell in spec.cells) and args.protocol:
            raise VerpoLaunchConfigError(
                "matrix owns the protocol axis; omit --protocol"
            )
        if any(cell.teacher is not None for cell in spec.cells) and args.teacher:
            raise VerpoLaunchConfigError("matrix owns the Teacher axis; omit --teacher")
        cells = tuple(
            (
                cell.identifier,
                cell.protocol or args.protocol,
                cell.teacher or args.teacher,
                cell.arm,
            )
            for cell in spec.cells
        )
        if any(
            protocol is None or teacher is None for _, protocol, teacher, _ in cells
        ):
            raise VerpoLaunchConfigError("matrix requires protocol and teacher axes")
        matrix_overrides = spec.overrides
        cell_overrides = {cell.identifier: cell.overrides for cell in spec.cells}
    resolved_runs = [
        (
            cell_id,
            resolve_verpo_launch(
                model=args.model,
                finetuning=args.finetuning,
                hardware=args.hardware,
                protocol=protocol,
                teacher=teacher,
                arm=arm,
                cli_overrides=matrix_overrides
                + cell_overrides.get(cell_id, ())
                + tuple(args.overrides),
            ),
        )
        for cell_id, protocol, teacher, arm in cells
    ]
    if args.print_config or args.print_command:
        for index, (cell_id, resolved) in enumerate(resolved_runs):
            if index:
                print("---")
            environment, _ = _runtime_environment(
                resolved,
                matrix=args.matrix is not None,
                matrix_id=args.matrix,
                cell_id=cell_id,
            )
            _print_resolved(
                resolved, command=args.print_command, environment=environment
            )
        return 0
    for cell_id, resolved in resolved_runs:
        command, _ = _engine_command(resolved)
        environment, output_root = _runtime_environment(
            resolved,
            matrix=args.matrix is not None,
            matrix_id=args.matrix,
            cell_id=cell_id,
        )
        if not environment.get("SWANLAB_API_KEY"):
            raise VerpoLaunchConfigError(
                "SWANLAB_API_KEY is required for formal SDPO training"
            )
        native_verl_projection(resolved, environment)  # fail before writing outputs
        _write_manifest(resolved, output_root, environment)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerpoLaunchConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
