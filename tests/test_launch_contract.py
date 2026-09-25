import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from risk_aware_opsd.verpo_launch_config import (
    CONFIG_ROOT,
    ROOT,
    ResolvedVerpoLaunch,
    VerpoLaunchConfigError,
    load_launch_matrix_spec,
    native_verl_projection,
    resolve_verpo_launch,
)
from scripts.launch_verpo_verl import _runtime_environment, _write_manifest


def resolve(**kwargs: str) -> ResolvedVerpoLaunch:
    defaults = {
        "model": "qwen3_4b",
        "hardware": "a800_8x_80gb",
        "protocol": "sdpo_section3_biology",
        "teacher": "ema_095",
        "arm": "fec_fkl",
    }
    return resolve_verpo_launch(**(defaults | kwargs))


@pytest.mark.parametrize(
    "kind,key",
    [
        ("models", "model"),
        ("hardware", "hardware"),
        ("teachers", "teacher"),
        ("arms", "arm"),
        ("protocols", "protocol"),
        ("finetuning", "finetuning"),
    ],
)
def test_registered_config_axes_reach_actual_engine(kind: str, key: str) -> None:
    for path in (CONFIG_ROOT / kind).glob("*.yaml"):
        r = resolve(**{key: path.stem})
        projection = native_verl_projection(r)
        c = r.config
        actor = projection["actor_rollout_ref"]["actor"]
        assert (
            projection["trainer"]["total_training_steps"]
            == c["protocol"]["total_training_steps"]
        )
        assert projection["trainer"]["save_freq"] == c["protocol"]["save_frequency"]
        assert actor["verpo"]["evidence_source"] == "rollout_group"
        assert (
            actor["verpo"]["advantage_modulation"]
            == c["method"]["advantage_modulation"]
        )
        assert float(actor["verpo"]["lambda_evi"]) == c["method"]["lambda_evi"]
        assert (
            projection["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"]
            == c["hardware"]["vllm_gpu_memory_utilization"]
        )
        assert actor["loss_mode"] == c["method"]["actor_loss_mode"]


def test_paper_matrix_all_backbones() -> None:
    spec = load_launch_matrix_spec("paper_main")
    assert len(spec.cells) == 10
    for model in ("qwen3_4b", "qwen3_8b", "llama3_2_1b_instruct"):
        for cell in spec.cells:
            r = resolve_verpo_launch(
                model=model,
                hardware="a800_8x_80gb",
                protocol=cell.protocol,
                teacher=cell.teacher,
                arm=cell.arm,
                cli_overrides=spec.overrides + cell.overrides,
            )
            projection = native_verl_projection(r)
            assert projection["trainer"]["total_training_steps"] == 200
            assert projection["trainer"]["max_actor_ckpt_to_keep"] is None
            actor = projection["actor_rollout_ref"]["actor"]
            assert actor["verpo"]["lambda_evi"] == (
                0.0 if cell.arm == "fec_advmod_fkl" else 1.0
            )


def test_explicit_files_and_protocol_resume_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIN_FILE", str(tmp_path / "custom-train.parquet"))
    monkeypatch.setenv("VAL_FILE", str(tmp_path / "custom-val.parquet"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "run"))
    r = resolve()
    env, output = _runtime_environment(r, matrix=False, matrix_id=None, cell_id=None)
    assert env["TRAIN_FILE"] == str((tmp_path / "custom-train.parquet").resolve())
    assert env["VAL_FILE"] == str((tmp_path / "custom-val.parquet").resolve())
    assert env["SDPO_AUTO_DOWNLOAD"] == "false"
    _write_manifest(r, output)
    _write_manifest(r, output)
    with pytest.raises(VerpoLaunchConfigError, match="another config"):
        _write_manifest(resolve(arm="fec_advmod_fkl"), output)


def test_overrides_survive_and_unsafe_memory_rejected() -> None:
    r = resolve_verpo_launch(
        model="qwen3_4b",
        hardware="a800_8x_80gb",
        protocol="sdpo_section3_biology",
        teacher="ema_095",
        arm="fec_fkl",
        cli_overrides=(
            "protocol.total_training_steps=7",
            "protocol.validation_frequency=2",
            "protocol.save_frequency=0",
        ),
    )
    result = native_verl_projection(r)
    assert result["trainer"]["total_training_steps"] == 7
    assert result["trainer"]["test_freq"] == 2
    assert result["trainer"]["save_freq"] == 0
    with pytest.raises(VerpoLaunchConfigError, match="utilization"):
        resolve_verpo_launch(
            model="qwen3_4b",
            hardware="a800_8x_80gb",
            protocol="sdpo_section3_biology",
            teacher="ema_095",
            arm="fec_fkl",
            cli_overrides=("hardware.vllm_gpu_memory_utilization=0.88",),
        )


@pytest.mark.parametrize(
    "module",
    [
        "risk_aware_opsd.train_trl_verpo",
        "scripts.prepare_sdpo_data",
        "scripts.package_sdpo_training_data",
        "scripts.sync_sdpo_public_bundle",
        "scripts.summarize_results",
    ],
)
def test_cli_help_without_torch(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_yaml_documents_parse() -> None:
    for path in (ROOT / "configs").rglob("*.yaml"):
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)


def test_runtime_identity_rejects_changed_dataset(tmp_path: Path) -> None:
    from scripts.check_runtime_identity import check_identity

    train, validation, model = (
        tmp_path / "train.jsonl",
        tmp_path / "val.jsonl",
        tmp_path / "model",
    )
    train.write_text("original", encoding="utf-8")
    validation.write_text("validation", encoding="utf-8")
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "run"
    check_identity(root, train, validation, model, "revision")
    check_identity(root, train, validation, model, "revision")
    train.write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        check_identity(root, train, validation, model, "revision")


def test_all_matrices_resolve_and_preserve_cell_overrides() -> None:
    for path in (CONFIG_ROOT / "matrices").glob("*.yaml"):
        spec = load_launch_matrix_spec(path.stem)
        assert len({cell.identifier for cell in spec.cells}) == len(spec.cells)
        for cell in spec.cells:
            r = resolve_verpo_launch(
                model="qwen3_8b",
                hardware="a800_8x_80gb",
                protocol=cell.protocol or "sdpo_section3_biology",
                teacher=cell.teacher or "ema_095",
                arm=cell.arm,
                cli_overrides=spec.overrides + cell.overrides,
            )
            assert (
                native_verl_projection(r)["actor_rollout_ref"]["actor"]["verpo"][
                    "evidence_source"
                ]
                == "rollout_group"
            )


def test_models_use_separate_matrix_output_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MATRIX_OUTPUT_ROOT", raising=False)
    roots = [
        _runtime_environment(
            resolve(model=model),
            matrix=True,
            matrix_id="paper_main",
            cell_id="biology_lw",
        )[1]
        for model in ("qwen3_4b", "qwen3_8b", "llama3_2_1b_instruct")
    ]
    assert len(set(roots)) == 3


def test_manifest_uses_effective_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIN_FILE", str(tmp_path / "train.parquet"))
    monkeypatch.setenv("VAL_FILE", str(tmp_path / "val.parquet"))
    r = resolve()
    env, _ = _runtime_environment(r, matrix=False, matrix_id=None, cell_id=None)
    output = tmp_path / "out"
    _write_manifest(r, output, env)
    import json

    saved = json.loads(
        (output / "formal/provenance/native_verl_projection.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["data"]["train_files"] == env["TRAIN_FILE"]
    assert saved["data"]["val_files"] == env["VAL_FILE"]


def test_unknown_and_nonfinite_override_rejected() -> None:
    for expression in (
        "method.lambda_evi_typo=1",
        "method.lambda_evi=.nan",
        "protocol.learning_rate=.inf",
    ):
        with pytest.raises(VerpoLaunchConfigError):
            resolve_verpo_launch(
                model="qwen3_4b",
                hardware="a800_8x_80gb",
                protocol="sdpo_section3_biology",
                teacher="ema_095",
                arm="fec_fkl",
                cli_overrides=(expression,),
            )


@pytest.mark.parametrize("flag", ["--print-config", "--print-command"])
def test_public_dry_run_creates_no_output(
    tmp_path: Path, flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    env = os.environ.copy()
    env.update(
        {
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "SDPO_DATA_DIR": str(tmp_path / "data"),
        }
    )
    if flag == "--print-config":
        env["VERPO_BASH"] = str(tmp_path / "missing-bash")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.launch_verpo_verl",
            "--model",
            "qwen3_4b",
            "--hardware",
            "a800_8x_80gb",
            "--protocol",
            "sdpo_section3_biology",
            "--teacher",
            "ema_095",
            "--arm",
            "fec_fkl",
            flag,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    if flag == "--print-command":
        projection_text = result.stdout.split("# native_verl_projection\n", 1)[1].split(
            "# trl_verpo_projection", 1
        )[0]
        projection = yaml.safe_load(projection_text)
        assert projection["trainer"]["logger"] == ["console"]
        assert projection["trainer"]["project_name"] == "verpo"
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "data").exists()


def test_changed_runtime_paths_do_not_overwrite_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "out"))
    r = resolve()
    env, output = _runtime_environment(r, matrix=False, matrix_id=None, cell_id=None)
    _write_manifest(r, output, env)
    path = output / "formal/provenance/native_verl_projection.json"
    before = path.read_bytes()
    env["TRAIN_FILE"] = str(tmp_path / "other.parquet")
    with pytest.raises(VerpoLaunchConfigError, match="data inputs changed"):
        _write_manifest(r, output, env)
    assert path.read_bytes() == before


def test_training_dispatch_needs_no_tracking_credentials(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from scripts import launch_verpo_verl as launcher

    environment = {}
    monkeypatch.setattr(
        launcher, "_runtime_environment", lambda *args, **kwargs: (environment, tmp_path)
    )
    manifest = MagicMock()
    dispatch = MagicMock()
    real_run = subprocess.run

    def run_or_record(command, **kwargs):
        if kwargs.get("capture_output"):
            return real_run(command, **kwargs)
        return dispatch(command, **kwargs)

    monkeypatch.setattr(launcher, "_write_manifest", manifest)
    monkeypatch.setattr(launcher.subprocess, "run", run_or_record)
    assert launcher.main([
        "--model", "qwen3_4b", "--hardware", "a800_8x_80gb",
        "--protocol", "sdpo_section3_biology", "--teacher", "ema_095",
        "--arm", "fec_fkl",
    ]) == 0
    manifest.assert_called_once()
    dispatch.assert_called_once()
    assert dispatch.call_args.kwargs["env"] == {}
