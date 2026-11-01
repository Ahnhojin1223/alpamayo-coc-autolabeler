#!/usr/bin/env python3
"""Run CoC labeling and validation with subscription CLIs from one JSON config."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = REPO_ROOT / "scripts" / "coc_validation_stage"
DEFAULT_CONFIG = REPO_ROOT / "config" / "pipeline.json"

BACKEND_ENV = {
    "codex": ("CODEX_CLI_MODEL", "CODEX_CLI_THINKING"),
    "claude_code": ("CLAUDE_CLI_MODEL", "CLAUDE_CLI_THINKING"),
}
CROSS_VALIDATOR = {"codex": "claude_code", "claude_code": "codex"}
STAGE_SCRIPTS = {
    1: "eval_stage1_coc_evidence_vllm.py",
    2: "eval_stage2_coc_behavior_vllm.py",
    3: "eval_stage3_coc_causal_vllm.py",
}
VALIDATION_MODES = ("cross", "same")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the pipeline config and reject settings the run could not honor."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for backend in config["labeling"]["backends"]:
        if backend not in BACKEND_ENV:
            raise ValueError(f"Unknown labeling backend: {backend}")
    for backend in BACKEND_ENV:
        if backend not in config["backends"]:
            raise ValueError(f"Missing backends.{backend} in {config_path}")
    if config["validation"]["mode"] not in VALIDATION_MODES:
        raise ValueError(f"validation.mode must be one of {VALIDATION_MODES}")
    for stage in config["validation"]["stages"]:
        if stage not in STAGE_SCRIPTS:
            raise ValueError(f"Unknown validation stage: {stage}")
    return config


def resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve configured paths against the repository root."""
    return {name: (REPO_ROOT / value).resolve() for name, value in config["paths"].items()}


def backend_env(config: dict[str, Any], backend: str) -> dict[str, str]:
    """Build the environment that selects a backend's model and thinking level."""
    env = dict(os.environ)
    model_env, thinking_env = BACKEND_ENV[backend]
    for name, key in ((model_env, "model"), (thinking_env, "thinking")):
        env.pop(name, None)
        value = config["backends"][backend].get(key)
        if value:
            env[name] = value
    return env


def run_labeling(config: dict[str, Any], paths: dict[str, Path], backend: str) -> None:
    """Generate CoC labels with one backend; existing labels for the segments are overwritten."""
    override = {
        "data.data_dir": paths["data_dir"],
        "data.cache_dir": paths["cache_dir"],
        "data.video_dir": paths["video_dir"],
        "data.meta_action_dir": paths["meta_action_dir"],
        "data.segment_config_path": paths["segment_config_path"],
        "resume_exp_dir": paths["output_dir"] / f"coc_{backend}",
        "hydra.run.dir": paths["output_dir"] / "ing" / f"hydra_{backend}",
    }
    command = [
        sys.executable,
        "-m",
        "coc_labeling.data_labeling",
        "--config-name=base_config_vlm_rel_ts",
        f"model_name={backend}",
        f"exp_name={backend}",
        "data_loader.video.save_segment_videos=false",
        f"data_loader.vector.trajdata.num_workers={config['trajdata_num_workers']}",
        *(f"{key}='{value.as_posix()}'" for key, value in override.items()),
    ]
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=backend_env(config, backend))


def clear_stale_validation(paths: dict[str, Path], label_backend: str) -> None:
    """Remove scores of replaced labels, which validation --resume would otherwise keep."""
    for run_dir in (paths["output_dir"] / "validation").glob(f"{label_backend}_by_*"):
        shutil.rmtree(run_dir)


def run_validation(
    config: dict[str, Any], paths: dict[str, Path], label_backend: str
) -> list[tuple[Path, int, int]]:
    """Validate one backend's labels and return (output, ok rows, total rows) per stage."""
    mode = config["validation"]["mode"]
    validator = CROSS_VALIDATOR[label_backend] if mode == "cross" else label_backend
    out_dir = paths["output_dir"] / "validation" / f"{label_backend}_by_{validator}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for stage in config["validation"]["stages"]:
        json_out = out_dir / f"stage{stage}.json"
        command = [
            sys.executable,
            str(VALIDATION_DIR / STAGE_SCRIPTS[stage]),
            "--coc-result-dir",
            str(paths["output_dir"] / f"coc_{label_backend}"),
            "--video-root",
            str(paths["video_dir"]),
            "--model-name",
            validator,
            "--resume",
            "--jsonl-out",
            str(out_dir / f"stage{stage}.jsonl"),
            "--json-out",
            str(json_out),
        ]
        subprocess.run(command, check=True, cwd=VALIDATION_DIR, env=backend_env(config, validator))
        rows = json.loads(json_out.read_text(encoding="utf-8"))
        summary.append((json_out, sum(bool(row.get("ok")) for row in rows), len(rows)))
    return summary


def main() -> int:
    """Run the configured labeling and validation steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = resolve_paths(config)
    backends = config["labeling"]["backends"]

    if config["labeling"]["enabled"]:
        for backend in backends:
            run_labeling(config, paths, backend)
            clear_stale_validation(paths, backend)

    all_ok = True
    if config["validation"]["enabled"]:
        for backend in backends:
            for json_out, ok_rows, total_rows in run_validation(config, paths, backend):
                print(f"{json_out}: {ok_rows}/{total_rows} rows ok")
                all_ok = all_ok and total_rows > 0 and ok_rows == total_rows
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
