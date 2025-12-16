"""Loader utilities for the Streamlit SNN results viewer.

This module bridges the Streamlit app with the "real" thesis repository. It
provides two public helpers:

* list_runs(results_root) – discover experiment/run folders.
* load_run(run_path) – load config, spikes, weights and summary metrics.

Both functions are intentionally defensive: if the thesis repo is not
available we fall back to a deterministic stub so the UI continues to render.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Path to your thesis codebase (source of truth for data/analysis).
# Override via the SNN_RESULTS_VIEWER_THESIS_PATH environment variable when
# needed (e.g. for local development with a different checkout path).
THESIS_REPO_PATH = os.environ.get(
    "SNN_RESULTS_VIEWER_THESIS_PATH",
    "/Users/erik/Documents/Uni/Bachelor Thesis/CodeBase",
)

# Ensure quick sys.path import works for prototypes. For a robust setup prefer
# editable install (pip install -e) inside the thesis repo.
THESIS_SRC_PATH = str(Path(THESIS_REPO_PATH) / "src")
for candidate in (THESIS_REPO_PATH, THESIS_SRC_PATH):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:  # Optional imports – available once the thesis repo is accessible
    from analysis.util import load_run as thesis_load_run  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    thesis_load_run = None

try:
    from analysis.summary_metrics import compute_summary_metrics  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    compute_summary_metrics = None

try:
    from analysis.metrics import (  # type: ignore
        build_weight_matrix,
        normalize_weight_matrix,
    )
except Exception:  # pragma: no cover - optional dependency
    build_weight_matrix = None
    normalize_weight_matrix = None


def _resolve_results_root(results_root: Optional[str]) -> Path:
    if results_root:
        root = Path(results_root).expanduser()
    else:
        root = Path(THESIS_REPO_PATH) / "results"
    return root


def _looks_like_run_folder(path: Path) -> bool:
    """Heuristic to detect if a folder contains a single run."""
    if not path.is_dir():
        return False

    config_candidates = (
        "config_resolved.yaml",
        "config.yaml",
        "config.yml",
        "metadata.yaml",
        "metadata.json",
    )
    for candidate in config_candidates:
        if (path / candidate).is_file():
            return True

    for child in path.iterdir():
        if child.suffix.lower() in {".npz", ".npy", ".png", ".jpg", ".jpeg", ".svg"}:
            return True
    return False


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _collect_file_info(run_dir: Path) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for entry in sorted(run_dir.iterdir()):
        info = {
            "name": entry.name,
            "path": str(entry),
            "is_dir": entry.is_dir(),
            "ext": entry.suffix.lower(),
        }
        try:
            info["size"] = entry.stat().st_size
        except Exception:
            info["size"] = None
        files.append(info)
    return files


def _build_run_info(run_dir: Path, experiment: str | None = None) -> Dict[str, Any]:
    metadata = _read_yaml(run_dir / "metadata.yaml")
    return {
        "name": run_dir.name,
        "experiment": experiment or run_dir.parent.name,
        "path": str(run_dir),
        "timestamp": metadata.get("timestamp"),
        "metadata": metadata,
    }


def list_runs(results_root: str | None = None) -> List[Dict[str, Any]]:
    """Return a list of runs grouped by experiment folders."""
    root = _resolve_results_root(results_root)

    runs: List[Dict[str, Any]] = []
    if not root.exists():
        return [{"name": "run_stub_2025-01-01", "path": "/tmp/run_stub_2025-01-01", "experiment": "stub"}]

    if _looks_like_run_folder(root):
        return [_build_run_info(root, experiment=root.parent.name if root.parent else "")]  # type: ignore[arg-type]

    for experiment_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        experiment_name = experiment_dir.name
        for run_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
            if run_dir.name.startswith("run_") or _looks_like_run_folder(run_dir):
                runs.append(_build_run_info(run_dir, experiment_name))

    if not runs:
        # Fallback: expose every subdirectory and let the user pick manually.
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            runs.append(_build_run_info(run_dir, experiment=root.name))

    if not runs:
        runs = [{"name": "run_stub_2025-01-01", "path": "/tmp/run_stub_2025-01-01", "experiment": "stub"}]

    return runs


def _load_npz_arrays(path: Path, allow_pickle: bool = False) -> Dict[str, Any]:
    with np.load(path, allow_pickle=allow_pickle) as data:  # type: ignore[no-untyped-call]
        return {key: data[key] for key in data.files}


def _ensure_preview_series(summary: Optional[Dict[str, Any]], data: Dict[str, Any]) -> np.ndarray:
    candidates: List[np.ndarray] = []
    if summary:
        for key in ("R", "mean_rates_per_neuron", "K_post"):
            arr = summary.get(key)
            if isinstance(arr, np.ndarray) and arr.size > 0:
                candidates.append(arr)
    spikes_E = data.get("spikes_E")
    if spikes_E and isinstance(spikes_E.get("times"), np.ndarray):
        candidates.append(spikes_E["times"])

    for arr in candidates:
        flat = np.asarray(arr).ravel()
        if flat.size > 1:
            return flat

    # deterministic stub fallback
    t = np.linspace(0, 2 * np.pi, 200)
    return np.sin(t)


def load_run(run_path: str | None = None) -> Dict[str, Any]:
    """Load config, spikes, weights, and summary metrics for a run."""
    run_dir = Path(run_path) if run_path else None
    if run_dir is None or not run_dir.exists():
        t = np.linspace(0, 2 * np.pi, 200)
        return {
            "config": {
                "experiment": "stub",
                "notes": "Thesis repo not available; using deterministic data.",
            },
            "metadata": {},
            "files": [],
            "data": {},
            "weights_final": None,
            "weights_trajectory": None,
            "summary": None,
            "preview_series": np.sin(t),
            "weight_matrix": None,
            "weight_matrix_error": None,
        }

    cfg: Dict[str, Any] = {}
    data: Dict[str, Any] = {}
    weights_final: Optional[Dict[str, Any]] = None
    weights_over_time: Optional[Dict[str, Any]] = None

    if thesis_load_run is not None:
        try:
            cfg, data, weights_final, weights_over_time = thesis_load_run(run_dir)
        except Exception:
            cfg, data, weights_final, weights_over_time = {}, {}, None, None

    if not cfg:
        cfg = _read_yaml(run_dir / "config_resolved.yaml")
        if not cfg:
            cfg = _read_yaml(run_dir / "config.yaml")

    if not data:
        def load_spikes(name: str) -> Optional[Dict[str, np.ndarray]]:
            fpath = run_dir / f"{name}.npz"
            if not fpath.is_file():
                return None
            arrs = _load_npz_arrays(fpath)
            if "times" in arrs and "senders" in arrs:
                return {"times": arrs["times"], "senders": arrs["senders"]}
            return arrs

        data = {
            "spikes_E": load_spikes("spikes_E"),
            "spikes_IH": load_spikes("spikes_IH"),
            "spikes_IA": load_spikes("spikes_IA"),
        }

    if weights_final is None:
        wf_path = run_dir / "weights_final.npz"
        if wf_path.is_file():
            weights_final = _load_npz_arrays(wf_path, allow_pickle=True)

    if weights_over_time is None:
        wt_path = run_dir / "weights_trajectory.npz"
        if wt_path.is_file():
            weights_over_time = _load_npz_arrays(wt_path)

    metadata = _read_yaml(run_dir / "metadata.yaml")

    weight_matrix: Optional[np.ndarray] = None
    weight_matrix_error: Optional[str] = None
    network_cfg = cfg.get("network") if isinstance(cfg, dict) else None
    expected_weight_keys = {"sources", "targets", "weights"}
    if (
        weights_final
        and network_cfg
        and build_weight_matrix is not None
        and normalize_weight_matrix is not None
    ):
        if isinstance(weights_final, dict) and expected_weight_keys.issubset(weights_final.keys()):
            try:
                required_keys = {"N_E", "N_IH", "N_IA_1", "N_IA_2"}
                if not required_keys.issubset(network_cfg.keys()):
                    raise KeyError("Missing network sizes required for weight matrix")

                sources = weights_final.get("sources")
                targets = weights_final.get("targets")
                weights_vals = weights_final.get("weights")

                if sources is not None and targets is not None and weights_vals is not None:
                    N_total = (
                        int(network_cfg["N_E"])
                        + int(network_cfg["N_IH"])
                        + int(network_cfg["N_IA_1"])
                        + int(network_cfg["N_IA_2"])
                    )
                    dense = build_weight_matrix(sources, targets, weights_vals, N_total)
                    weight_matrix = normalize_weight_matrix(dense, cfg)
            except Exception as exc:  # pragma: no cover - diagnostics only
                weight_matrix_error = str(exc)
        else:
            weight_matrix_error = (
                "Weights stored in unsupported legacy format. "
                "Please regenerate the run with the updated exporter."
            )

    summary: Optional[Dict[str, Any]] = None
    if compute_summary_metrics and cfg and data.get("spikes_E"):
        try:
            summary = compute_summary_metrics(cfg, data, weights_over_time)
        except Exception as exc:  # pragma: no cover - diagnostics only
            summary = {"error": str(exc)}

    preview_series = _ensure_preview_series(summary, data)

    return {
        "config": cfg,
        "metadata": metadata,
        "files": _collect_file_info(run_dir),
        "data": data,
        "weights_final": weights_final,
        "weights_trajectory": weights_over_time,
        "summary": summary,
        "preview_series": preview_series,
        "weight_matrix": weight_matrix,
        "weight_matrix_error": weight_matrix_error,
    }


def list_subdirectories(results_root: str | None = None) -> List[Dict[str, Any]]:
    """Return immediate sub-directories for navigation purposes."""
    root = _resolve_results_root(results_root)
    entries: List[Dict[str, Any]] = []
    if not root.exists():
        return entries

    for child in sorted(root.iterdir()):
        if child.is_dir():
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_run": _looks_like_run_folder(child),
                }
            )
    return entries
