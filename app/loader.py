"""Loader utilities for the Streamlit SNN results viewer.

This module bridges the Streamlit app with the "real" thesis repository. It
provides two public helpers:

* list_runs(results_root) – discover experiment/run folders.
* load_run(run_path) – load config, spikes, weights and summary metrics.

Both functions are intentionally defensive: if the thesis repo is not
available we fall back to a deterministic stub so the UI continues to render.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Parameters for MR regression plot (dt configurable via env; default 1 ms).
MR_DEFAULT_DT_MS = float(os.environ.get("SNN_RESULTS_VIEWER_MR_DT_MS", "1.0"))
MR_FIT_MIN_MS = 10.0
MR_FIT_MAX_MS = 60.0
MR_MIN_FIT_POINTS = 3
MR_FIT_USE_OFFSET = True

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
    from analysis.util import (  # type: ignore
        load_run as thesis_load_run,
        combine_spikes,
    )
except Exception:  # pragma: no cover - optional dependency
    thesis_load_run = None
    combine_spikes = None

try:
    from analysis.summary_metrics import compute_summary_metrics  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    compute_summary_metrics = None

try:
    from analysis.metrics import (  # type: ignore
        average_inter_event_interval,
        empty_bin_fraction,
        branching_ratios_binned_global,
        branching_ratio_mr_estimator,
        MRESTIMATOR_AVAILABLE,
        build_weight_matrix,
        normalize_weight_matrix,
    )
except Exception:  # pragma: no cover - optional dependency
    average_inter_event_interval = None
    empty_bin_fraction = None
    branching_ratios_binned_global = None
    branching_ratio_mr_estimator = None
    MRESTIMATOR_AVAILABLE = False
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


def _compute_criticality_report(
    run_dir: Path,
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    mr_dt_ms: Optional[float] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    if combine_spikes is None:
        raise RuntimeError("combine_spikes helper unavailable")
    if (
        average_inter_event_interval is None
        or empty_bin_fraction is None
        or branching_ratios_binned_global is None
    ):
        raise RuntimeError("Criticality metrics dependencies are missing")

    experiment_cfg = cfg.get("experiment") if isinstance(cfg, dict) else None
    stimulation_cfg = cfg.get("stimulation", {}) if isinstance(cfg, dict) else {}
    if not experiment_cfg:
        raise ValueError("Experiment configuration missing")

    simtime_ms = float(experiment_cfg.get("simtime_ms", 0.0))
    t_start_ms = 0.0
    t_stop_ms = simtime_ms

    pattern_cfg = stimulation_cfg.get("pattern", {}) if isinstance(stimulation_cfg, dict) else {}
    dc_cfg = stimulation_cfg.get("dc", {}) if isinstance(stimulation_cfg, dict) else {}
    dc_enabled = bool(dc_cfg.get("enabled", False))

    if dc_enabled:
        t_off_val = pattern_cfg.get("t_off_ms")
        if t_off_val is None:
            t_off_val = dc_cfg.get("t_off_ms", t_stop_ms)
        t_off_ms = float(t_off_val)
        t_start_ms = min(max(t_off_ms, 0.0), t_stop_ms)

    spikes_all = combine_spikes(
        data,
        ["spikes_E", "spikes_IH", "spikes_IA"],
    )
    if not spikes_all:
        raise ValueError("Spike data not available for criticality metrics")

    times_full = np.asarray(spikes_all.get("times"))
    senders_full = np.asarray(spikes_all.get("senders"))
    if times_full.size == 0 or senders_full.size == 0:
        raise ValueError("Spike data empty for selected window")

    mask = (times_full >= t_start_ms) & (times_full <= t_stop_ms)
    times = times_full[mask]

    if times.size < 2:
        raise ValueError("Not enough spikes within analysis window")

    net_cfg = cfg.get("network", {})
    N_total = int(
        net_cfg.get("N_E", 0)
        + net_cfg.get("N_IH", 0)
        + net_cfg.get("N_IA_1", net_cfg.get("N_IA", 0))
        + net_cfg.get("N_IA_2", 0)
    )

    aiei_ms, _ = average_inter_event_interval(
        times_ms=times,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
    )

    if not np.isfinite(aiei_ms) or aiei_ms <= 0.0:
        raise ValueError("Average inter-event interval is not defined")

    dt_factors = np.array([0.5, 1.0, 2.0], dtype=float)
    dt_min_ms = 1.0
    effective_mr_dt = float(mr_dt_ms if mr_dt_ms is not None else MR_DEFAULT_DT_MS)
    if effective_mr_dt <= 0.0:
        effective_mr_dt = MR_DEFAULT_DT_MS

    lines = [
        f"**Run dir:** `{run_dir}`",
        f"**Time window (ms):** ({t_start_ms:.1f}, {t_stop_ms:.1f})",
        f"**N_total:** {N_total}",
        f"**AIEI (ms):** {aiei_ms:.3f}",
        "**dt factors:** " + ", ".join(f"{f:.2f}" for f in dt_factors),
    ]
    mr_available = bool(MRESTIMATOR_AVAILABLE and branching_ratio_mr_estimator is not None)
    if mr_available:
        lines.append("**MR estimator:** available")
    else:
        lines.append("**MR estimator:** unavailable (install 'mrestimator')")
    lines.append(f"**MR dt (ms):** {effective_mr_dt:.3f}")

    lines.extend(
        [
            "",
            "| dt/AIEI | dt (ms) | p0(dt) | sigma_global | sigma_MR | tau_MR (ms) |",
            "|--------|---------|--------|--------------|----------|-------------|",
        ]
    )

    dt_candidates: List[float] = []
    for factor in dt_factors:
        dt_raw = float(factor * aiei_ms)
        dt_candidates.append(max(dt_raw, dt_min_ms))
    # Allow dedicated MR plot dt (1 ms) even if below dt_min.
    final_dt_values: List[float] = []

    def _append_unique(value: float) -> None:
        if value <= 0:
            return
        if not any(
            math.isclose(value, existing, rel_tol=1e-9, abs_tol=1e-9)
            for existing in final_dt_values
        ):
            final_dt_values.append(value)

    for value in dt_candidates:
        _append_unique(value)

    _append_unique(effective_mr_dt)

    def _fmt_or_na(value: float) -> str:
        return f"{value:.3f}" if np.isfinite(value) else "n/a"

    mr_plot_payload: Optional[Dict[str, Any]] = None

    for dt_ms in final_dt_values:
        p0 = empty_bin_fraction(
            times_ms=times,
            t_start_ms=t_start_ms,
            t_stop_ms=t_stop_ms,
            dt_ms=dt_ms,
        )

        sigma_binned, _, counts = branching_ratios_binned_global(
            times_ms=times,
            t_start_ms=t_start_ms,
            t_stop_ms=t_stop_ms,
            dt_ms=dt_ms,
            min_aval_len=2,
        )

        sigma_mr = np.nan
        tau_mr = np.nan
        coeffs = None
        fit_result = None
        wants_plot_details = mr_available and math.isclose(dt_ms, effective_mr_dt, rel_tol=1e-9, abs_tol=1e-9)
        if mr_available and wants_plot_details:
            try:
                result = branching_ratio_mr_estimator(
                    spike_counts=counts,
                    dt_ms=dt_ms,
                    fit_lag_ms_min=MR_FIT_MIN_MS,
                    fit_lag_ms_max=MR_FIT_MAX_MS,
                    fit_use_offset=MR_FIT_USE_OFFSET,
                    min_fit_points=MR_MIN_FIT_POINTS,
                    return_details=wants_plot_details,
                )
                if wants_plot_details:
                    sigma_mr, tau_mr, coeffs, fit_result = result  # type: ignore[assignment]
                else:
                    sigma_mr, tau_mr = result  # type: ignore[misc]
            except Exception:
                sigma_mr = np.nan
                tau_mr = np.nan
                coeffs = None
                fit_result = None
        elif mr_available:
            # Skip MR computation for other dt values
            sigma_mr = np.nan
            tau_mr = np.nan

        dt_over_aiei = dt_ms / aiei_ms
        lines.append(
            "| {dt_over_aiei:6.3f} | {dt_ms:7.3f} | {p0:6.3f} | {sigma:12.3f} | {sigma_mr:>8} | {tau_mr:>11} |".format(
                dt_over_aiei=dt_over_aiei,
                dt_ms=dt_ms,
                p0=p0,
                sigma=sigma_binned,
                sigma_mr=_fmt_or_na(sigma_mr),
                tau_mr=_fmt_or_na(tau_mr),
            )
        )

        if wants_plot_details and coeffs is not None and fit_result is not None and mr_plot_payload is None:
            steps = np.asarray(getattr(coeffs, "steps", []), dtype=float)
            coeff_vals = np.asarray(getattr(coeffs, "coefficients", []), dtype=float)
            dt_coeff = float(getattr(coeffs, "dt", dt_ms))
            time_ms = steps * dt_coeff
            fitfunc = getattr(fit_result, "fitfunc", None)
            params = getattr(fit_result, "popt", None)
            if callable(fitfunc) and params is not None:
                try:
                    fit_vals = fitfunc(steps, *params)
                except Exception:
                    fit_vals = np.full_like(time_ms, np.nan)
            else:
                fit_vals = np.full_like(time_ms, np.nan)

            if time_ms.size and coeff_vals.size:
                mr_plot_payload = {
                    "dt_ms": dt_ms,
                    "time_ms": time_ms,
                    "rk": coeff_vals,
                    "fit": fit_vals,
                    "title": f"MR regression (dt={dt_ms:.3f} ms)",
                }

    return "\n".join(lines), mr_plot_payload


def load_run(run_path: str | None = None, *, mr_dt_ms: Optional[float] = None) -> Dict[str, Any]:
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
    if weights_final:
        if (
            network_cfg
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
        else:
            weight_matrix_error = (
                "Weight matrix helpers unavailable or weight data missing"
            )

    summary: Optional[Dict[str, Any]] = None
    if compute_summary_metrics and cfg and data.get("spikes_E"):
        try:
            summary = compute_summary_metrics(cfg, data, weights_over_time)
        except Exception as exc:  # pragma: no cover - diagnostics only
            summary = {"error": str(exc)}

    preview_series = _ensure_preview_series(summary, data)

    criticality_markdown: Optional[str] = None
    criticality_mr_plot: Optional[Dict[str, Any]] = None
    criticality_error: Optional[str] = None
    if cfg and data:
        try:
            criticality_markdown, criticality_mr_plot = _compute_criticality_report(
                run_dir,
                cfg,
                data,
                mr_dt_ms=mr_dt_ms,
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            criticality_error = str(exc)

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
        "criticality_markdown": criticality_markdown,
        "criticality_mr_regression": criticality_mr_plot,
        "criticality_error": criticality_error,
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
