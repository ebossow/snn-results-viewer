"""Streamlit-based UI for inspecting SNN experiment runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app import loader, ui


st.set_page_config(page_title="SNN Results Viewer", layout="wide")


def _trigger_rerun() -> None:
    """Call the modern rerun API with a fallback for older Streamlit versions."""
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    legacy_fn = getattr(st, "experimental_rerun", None)
    if callable(legacy_fn):
        legacy_fn()
        return
    raise AttributeError("Streamlit rerun API is unavailable in this environment")


def normalize_results_root(path: str) -> str:
    """Return a usable path (auto-append /results if needed)."""
    if not path:
        return path
    try:
        if os.path.isdir(path):
            entries = [
                name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))
            ]
            if any(name.startswith("run_") for name in entries):
                return path
        candidate = os.path.join(path, "results")
        if os.path.isdir(candidate):
            return candidate
    except Exception:
        pass
    return path


def _format_bytes(num: int | None) -> str:
    if num is None:
        return "?"
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def _array_summary(arr: np.ndarray) -> dict:
    summary = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if arr.size and arr.size <= 200_000:
        summary.update(
            {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }
        )
    return summary


def render_file_preview(file_info: Dict[str, Any]) -> None:
    path = Path(str(file_info["path"]))
    if file_info.get("is_dir"):
        st.info("Directory – open it locally to inspect contents.")
        return

    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            st.code(path.read_text(), language="yaml")
        elif suffix == ".json":
            st.json(json.load(path.open()))
        elif suffix in {".txt", ".log"}:
            st.text(path.read_text())
        elif suffix in {".png", ".jpg", ".jpeg", ".svg"}:
            st.image(str(path))
        elif suffix == ".npz":
            with np.load(path, allow_pickle=True) as data:
                st.write({key: _array_summary(data[key]) for key in data.files})
                if data.files:
                    first = data[data.files[0]]
                    if isinstance(first, np.ndarray) and first.ndim == 1 and first.size <= 2000:
                        st.line_chart(first)
        elif suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
            st.write(_array_summary(arr))
            if arr.ndim == 1 and arr.size <= 2000:
                st.line_chart(arr)
        else:
            st.info("Preview not available for this file type. Use the download button above.")
    except Exception as exc:
        st.error(f"Failed to preview file: {exc}")


def render_spike_raster(data: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    spikes_E = data.get("spikes_E")
    spikes_IH = data.get("spikes_IH")
    spikes_IA = data.get("spikes_IA")
    if not any((spikes_E, spikes_IH, spikes_IA)):
        st.info("No spike data available in this run.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    def _scatter_population(spikes: Optional[Dict[str, np.ndarray]], color: str, label: str) -> bool:
        if not spikes:
            return False
        times = np.asarray(spikes.get("times"))
        senders = np.asarray(spikes.get("senders"))
        if times.size == 0 or senders.size == 0:
            return False
        ax.scatter(times, senders - 1, s=4, c=color, label=label)
        return True

    _scatter_population(spikes_E, "#d62728", "Excitatory (E)")

    inhib_times: list[np.ndarray] = []
    inhib_senders: list[np.ndarray] = []
    for spikes in (spikes_IH, spikes_IA):
        if not spikes:
            continue
        times = np.asarray(spikes.get("times"))
        senders = np.asarray(spikes.get("senders"))
        if times.size == 0 or senders.size == 0:
            continue
        inhib_times.append(times)
        inhib_senders.append(senders - 1)

    if inhib_times:
        ax.scatter(
            np.concatenate(inhib_times),
            np.concatenate(inhib_senders),
            s=4,
            c="#1f77b4",
            label="Inhibitory (IH/IA)",
        )

    simtime = cfg.get("experiment", {}).get("simtime_ms")
    if isinstance(simtime, (int, float)):
        ax.set_xlim(0, simtime)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron index (GID)")
    ax.set_title("Spike raster")
    if ax.has_data():
        ax.legend(loc="upper right")
    st.pyplot(fig)


def render_weight_matrix(matrix: Any, cfg: Dict[str, Any], error: str | None) -> None:
    if error:
        st.error(f"Could not build weight matrix: {error}")
        return
    if matrix is None:
        st.info("No weight matrix available.")
        return
    arr = np.asarray(matrix)
    if arr.size == 0:
        st.info("Weight matrix is empty.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, origin="lower", vmin=-1.0, vmax=1.0, cmap="bwr")
    ax.set_xlabel("Pre-synaptic neuron index")
    ax.set_ylabel("Post-synaptic neuron index")
    ax.set_title("Normalized weight matrix")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)


def render_mr_regression(plot_data: Dict[str, Any]) -> None:
    time_ms = np.asarray(plot_data.get("time_ms"))
    rk = np.asarray(plot_data.get("rk"))
    fit_vals = np.asarray(plot_data.get("fit"))
    title = plot_data.get("title") or "MR regression"

    if time_ms.size == 0 or rk.size == 0:
        st.info("MR regression data unavailable for this run.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_ms, rk, "o", label=r"$r_k$")
    if fit_vals.size:
        ax.plot(time_ms, fit_vals, "-", label="MR fit")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel(r"Correlation $r_k$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


st.title("SNN Results Viewer")

default_results_root = str(Path(loader.THESIS_REPO_PATH) / "results")

if "results_root_text" not in st.session_state:
    st.session_state["results_root_text"] = default_results_root
if "current_results_root" not in st.session_state:
    st.session_state["current_results_root"] = normalize_results_root(default_results_root)

results_root_input = st.sidebar.text_input(
    "Thesis results root",
    value=st.session_state["results_root_text"],
    help="Full path to the thesis 'results' folder (e.g. /Users/erik/.../CodeBase/results)",
)

def _set_current_root(path: str) -> None:
    normalized = normalize_results_root(path)
    st.session_state["current_results_root"] = normalized
    st.session_state["results_root_text"] = normalized

if results_root_input != st.session_state["results_root_text"]:
    _set_current_root(results_root_input)

refresh_requested = st.sidebar.button("Refresh runs")

nav_subdirs = loader.list_subdirectories(st.session_state["current_results_root"])
subdir_options = ["(select subfolder)"] + [entry["name"] for entry in nav_subdirs]
selected_subdir = st.sidebar.selectbox("Browse subfolders", subdir_options)

if st.sidebar.button("Enter folder", disabled=selected_subdir == subdir_options[0]):
    target = next((entry for entry in nav_subdirs if entry["name"] == selected_subdir), None)
    if target:
        _set_current_root(target["path"])
        _trigger_rerun()
    else:
        st.sidebar.warning("Selected folder is no longer available. Please pick another entry.")

parent_path = str(Path(st.session_state["current_results_root"]).parent)
if st.sidebar.button("Go up", disabled=parent_path == st.session_state["current_results_root"]):
    _set_current_root(parent_path)
    _trigger_rerun()

current_root = st.session_state["current_results_root"]
if not os.path.isdir(current_root):
    st.sidebar.error(f"Path does not exist: {current_root}")

runs = loader.list_runs(results_root=current_root)

if refresh_requested:
    _trigger_rerun()

if runs and runs[0]['path'].startswith('/tmp/run_stub_'):
    st.warning(
        "No runs detected under the provided path. "
        "Please confirm it points to the thesis 'results' directory."
    )

if not runs:
    st.stop()

experiments = sorted({r.get('experiment', 'ungrouped') for r in runs})
selected_experiment = st.selectbox("Experiment", experiments)

runs_in_exp = [r for r in runs if r.get('experiment', 'ungrouped') == selected_experiment]
if not runs_in_exp:
    runs_in_exp = runs

run_options = [
    {
        "label": f"{r['name']}" + (f" ({r.get('timestamp')})" if r.get('timestamp') else ""),
        "info": r,
    }
    for r in runs_in_exp
]
run_labels = [opt["label"] for opt in run_options]
selected_label = st.selectbox("Run", run_labels)
selected_run_info = next(opt["info"] for opt in run_options if opt["label"] == selected_label)

run_details = loader.load_run(selected_run_info['path'])
metadata = run_details.get('metadata') or {}
config = run_details.get('config') or {}
summary = run_details.get('summary') or {}
data = run_details.get('data') or {}

with st.expander("Run overview", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Experiment", selected_experiment)
    col_b.metric("Run folder", selected_run_info['name'])
    sim_time = config.get('experiment', {}).get('simtime_ms')
    col_c.metric("Simtime (ms)", sim_time if sim_time is not None else "n/a")
    st.caption(f"Path: {selected_run_info['path']}")

with st.expander("Summary metrics", expanded=False):
    if isinstance(summary, dict) and summary:
        if 'error' in summary:
            st.error(f"Metric computation failed: {summary['error']}")
        else:
            metric_cols = st.columns(3)
            metric_cols[0].metric("Mean rate (Hz)", f"{summary.get('mean_rate_Hz', float('nan')):.2f}")
            metric_cols[1].metric("Mean CV", f"{summary.get('mean_cv', float('nan')):.2f}")
            metric_cols[2].metric("Mean R", f"{summary.get('mean_R_all', float('nan')):.2f}")
    else:
        st.info("Summary metrics not available for this run.")

criticality_markdown = run_details.get('criticality_markdown')
criticality_error = run_details.get('criticality_error')
criticality_mr_plot = run_details.get('criticality_mr_regression')

with st.expander("Criticality metrics", expanded=False):
    if criticality_markdown:
        if criticality_mr_plot:
            table_col, plot_col = st.columns((3, 2))
            with table_col:
                st.markdown(criticality_markdown)
            with plot_col:
                dt_ms = criticality_mr_plot.get("dt_ms") if isinstance(criticality_mr_plot, dict) else None
                if isinstance(dt_ms, (int, float)) and np.isfinite(dt_ms):
                    st.markdown(f"#### MR regression (dt = {dt_ms:.1f} ms)")
                else:
                    st.markdown("#### MR regression")
                render_mr_regression(criticality_mr_plot)
        else:
            st.markdown(criticality_markdown)
    elif criticality_error:
        st.error(f"Criticality metrics unavailable: {criticality_error}")
    else:
        st.info("Criticality metrics not available for this run.")

with st.expander("Visualizations", expanded=True):
    col_raster, col_weight = st.columns(2)
    with col_raster:
        st.markdown("#### Spike raster")
        render_spike_raster(data, config)
    with col_weight:
        st.markdown("#### Weight matrix")
        render_weight_matrix(
            run_details.get('weight_matrix'),
            config,
            run_details.get('weight_matrix_error'),
        )

with st.expander("Config", expanded=False):
    st.code(ui.config_to_yaml_text(config))

with st.expander("Metadata", expanded=False):
    if metadata:
        st.code(ui.config_to_yaml_text(metadata))
    else:
        st.info("No metadata available for this run.")

with st.expander("Files", expanded=False):
    files = run_details.get('files', [])
    if not files:
        st.info("No files found in run folder.")
    else:
        file_table_data = [
            {
                "Name": f['name'],
                "Type": f['ext'] or ('dir' if f['is_dir'] else ''),
                "Size": _format_bytes(f.get('size')),
            }
            for f in files
        ]
        st.dataframe(file_table_data, width="stretch")

        selectable_files = [f for f in files if not f.get('is_dir')]
        if selectable_files:
            file_names = [f["name"] for f in selectable_files]
            selected_file = st.selectbox("Preview file", file_names)
            file_info = next(f for f in selectable_files if f["name"] == selected_file)
            render_file_preview(file_info)
        else:
            st.info("No previewable files in this run.")
