# snn_results_viewer — Prototype

Small Streamlit prototype to inspect runs produced by the thesis NEST project.

Quick start

1. Create a virtual environment (recommended) and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Thesis repo integration

- This prototype uses a quick sys.path import to the thesis repo. Edit
  `app/loader.py` and change `THESIS_REPO_PATH` to point at your source tree
  if needed.
- For a more robust development workflow you can `pip install -e` the thesis
  project (if it has a setup.py/pyproject). That way changes to the thesis
  source are immediately visible to the app.

Next steps

- Replace loader stubs with real parsing of run folders and additional file
  previews (images, numpy arrays, metrics).
- Add tests in `tests/` for loader functions.
