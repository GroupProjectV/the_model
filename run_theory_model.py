"""Build a small executable notebook with the theory-driven section using saved artifacts.
Then merge its outputs into the previously-executed full notebook and export HTML."""
import json, shutil, subprocess, os

# 1. Build a self-contained notebook that loads artifacts + runs the new section
def mkmd(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}
def mkcode(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

setup = mkcode([
    "# Setup: load artifacts from the previous full run\n",
    "import pandas as pd, numpy as np, json\n",
    "import plotly.express as px, plotly.graph_objects as go\n",
    "from plotly.subplots import make_subplots\n",
    "import warnings; warnings.filterwarnings('ignore')\n",
    "RANDOM_STATE = 42\n",
    "\n",
    "df_sample = pd.read_parquet('app_artifacts/df_sample.parquet')\n",
    "df_model = pd.read_parquet('app_artifacts/df_model.parquet')\n",
    "with open('app_artifacts/topic_labels.json') as f: topic_labels = {int(k): v for k,v in json.load(f).items()}\n",
    "with open('app_artifacts/political_topic_ids.json') as f: political_topic_ids = json.load(f)\n",
    "\n",
    "# Reconstruct intermediate frames needed by the theory-driven cells\n",
    "df_states = df_sample.dropna(subset=['state_code']).copy()\n",
    "df_pol_states = df_states[df_states['topic'].isin(political_topic_ids)].copy()\n",
    "\n",
    "# state_stance_valid -> stance_lean from df_model.partisan_lean\n",
    "state_partisan_lean = df_model.set_index('state_code')['partisan_lean']\n",
    "\n",
    "print(f'df_sample: {df_sample.shape}, df_model: {df_model.shape}')\n",
    "print(f'df_pol_states: {df_pol_states.shape}, political topics: {len(political_topic_ids)}')\n"
])

# Now copy the new theory cells from the merged notebook
with open('the model.ipynb') as f:
    src_nb = json.load(f)
# new cells are at indices 51..57 (7 cells inserted)
theory_cells = src_nb['cells'][51:58]

# Patch the code_features cell to use state_partisan_lean instead of df_llm
# Find the stance_lean construction and replace
def patch_stance(cells):
    for c in cells:
        if c['cell_type'] != 'code': continue
        src = ''.join(c['source'])
        if 'df_llm_geo' in src:
            new_src = src.replace(
                "# --- LLM stance lean per state (net D - R share) ---\n"
                "df_llm_geo = df_llm.dropna(subset=['state_code']).copy()\n"
                "stance_pivot = (df_llm_geo.groupby(['state_code', 'stance_label']).size().unstack(fill_value=0))\n"
                "stance_total = stance_pivot.sum(axis=1).replace(0, 1)\n"
                "stance_lean = ((stance_pivot.get('supports Democratic party', 0) - stance_pivot.get('supports Republican party', 0))\n"
                "               / stance_total)\n",
                "# --- LLM stance lean per state (loaded from saved artifacts) ---\n"
                "stance_lean = state_partisan_lean.fillna(0)\n"
            )
            c['source'] = new_src.splitlines(keepends=True)
    return cells

theory_cells = patch_stance(theory_cells)

# Build standalone notebook
standalone = {
    "cells": [setup] + theory_cells,
    "metadata": src_nb.get('metadata', {}),
    "nbformat": src_nb.get('nbformat', 4),
    "nbformat_minor": src_nb.get('nbformat_minor', 5),
}
with open('theory_standalone.ipynb', 'w') as f:
    json.dump(standalone, f, indent=1)
print(f"Wrote theory_standalone.ipynb with {len(standalone['cells'])} cells")
