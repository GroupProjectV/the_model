"""Swap BART zero-shot for m-newhauser/distilbert-political-tweets in the source notebook."""
import json, shutil

shutil.copy('the model.ipynb', 'the model.ipynb.bak3')

with open('the model.ipynb') as f:
    nb = json.load(f)

# Locate the BART cells by content
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    src = ''.join(c['source'])
    if 'facebook/bart-large-mnli' in src:
        bart_setup_idx = i
        print(f"BART setup cell at index {i}")
    if 'candidate_labels=candidate_labels' in src and 'batch_size' in src:
        bart_run_idx = i
        print(f"BART classify cell at index {i}")

# New cell 34: load domain-specific classifier
new_setup = [
    "# --- Domain-specific partisan tweet classifier ---\n",
    "# Swap generic BART zero-shot (which hedged at 63% \"neutral\") for a model trained on political tweets.\n",
    "# m-newhauser/distilbert-political-tweets is DistilBERT fine-tuned on U.S. congressional tweets\n",
    "# to classify Democrat vs Republican stance.\n",
    "from transformers import pipeline\n",
    "import torch\n",
    "\n",
    "DEVICE = 0 if torch.cuda.is_available() else -1\n",
    "classifier = pipeline(\n",
    "    'text-classification',\n",
    "    model='m-newhauser/distilbert-political-tweets',\n",
    "    device=DEVICE,\n",
    "    top_k=None,                  # return both class probabilities\n",
    "    truncation=True,\n",
    "    max_length=128,\n",
    ")\n",
    "\n",
    "# We keep candidate_labels in the same naming the downstream code expects\n",
    "# so we can drop the new classifier in without changing the aggregation logic.\n",
    "candidate_labels = ['supports Democratic party', 'supports Republican party', 'neutral or nonpartisan']\n",
    "\n",
    "# Confidence threshold below which we call a tweet 'neutral or nonpartisan'\n",
    "# (the model itself is binary D/R; we derive neutrality from low-margin predictions)\n",
    "NEUTRAL_BAND = 0.15  # if |p(D) - p(R)| < this, call it neutral\n",
    "\n",
    "print('Loaded m-newhauser/distilbert-political-tweets')\n",
    "print(f'Device: {\"cuda\" if DEVICE == 0 else \"cpu\"}')\n",
]

# New cell 35: classify
new_run = [
    "# --- Run domain-specific classifier on the political-tweet sample ---\n",
    "from tqdm import tqdm\n",
    "\n",
    "llm_sample_size = min(3000, len(df_political))\n",
    "df_llm = df_political.sample(n=llm_sample_size, random_state=RANDOM_STATE).copy()\n",
    "print(f'Classifying {len(df_llm)} political tweets with DistilBERT (CPU expected ~5-10 min)...')\n",
    "\n",
    "texts = df_llm['tweet_clean'].astype(str).tolist()\n",
    "stance_results = []\n",
    "batch_size = 32\n",
    "for i in tqdm(range(0, len(texts), batch_size), desc='Stance classification'):\n",
    "    batch = texts[i:i+batch_size]\n",
    "    out = classifier(batch, batch_size=batch_size)\n",
    "    if not isinstance(out, list): out = [out]\n",
    "    for r in out:\n",
    "        # r is a list of dicts [{'label':'Democrat','score':...}, {'label':'Republican','score':...}]\n",
    "        d = {x['label']: x['score'] for x in r}\n",
    "        p_d, p_r = d.get('Democrat', 0.0), d.get('Republican', 0.0)\n",
    "        margin = p_d - p_r\n",
    "        if abs(margin) < NEUTRAL_BAND:\n",
    "            label = 'neutral or nonpartisan'\n",
    "            score = max(p_d, p_r)\n",
    "        elif p_d > p_r:\n",
    "            label = 'supports Democratic party'\n",
    "            score = p_d\n",
    "        else:\n",
    "            label = 'supports Republican party'\n",
    "            score = p_r\n",
    "        stance_results.append({'stance_label': label, 'stance_score': float(score),\n",
    "                                'p_dem': float(p_d), 'p_rep': float(p_r), 'partisan_margin': float(margin)})\n",
    "\n",
    "df_llm['stance_label']    = [r['stance_label']    for r in stance_results]\n",
    "df_llm['stance_score']    = [r['stance_score']    for r in stance_results]\n",
    "df_llm['p_dem']           = [r['p_dem']           for r in stance_results]\n",
    "df_llm['p_rep']           = [r['p_rep']           for r in stance_results]\n",
    "df_llm['partisan_margin'] = [r['partisan_margin'] for r in stance_results]\n",
    "\n",
    "print(f'\\nStance distribution:')\n",
    "print(df_llm['stance_label'].value_counts())\n",
    "print(f'\\nMean confidence (winning class): {df_llm[\"stance_score\"].mean():.3f}')\n",
    "print(f'Mean |partisan_margin|:          {df_llm[\"partisan_margin\"].abs().mean():.3f}')\n",
    "print(f'(Compare to BART zero-shot mean confidence ~0.645 with 63% labeled neutral)')\n",
]

nb['cells'][bart_setup_idx]['source'] = new_setup
nb['cells'][bart_run_idx]['source']   = new_run
nb['cells'][bart_setup_idx]['outputs'] = []
nb['cells'][bart_run_idx]['outputs']   = []
nb['cells'][bart_setup_idx]['execution_count'] = None
nb['cells'][bart_run_idx]['execution_count']   = None

with open('the model.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Swapped classifier in cells {bart_setup_idx} and {bart_run_idx}")
