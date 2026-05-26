"""Merge executed theory cells into the full executed notebook + export HTML."""
import json, shutil

with open('the model.executed.ipynb') as f:
    full = json.load(f)
with open('theory_standalone.executed.ipynb') as f:
    theory = json.load(f)

# theory cells: index 0 = setup (skip), 1..7 = our new section
# In the merged "the model.ipynb" they were inserted at index 51 (after §6.6 Prediction Map at 50, before §7 markdown at originally 51).
# But "the model.executed.ipynb" was executed BEFORE the new cells were inserted, so it has the original 75-cell layout.
# So we insert at index 51 of the executed notebook.

new_cells = theory['cells'][1:]  # skip setup
INSERT_AT = 51

# Strip "setup-only" warnings from outputs if any — keep them as-is
merged_cells = full['cells'][:INSERT_AT] + new_cells + full['cells'][INSERT_AT:]

# Renumber section headers downstream: existing §7..§10 are unchanged in number (we used 6.6)
# The current model.executed.ipynb has §7 Bias Audit at index 51, §9 Summary later, §10 Export later. No renumbering needed since we used 6.6.

full['cells'] = merged_cells
with open('the model.final.ipynb', 'w') as f:
    json.dump(full, f, indent=1)
print(f"Merged notebook: {len(merged_cells)} cells")
