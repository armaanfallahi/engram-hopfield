# -*- coding: utf-8 -*-
from pathlib import Path

import nbformat
from nbformat.validator import normalize

p = Path("notebooks/04_fear_extinction.ipynb")
nb = nbformat.read(p, 4)

RATIONALE = (
    "**Part 1 rationale.** Increasing CUE_FRAC from 3% to 5% means each cue "
    "component activates 100 neurons instead of 60. With P_OLD=25 the recurrent "
    "background is weaker. Together these increase the cue-to-recurrent signal "
    "ratio so that CS, US, and context neurons reliably win the top-k competition.\n"
)
nb["cells"].insert(
    2,
    {"cell_type": "markdown", "metadata": {}, "source": [RATIONALE]},
)


def src_join(c):
    return "".join(c.get("source", []))


def set_src(c, text):
    if not text.endswith("\n"):
        text += "\n"
    c["source"] = [text]


for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = src_join(c)
    if 'f"Fear-{name}:' in s:
        set_src(c, s.replace('f"Fear-{name}:', 'f"Fear vs {name}:'))

for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = src_join(c)
    if "Part 7 complete: multi-trial retrieval" in s and "Which memory wins" not in s:
        old = "    )\n\nfig, ax = plt.subplots(figsize=(12, 5))"
        new = (
            "    )\n\n"
            'print("\\nWhich memory wins (FEAR / EXT / TIE) — by mean overlap:")\n'
            "for label in test_labels:\n"
            '    fm = np.mean(all_results[label]["fear"])\n'
            '    em = np.mean(all_results[label]["ext"])\n'
            '    print(f"  {label}: {memory_winner(fm, em)}")\n\n'
            "fig, ax = plt.subplots(figsize=(12, 5))"
        )
        if old in s:
            set_src(c, s.replace(old, new))

for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = src_join(c)
    if s.strip().startswith('print("=== Summary'):
        set_src(
            c,
            """print("=== Summary (Parts 1–8) ===")
print(f"N={N}, K={K}, P_OLD={P_OLD}, N_CUE={N_CUE}, BETA={BETA}, SIGMA_THETA={SIGMA_THETA}")
print(f"Cue-to-recurrent ratio (Part 3): {ratio:.3f}")
print(f"Phi total: {np.sum(fear_engram * ext_engram) / K:.3f}")
print(
    "Retrieval winners: "
    + ", ".join(
        f"{lab}={memory_winner(np.mean(all_results[lab]['fear']), np.mean(all_results[lab]['ext']))}"
        for lab in test_labels
    )
)
print("Part 8 complete: summary")
""",
        )

normalize(nb)
nbformat.write(nb, p)
print("Updated", p)
