"""Builds the README figures in assets/ from scoreboard.json (and the real output of show.py)."""
import json
import subprocess
import sys
import numpy as np
from figstyle import *

P = theme("rose")
S = json.load(open("scoreboard.json"))
C, T = S["conditions"]["strict"], S["tests"]["strict"]
SEL = S["selected_direction"]
CONDS = [("baseline", "unmodified", 0.3, "fg"), ("ablated", "refusal direction removed", 1.0, "acc"),
         ("random_pooled", "random direction removed (3 pooled)", 0.3, "acc")]
SETS = [("harmful_all", "harmful\n(JBB + AdvBench)"), ("borderline_jbb_benign", "sensitive but\nharmless"),
        ("harmless_alpaca", "ordinary\n(Alpaca)")]


def fp(p):
    return f"{p:.1e}".replace("e-0", "e-") if p < 1e-3 else f"{p:.2f}"


# 0. banner: two clouds of activations, split by a single bright direction
rng = np.random.default_rng(23)
field = np.full((100, 400), 0.04)
for (cy, cx, n) in [(34, 250, 260), (68, 330, 260)]:
    for _ in range(n):
        y, x = int(rng.normal(cy, 9)), int(rng.normal(cx, 26))
        if 0 <= y < 99 and 150 <= x < 399:
            field[y:y + 2, x:x + 2] = rng.uniform(0.35, 0.8)
xs = np.arange(150, 400)
for x in xs:                                                   # the direction: a line from one cloud's mean to the other's
    t = (x - 150) / 250
    y = int(20 + 62 * t)
    field[max(y - 1, 0):y + 1, x] = 1.0
banner("banner.png", "refusal-direction",
       "removing one direction from a small model's residual stream switches off refusal.", field,
       meta="> replication · qwen2.5-0.5b · cpu only")

# 1. refusal rates
f, ax = fig(6.4, bottom_in=1.8)
x = np.arange(len(SETS)); w = 0.26
for k, (cond, lab, lv, ramp) in enumerate(CONDS):
    for i, (key, _) in enumerate(SETS):
        c = C[cond][key]
        v = 100 * c["refused"] / c["n"]
        xi = x[i] + (k - 1) * w
        dbar(ax, xi, max(v, 0.6), lv, ramp, width=w * 0.88)
        label(ax, xi, v + 2, f"{v:.0f}%", color=getattr(P, ramp), ha="center", va="bottom", fontsize=12)
ax.set_xticks(x); ax.set_xticklabels([s[1] for s in SETS])
ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75]); ax.set_ylabel("% of prompts refused")
ax.grid(axis="x", visible=False)
ax.legend(handles=[swatch(lv, ramp, lab) for _, lab, lv, ramp in CONDS], loc="upper center",
          bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12)
h = T["harmful_all"]
title(ax, "remove one direction and the model stops refusing",
      f"harmful refusals {C['baseline']['harmful_all']['refused']}/182 -> {C['ablated']['harmful_all']['refused']}/182 "
      f"(p = {fp(h['ablated_vs_baseline_p'])}) · random direction: p = {h['random_pooled_vs_baseline_p']:.2f}")
save(f, "refusal-rates.png")

# 2. per-layer separation
L = S["layer_separation"]
layers = [r["layer"] for r in L]
f, ax = fig(5.8, right=0.78)
ax.axvline(SEL["layer"], color=P.acc, lw=2, ls=(0, (2, 2)))
ax.plot(layers, [r["cohens_d_last_pos"] for r in L], color=P.dim, ls=(0, (2, 2)), lw=2.5, marker="s", ms=6)
ax.plot(layers, [r["cohens_d_selected_pos"] for r in L], color=P.acc, marker="s")
last = L[-1]
label(ax, layers[-1] + 0.6, last["cohens_d_selected_pos"], f"selected position\n({SEL['position_from_end']} from end)",
      color=P.acc, va="center", fontsize=12)
label(ax, layers[-1] + 0.6, last["cohens_d_last_pos"], "last token", color=P.dim, va="center", fontsize=12, fontweight="normal")
ymin = min(r["cohens_d_selected_pos"] for r in L if r["cohens_d_selected_pos"] == r["cohens_d_selected_pos"])
label(ax, SEL["layer"] + 0.3, ymin, f"chosen: layer {SEL['layer']}\nkl {SEL['kl_on_harmless_val']:.3f}",
      color=P.acc, va="bottom", fontsize=12, boxed=True)
ax.set_xlim(-0.5, layers[-1] + 0.5)
ax.set_xlabel("layer"); ax.set_ylabel("separation (cohen's d)")
title(ax, "harmful and harmless prompts separate sharply from layer 15 on",
      "difference of means along each candidate direction, on the training prompts")
save(f, "layer-separation.png")

# 3. the real terminal readout
out = subprocess.run([sys.executable, "-c", "import runpy; runpy.run_path('./show.py', run_name='__main__')"],
                     capture_output=True, text=True, encoding="utf8", check=True).stdout
term_capture("show.png", "python show.py", out)
