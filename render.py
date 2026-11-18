import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from style import *
S = json.load(open("scoreboard.json"))
C = S["conditions"]["strict"]; T = S["tests"]["strict"]
TITLE = "refusal-direction \u2014 qwen 2.5 0.5b instruct, local cpu"
def rt(d, xr, y, t):
    f = font(15, True); d.text((xr - d.textlength(t, font=f), y), t, font=f, fill=FG)
def pct(x): return f"{round(100 * x['refused'] / x['n'])}%"
# ---------- card grid
W, H = 1200, 1060
im, d = canvas(W, H, TITLE)
d.text((30, 64), "one direction inside the model controls refusal", font=font(28, True), fill=FG)
d.text((30, 104), f"remove it and the model stops refusing. remove a random direction and nothing happens.", font=font(17), fill=DIM)
cards = [("unmodified model", "no change", "baseline", GRN),
         ("refusal direction removed", f"projected out at every layer (found at layer {S['selected_direction']['layer']})", "ablated", RED),
         ("random direction removed", "control: 3 random directions, same method", "random_pooled", YEL)]
CW, CH = 365, 530; xs = [30, 418, 806]; y0 = 150
for (name, sub, key, col), x in zip(cards, xs):
    c = C[key]
    d.rounded_rectangle([x, y0, x + CW, y0 + CH], 14, fill=PAN, outline=col, width=2)
    d.text((x + 20, y0 + 16), name, font=font(21, True), fill=col)
    import textwrap
    for i, ln in enumerate(textwrap.wrap(sub, 38)[:2]): d.text((x + 20, y0 + 48 + 20 * i), ln, font=font(14), fill=DIM)
    h = c["harmful_all"]
    d.text((x + 20, y0 + 100), "refuses harmful requests", font=font(16), fill=FG)
    d.text((x + 20, y0 + 124), pct(h), font=font(68, True), fill=col)
    d.text((x + 20, y0 + 210), f"{h['refused']} / {h['n']} held-out prompts", font=font(16), fill=DIM)
    bar(d, x + 20, y0 + 240, CW - 40, 18, h["refused"] / h["n"], col)
    yy = y0 + 290
    for lab, k in [("jailbreakbench (other source)", "harmful_jbb"), ("advbench (held out)", "harmful_advbench_heldout")]:
        d.text((x + 20, yy), lab, font=font(14), fill=DIM); rt(d, x + CW - 20, yy, f"{C[key][k]['refused']}/{C[key][k]['n']}"); yy += 26
    yy += 18; d.line([x + 20, yy, x + CW - 20, yy], fill=BOR); yy += 16
    d.text((x + 20, yy), "over-refusal on harmless prompts", font=font(15), fill=FG); yy += 32
    for lab, k in [("everyday (alpaca)", "harmless_alpaca"), ("sensitive topic, harmless", "borderline_jbb_benign")]:
        d.text((x + 20, yy), lab, font=font(14), fill=DIM); rt(d, x + CW - 20, yy, f"{C[key][k]['refused']}/{C[key][k]['n']}")
        bar(d, x + 20, yy + 22, CW - 40, 10, C[key][k]["refused"] / C[key][k]["n"], MAG); yy += 52
p_ab = T["harmful_all"]["ablated_vs_baseline_p"]; p_rn = T["harmful_all"]["random_pooled_vs_baseline_p"]
def fp(p): return "< 0.0001" if p < 1e-4 else f"= {p:.2f}"
d.text((30, 705), f"removed vs unmodified: fisher p {fp(p_ab)}.   random vs unmodified: p {fp(p_rn)}.", font=font(18), fill=YEL)
lines = ["direction = mean activation on 128 harmful (advbench) minus 128 harmless (alpaca) instructions.",
         "picked on a separate 32+32 validation set. scored on prompts it never saw, incl. a different dataset.",
         "refused / complied judged by a keyword rule on the reply's first sentence. no replies are shown.",
         "greedy decoding, 64 new tokens. one small model; weights were never saved, only the direction."]
for i, ln in enumerate(lines): d.text((30, 747 + 26 * i), ln, font=font(15), fill=DIM)
d.text((30, 865), "after Arditi et al. 2024, 'Refusal in language models is mediated by a single direction'", font=font(15), fill=DIM)
save(im.crop((0, 0, W, 905)), "img1-scoreboard-grid.png"); print("grid ok")

# ---------- per-layer separation chart
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ls = S["layer_separation"]; L = [r["layer"] for r in ls]
fig, ax = plt.subplots(figsize=(12, 6.2), dpi=100)
fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#161b22")
ax.plot(L, [r["cohens_d_selected_pos"] for r in ls], color="#f85149", lw=3, marker="o", label=f"selected token position ({S['selected_direction']['position_from_end']} from end)")
ax.plot(L, [r["cohens_d_last_pos"] for r in ls], color="#58a6ff", lw=2, marker="o", alpha=0.8, label="last token position")
sl = S["selected_direction"]["layer"]
ax.axvline(sl, color="#d29922", ls="--", lw=1.5); ax.text(sl + 0.3, ax.get_ylim()[1] * 0.92, f"direction used\n(layer {sl})", color="#d29922", fontsize=12, family="DejaVu Sans Mono")
ax.set_xlabel("layer (0 = embeddings)", color="#c9d1d9", fontsize=13, family="DejaVu Sans Mono")
ax.set_ylabel("separation, harmful vs harmless (Cohen's d)", color="#c9d1d9", fontsize=13, family="DejaVu Sans Mono")
ax.set_title("where the model tells harmful from harmless apart\n32 + 32 validation prompts projected on each layer's direction", color="#c9d1d9", fontsize=15, family="DejaVu Sans Mono", loc="left")
ax.tick_params(colors="#7d8590"); [s.set_color("#30363d") for s in ax.spines.values()]
ax.grid(color="#30363d", lw=0.6); leg = ax.legend(facecolor="#161b22", edgecolor="#30363d", fontsize=12, prop={"family": "DejaVu Sans Mono"})
[t.set_color("#c9d1d9") for t in leg.get_texts()]
fig.tight_layout(); fig.savefig("img2-layer-separation.png", facecolor=fig.get_facecolor(), metadata={"Software": None}); print("chart ok")
