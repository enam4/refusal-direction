# build scoreboard.json from runs/ (eval2 labels = reported numbers; v1 labels kept for transparency)
import json
from math import comb
R = json.load(open("runs/eval2_labels.json"))
sel = json.load(open("runs/selection.json"))["selected"]
sep = json.load(open("runs/layer_separation.json"))
sizes = json.load(open("runs/split_sizes.json"))

def fisher2(a, n1, b, n2):
    """two-sided Fisher exact test on [[a, n1-a], [b, n2-b]]"""
    K = a + b; N = n1 + n2; tot = comb(N, K)
    p0 = comb(n1, a) * comb(n2, K - a)
    return min(1.0, sum(comb(n1, k) * comb(n2, K - k) for k in range(max(0, K - n2), min(n1, K) + 1)
                        if comb(n1, k) * comb(n2, K - k) <= p0 * (1 + 1e-9)) / tot)

def cnt(cond, sets, judge="strict"):
    rows = [r for s in sets for r in R[cond][s]]
    return {"refused": sum(r[judge] == "refused" for r in rows), "incoherent": sum(r[judge] == "incoherent" for r in rows), "n": len(rows)}

H = ["test_harmful_jbb", "test_harmful_advbench_heldout"]
S = {"model": "Qwen2.5-0.5B-Instruct (bf16)", "paper": "arxiv.org/abs/2406.11717",
     "split_sizes": sizes, "advbench_heldout_n": 100,
     "selected_direction": {"layer": sel["layer"], "position_from_end": sel["pos"], "kl_on_harmless_val": sel["kl_harmless"]},
     "conditions": {}, "tests": {}}
for judge in ["strict", "loose"]:
    S["conditions"][judge] = {}
    for c in ["baseline", "ablated", "random0", "random1", "random2"]:
        S["conditions"][judge][c] = {
            "harmful_jbb": cnt(c, ["test_harmful_jbb"], judge), "harmful_advbench_heldout": cnt(c, ["test_harmful_advbench_heldout"], judge),
            "harmful_all": cnt(c, H, judge), "harmless_alpaca": cnt(c, ["test_harmless_alpaca"], judge),
            "borderline_jbb_benign": cnt(c, ["test_borderline_jbb_benign"], judge)}
    rnd = {k: {"refused": sum(S["conditions"][judge][f"random{i}"][k]["refused"] for i in range(3)),
               "n": sum(S["conditions"][judge][f"random{i}"][k]["n"] for i in range(3))} for k in S["conditions"][judge]["baseline"]}
    S["conditions"][judge]["random_pooled"] = rnd
    b = S["conditions"][judge]["baseline"]; a = S["conditions"][judge]["ablated"]
    S["tests"][judge] = {k: {"ablated_vs_baseline_p": fisher2(a[k]["refused"], a[k]["n"], b[k]["refused"], b[k]["n"]),
                             "random_pooled_vs_baseline_p": fisher2(rnd[k]["refused"], rnd[k]["n"], b[k]["refused"], b[k]["n"])} for k in b}
best = sel["pos"] + 5
S["layer_separation"] = [{"layer": r["layer"], "cohens_d_selected_pos": r[f"pos{sel['pos']}"]["cohens_d"], "auroc_selected_pos": r[f"pos{sel['pos']}"]["auroc"],
                          "cohens_d_last_pos": r["pos-1"]["cohens_d"]} for r in sep]
json.dump(S, open("scoreboard.json", "w"), indent=1)
print(json.dumps(S["conditions"]["strict"], indent=0)[:1500]); print(json.dumps(S["tests"], indent=0))
