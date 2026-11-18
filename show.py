import json, sys
S = json.load(open(__file__.rsplit("/", 1)[0] + "/scoreboard.json"))
C = S["conditions"]["strict"]; T = S["tests"]["strict"]
B, R0, G, Rd, Y, Cy, Dm = "\x1b[1m", "\x1b[0m", "\x1b[32m", "\x1b[31m", "\x1b[33m", "\x1b[36m", "\x1b[90m"
print(f"{B}refusal direction{R0} \u00b7 qwen2.5 0.5b instruct \u00b7 difference in means \u00b7 local cpu")
print(f"{Dm}direction from layer {S['selected_direction']['layer']}, removed at every layer \u00b7 greedy \u00b7 keyword judge (first sentence){R0}")
print()
print(f"{B}{'condition':<26}{'harmful refused':>18}{'alpaca refused':>17}{'borderline refused':>21}{R0}")
for lab, k, col in [("unmodified", "baseline", G), ("refusal dir removed", "ablated", Rd), ("random dir removed (x3)", "random_pooled", Y)]:
    c = C[k]
    print(f"{lab:<26}{col}{c['harmful_all']['refused']:>8} / {c['harmful_all']['n']:<7}{R0}{c['harmless_alpaca']['refused']:>8} / {c['harmless_alpaca']['n']:<6}{c['borderline_jbb_benign']['refused']:>10} / {c['borderline_jbb_benign']['n']}")
print()
p = T["harmful_all"]
f = lambda x: "<0.0001" if x < 1e-4 else f"{x:.3f}"
print(f"{Cy}fisher, harmful refused:{R0} removed vs unmodified p = {f(p['ablated_vs_baseline_p'])}   random vs unmodified p = {f(p['random_pooled_vs_baseline_p'])}")
print(f"{Dm}held-out harmful = 82 jailbreakbench (non-advbench) + 100 unseen advbench. no completions stored or shown.{R0}")
