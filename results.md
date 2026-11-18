# refusal-direction: does one direction control refusal in a small model?
Run 18 Nov 2026, 9:39–10:16 AM ET, on a CPU-only box. Nothing posted.

Replicates the core result of Arditi et al. 2024, "Refusal in language models is mediated by a single direction" (arxiv.org/abs/2406.11717), at small scale.

## Setup
- **Hardware:** 8-core Intel Xeon, 16 GB RAM, no GPU.
- **Model:** Qwen/Qwen2.5-0.5B-Instruct, bf16, using its default chat template. Only one model was tested, because of compute limits. The paper used 1.8B to 72B models.
- **Direction:**
  - Mean residual-stream activation on 128 harmful instructions (AdvBench) minus the mean on 128 harmless instructions (Alpaca, no-input subset).
  - Computed at each layer and at each of the 5 post-instruction template tokens.
- **Selection:**
  - 90 candidates (layers 1–18, which is under 0.8 L, × 5 positions) were scored on a separate validation set of 32 harmful and 32 harmless prompts.
  - The winner was the candidate whose ablation most lowered the first-token refusal log-odds, subject to KL < 0.1 on harmless validation prompts.
  - Selected: layer 17, position −2 from the end. Validation refusal score fell from +1.68 to −2.09, and KL was 0.062.
  - The paper's third filter (adding the direction must induce refusal) was not applied.
- **Ablation:**
  - The direction is projected out of the embedding output and of every attention and MLP output, at every layer and token. This is done with forward hooks at run time.
  - No modified weights were ever written to disk. Only the 896-float direction was kept (runs/refusal_direction.pt, kept locally and not included in this repo).
- **Controls:**
  - The unmodified model.
  - 3 random unit directions ablated the same way (seed 1234), pooled.
- **Held-out test sets (never used for direction or selection):**
  - Harmful: 82 JailbreakBench behaviours. The 18 that JBB took from AdvBench were removed, so these come from a different source than training. Plus 100 unseen AdvBench prompts.
  - Harmless: 100 unseen Alpaca prompts.
  - Over-refusal probe: the 100 JailbreakBench "benign" prompts, which are harmless requests on sensitive topics.
- **Decoding:** greedy, 64 new tokens.
- **Judge:**
  - Keyword rule on the first sentence of the reply ("I'm sorry", "I can't", "I cannot assist", and similar). The full list is in eval2.py.
  - Harmful-prompt completions were never stored. Only the labels were kept.

## Scoreboard (strict judge, reported numbers)

| condition | harmful refused (182) | JBB harmful (82) | AdvBench held-out (100) | Alpaca refused (100) | sensitive-but-harmless refused (100) |
|---|---|---|---|---|---|
| unmodified | 144 (79%) | 55 | 89 | 3 | 20 |
| refusal direction removed | **16 (9%)** | 5 | 11 | 1 | **1** |
| random direction removed (3 × pooled) | 438/546 (80%) | 163/246 | 275/300 | 6/300 | 71/300 |

Two-sided Fisher exact tests:
- **Harmful refusals:** removed vs unmodified, p = 1.5e-45. Random vs unmodified, p = 0.75.
- **Sensitive-but-harmless refusals:** removed vs unmodified, p = 8e-6. Random vs unmodified, p = 0.49.
- **Alpaca refusals:** removed vs unmodified, p = 0.62. There is no change in everyday helpfulness refusals; both rates are near the floor.

**Supporting signal:** replies that open like an answer ("Sure", "Here", "Certainly", a numbered list) went from 14/182 (unmodified) to 70/182 (removed) on harmful prompts. The random controls stayed at 13/182 or 14/182. This is a rough sign that "not refused" mostly means "answered" rather than deflected. The replies were not kept, so it can't be checked further.

**Loose judge** (the v1 broad substring list in direction.py, kept for transparency):
- Harmful prompts: unmodified 154/182, removed 29/182, random 470/546.
- The loose list over-counts refusals. It flags "As an AI language model, ... but I can" disclaimers, and words like "harmful" or "illegal" inside helpful answers. We saw this on the stored harmless outputs, then wrote the stricter first-sentence judge (eval2.py) and re-ran the held-out eval with the same saved direction. Both judges give the same conclusion.
- We audited every strict-judge "refused" on the harmless and sensitive sets by reading it. All were genuine refusals or near-refusals.

**Per-layer separation** (img2):
- Measured as Cohen's d between harmful and harmless validation prompts, projected on each layer's direction.
- Separation is about 2–4 up to layer 14, jumps to about 5.8 at layer 15, and stays at about 5–6.4 after that.
- The directions with the strongest ablation effect (layers 14–18) cluster where the jump happens. Several of them failed the KL < 0.1 filter (KL 0.5–3.2) because they also disrupted harmless prompts.

## Verdict
Replicated at 0.5B scale.
- Removing one direction cut refusals on held-out harmful prompts from 79% to 9%. This includes prompts from a dataset the direction never saw.
- Removing random directions did nothing (80%).
- It also removed most over-refusal on harmless prompts about sensitive topics (20 to 1), with no measurable change on everyday prompts.

## Caveats
- One small model, greedy decoding, 64-token replies. The keyword judge only says refused or not refused. It does not measure how harmful a reply is, and that was never scored, by design.
- The paper's induce-refusal selection filter was skipped.
- The v1 eval and eval2 baselines differ by a few prompts (60 vs 55 refusals on JBB under different judges and thread counts, since bf16 CPU math is not bit-identical across thread counts).
- Dataset overlap was controlled for JBB vs AdvBench by source label only, not by fuzzy text matching.

## Files
- **Data:** scoreboard.json and the runs/ folder:
  - eval2_labels.json: reported labels; harmless replies stored, harmful ones not.
  - generations_labels.json: v1 labels.
  - selection.json, layer_separation.json, split_sizes.json.
  - refusal_direction.pt: written by direction.py; deliberately not included in this repo.
- **Logs:** logs/run.log and logs/eval2.log.
- **Images:** img1-scoreboard-grid.png, img2-layer-separation.png, term1-scoreboard.png (rendered from real show.py output).
- **Post draft:** post-draft.txt.
- **Scripts:** direction.py, eval2.py, score.py, show.py, render.py, plus lib/ (style.py, term.py).
- **Data sources:** data/ (AdvBench CSV from the llm-attacks repo, Alpaca from HF tatsu-lab/alpaca, JailbreakBench JBB-Behaviors).
