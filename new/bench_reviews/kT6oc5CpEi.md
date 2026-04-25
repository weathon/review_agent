Now let me search for calibration anchors to properly score the paper.Now I have enough to write the final review.

---

## Summary
BlackDAN proposes a black-box jailbreak attack framework for LLMs using the NSGA-II multi-objective evolutionary algorithm. Beyond the standard single-objective attack success rate (ASR), it adds semantic consistency between the jailbreak response and the original harmful query as a second fitness objective. The key claim is that multi-objective optimization produces higher attack success rates and more contextually relevant jailbreak outputs than existing single-objective approaches, evaluated on 9+ LLMs and multimodal LLMs.

---

## Strengths

- **Direct MO vs. SO advantage demonstrated in Figure 3**: The final row of the heatmap shows the multi-objective approach consistently achieves 93.1–100% ASR across all 9 target models, matching or exceeding each model's own single-objective self-attack rate (diagonal). This is a direct, controlled comparison that supports the core claim.

- **Strong ASR and GPT4-Metric improvements on open-source targets (Table 2)**: BlackDAN achieves 95.4% ASR / 93.8 GPT4-Metric on Llama2-7b and 97.5% / 96.0 on Vicuna-7b, substantially outperforming DeepInception (77.5%/31.2 and 92.7%/41.5). The GPT-4 judge metric being much higher than competing methods on these models provides credible corroboration.

- **Extension to multimodal LLMs (Figure 4)**: The paper tests on llava-v1.6-mistral-7b-hf and llava-v1.6-secure-7b-hf across three multimodal scenarios (SD, SD+Typo, Typo), with MO consistently outperforming SO—a useful scope extension beyond prior work.

- **Time efficiency (Table 1)**: ~2 minutes per sample vs. ~15 min (GCG) and ~12 min (AutoDAN), making black-box querying practical.

---

## Weaknesses

### Fatal
None.

### Major

- **The "stealthiness" claim directly contradicts the fitness function definition.** The paper's abstract, introduction, and conclusion repeatedly claim that BlackDAN "minimizes detectability" and optimizes for "stealthiness." However, the actual f₁ is `log P(unsafe token | R)` from Llama Guard 2 (§3.1), which *maximizes* the probability that the response is flagged as unsafe — the exact opposite of stealthiness. A stealthy attack should minimize detectability by safety classifiers, not maximize it. This is not a framing issue; there is simply no fitness term in the optimization that reduces detectability. The semantic consistency term (f₂) is real and useful, but it should be called what it is: relevance or contextual consistency, not stealthiness. This mislabeling propagates through the abstract, contributions, and conclusion.

- **On the most credible evaluation target, BlackDAN underperforms PAIR.** Table 2 shows that on GPT-4 with the GPT-4 Metric (the most reliable evaluation), BlackDAN scores 28.0% while PAIR scores 30.0%. The paper does not acknowledge this inversion and instead claims "consistently outperforms all other methods." The headline numbers for BlackDAN's superiority are driven by open-source, weakly-aligned models (Llama2-7b, Vicuna-7b) where the absolute performance gap to baselines is large; on the hardest and most safety-aligned target this advantage evaporates.

- **Table 2 does not isolate the multi-objective contribution.** The paper's central claim is that multi-objective > single-objective optimization. Table 2 compares BlackDAN against PAIR (iterative LLM-refinement), TAP (tree-of-attacks-with-pruning), and DeepInception (nested role-playing) — all fundamentally different attack paradigms. Outperforming them does not establish that *adding semantic consistency to an evolutionary attack framework* is what drives performance. Figure 3 partially provides this comparison, but it covers only ASR (not GPT-4 Metric) and excludes GPT-4 and GPT-3.5 as targets. The claim requires a single-objective BlackDAN vs. multi-objective BlackDAN comparison in the same setup as Table 2.

### Minor

- **Keyword-based ASR metric is over-permissive.** The primary metric (§4.1) counts a response as a success if it lacks rejection phrases like "I'm sorry." This is well-known to produce inflated numbers — a model can hedge without any rejection keyword. ASRs of 95%+ on safety-fine-tuned models are suspicious under this metric. The GPT-4 Metric partially addresses this but is underemphasized in the analysis given that it tells a different story on GPT-4.

- **GPT-2-XL should not be included as a target model.** GPT-2-XL has no safety fine-tuning and complies with virtually any instruction. Including it inflates cross-model transferability numbers in Figure 3 without contributing informative evidence about jailbreak effectiveness.

- **Rank Boundary Hypothesis visualization is near-circular.** Figures 5–6 show that Pareto-rank-1 solutions occupy distinct embedding regions from rank-N solutions. This is expected by construction: NSGA-II explicitly selected rank-1 solutions to score highest on f₁ and f₂. Without a null baseline (e.g., would random selection produce the same clustering?) or a demonstrated predictive use for this boundary (e.g., can it identify effective jailbreaks without running optimization?), this analysis does not provide independent validation.

- **The black-box label is partially questionable.** Computing f₁ requires running Llama Guard 2 locally and extracting log-probabilities of specific tokens — going beyond query-only access. The attack is black-box with respect to the *target* LLM, but requires model-internal access to a safety classifier. This should be disclosed as part of the threat model rather than simply labeled "Black-box."

### Trivial
- None significant beyond the above.

---

## Nice-to-Haves
- A single-objective BlackDAN ablation should be added to Table 2 alongside the main baselines to cleanly isolate the multi-objective contribution.
- An independent stealthiness evaluation (using a held-out classifier not involved in f₁) is needed to substantiate the detectability claims. Evaluating stealthiness against the same Llama Guard 2 used as f₁ is circular.
- Qualitative side-by-side comparison of single-objective vs. multi-objective responses with human ratings would make the "contextual relevance" advantage tangible.

---

## Removed Points
*These points were removed; treat with caution.*

- **GCG unfair transfer comparison** (Harsh Critic): While technically correct that GCG isn't designed for transfer, the comparison in Table 1 is primarily about time efficiency, not transfer performance. The paper doesn't specifically overclaim GCG's transfer ability is inferior by design.

- **GPT-4 Metric shows BlackDAN underperforms PAIR on GPT-4** (flagged as removed by Strength Finder): This is actually a KEPT weakness — the Strength Finder's claim that "GPT4-Metric scores follow the same pattern" conflicts with Table 2 which clearly shows BlackDAN (28.0%) < PAIR (30.0%) on GPT-4. The Strength Finder's claim for that specific data point is inaccurate.

- **Strength: "Customizable Pareto front for user preferences"** (Strength Finder): Too generic and not backed by concrete experimental evidence in the paper — the paper mentions this in passing but provides no user study or demonstration. Removed.

- **Strength: "Interpretable genetic operators"** (Strength Finder): Generic claim; synonym substitution and sentence swapping are standard GA operations, not a specific contribution of this paper. Removed.

- **Strength: "Strong transferability numbers"** (Strength Finder, citing Figure 3): The high transfer numbers in Figure 3 include GPT-2-XL (no safety fine-tuning) as both source and target, and the numbers across all models include models with limited alignment. These inflate the transferability claim. Kept as a minor concern rather than a verified strength.

---

## Novel Insights
The paper's most genuinely novel observation is that optimizing jailbreaks on a single objective (ASR or harmfulness alone) can produce responses that are on-topic in only a surface sense — the jailbreak prompt forces a compliance token but the response wanders away from the actual harmful query. Adding semantic consistency between Q and R as a Pareto objective creates selection pressure toward responses that are simultaneously harmful *and* on-topic. This is a real insight that could benefit future red-teaming frameworks. However, the paper frames this under "stealthiness" — a different and unsupported claim — obscuring the actual contribution.

---

## Suggestions
1. Rename the second objective as "semantic relevance" or "contextual consistency" throughout, and remove all "stealthiness" framing, since f₁ maximizes — not minimizes — unsafe detection probability.
2. Add a dedicated Table 2 row for single-objective BlackDAN (NSGA-I, or BlackDAN with only f₁) to directly validate the multi-objective contribution.
3. Acknowledge that on GPT-4 with GPT-4 Metric, PAIR outperforms BlackDAN, and discuss why — this would strengthen credibility rather than weaken the paper.
4. Remove GPT-2-XL from Table 2 / Figure 3 to avoid inflating results.

---

## Score and Decision

**Calibration:**

| Paper | Path | Avg Score | Notes vs. BlackDAN |
|---|---|---|---|
| KDA (LLM jailbreak, low anchor) | UWuTZYPSxJ.md | 2.5 | Rejected; fundamental claim errors, evaluation methodology flawed. BlackDAN is stronger — real SO vs. MO evidence. |
| Multi-round contextual jailbreak | w0b7fCX2nN.md | 3.75 | Withdrawn; limited novelty, missing key ablations. BlackDAN is comparable but somewhat stronger. |
| DAG-Jailbreak | xQIJ5fjc7q.md | 5.5 | Rejected; comprehensive but has clarity issues and contribution concerns. BlackDAN is similar in quality with more specific weaknesses. |
| SoC MAB jailbreak | jCDF7G3LpF.md | 6.25 | Accepted; clear theoretical analysis + strong results. BlackDAN is weaker — no clean ablation, overclaims stealthiness. |
| AutoDAN-Turbo | bhK7U37VW8.md | 7.17 | Accepted (Spotlight); clean methodology, strong results, honest claims. BlackDAN is clearly weaker in rigor and honesty. |

**Assessment:** BlackDAN sits between the 3.75 (rejected with insufficient novelty/ablations) and the 5.5 (rejected DAG-Jailbreak) anchors. The stealthiness contradiction is a non-trivial honesty issue: the claim in the abstract conflicts with the actual optimization objective defined two sections later. The GPT-4 evaluation reveals that the paper's strongest competitor (PAIR) actually outperforms BlackDAN on the most credible target, which the paper ignores. The ablation structure (no SO-BlackDAN row in Table 2) leaves the central claim incompletely supported. On the positive side, Figure 3/4 do provide genuine evidence for MO > SO, and the contextual consistency framing has value.

The paper is below the acceptance bar as submitted. The core ideas are worth developing, but the current form has misleading claims and an incomplete evaluation. Score: **4.0**.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>