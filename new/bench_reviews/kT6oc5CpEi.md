Now I have a solid understanding of the calibration landscape. Let me synthesize my final review.

## Summary

BlackDAN proposes a black-box multi-objective jailbreak framework for LLMs and MLLMs, using NSGA-II to simultaneously optimize for harmfulness (unsafe token probability via llama_guard_2) and semantic consistency (cosine similarity via all-MiniLM-L6-v2). The paper claims this multi-objective approach yields jailbreaks that are both effective and contextually relevant, with interpretability via Pareto fronts and a "Rank Boundary Hypothesis" about embedding space structure.

## Strengths

- **Conceptually motivated multi-objective framing**: The shift from single-objective ASR optimization to jointly optimizing harmfulness and semantic consistency addresses a real gap. Single-objective attacks often produce irrelevant or off-topic outputs; ensuring semantic relevance makes attacks more realistic threat models. This conceptual contribution is valuable even if the empirical validation has gaps.
- **Principled optimization framework**: Using NSGA-II with Pareto-dominance and crowding distance provides a transparent, interpretable optimization process compared to gradient-based or heuristic end-to-end methods. The Pareto front naturally offers user-controllable trade-offs between objectives.
- **Broad empirical scope**: The paper evaluates across 9+ open-source LLMs and 2 multimodal LLMs on standard benchmarks (AdvBench, MM-SafetyBench). Multi-objective optimization consistently outperforms single-objective across all tested models (Fig 3-4, Table 2).
- **Embedding space analysis**: Figures 5-6 attempt to connect Pareto rank structure with embedding geometry, providing more interpretability analysis than typical jailbreak papers.

## Weaknesses

### Major:

- **Multi-objective gains are not directly evaluated**: The core claimed novelty is multi-objective optimization that simultaneously improves harmfulness, semantic consistency, and stealthiness. However, all headline quantitative results (Table 1, Table 2, Fig 3-4) report only ASR or GPT-4 harm scores. No table or figure reports the semantic consistency objective ($f_2$) or any stealthiness metric for BlackDAN vs. baselines. The GPT-4 metric in Table 2 judges harmfulness, not relevance. Without measuring whether BlackDAN dominates baselines on *both* objectives, the central claim is not substantiated—the method could simply be optimizing harmfulness alone and the experiments would look identical. This is the paper's most significant gap.

- **Stealthiness is claimed but never measured**: The abstract, introduction, and conclusion all claim BlackDAN improves "stealthiness" and "minimizing detectability," but no stealthiness metric (e.g., perplexity-based detection, classifier detection rate) is reported anywhere. The keyword-based ASR metric's absence of refusal keywords is a very weak proxy for stealth, as many non-harmful responses also avoid those phrases.

- **llama_guard_2 fitness function is underspecified**: Section 3.1 defines $f_1(P,Q) = \log P(\text{unsafe token} \mid R)$ using llama_guard_2, but llama_guard_2 is a classifier that outputs discrete safe/unsafe labels with reasoning, not a standard language model providing token-level log-probabilities. The paper does not explain: (1) what exactly constitutes an "unsafe token," (2) how llama_guard_2 is coerced into providing log-probabilities rather than hard labels, or (3) the discrepancy between §3.1 (conditioned only on R) and Fig 2 (conditioned on Q+R). This ambiguity in a core fitness function undermines methodological clarity.

- **No evaluation against defense mechanisms**: The paper does not test BlackDAN against any defenses (perplexity filtering, safety classifiers, SmoothLLM, guardrail models). If generated prompts are easily filtered, the practical significance of high ASR on undefended models is limited. This is a standard expectation in the jailbreak literature—AutoDAN (accepted as poster) specifically evaluated against perplexity-based defenses.

### Minor:

- **Keyword-based ASR is a weak metric**: The ASR metric (§4.1) counts success as the absence of rejection keywords like "I'm sorry." This is known to be brittle and easily gamed—responses avoiding these keywords may still be vacuous or non-harmful. The large gap between ASR and GPT4-Metric on GPT-4 (71.4% vs 28.0%) further suggests this metric substantially overestimates attack success.

- **No statistical analysis**: Evolutionary algorithms are inherently stochastic, yet no standard deviations or confidence intervals are reported for any result. Without this, it is impossible to determine whether reported improvements are statistically significant.

- **Missing key hyperparameters**: Population size, number of generations, crossover rate, mutation rate, and selection criteria are deferred to the appendix rather than summarized in the main text, making it difficult to assess query cost, convergence, or feasibility.

- **Query efficiency not reported**: While Table 1 reports "~2 min per sample," the number of target-model queries required per sample is not disclosed. In black-box settings with API rate limits and costs, query count is the primary constraint, not wall-clock time. A population-based evolutionary approach could easily require hundreds of queries per prompt.

- **"Rank Boundary Hypothesis" is not rigorously tested**: The paper claims each Pareto rank has "distinct boundaries in the embedding space" enabling "better differentiation between toxic and non-toxic prompts." However, Figures 5-6 only show post-hoc visualizations of embedding clusters by rank. There is no quantitative test (e.g., classifier accuracy on rank prediction, silhouette scores, correlation between geometric separation and attack transfer success). The jump from "ranks cluster in embedding space" to "better differentiation of toxic/non-toxic prompts" is speculative.

- **Simplistic genetic operators**: Mutation replaces single words with WordNet synonyms (§3.3), and crossover swaps sentences between parents. For jailbreak prompts relying on highly specific patterns, naïve synonym substitution often destroys effectiveness. No ablation studies validate whether mutation actually contributes to optimization or is just noise.

### Trivial:

- **GPT-4 metric underperforms PAIR on GPT-4 model**: In Table 2, BlackDAN's GPT4-Metric on GPT-4 (28.0) is slightly below PAIR's (30.0). This counter-result on the strongest commercial model is not discussed.

## Nice-to-Haves

- Pareto front visualization (scatter plot of harmfulness vs. semantic consistency) showing actual trade-off curves
- Ablation comparing NSGA-II (MO) against single-objective optimization + post-hoc semantic filtering, to isolate whether MO optimization itself contributes beyond simple reranking
- Side-by-side qualitative output examples comparing SO vs. MO responses for the same query
- Evaluation on more robust closed-source models (Claude, Gemini)

## Removed Points

*These points were flagged but removed—treat with caution:*

- **"Paper does not test on truly closed-source models"** — The paper does include GPT-3.5-turbo and GPT-4 in Table 2, so it partially addresses this. However, the evaluation is limited to only these two API models, which warrants keeping as a minor/nice-to-have concern rather than a major one.
- **"Reproducibility concerns about missing code or training logs"** — This is a nitpick about artifacts impractical to include in submission.
- **"The paper's framing about safe constraints is misleading for an attack paper"** — While the language about "safe constraints" is somewhat overblown, this is more of a framing critique than a substantive weakness; the methodology itself is transparent about being an attack method.
- **"Baselines not re-run under same settings"** — The paper does provide multiple baselines (GCG, AutoDAN, PAIR, TAP, DeepInception) across different settings. While equal-budget comparisons would strengthen the paper, the baselines cover different attack paradigms (white-box, gray-box, black-box) which makes direct budget comparison complex. This is weakened to a minor concern about query efficiency transparency.

## Novel Insights

The paper's most interesting observation is that Pareto ranking from multi-objective optimization produces embeddings that are geometrically separable in embedding space (Figures 5-6). However, the paper overinterprets this as evidence for a "Rank Boundary Hypothesis" about toxic/non-toxic differentiation—when it more naturally reflects the tautology that optimizing a fitness function creates solutions whose embeddings correlate with that fitness. The practical question of whether this geometric structure can be exploited for defense (e.g., pre-filtering jailbreakable prompts) remains unexplored and could be a genuinely valuable direction.

## Suggestions

1. **Add a table reporting semantic consistency scores** ($f_2$) and any stealth metric alongside ASR for both BlackDAN and baselines. This is the single most impactful addition—without it, the central claim of multi-objective improvement is unsupported.
2. **Report query counts** per sample alongside time costs, so practitioners can assess feasibility against API budgets.
3. **Clarify the llama_guard_2 fitness function**: specify exactly how "unsafe token probability" is extracted from what is fundamentally a classification model, and resolve the discrepancy between inputs (R only vs. Q+R).
4. **Add a Pareto front plot** showing the actual trade-off surface discovered by NSGA-II—this is a natural and expected visualization for any multi-objective optimization paper.
5. **Evaluate against at least one standard defense** (e.g., perplexity filtering) to demonstrate that the claimed stealth improvements have practical implications.

## Score and Decision

**Calibration comparison:**

| Paper | Topic | Key Issues | Score Range | Decision |
|-------|-------|-----------|-------------|----------|
| AutoDAN (7Jwpw4qKkb.md) | GA-based jailbreak, stealth, single-objective | Method unclear, needs std errors, but good results | 6-8 | Accept (Poster) |
| Open Sesame (QXCjvHnDmu.md) | GA-based black-box jailbreak, single-objective | Loose formulation, limited baselines, query efficiency | 5 | Reject |
| ASE (xF5st2HtYP.md) | GA-based jailbreak, modular strategies | Missing params, no statistical results, query concerns | 3-6 | Reject |
| DGAttack (GnBBSlUb0S.md) | Multi-objective black-box attack, NSGA-II | Limited novelty, no closed-source eval, cost concerns | 1-6 | Reject |
| Jailbreaking as Reward Misspecification (uBnM3EFovQ.md) | Novel red-teaming perspective | Overclaiming, unclear definitions | 5-6 | Accept (Poster) |

BlackDAN sits in a similar space to DGAttack (multi-objective + NSGA-II for black-box attack) and ASE/ Open Sesame (GA-based black-box jailbreak). It shares DGAttack's core issue of claiming multi-objective optimization but not demonstrating the multi-objective gains in evaluation. Compared to AutoDAN (which was accepted), BlackDAN has broader empirical scope but weaker methodological clarity and a more fundamental gap between claims and evidence. AutoDAN tested against defenses and measured stealth (perplexity); BlackDAN does neither. Compared to ASE (rejected), BlackDAN has similar issues with missing statistical analysis and query budgets, but stronger empirical numbers.

The paper's most fundamental problem is the gap between its multi-objective claims and single-objective evaluation—this is not a minor omission but undermines the central novelty. The method may well be effective, but the paper does not demonstrate that the multi-objective formulation provides benefits beyond what a strong single-objective attack achieves. This places it firmly below the acceptance threshold.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>