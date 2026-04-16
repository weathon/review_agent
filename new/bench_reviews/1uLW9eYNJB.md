Now let me search for calibration papers to anchor the score.Now I have sufficient context. Let me produce the final consolidated review.

---

## Summary

Mixture of Shards (MoS) proposes a parameter-efficient LoRA fine-tuning method that combines inter-layer and intra-layer parameter sharing with four lightweight differentiation strategies: subset selection, pair dissociation, vector sharding, and shard privatization. The method constructs low-rank matrices by selecting and concatenating shards from global pools using a static, randomly initialized index mechanism. Empirically, at a fixed 5M parameter budget, MoS outperforms all baselines (LoRA, Tied LoRA, ProLoRA), and at 19.99M parameters it matches LoRA at rank 64 (159.91M), claiming approximately 8× parameter savings.

---

## Strengths

- **Clear and empirically motivated design principle**: Section 2's analysis of pure sharing vs. differentiated sharing (Table 1) cleanly establishes that sharing alone can hurt performance and differentiation is needed. This is a simple but genuine insight backed by a well-scoped experiment, providing grounded motivation for all subsequent design choices.
- **Consistent improvement at matched parameter budget**: At 5M parameters, MoS outperforms LoRA (36.39 vs. 34.98), Tied LoRA (35.26), and ProLoRA (36.03) across all five individual benchmarks simultaneously—underlined values in Table 2—not merely on average. This consistency across diverse tasks (factuality, reasoning, multilinguality, coding) is a meaningful empirical result.
- **Meaningful ablation study**: Table 2's ablation isolates pair dissociation (>1% average drop) and shard privatization (>1% drop) as the most impactful components, while vector sharding provides only incremental gains. This granular accounting supports the method's design rather than treating it as an opaque ensemble.
- **Reasonable scalability**: Experiments on three model sizes (LLaMA2-7B, LLaMA2-13B, LLaMA3.2-3B) show consistent improvement over LoRA and ProLoRA, with ~1% average gain per step in the chain LoRA → ProLoRA → MoS on 13B.
- **Code released**: Public code availability facilitates reproducibility and adoption.

---

## Weaknesses

### Fatal
*None.*

### Major

- **The headline "8× parameter savings" claim is not sufficiently supported.** The comparison rests on a single-run point estimate: MoS at 19.99M achieves an average of 37.63 vs. LoRA-64's 37.53—a difference of 0.10 points. There is no variance estimate, no repeated runs in the main table, and no statistical equivalence test. Since the entire "8× savings" conclusion requires demonstrating performance parity, and LoRA at rank 16 already achieves 36.97 (close to MoS's 37.63), the margin is thin and its significance is unverified. The authors provide some random-seed results for LLaMA3.2-3B in Appendix B.3, but the main claim in Section 4.2 is unsupported by uncertainty estimates.

- **The deployment/serving motivation is asserted but not experimentally validated.** The paper's primary framing—alleviating GPU memory overhead "when numerous customized models are served simultaneously"—is never evaluated. There is no measurement of actual adapter memory footprint in a multi-tenant scenario, no adapter loading/switching benchmark, and no comparison of inference latency or memory residency. The contribution is more accurately described as "reduced trainable parameter count for similar benchmark accuracy," which is meaningful but narrower than the serving efficiency claim the introduction and abstract emphasize.

- **Missing ablation of inter-layer vs. intra-layer sharing contributions.** The ablation study (Table 2) only removes individual differentiation strategies; it does not isolate whether inter-layer sharing alone, intra-layer sharing alone, or their combination is driving the gains over ProLoRA (which uses intra-layer sharing only). Given that one of the paper's stated advances over ProLoRA is the addition of inter-layer sharing, the absence of this ablation is a real gap in the evidence.

### Minor

- **Scalability analysis excludes benchmarks selectively.** Table 3 (LLaMA2-13B) drops TyDi QA and HumanEval because "finetuning with vanilla LoRA does not yield consistent improvements." This is a reasonable empirical choice, but it reduces the 13B results to three tasks and weakens the scalability claim. Including a note on why LoRA fails on these tasks—and whether MoS also fails—would be more informative.

- **"MoE-like routing" is a misnomer.** The routing mechanism is static: indices are randomly sampled at initialization and frozen for the entire training run (Section 3.2: "randomly sampled during initialization, remains fixed during the finetuning process"). This is not routing in the MoE sense (no input-conditioned dispatch, no learned router). The paper's claim to be "the first to apply MoE-like mechanism for parameter savings in a single-task LoRA" (Section 5) is imprecise and could mislead readers about the nature of the mechanism.

- **Equation (5) contains a likely A/B label swap.** In Eq. (4), Route^c (column retrieval) operates on **B**^p with index **I**_b, and Route^r (row retrieval) operates on **A**^p with index **I**_a. In Eq. (5) as rendered, Route^c receives Concat(**A**^pub, **A**^pri) and Route^r receives Concat(**B**^pub, **B**^pri)—the opposite of what Eq. (4) implies. The surrounding text correctly describes the substitution, suggesting a typographical error in the equation itself. This should be corrected.

### Trivial

- At the fixed 5M parameter budget, MoS's gain over ProLoRA is modest (36.39 vs. 36.03, +0.36 average), with the four differentiation mechanisms adding complexity. This margin is real but small; future work might look at whether a simpler subset of strategies can recover most of the gain.

---

## Nice-to-Haves

- A **Pareto efficiency curve** plotting benchmark average vs. parameter count for both MoS and LoRA at multiple budgets (e.g., 5M, 10M, 20M, 40M, 160M) would make the efficiency story far more convincing than a two-point comparison.
- A **learned-routing variant** (even a lightweight layer-index-conditioned linear selector) would help clarify whether the static random indexing is sufficient by design or merely a convenience, and whether the "MoE-like" framing could be earned rigorously.
- **Hyperparameter sensitivity analysis** (pool size, shard size l, public/private split ratio) would help practitioners configure MoS for new settings.
- Testing on **at least one non-LLaMA model family** (e.g., Mistral, Qwen) would help establish that the sharing-differentiation dynamics are not architecture-specific.

---

## Removed Points

*These points are flagged to be removed; treat with caution.*

- **Harsh Critic: "VeRA comparison is unfair."** The paper explicitly acknowledges VeRA cannot be scaled to 5M parameters due to OOM, states clearly it does not claim superiority over VeRA in parameter efficiency, and includes VeRA as context rather than a primary comparison. Furthermore, under the hard rules, if asymmetric comparison favors the baseline it should not be penalized. Removed as the authors handle this responsibly.

- **Spark: "No modern model (LLaMA-3 8B)."** LLaMA3.2-3B experiments are reported (Appendix B.3), and 13B is in the main paper. While a larger current-generation model would be nice, this falls in "nice-to-have" territory given the paper's scope.

- **Neutral Reviewer: "Theoretical motivation is thin."** For an empirical systems-style PEFT paper, demanding formal approximation bounds or expressiveness proofs is not a community standard. Moved to nice-to-have territory.

- **Neutral Reviewer and Spark: "No compared analysis of compute overhead."** The paper's claim is about trainable parameter count, not FLOPs; evaluating it on FLOPs is scope creep unless the paper itself claims compute efficiency. Removed.

---

## Novel Insights

The most genuinely useful observation in this paper is the empirical decomposition in Table 1: pure parameter sharing at matched parameter count can *underperform* vanilla LoRA, and the introduction of differentiation strategies can *reverse* this degradation and surpass LoRA. While this intuition is implicit in prior work, the controlled isolation of "pure sharing" as a distinct baseline and the quantification of how each differentiation technique contributes represents a real pedagogical and design contribution. The hierarchy among differentiation strategies—pair dissociation and shard privatization matter most, vector sharding contributes marginally—is also a concrete and actionable finding that prior work (VeRA, ProLoRA, Tied LoRA) had not surfaced. Future parameter-sharing LoRA designs could use this hierarchy to prioritize which mechanisms are worth their complexity budget.

---

## Suggestions

1. Add repeated-run variance estimates (at minimum ±std over 3 seeds) to the 8× savings comparison in Table 2; this is the paper's headline claim and needs statistical backing.
2. Replace or supplement the single-sentence serving argument with a concrete multi-adapter memory measurement (e.g., memory used by 100 concurrent MoS adapters vs. 100 LoRA adapters in a mock serving setup).
3. Add an ablation row that removes *both* inter-layer and intra-layer sharing together (i.e., pure intra-only, pure inter-only) to properly credit each sharing dimension independently.
4. Correct Eq. (5): Route^c should receive Concat(**B**^pub, **B**^pri) and Route^r should receive Concat(**A**^pub, **A**^pri), consistent with Eq. (4).
5. Rename the routing mechanism to "static index-based shard selection" or similar, reserving "MoE-like" only if a learned router is actually implemented.

---

## Score and Decision

**Calibration:**

| Paper | Decision | Scores | Notes |
|-------|----------|--------|-------|
| RaSA (GdXI5zCoAt) | Accept Poster | 8,8,6,6 | Rank-sharing LoRA with theory + strong experiments |
| RandLoRA (Hn5eoTunHN) | Accept Poster | 6,6,6,6 | Random matrix PEFT, solid but incremental |
| Bi-Share LoRA (Thv66GmqZS) | Reject | 3,6,6,6 | Combined intra/inter sharing LoRA, similar concept, weaker results |
| ShareLoRA (O6QZ4W6GXt) | Reject | 3,5,3,5,5 | Layer similarity sharing, limited savings, heuristic design |
| LoRTA (ayPfZIkovt) | Reject | 3,5,6,3 | Tensor PEFT, rejected for limited gains |

**Positioning:** MoS is clearly above Bi-Share LoRA and ShareLoRA in terms of empirical gains (8× vs. ~50% savings), methodological clarity, and ablation rigor. It is below RaSA, which has both theoretical grounding and broader evaluation. It is comparable to RandLoRA: a solid empirical paper with an interesting core idea, good ablation, but limited theoretical depth and some overclaiming. The unsupported 8× headline claim and missing serving experiments prevent a higher score, but the consistent per-task wins at 5M parameters, multi-scale evaluation, and meaningful ablation prevent a low score. I place this at **5.5**—borderline, leaning toward weak accept if the statistical concerns can be addressed, but currently sitting just below RandLoRA-level quality due to the overclaimed primary result.

**Axes summary:**
- *Originality*: Moderate. Combining existing sharing mechanisms with a principled differentiation analysis is incremental but useful.
- *Importance*: Moderate-high. The multi-adapter serving problem is real and relevant.
- *Claim support*: Weak-moderate. Core results hold directionally; the headline 8× claim needs variance.
- *Experimental soundness*: Moderate. Broad task coverage and sensible baselines, but no statistical testing and selective benchmark dropping for 13B.
- *Clarity*: Good, with one equation error.
- *Community value*: Moderate. The sharing-differentiation hierarchy is a useful design lesson.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>