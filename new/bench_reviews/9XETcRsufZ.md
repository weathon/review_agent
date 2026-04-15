Now let me look at some calibration papers to help score this work.Now I have enough information. Let me write the consolidated final review.

---

## Summary

This paper investigates whether Mixture-of-Experts (MoE) transformers offer a "free lunch" over dense transformers, arguing that experts disproportionately improve memorization over reasoning. The authors support this thesis through (1) theoretical lower/upper bounds using communication-complexity arguments for one-layer transformers, (2) synthetic experiments on phone-book memorization and shortest-path tasks, and (3) pretraining experiments up to 2.1B parameters evaluated on 18 benchmarks in NLP and math.

---

## Strengths

- **Clear and important thesis with convergent evidence.** The memorization-vs-reasoning tradeoff for MoEs is a genuinely important architectural question for the community. The paper brings together theory, synthetic experiments, and large-scale pretraining, which collectively tell a coherent story. Specifically, Figure 4a (phone-book) and Figure 4b (shortest path) offer a striking visual split, and Figure 6 (fixed-perplexity comparison) is a particularly thoughtful design that reveals architectural biases rather than just parameter-count effects.

- **Non-trivial theoretical contributions.** Theorem 3.2 cleanly extends prior communication-complexity lower bounds (Sanford et al., 2024) to the sparse transformer class. Corollary 3.4 provides a concrete parameter-matched separation. Theorems 3.5–3.6 establish an active-parameter separation for memorization (MoEs require Õ(√nm) active parameters vs. Ω̃(n) for dense), with the elegant choice K=√(n/m) being a clean analytical result.

- **Strong synthetic memorization experiment.** The phone-book task is a direct test of storage capacity, and the result is persuasive: MoEs match dense models at equal total parameters while requiring far fewer active parameters, consistent with the theory.

- **Scale of pretraining.** Models are trained up to 2.1B parameters on 65B tokens with multiple MoE active-parameter variants, evaluated across 18 benchmarks—a substantial empirical investment for a theory+empirical paper.

- **Generalization gap analysis (Figure 5).** The observation that MoEs show a larger train-test gap on math problems is a concrete, practically meaningful warning about deployment of MoE models on reasoning tasks.

---

## Weaknesses

### Fatal
*None.* The paper has real problems, but they do not invalidate the core thesis.

### Major

- **Training-set contamination confounds the world-knowledge evaluation.** Section 5.1 explicitly states: *"The natural language dataset is a mixture constituted of FineWeb-edu, Cosmopedia, Wikipedia and the **training sets of the downstream tasks we evaluate on**."* The world-knowledge benchmarks (TriviaQA, NQ, HotpotQA, WebQuestions, ComplexWebQuestions) have their training sets included in pretraining. Because both MoE and dense models see the same data, the architecture comparison remains valid — but the interpretation changes: the paper cannot cleanly claim that MoEs are better at *general factual recall from broad corpora*; they may simply absorb benchmark-specific answer patterns more efficiently. This weakens the link between the world-knowledge benchmark results and the "memorization" framing the paper uses throughout. The perplexity-controlled analysis (Figure 6a) partially rescues the claim, but the benchmark-overlap issue should be explicitly discussed as a limitation on the interpretation.

- **Compute-matched comparison is missing.** MoEs with fewer active parameters process fewer FLOPs per token. At fixed training tokens (65B), models with 18M vs. 200M active parameters receive dramatically different total training compute. Whether the observed reasoning gap is an architectural effect or a training-efficiency/compute effect is never tested. Training models to matched compute (FLOPs × tokens) rather than only matched parameter counts would be needed to cleanly attribute the reasoning gap to architecture rather than data efficiency. This is arguably the most important confound left uncontrolled in Section 5.

- **Theory-to-experiment gap is wide and unaddressed.** The theory is for depth-1 transformers with top-1 routing and log-precision; the experiments use 12–20 layer models with top-2 routing. The paper offers no discussion of whether multi-layer architectures can circumvent the width bottleneck through distributed cross-layer computation. This gap is real: a 12-layer transformer can pass information across layers in ways that one-layer bounds do not capture. The theory is interesting on its own terms, but it is used to motivate empirical claims in a setting where its conditions do not hold.

### Minor

- **Benchmark categorization is asserted, not validated.** The paper assumes world-knowledge QA = memorization, commonsense QA = reasoning. No validation (e.g., correlation analysis between total vs. active parameter count and per-task accuracy) is provided for this split. As noted in Section 5.1, aggregate category averages hide per-task variance; commonsense benchmarks like SciQ and ARC-E heavily reward pattern memorization, potentially inflating the "reasoning" signal from that category. The appendix reportedly has per-task results, but the main claims should be more precisely grounded.

- **Graph reasoning experiment uses i.i.d. held-out test data.** The test set is "sampled from the *same* distribution as the training examples" (Section 4.1). The paper tests whether models generalize to held-out graphs from the same family, which is the appropriate and standard setup for this type of task. However, the paper does not sweep graph sizes to show that the MoE performance bottleneck corresponds quantitatively to the theoretical critical-width prediction, weakening the theory-experiment connection.

- **Non-standard FFN intermediate dimension (d instead of 4d) lacks ablation.** The paper follows OLMoE's choice of setting the FFN intermediate dimension to d rather than 4d, which halves active parameters relative to standard architectures. This affects the active/total parameter ratio for both architectures consistently, but since the claim is about active parameters as the key driver for reasoning, it could alter the measured tradeoffs. The paper justifies this by citing OLMoE, but a brief ablation would clarify whether conclusions generalize to standard configurations.

### Trivial

- The generalization-gap interpretation (Figure 5) should be hedged: a larger train-test gap for MoEs could also reflect distribution shift between OpenMathInstruct (training examples) and GSM8k/MATH (test sets), or difference in problem formatting, rather than memorization per se.

---

## Nice-to-Haves

- Expert specialization visualization: routing heatmaps showing which experts activate on fact-retrieval vs. multi-step reasoning tokens would directly support the mechanistic claim.
- Experiments with shared experts (as in DeepSeek-MoE) or expert-choice routing to test whether the reasoning gap is specific to token-choice top-k routing or more fundamental to the MoE paradigm.
- A scaling experiment varying graph size (n) to verify that the performance wall for MoEs occurs at the theoretically predicted critical width, quantitatively connecting theory and experiment.
- Longer training experiments to test whether the reasoning gap closes with more tokens, which would reframe it as a data-efficiency issue rather than an architectural one.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Bit precision unrealism" (Human Finder, citing UGVYezlLcZ):** The Human Finder reviewer imports a concern from a different memorization paper about log-bit precision being unrealistic. However, the paper's theoretical setting explicitly uses log-precision throughout (Section 3.1), and the separation results hold within that setting. This is a standard theoretical assumption in transformer expressivity work, and criticizing it here without showing it breaks the paper's specific argument is a reviewer knowledge-gap issue rather than a paper flaw.

- **"Routing mechanism generalization" as a weakness (Neutral Reviewer, weakness #2; Spark, missing experiment #4):** The paper explicitly scopes this out: *"We leave the study of MoEs trained with other routing mechanisms for future work."* The core contribution does not depend on this being resolved; it is a legitimate next step, moved to Nice-to-Haves.

- **"Binary task categorization is too simplistic" (Neutral Reviewer, weakness #4) raised as a major weakness:** This is a known limitation of benchmark-based evaluation in NLP, but the paper does make an effort to use three distinct categories and report per-task results in the appendix. The criticism is valid but belongs as a minor note, not a major weakness.

---

## Novel Insights

The most genuinely novel insight this paper offers beyond prior scaling-law papers is the *implicit architectural bias* framing in Figure 6: two models achieving the same validation perplexity need not learn equivalent capabilities — MoEs and dense models exploit different mixtures of memorization vs. reasoning strategies to minimize the same training objective. This suggests that perplexity, the dominant pretraining metric, is an insufficient specification for downstream capability, and that architectural choice implicitly shapes the capability profile of a model at fixed perplexity. This observation could have broad implications for how practitioners select architectures depending on the capability profile they need.

---

## Suggestions

1. **Explicitly discuss and quantify the training-set overlap** for world-knowledge benchmarks and assess whether the memorization advantage persists when evaluating on test examples whose training counterparts are strictly excluded.
2. **Add compute-matched experiments** (total FLOPs = active_params × tokens) as either a figure or ablation to disentangle compute efficiency from architectural bias.
3. **Add informal discussion in Section 3 of whether multi-layer MoEs could circumvent the single-layer width bottleneck**, even without formal theorems — this would better calibrate how much the theory supports the empirical claims.
4. **Report per-task results in the main body** for at least selected benchmarks, showing whether the memorization/reasoning split is consistent across all tasks within each category or driven by a few outliers.

---

## Score and Decision

**Calibration papers considered:**

| Paper | Type | Scores | Decision |
|---|---|---|---|
| *On the Optimal Memorization Capacity of Transformers* (UGVYezlLcZ) | Theory + light empirical | 6, 6, 8, 6 | Accept poster |
| *Chain of Thought Empowers Transformers* (3EWTEy9MTM) | Theory + empirical | 8, 6, 8, 3, 8, 5 | Accept poster |
| *FLAN-MoE* (6mLjDwYte5) | Empirical MoE comparison | 8, 5, 6, 8 | Accept poster |
| *Dense Backpropagation for MoE* (huy8g3iKy0) | MoE training method | 6, 3, 5, 8 | Reject |

**Reasoning:** This paper is stronger than the MoE training papers that were rejected (huy8g3iKy0) in that it provides a theoretical foundation and a systematic empirical program rather than a narrow engineering contribution. It is comparable in scope to the memorization capacity paper (UGVYezlLcZ, avg ~6.5, accepted poster), but benefits from scale of empirical evidence (18 benchmarks, 2.1B models) while suffering from the training-contamination concern and compute-matching gap that weaken its strongest claims. The paper does not reach the level of the CoT expressivity papers, which have cleaner theory-practice links. The human reviewer panel for this paper suggested Accept (poster) at scores consistent with 6–7 on average. The training contamination is a real but not fatal issue; the core architecture-level comparisons are still valid. The compute-matching gap is a more serious confound but is unfortunately common in architecture comparison papers at this scale.

**Assessment on axes:**
- *Originality:* Good — the memorization/reasoning split for MoEs is new and important.
- *Importance of research question:* High — MoE is the de facto architecture for frontier models; this tradeoff matters enormously.
- *Claims well-supported:* Partially — the phone-book and perplexity experiments are strong; the world-knowledge interpretation is weakened by contamination; compute is uncontrolled.
- *Soundness of experiments:* Moderate — systematic but with the two major methodological gaps noted.
- *Clarity of writing:* Good — the paper is well-organized and clearly written.
- *Value to community:* High — a practical warning about MoE limitations for reasoning is broadly useful.

Final score: **6.5** (above the threshold for a poster; the contribution is genuine and the paper's synthesis of theory, synthetic experiments, and pretraining evaluations is broadly sound, but two major methodological concerns prevent a higher score).

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>