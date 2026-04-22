Now I have all the information I need. Let me compile the final review.

## Summary

This paper investigates output volatility in long-form LLM generation through a three-stage approach: (1) VOLTBench, a benchmark that introduces multi-sample evaluation with volatility metrics (LSD, LVC, MLA) across structured and unstructured tasks; (2) attention trace analysis identifying "Attention Collapse" and "Attention Instability" as internal failure patterns; and (3) SELB, a decoding-stage method that boosts section-title logits and bans EOS/filler tokens to enforce structural adherence and prevent premature termination.

## Strengths

- **VOLTBench fills a genuine evaluation gap**: Table 1 explicitly shows no prior benchmark includes "Multiple Sampling" or "Stability Eval." The formally defined metrics (LSD, LVC, MLA) in Section 3.2 provide a rigorous quantitative framework for measuring cross-generation instability—a phenomenon prior work overlooked by focusing on single-generation results.

- **Heterogeneous task design surfaces important cross-format findings**: VOLTBench covers both unstructured and structured tasks with fine-grained constraints, and Figure 3 reveals that models generate longer and more stable text for structured tasks—a useful empirical finding attributed to format constraints providing stronger generation guidance.

- **Fine-grained constraint framework enables automated quality evaluation**: Section 4.2 introduces character-level, keyword, and theme constraints for programmatic verification of instruction adherence in narrative tasks. Section 4.3.1 quantifies a stark collapse: at the 500-section task, no model delivered more than 40 correct sections out of 100 required.

- **Descriptive failure mode analysis is valuable**: The identification of "Incomplete Generation" and "Section Skipping" patterns (Section 4.3), and the finding that constraint-following degrades systematically with output length, are useful empirical observations for the community.

- **SELB demonstrates practical effectiveness**: Figure 5 shows SELB-equipped models closely tracking the target-length reference line across all required lengths, while baselines degrade severely. The method achieves 100% SCA on structured tasks and 86.7% UCA, confirming the forced content is not just longer but also qualitatively correct.

## Weaknesses

### Fatal
None.

### Major

- **Misleading headline claims in the abstract**: The abstract states SELB "improves the mean output length of the base model by 148% and reduces the length volatility by 69%." Both figures are computed relative to LongWriter-8B—not the base model (Qwen2.5-7B) to which SELB is applied. Against the actual base model, the LVC reduction is only 17.5% (17.0% → 14.02%), not 69%, and the length increase is ~3417%, not 148%. Section 6.3 is transparent about the comparison baseline ("a 69% reduction in volatility compared to 45.4% for LongWriter-8B"), but the abstract and conclusion use the phrasing "of the base model," which clearly implies both improvements are relative to the same base. This misrepresentation of the primary quantitative claims is a serious presentation issue that could mislead readers who do not read the fine print.

- **SELB's methodological contribution is limited and task-specific**: The core mechanism (Equations 2–3) consists of (a) boosting logits of known section-title tokens when section length exceeds a threshold, and (b) setting EOS and filler-token logits to −∞ before the final section. This is a hard structural constraint that requires knowing the exact output format (section titles V_title, total section count P_total, target section length τ_max) in advance. The length increase is a direct consequence of preventing termination, not an indication that the model has improved its generation capability. While the SELB-Hybrid extension for free-form generation (Section 6.4) addresses this limitation, it is described only briefly with details in the appendix—despite free-form generation being the more practically relevant and challenging scenario. The paper's headline claims rest almost entirely on the structured, task-specific version.

### Minor

- **Attention analysis is correlational, not causal**: The abstract claims to identify "internal patterns that cause this volatility," and Section 1 references "root causes," but the evidence in Section 5 is purely observational—attention drops are correlated with generation failures. No causal intervention is performed (e.g., manipulating attention toward constraint tokens and showing changed output behavior). The more careful phrasing in Section 5 ("closely linked to and preceded by") is appropriate, but the abstract overclaims causality. That said, the success of SELB (designed based on these patterns) provides indirect support for the diagnostic value of the attention traces.

- **Attention pattern analysis is anecdotal**: "Attention Collapse" and "Attention Instability" are defined based on visual inspection of two traces (Qwen2.5-7B and Qwen2.5-3B, Figure 4) on one task (diary generation, 40 sections). There is no quantitative characterization—no report of how often these patterns occur, across how many samples, tasks, or models, or what thresholds distinguish "collapse" from normal fluctuation. This limits the scientific generality of these patterns.

- **N=5 samples per prompt may be insufficient for reliable volatility estimation**: Section 3.2 sets N=5 for computing SD and LVC. The standard error of the sample standard deviation is approximately SD/√(2(N-1)) ≈ 0.35·SD, meaning the volatility metrics themselves have substantial variance. This is a practical constraint but should be acknowledged as a limitation.

- **Missing ablation against naive structural enforcement**: No baseline tests the simple intervention of forcing section headers and banning EOS tokens without the logits-boosting formulation. Such an ablation would help determine whether SELB's gains come from the specific logit manipulation or from the structural constraints themselves. Without this, it is difficult to isolate the contribution of the proposed method from the injection of task-specific knowledge.

### Trivial
None.

## Nice-to-Haves

- Sensitivity analysis for β and τ_max hyperparameters would clarify practical utility and robustness.
- Content quality analysis comparing early (naturally generated) vs. late (forced) sections would reveal whether forced continuation maintains coherence or produces degenerate text masked by correct structural formatting.
- Quantitative characterization of attention patterns across models and tasks (frequency, thresholds) would strengthen the probing contribution from anecdotal to systematic.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh critic: "The 148% improvement in mean output length is a trivial consequence of banning the EOS token"** — While the mechanism does prevent termination, the SCA=100% and UCA=86.7% demonstrate the forced content is genuinely correct and coherent, not merely longer. The quality metrics partially address this concern, though the point about forced content quality in late sections remains valid as a minor concern.

- **Harsh critic: "SCA=100% reflects individually correct code snippets forced into a scaffolded structure, not coherent codebase"** — This is speculative without evidence. The paper's SCA metric measures correctness of individual sections, and the claim that this doesn't reflect "coherent codebase" is an unverified assertion.

- **Harsh critic: "Unfair baseline comparisons — baselines don't have task-specific structural information"** — The asymmetry actually favors the baselines (they don't need structural knowledge), not the proposed method, per the hard rules. However, the missing ablation against a naive structural enforcement baseline IS a valid minor concern, kept above.

- **Harsh critic: "A baseline that simply forces section headers and bans EOS tokens would likely achieve similar results"** — This is speculative. Without running this experiment, we cannot assume this baseline would achieve similar results, and the rule against unfair comparison asymmetry applies.

- **Strength finder: "SELB-Hybrid generalizes to free-form generation with MLA 97% and LVC 12.1%"** — This strength is based on appendix results that cannot be verified from the parsed text. The claim is kept as the paper states it in Section 6.4, but the evidence is thin (brief description, appendix-only detail).

- **Harsh critic: "LIFEBench does measure length adherence, and attention analysis for generation failures is not entirely new"** — These are "missing related work" critiques, which per the rules should be removed since we cannot verify the existence or relevance of uncited works.

- **Harsh critic: Formatting/presentation nitpicks about appendix placement** — Per rules, formatting complaints are removed.

## Novel Insights

The paper reveals an important asymmetry in how LLMs handle structured vs. unstructured long-form generation: models produce longer and more stable outputs for structured tasks (Figure 3), likely because format constraints provide implicit generation scaffolding. This observation has a paradoxical implication for the paper's own method—SELB's success may largely stem from externalizing exactly the kind of structural scaffolding that models already handle better internally, rather than from addressing the underlying attention dynamics. The paper's benchmark contribution (measuring cross-generation volatility) is more durable than its method contribution precisely because the benchmark exposes a real problem, while the method works around it via hardcoded structural knowledge.

## Suggestions

- Rewrite the abstract and conclusion to honestly frame the comparative claims: state the SELB vs. Qwen2.5-7B (base model) improvements directly (LVC: 17.0%→14.02%, a 17.5% reduction; length: 445→15,651 words), and if comparing to LongWriter-8B, explicitly name it as the comparison target rather than using the ambiguous "of the base model" phrasing.
- Add a naive structural enforcement ablation (ban EOS + force section headers without logits boosting) to isolate SELB's specific contribution from the effect of injecting task-specific structural knowledge.
- Elevate the SELB-Hybrid free-form results to the main paper with full experimental detail, since the paper's title and framing suggest a general solution to "long-form generation stability."

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LongWriter (long-form generation benchmark + data-centric method) | kQ5s9Yh0WI | 6.00 | Direct comparator; accepted as poster. Our paper has a benchmark + analysis + method but with misleading claims and a weaker method contribution. Below this. |
| Forking Paths (output volatility, statistical analysis) | 8RCmNLeeXx | 6.33 | Analyzes output volatility via forking tokens. Novel hypothesis but limited practical mitigation. Our paper has practical method but overstated claims. Comparable. |
| DoLa (decoding-time method, strong improvements) | Th6NyL07na | 7.25 | Decoding-time method with consistent improvements. Overclaimed hypothesis but solid method. Our paper is weaker due to misleading claims and task-specific method. Below this. |
| HelloBench (long-text generation benchmark) | QM2WoPu1It | 4.75 | Benchmark-only for long-text generation. Our paper has more dimensions + a method, but HelloBench was rejected. Our paper is slightly stronger. |
| Causal Token Decoding (attention analysis + decoding algorithm) | 6o9QUqUq9f | 4.67 | Very similar structure (mechanistic probing + decoding intervention). Rejected for limited novelty and missing baselines. Our paper has a stronger benchmark contribution. Above this. |
| MixAttention (zero novelty) | 2DD4AXOAZ8 | 2.00 | Genuinely poor paper with no novelty. Our paper is far above this. |
| From Loops to Oops (LLM fallback behaviors under uncertainty) | tFwEsrx1hm | 5.75 | Analyzes degenerate LLM behaviors, rejected. Our paper has similar analytical contribution plus a method but with misleading claims. |

This paper sits between HelloBench (4.75) and LongWriter/Forking Paths (6.0–6.33). The benchmark contribution is genuine and valuable, and the empirical findings about volatility are useful. However, the misleading abstract claims (69% volatility reduction and 148% length improvement, both computed against LongWriter-8B rather than the base model) and the limited methodological novelty of SELB (essentially hardcoded structural constraints) pull the score down. The paper is stronger than HelloBench (which was benchmark-only and rejected) and Causal Token Decoding (which had missing baselines), but weaker than LongWriter (transparent claims, clear contribution) and DoLa (consistent improvements across tasks).

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>