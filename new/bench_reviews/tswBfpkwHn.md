Now I have enough information to write the final review. Let me consolidate my analysis.

## Summary

This paper provides the first theoretical analysis of one-layer Mamba's training dynamics and in-context learning (ICL) generalization on binary classification tasks with additive outliers in the prompt. The analysis decomposes Mamba's ICL mechanism into a linear attention component (which selects examples sharing the query's relevant pattern) and a nonlinear gating component (which suppresses outlier-containing examples and induces exponential positional decay). The main theoretical results (Theorems 1–4) characterize convergence conditions, sample complexity, and ICL generalization bounds for both Mamba and a one-layer single-head linear Transformer, showing that Mamba can tolerate outlier fractions approaching 1 at test time, while the linear Transformer tolerates at most 1/2.

## Strengths

- **First rigorous training dynamics analysis of Mamba for ICL**: The paper fills a genuine gap—prior theoretical work on Mamba's ICL capabilities was largely empirical. The convergence guarantee (Theorem 1) with explicit sufficient conditions on batch size, iterations, outlier magnitude, and prompt length is a non-trivial and novel contribution.
- **Clean mechanistic decomposition**: Corollaries 1–2 provide an interpretable characterization of how Mamba implements ICL: linear attention selects same-pattern examples (Corollary 1, Eq. 16), while gating suppresses outliers (Eq. 17) and imposes exponential positional decay (Eq. 18). This mechanism-level insight is valuable regardless of the Mamba-vs-Transformer framing.
- **Quantitative separation in outlier robustness**: Theorem 2 condition (c) shows Mamba generalizes when α < min(1, p_a·l_tr/l_ts), while Theorem 4 requires α < 1/2 for the linear Transformer. This is an order-wise separation, not merely a constant-factor gap, directly supporting the claim that sigmoid gating provides meaningful additional robustness.
- **Explicit trade-off characterization**: The paper honestly identifies that Mamba's superior robustness comes at the cost of harder optimization—Theorem 1 requires T_M = Θ(l_tr) · T_T more iterations than the linear Transformer (Remark 4)—giving concrete guidance about when each architecture is preferable.

## Weaknesses

### Fatal
None.

### Major

- **The "Mamba vs. Transformers" framing overreaches what the evidence supports**: The paper's central narrative is that Mamba outperforms Transformers on ICL with outliers. However, the comparison in Section 3.4 is specifically between Mamba (with sigmoid gating) and a one-layer single-head *linear* Transformer (obtained by setting all gating values G=1). This strips the baseline of the mechanisms—softmax normalization, multi-head structure—that give practical Transformers their outlier robustness. The paper acknowledges this in Remark 6 ("the gating is the only difference between the two architectures") and notes that "Large Transformer models, with appropriate training methods and ICL prompt design, can indeed achieve favorable robustness." However, the abstract claims Mamba "maintains accurate predictions even when the proportion of outliers exceeds the threshold that a linear Transformer can tolerate," and the introduction frames the question as whether Mamba "can match or surpass the capabilities of Transformer models." This broader framing is not supported by the evidence. What *is* established—and is a valid and important contribution—is that sigmoid gating provides outlier robustness that uniform weighting does not. The framing should be calibrated accordingly.

- **Exponential positional decay creates a significant conditional vulnerability that undercuts the robustness claims**: Corollary 2(ii) and equation (18) show that gating values decay exponentially with index distance from the query. Table 1 confirms the practical consequence: when outliers are placed closest to the query (CQ setting), Mamba's accuracy drops to 82.73% while the linear Transformer stays at 93.96%. The paper acknowledges this in the conclusion, but the abstract and introduction make unconditional robustness claims ("maintains accurate predictions even when the proportion of outliers exceeds the threshold"). The robustness guarantee is conditional on favorable outlier placement, which an adversary can easily violate. This is a real architectural limitation of Mamba's recurrence-based gating, and the paper should foreground it as a significant qualifier on the robustness claims.

### Minor

- **Gap between the motivation (adversarial labels) and the theoretical setup (random labels)**: Example 1 in Section 3.2 motivates the work with data poisoning where outlier labels are *deterministically* adversarial (e.g., always positive). But Definition 1 specifies that training-time outlier labels are random (uniform over {+1,−1}). Random labels provide no gradient signal, so SGD naturally learns to ignore them—this is partly an optimization artifact rather than a purely architectural mechanism. At test time (Definition 2), labels can be arbitrary, which is the practically important guarantee, but the training-time assumption distances the theory from the adversarial motivation. This is a reasonable modeling choice for tractability, but the gap deserves explicit acknowledgment.

- **The condition α < min(1, p_a·l_tr/l_ts) is less permissive than the abstract suggests**: The paper emphasizes that α "can be close to 1" (Remark 3), but when l_tr = l_ts (equal training and testing prompt lengths), this requires α < p_a, meaning one can only tolerate at test time what was seen during training. Achieving α close to 1 requires either p_a ≈ 1 (training with nearly all outliers) or l_tr ≫ l_ts (much longer training prompts). These are unusual regimes. The claim that α approaches 1 is technically correct but practically constrained.

## Nice-to-Haves

- Experiments with softmax attention as a baseline, even if only to empirically demonstrate that the advantage comes specifically from gating rather than from any nonlinearity that can suppress outliers.
- Analysis or experiments with adversarial (non-random) training-time outlier labels to test whether the robustness depends on the random-label assumption.
- Evolution of gating weights during training (not just the final step) to strengthen mechanistic claims.

## Removed Points

- **Claim that the Appendix B.1 softmax experiments are absent and therefore the broader comparison is unsupported**: The paper references additional experiments with softmax/multi-head attention in Appendix B.1. This appendix exists in the original submission; it is only stripped from the parsed version. The paper explicitly mentions these experiments, so this is not a missing contribution.
- **Formatting/notation complaints**: Various parser artifacts and notation issues are removed per instructions.
- **Reproducibility concerns about undisclosed hyperparameters**: The paper provides explicit parameter settings (d=30, M1=6, M2=10, etc.) for synthetic experiments. Minor implementation details are not substantive weaknesses for a theory paper.
- **Demand for larger-scale experiments on real data**: The paper's scope is theoretical analysis on a well-defined synthetic setting consistent with the ICL theory literature. Requesting real-data experiments goes beyond the paper's stated scope.
- **Strength claim about "sharp quantitative separation"**: The separation is order-wise valid, but it's between specific architectural variants, not between Mamba and practical Transformers. I've adjusted the framing in my kept strengths.

## Novel Insights

The most interesting insight that emerges from synthesizing the reviews is the architectural trade-off revealed by the mechanistic analysis: Mamba's sigmoid gating simultaneously provides outlier suppression (Eq. 17) and positional locality bias (Eq. 18), and these are two sides of the same coin. The gating mechanism that enables robustness to high outlier fractions is also what creates the CQ vulnerability—when outliers are positioned near the query, the exponential decay that normally protects Mamba by downweighting distant examples instead downweights the clean examples that are further away. This is not merely a weakness but a structural consequence of the design. The paper's contribution is most honestly understood as characterizing this fundamental trade-off rather than as demonstrating unconditional superiority of one architecture over another.

## Suggestions

- Recalibrate the abstract and introduction to frame the contribution as "characterizing the role of sigmoid gating in Mamba's ICL robustness" rather than "Mamba outperforms Transformers," since the theoretical comparison is between gated vs. ungated linear attention, not between Mamba and practical Transformers.
- Frontend the CQ limitation prominently (e.g., in the abstract or as a highlighted remark) rather than relegating it to the conclusion, since it represents a genuine and architecturally inherent constraint on the robustness claims.
- Add a brief discussion of how commonly l_tr ≫ l_ts holds in practice when claiming α can approach 1.

## Score and Decision

**Calibration anchors:**
- wkHcXDv7cv (SSM frequency bias, avg 7.5): Strong SSM theory paper with both theory and practical implications. This paper has a narrower empirical contribution (synthetic only) and an overclaim issue that the high-scoring paper avoids.
- vSh5ePa0ph (ICL linear regression, avg 6.75): Clean theoretical ICL analysis with Bayes-optimal bounds. Comparable scope and rigor, but this paper has a more significant framing gap.
- kxpswbhr1r (nonlinear transformer ICL, avg 5.75): Similar scope—first softmax attention training dynamics for ICL. This paper has comparable depth but slightly stronger framing issues.
- m60n31iYMw (SSM clean-label poisoning, avg 5.75): SSM theory with conditional results and a contrived example. This paper has broader applicability than that one.
- 2L1OxhQCwS (LSTM vs Transformers, avg 3.33): Overclaimed comparison with narrow evaluation. This paper is substantially stronger—it has genuine theoretical contributions and honest experimental acknowledgment of limitations.

The paper sits above the medium-theory anchors (which were rejected) mainly due to its genuine novelty as the first Mamba training dynamics analysis, but below the high-scoring theory papers due to the framing overreach and the conditional nature of its robustness claims. The core contribution—the mechanistic decomposition and the formal separation between gated and ungated attention—is sound and valuable, but these are wrapped in broader claims the evidence cannot fully support.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>