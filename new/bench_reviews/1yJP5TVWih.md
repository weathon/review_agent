Now let me carefully analyze the paper content and the reviewer claims. Let me verify some of the harsh critic's key claims:

1. **Claim: The guarantee is conditional and practically weak** - The paper does state the input condition μ(Y^(0))² ≥ b in Theorem 4.1, and b depends on 1/a^K, which means for deeper networks the condition becomes harder to satisfy. This is a valid concern. Also, Remark 4.1 concedes that for Mamba, a<1, so the bound is still decaying. These are substantive issues.

2. **Claim: Necessity is not established** - The paper title says "the architectural component that PREVENTS rank collapse" but Section 4.2 says "we do not provide a formal necessary condition." The experiments in Fig 3 show that gating + LN without λ-skip can also prevent collapse. So this is a valid concern about overclaiming.

3. **Claim: μ is a weak metric for rank collapse** - This is a softer claim. The μ metric is standard in the literature (Dong et al. 2023; Wu et al. 2024a). The paper acknowledges limitations in the conclusion. This is a minor concern, not a major one.

4. **Claim: Experiments are weak** - The experiments on pre-trained Mamba-2 with removed gating is valid. Table 1 is small. No training dynamics metrics. This is valid but somewhat expected for a theoretically-oriented paper.

Let me also consider the positive aspects more carefully:
- The paper does extend rank collapse analysis to SSMs, which is novel
- The unifying framework is clean
- The theoretical results (Theorem 4.3 for selective SSMs, doubly exponential collapse) are genuine contributions
- The tightness result (Proposition 4.3.2) is valuable

Now let me think about scoring relative to calibration:
- X6xzYP2cMk (Mind the Gap): 5,5,6,3 — rejected, had limited theoretical framework and small experiments
- cxKLRM3KhC (Residual Connections Harm): 6,5,5,6 — rejected, similar concerns about conservative conditions and limited evaluation
- dIoLjHet58 (GPAM): 6,5,5,6 — rejected, similar concerns about rank collapse theory with limited experimental validation
- 9FqARW7dwB (Hyper-Connections): 5,8,6,6 — accepted, had stronger practical results
- mOTiVzTgF2 (ResiDual): 6,5,1,6,3 — rejected, theoretical contribution with limited experiments
- RtDok9eS3s (Simplifying Transformer Blocks): 8,8,6 — accepted, strong empirical results
- x7cuUZxwFS (Transformers Get Stable): 3,6,5,6 — rejected, accused of lack of novelty

The current paper has genuine theoretical contributions (SSM rank collapse analysis, λ-skip analysis) but with significant caveats (conservative bounds, excluded gating, limited experiments). The overclaiming in the title and abstract is notable. The paper is comparable to "Mind the Gap" (rank collapse in transformers, limited scope) and "Residual Connections Harm" (skip connection parameter, conservative conditions).

I'd place this paper somewhat below the rejected papers in the 5-6 range, given the overclaiming issue and the gap between theory and practice. Maybe around 4.5-5.

Actually, let me reconsider. The paper does make genuine contributions:
- Novel theoretical extension to SSMs
- Unified framework for both transformers and SSMs
- Concrete λ-skip connection analysis with sufficient conditions
- New finding about gating mechanisms preventing rank collapse

The main issues are:
- Overclaiming (title says "the" architectural component, but necessity isn't shown)
- Conservative bounds that aren't practically useful
- Excluded gating from theory
- Limited experimental validation

Given the calibration papers, I think a 5 is fair. It's similar to Mind the Gap (5,5,6,3 = ~4.75 avg → reject) and Residual Connections Harm (6,5,5,6 = ~5.5 avg → reject). The paper has similar characteristics: genuine but limited theoretical contribution, overclaimed results, and modest experiments. I'll give it a 4.5, which is below acceptance threshold.

Wait, let me reconsider more carefully. The paper does have some nice theoretical work, and it's the first to study rank collapse in SSMs. But the overclaiming is substantial. The title literally says "THE architectural component that prevents rank collapse" as if it's the unique solution, which it isn't. And the "guarantee" in Theorem 4.1 is conditional with a K-dependent input condition that gets harder to satisfy as depth increases. Plus for Mamba it can only guarantee geometric decay, not actual prevention.

I think 4.5 is right, but let me think about it more carefully in terms of whether it should be 5. The paper is not terrible - it has clear theoretical contributions, clean analysis, and novel insights. But the overclaiming and gap between claims and evidence is significant.

Actually, looking at the scores more carefully:
- GPAM (6,5,5,6): rejected, theoretical analysis + limited empirical results on rank collapse
- Mind the Gap (5,5,6,3): rejected, rank collapse theory + limited experiments
- Residual Connections Harm (6,5,5,6): rejected, skip connection parameter + limited experiments

This paper seems comparable to these. I'll go with 4.5, which is borderline Reject.</think>Now let me write the final review.

## Summary

This paper extends the theory of rank collapse from transformers to State Space Models (SSMs) using a unifying sequence model framework. It introduces λ-skip connections (skip connections with a learnable strength parameter λ) and provides sufficient conditions on λ, under which rank collapse can be prevented across transformers, LTI SSMs, and selective SSMs. The paper also shows that removing skip connections leads to exponential (with LayerNorm) or doubly exponential (without LayerNorm) rank collapse in selective SSMs, empirically identifies gating mechanisms as important for preventing rank collapse, and validates findings on Mamba-2.

## Strengths

- **Novel theoretical extension to SSMs**: This is, to my knowledge, the first formal analysis of rank collapse in State Space Models, addressing a real gap in the literature. The extension of collapse analysis (Theorems 4.2 and 4.3, plus the doubly exponential results in Appendix A.9) from transformers to selective SSMs is a genuine and valuable contribution.

- **Clean unifying framework**: The use of the O = M·V formulation (Eq. 1) to capture both attention and SSM blocks in a single analysis is elegant and allows the main theoretical results to apply broadly. This abstraction makes the results generalizable beyond specific architectures.

- **Constructive analytical examples**: Propositions 4.3.1 and 4.3.2 demonstrate both that λ-skip connections can fail under poor parameter choices and that the lower bound in Theorem 4.1 is essentially tight. This honest characterization of the theory's limits strengthens credibility.

- **Empirical discovery of gating's role in rank collapse**: The finding (Figure 3) that gating mechanisms in Mamba-2 help prevent rank collapse is novel and practically relevant, as gating was originally designed for memory, not stability.

## Weaknesses

### Fatal
None.

### Major

- **Overclaiming: the title and abstract promise more than the theory delivers.** The title states that λ-skip connections are "the architectural component that prevents rank collapse," and the abstract claims this is "the first study that provides a general guarantee to prevent rank collapse." However: (1) Section 4.2 explicitly says "we do not provide a formal necessary condition"; (2) Theorem 4.1 only gives a *conditional* lower bound (requiring μ(Y^(0))² ≥ b where b depends on 1/a^K, making it harder to satisfy for deeper models); (3) Figure 3 shows that gating + LayerNorm can prevent rank collapse without any λ modification, contradicting the framing of λ-skips as "the" solution; (4) For SSMs like Mamba, Remark 4.1 concedes that a<1, meaning the bound still allows geometric decay — it does not *prevent* collapse in the asymptotic sense. The gap between the strong framing and the actual results is substantial and misleading.

- **Conservative sufficient conditions with unclear practical utility.** The condition |λ| > (a+√a)SC_M/(1−a) requires very large λ values to satisfy for reasonable a, and the paper itself acknowledges (Section 5.1, discussing Figure 2) that "the condition on λ in Theorem 4.1 is too conservative" and that "in practice much lower values of λ are good enough." A theorem whose conditions cannot be verified in practice (C_M requires bounding sup_k ||M^(k)||_F for input-dependent M^(k), which is generally intractable) and whose predictions are far from real behavior limits its utility as an architectural design principle. Proposition 4.3.2 shows the bound is tight, meaning this conservativeness is inherent to the approach without additional assumptions.

- **Gating mechanisms excluded from the theoretical analysis.** The paper identifies gating as playing "a crucial role in preventing rank collapse" (Section 5.2) and acknowledges this is the "first" such connection, yet explicitly excludes gating from all theoretical derivations ("we ignore these in the theoretical part of this paper for simplicity," Section 3.1). Since gating is a core component of Mamba and Mamba-2 — the architectures primarily studied — the theory does not cover the most practically relevant model configurations.

### Minor

- **Limited experimental validation of practical benefits.** Table 1 shows mixed results: learnable λ sometimes underperforms λ=1 (e.g., Transformer on MQAR: 99.6→98.9; Mamba-2 on Image LRA: 42.28→38.92). The main rank collapse experiments (Figures 1–3) modify a pre-trained model rather than training from scratch, so they probe forward-pass geometry, not training dynamics. The link between preventing rank collapse and improved training or downstream task performance is never established.

- **Assumption 4.1 restricts the SSM analysis.** Theorem 4.3 for selective SSMs requires A_t = αI (not input-dependent), which excludes the most interesting selective parameters in Mamba. The paper states this is "for ease of exposition" and provides empirical evidence for the general case, but the theoretical guarantee is narrower than claimed.

- **The input condition in Theorem 4.1 receives insufficient discussion.** The condition μ(Y^(0))² ≥ b with b = (1/a^K)(2λNdSC_M)/(λ² − a(SC_M+|λ|)²) depends on 1/a^K, which grows exponentially with depth K when a < 1 (the SSM case). This means deeper models require inputs further from rank collapse for the guarantee to hold — precisely the setting where rank collapse matters most. This important limitation is not discussed.

### Trivial
- The definition of the normalized rank collapse measure used in experiments (Section 5.1) is stated verbally ("normalized by the norm of the layer output") but differs from the formal definition in Eq. 5, which has no normalization. A precise mathematical definition would avoid confusion.

## Nice-to-Haves

- Train models from scratch with different λ values on standard benchmarks (e.g., language modeling perplexity) and measure both rank collapse metrics and training dynamics (gradient norms, loss curves) to establish the practical benefit of rank collapse prevention.
- Incorporating even a simplified model of gating (e.g., fixed gate values) into the theoretical framework would make the results directly applicable to Mamba-like architectures.
- Report learned λ values when using "var. λ" in Table 1 to understand what the model actually learns and how this relates to the theoretical predictions.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The experiments modify a pre-trained Mamba-2 model by removing gating and varying λ, which completely changes the architecture."** (From harsh critic #4) — While true, this is a standard approach in the rank collapse literature (Dong et al. 2023; Wu et al. 2024a also use pre-trained models). The experiments probe architectural properties, not training dynamics, and the paper is transparent about this. This is a limitation but not a methodological error.

- **"No comparison with alternative rank-collapse mitigation strategies (DeepScaleLM, ReZero, etc.)"** (From spark reviewer) — This demands the paper address problems outside its stated scope. The paper's stated scope is analyzing λ-skip connections and their effect on rank collapse, not a comprehensive comparison of mitigation strategies.

- **"The collapsed notion of rank collapse conflates expressivity, stability, and mere nondegeneracy"** (From harsh critic #3) — This is overly strong. The μ metric is the standard metric in the rank collapse literature (Dong et al. 2023; Wu et al. 2024a). The paper acknowledges limitations of this metric in Section 6. While the metric is imperfect, it is not a novel choice and its limitations are well-understood in the community.

- **"The paper should explicitly discuss negative λ"** (From spark reviewer) — This is a nice-to-have suggestion, not a weakness. The paper does show negative λ results empirically (Figure 2) and the theory applies symmetrically to |λ|.

## Novel Insights

The most novel insight is the connection between gating mechanisms (originally designed for selective memory in LSTMs) and rank collapse prevention in SSMs. The paper shows empirically (Figure 3) that gating serves a dual purpose: its multiplicative structure acts as an input-dependent skip connection that stabilizes representations, a function distinct from its original memory design motivation. This suggests that architectural components may serve stability functions beyond their intended purpose — a finding that could inform more principled SSM architecture design. Additionally, the observation that negative λ values can be more effective than positive ones for rank collapse prevention (Figure 2) suggests an interpretation as negative feedback control, which aligns with the paper's brief remark about λ as a "control gain" (Remark 4.1) but deserves further exploration.

## Suggestions

- **Soften the title and claims** to reflect that λ-skip connections are *sufficient* (not necessary, and not "the" component) for preventing rank collapse. A more accurate title might be "Lambda-Skip Connections: Sufficient Conditions for Preventing Rank Collapse in Sequence Models." The abstract should also be revised to use "sufficient conditions" rather than "guarantees."

- **Add a discussion of the input condition** μ(Y^(0))² ≥ b and its practical implications, particularly how it scales with depth K and what this means for very deep models.

- **Report learned λ values** in Table 1 experiments and compare them to the theoretical predictions from Theorem 4.1 to characterize the gap between theory and practice.

## Score and Decision

**Calibration**: Compared against papers with similar topics and quality profiles:
- "Mind the Gap" (rank collapse theory + limited experiments): scores 5,5,6,3 → Avg ~4.75, Reject
- "Residual Connections Harm" (skip connection parameter + conservative conditions): scores 6,5,5,6 → Avg ~5.5, Reject
- "GPAM" (rank collapse theory + limited empirical validation): scores 6,5,5,6 → Avg ~5.5, Reject
- "Hyper-Connections" (skip connection variant + strong empirical results): scores 5,8,6,6 → Avg ~6.25, Accept Poster

This paper shares characteristics with the rejected papers: genuine but limited theoretical contribution, overclaimed results, and modest experimental validation. It does not have the strong empirical results that distinguished "Hyper-Connections." The overclaiming in the title and abstract is the most serious issue — it promises a "general guarantee to prevent rank collapse" that is actually a conditional, conservative lower bound on a single metric, with crucial architectural components (gating) excluded from the theory. The SSM extension is novel and valuable, but the paper oversells its contributions.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>