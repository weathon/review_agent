Now I have a good calibration set. Let me synthesize my final review.

**Key assessment:**

The paper makes a genuine contribution by extending rank collapse theory from transformers to SSMs via a unifying framework. The unifying formulation (Eq. 6) and Theorem 4.1 are real, technically correct results. The necessity results (Theorems 4.2, 4.3) are also meaningful. The tightness result (Prop. 4.3.2) is a careful addition.

However, there are significant issues:

1. **Overclaiming in framing**: The title says "prevents rank collapse" and the abstract says "guarantees for rank collapse prevention." But the main theorem only guarantees μ(Y^(K))² ≥ a^K μ(Y^(0))² with a < 1 (for SSMs). This is a *rate control* result — for finite depth, it provides a lower bound, but in the infinite-depth limit (which is the paper's own definition of rank collapse), a^K → 0, so rank collapse still occurs asymptotically. The paper does qualify this in the body text ("finite layers setting"), but the title and abstract do not carry this qualification.

2. **The b condition**: b = (1/a^K) · factor, which grows exponentially with K. The paper doesn't discuss this dependence. For Mamba's specific case (a=0.9999, K=64), the factor is mild (1.006), but the conceptual issue remains and should be discussed.

3. **Experimental validation is indirect**: The experiments show that larger |λ| reduces rank collapse (trivially expected), but don't validate the specific conditions or tightness of Theorem 4.1. Table 1 results are mixed.

4. **Simplified model**: LayerNorm without centering and omission of gating are significant simplifications, though they are acknowledged.

Now for scoring calibration:
- High-scoring anchors (7+): Papers like "Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs" (7.0) and "Simplifying Transformer Blocks" (7.33) — these have strong empirical validation and tight theory-practice alignment.
- Medium anchors (4-6): Papers like "Residual Connections Harm Generative Representation Learning" (5.5) — overclaims somewhat, limited theory, mixed experiments; "Quantifying Memory Utilization with Effective State-Size" (5.6) — good idea but limited validation, presentation issues.
- Low anchors (<3): Papers like the Neural ODE generalization bounds (3.0) — vacuous bounds, overclaims; and the MIPS paper (2.0) — overclaims, trivial theory.

This paper is better than the low-scoring anchors — it has genuine theoretical contributions (the unifying framework, the necessity results, the tightness result). But it overclaims in framing like the medium-scoring anchors. It's worse than the high-scoring anchors because of the gap between claims and what the theorem actually provides. The empirical validation is also weaker than the high-scoring papers.

I'd place it around 5.0-5.5: it has real contributions but significant overclaiming and insufficient experimental validation.

Let me finalize my review.

## Summary

This paper extends the theory of rank collapse—previously studied primarily in transformers—to State Space Models (SSMs) by using a unifying framework that captures both architectures. The main theoretical contribution is Theorem 4.1, which provides a λ-dependent lower bound on the rank collapse metric μ(Y^(K))² ≥ a^K μ(Y^(0))², along with a sufficient condition on λ for this bound to hold (Eq. 7). The paper also provides necessity results showing exponential collapse without skip connections and doubly exponential collapse without both skip connections and LayerNorm. Analytical examples and experiments on a 2B-parameter Mamba-2 model complement the theory.

## Strengths

- **First unified rank collapse analysis for transformers and SSMs:** The unifying formulation Y^(k) = D^(k)(M^(k-1)Y^(k-1)C_V + λY^(k-1)) (Eq. 6) and the sufficient condition λ² − a(SC_M + |λ|)² > 0 (Eq. 7) that holds across both architecture families is a genuine conceptual contribution, extending prior transformer-only analyses (Dong et al., Wu et al.) to SSMs.

- **Insightful necessity results:** Theorems 4.2 and 4.3 establish that selective SSMs without skip connections suffer exponential rank collapse (even with LayerNorm), and Appendix Theorem A.10 shows doubly exponential collapse when both skip connections and LayerNorm are ablated. The identification that the quadratic input-dependence of M^(k) in selective SSMs causes the doubly exponential behavior mirrors the mechanism in attention-only transformers, providing a crisp unifying insight.

- **Tightness of the lower bound (Proposition 4.3.2):** The construction of a concrete system where μ(Y^(k))² = O(a^k μ(Y^(0))²) for λ satisfying Eq. 7 proves the bound cannot be improved without additional assumptions. This is a careful completeness result that distinguishes the analysis from a loose bound.

- **Empirical discovery connecting gating to rank collapse:** Figure 3 shows that removing gating from Mamba-2 causes collapse even with LayerNorm, while gating without LayerNorm leads to instability. The identification of gating mechanisms—originally designed for memory—as rank collapse prevention mechanisms is novel and worth reporting.

## Weaknesses

### Fatal
None.

### Major

- **Overclaiming "prevention" vs. "rate control":** The title ("the architectural component that **prevents** rank collapse"), abstract ("guarantees for rank collapse **prevention**"), and much of the framing promise prevention, but Theorem 4.1 only establishes μ(Y^(K))² ≥ a^K μ(Y^(0))² with a < 1 for SSMs. Since a^K → 0 as K → ∞, this is consistent with the paper's own definition of rank collapse (convergence to rank-1 in the infinite-depth limit, Section 3.2). What the theorem provides is a guarantee that collapse is slowed from doubly-exponential to single-exponential rate—a meaningful but qualitatively different claim from "prevention." The paper does qualify this as "finite layers setting" in the body (Section 4.1, line 128), but this critical qualifier is absent from the title and abstract. This gap between claim and result is the paper's most significant issue.

- **The b condition's exponential dependence on K is unacknowledged:** The theorem's input precondition μ(Y^(0))² ≥ b, where b = (1/a^K) · (2λNdSC_M)/(λ² − a(SC_M + |λ|)²), grows exponentially with depth since a < 1. For Mamba's specific parameters (a=0.9999, K=64), the factor 1/a^K ≈ 1.006 is mild, but the paper never discusses this K-dependence, its practical implications, or whether it becomes restrictive for deeper networks. This omission is significant because it determines when the theorem's guarantee actually applies.

- **Experimental validation is indirect and results are mixed:** Figures 1–2 show that increasing |λ| reduces rank collapse, but this is the trivially expected outcome of amplifying the identity path relative to the main mechanism. The experiments do not test whether Theorem 4.1's specific conditions are tight, whether the bound is accurate, or whether the b condition is necessary. Table 1 on learnable λ shows mixed results: Mamba-2 on Image drops from 42.28% to 38.92%, Linear Transformer on Image drops from 34.10% to 32.80%, while MQAR shows improvements for Mamba (81.5→85.3) and Mamba-2 (97.3→99.1). The paper's claim that "learning λ does not affect the performance and even outperforms" overstates what the data show—only 3 of 8 comparisons show improvement.

### Minor

- **Theoretical analysis excludes gating and uses simplified LayerNorm:** The paper analyzes a simplified LayerNorm (normalization without centering, following Wu et al. 2024a) and omits gating mechanisms (the primary skip-path in Mamba). Both simplifications are acknowledged, but they mean the theoretical results apply to a partial architecture that differs from how these models are actually used. The experiments partially address this (Figure 3), but the theory-practice gap remains.

- **Assumption 4.1 (A_t = αI, input-independent) limits generality:** Theorem 4.3 restricts A_t to be input-independent, which significantly limits applicability to selective SSMs where input-dependent A_t is a key feature. The paper notes this and Figure 1 (λ=0) provides some empirical validation, but the necessity result doesn't fully cover the selective setting.

- **Analytical examples (Proposition 4.3.1, N=2, d=2) are very low-dimensional:** While illustrative, these examples are too small to establish whether the λ threshold generalizes or whether Eq. 7 is tight in practical settings.

## Trivial
None.

## Nice-to-Haves

- Validation of Theorem 4.1's bound on randomly-initialized models with controlled C_M and S values, to test tightness vs. practice.
- A plot of b vs. K for typical parameter values, making the input condition's depth-dependence transparent.
- Training-from-scratch experiments comparing architectures with and without the λ-skip condition satisfied.
- Incorporation of MLPs into the lower bound, as the paper acknowledges they would improve it.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Theorem 4.1 does not prevent rank collapse at all" (Harsh Critic, Structural):** The harsh critic claims the theorem is "entirely consistent with rank collapse occurring." This overstates the issue. The paper explicitly says "finite layers setting" and for any fixed K, a^K > 0 guarantees μ > 0. The theorem does provide a meaningful guarantee for finite-depth networks—it just doesn't prevent asymptotic collapse as K → ∞. The issue is overclaiming, not vacuity.

- **"b grows exponentially and makes the condition vacuous" (Harsh Critic, Structural):** While b does depend on 1/a^K, for the paper's own concrete numbers (a=0.9999, K=64), this factor is ≈1.006, making the condition essentially irrelevant. The claim that it is "vacuous for practical network depths" is too strong—it is mild for the specific case the paper analyzes. The real concern is the lack of discussion, not vacuity in the presented case.

- **"Experiments confirm the trivial observation that amplifying identity path reduces collapse" (Harsh Critic, Evidential):** While partially true, this dismisses the value of empirical confirmation. Showing that λ=1 (the standard choice) leads to rank collapse in Mamba-2 while larger |λ| prevents it is not entirely trivial—it reveals that the default architecture may be suboptimally configured. The criticism that specific conditions aren't validated is valid though.

- **"Pre-trained model ablation doesn't validate theoretical claims" (Harsh Critic, Section 5.2):** While using a pre-trained model for ablation has limitations, removing components from a trained model and observing rank collapse is a legitimate empirical investigation that provides useful qualitative evidence.

- **"Missing experiments on b condition, tightness of bound, training-from-scratch" (Harsh Critic, Missing Experiments):** These are valid suggestions for strengthening the paper but are beyond what is strictly required. Moved to Nice-to-Haves.

- **Simplified LayerNorm criticism (Harsh Critic, Section 3.1):** The paper clearly states this follows Wu et al. (2024a). This is a standard theoretical simplification that is transparent and justified.

- **"Missing related works" suggestions:** Removed per rules.

## Novel Insights

The most novel insight emerging from this analysis is the parallel between the doubly-exponential rank collapse mechanism in attention-only transformers and selective SSMs: in both cases, the quadratic input-dependence of the mixing matrix M^(k) (through softmax(QK^T) in transformers vs. YWCW_B^TY^T in selective SSMs) drives the doubly-exponential decay, while LayerNorm converts it to single-exponential. This structural analogy between two seemingly different architectures is a genuine conceptual contribution that transcends the specific technical results.

## Suggestions

- Revise the title and abstract to accurately reflect the result as "rate control" or "mitigation" of rank collapse rather than "prevention," or explicitly add the "finite-depth" qualifier throughout.
- Add a discussion paragraph analyzing b's dependence on K and providing concrete numbers for when the input condition becomes restrictive.
- Soften the claim in Section 5.1 about learnable λ not affecting performance—acknowledge the mixed results honestly.
- Consider adding at least one experiment that directly tests whether Theorem 4.1's predicted λ threshold aligns with observed collapse thresholds.

## Score and Decision

**Calibration anchors:**
- **High (7+):** "Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs" (avg 7.0) — similarly extends rank collapse theory to a new architecture family (GNNs), but has tighter alignment between theory and experiment, including a practical method (GraphNormv2). "Simplifying Transformer Blocks" (avg 7.33) — strong signal propagation theory + thorough experiments. This paper is weaker than both due to the overclaiming gap and limited experimental validation.
- **Medium (4-6):** "Residual Connections Harm Generative Representation Learning" (avg 5.5) — proposes modifying skip connection strength, has mixed experimental results, overclaims somewhat. Closely comparable: both propose modifying the skip connection and both have mixed empirical results, though this paper has a more substantial theoretical framework. "Quantifying Memory Utilization with Effective State-Size" (avg 5.6) — good conceptual contribution for SSMs, limited empirical validation, presentation issues. This paper has stronger theory but a larger claim-reality gap.
- **Low (<3):** Neural ODE generalization bounds (avg 3.0) — vacuous theoretical bounds with overclaims. This paper's bounds are not vacuous (they provide real rate control), so it is clearly above this band.

The paper sits in the medium band, closer to 5 than 6. It has genuine theoretical contributions (unified framework, necessity results, tightness), but the overclaiming in the framing is significant, the experimental validation is indirect, and the b-dependence issue is unacknowledged. It is somewhat stronger than the 5.5 "Residual Connections Harm" paper because of its more substantial theoretical contribution, but the overclaiming pulls it back down.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>