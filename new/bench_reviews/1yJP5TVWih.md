## Summary

This paper extends the theory of rank collapse—where embedding vectors converge to a uniform token, degrading expressivity—from transformers to State Space Models (SSMs) using a unifying sequence model framework. The authors introduce “lambda-skip connections” (parameterized residual connections with strength λ) and prove a sufficient condition on λ that guarantees a lower bound on the rank collapse metric μ(Y^(K)) across both transformer and SSM architectures. They also analyze the necessity of skip connections by showing exponential/doubly-exponential rank collapse in ablated settings, and empirically demonstrate the role of both skip strength and gating mechanisms in mitigating rank collapse.

## Strengths

- **Unified theoretical framework for rank collapse across architectures:** The paper leverages the framework from Ali et al. (2024) / Dao and Gu (2024) to treat attention and SSM blocks uniformly via equation (6), enabling Theorem 4.1 to apply broadly. This unification is a meaningful contribution since SSMs had not been analyzed for rank collapse before.

- **First analysis of rank collapse in SSMs:** Theorem 4.3 establishes that selective SSMs without skip connections experience exponential rank collapse, and Theorem A.10 shows doubly exponential collapse when LayerNorm is also ablated—paralleling known results for transformers. This fills a genuine gap in the literature.

- **Comprehensive theoretical treatment:** The paper goes beyond sufficient conditions by studying necessity through ablation (Theorems 4.2, 4.3), analytical examples where specific λ values fail (Proposition 4.3.1), and tightness analysis (Proposition 4.3.2). This is a well-rounded theoretical package.

- **Novel empirical finding on gating mechanisms:** Figure 3 shows that gating mechanisms in Mamba-2 play a critical role in preventing rank collapse—connecting a component designed for memory with a structural stability issue. This is, to the authors' knowledge, the first such observation and provides valuable practical guidance.

## Weaknesses

### Major

- **Overstatement of the “prevention” guarantee (overclaiming):** The paper repeatedly states that Theorem 4.1 “prevents rank collapse” (Abstract, Sec. 4.1, Remark 4.1). What the theorem actually guarantees is μ(Y^(K))² ≥ a^K μ(Y^(0))² for some decay rate a. When a < 1—necessary for Mamba where S=1 (Remark 4.1)—this bound is fully compatible with exponential decay of μ to 0. Calling this “prevention of rank collapse” conflates slowing the rate of collapse with actually preventing it. Even for a = 1, the bound only ensures μ does not decrease monotonically, which says nothing about its absolute scale. The paper does note in Remark 4.1 that a^64 ≈ 0.993 for a = 0.9999, which is a finite-depth heuristic—but the formal claim is much stronger than what is proved. **This overclaiming undermines the core conceptual contribution.**

- **The sufficient condition on λ is extremely loose and not instantiated for any real model:** The condition λ² > a(SC_M + |λ|)² requires enormous |λ| once C_M reflects actual weight norms in deep models (e.g., C_M ≥ √N for transformers, or ∥W_A∥∥W_C∥_F for selective SSMs). The paper acknowledges in Section 5.1 that “our condition on λ in Theorem 4.1 is too conservative” and “in practice much lower values of λ are good enough to prevent rank collapse.” But this disconnect is not analyzed further—no concrete instantiation of C_M is computed for any standard model configuration, and no attempt is made to estimate these quantities from the checkpoints actually used in experiments. The “guarantee” is mathematically correct but practically vacuous without demonstrating feasibility. **The headline “guarantees for rank collapse prevention” is therefore misleading for practitioners.**

- **Significant gap between simplified theoretical assumptions and practical architectures:** The theoretical analysis excludes gating mechanisms (acknowledged in Section 6), uses a simplified LayerNorm (Eq. 4, no shifting), excludes MLPs, and restricts selective SSMs to A_t = αI (Assumption 4.1). Yet the primary experimental architecture (Mamba-2) uses gating, full LayerNorm, and input-dependent A_t. Figure 3 itself shows gating is empirically crucial. The exclusions amount to removing the very components that the most popular SSM relies on. While the paper acknowledges this in the Limitations section, the theoretical claims are presented as applying to the full architectures when they do not.

- **Experimental validation is limited and does not establish practical benefit:** (1) The main Mamba-2 experiments (Figures 1–2) modify a pre-trained model by removing gating and injecting λ-skip, creating an out-of-distribution architecture where observed behavior may be an artifact. No task performance is reported for these modified models. (2) Table 1 shows mixed results: Mamba-2 accuracy *drops* from 42.28% to 38.92% on Image LRA when using learnable λ, and Linear Transformer drops from 34.10% to 32.80%—yet the text claims “learning λ does not affect the performance and even outperforms.” (3) Only two small-scale tasks (LRA Image, MQAR) are used for training experiments, with no language modeling benchmarks. (4) No comparison is made with alternative rank collapse mitigation strategies (e.g., adjusting LayerNorm, MLP width, initialization), making it unclear that λ-skip is the right lever.

### Minor

- **The condition μ(Y^(0))² ≥ b in Theorem 4.1 is not discussed substantively, despite being non-trivial:** Since b = O(1/a^K · λ²/(λ² − a(SC_M+|λ|)²)), and a should be close to 1 for practical relevance (Remark 4.1), the factor 1/a^K can still be significant for deep models. This means the theorem only guarantees a non-trivial bound for inputs that are already sufficiently far from rank collapse—a circularity issue that is not acknowledged.

- **Assumption 4.1 (A_t = αI, fixed) restricts the SSM theory to a special case:** The paper shows experimental results for full Mamba (input-dependent A_t) but the theoretical results for selective SSMs require A_t to be fixed. The extension to input-dependent A is deferred without discussion of why the doubly-exponential result might or might not extend.

- **Mixed task performance contradicts the claim of no performance impact:** As noted above, Table 1 shows var.λ *decreases* performance on 3 out of 4 model-task pairs for Mamba-2 and Linear Transformer. The claim that “learning λ does not affect the performance and even outperforms” is not supported by the data.

### Trivial

- The definition of “collapse rate” a (Definition 4.1) describes the decay rate of the *lower bound* on the metric, not the metric itself. While the paper sometimes uses these interchangeably, this is a minor notational confusion since the bound-tightness analysis (Prop. 4.3.2) partially addresses it.

## Nice-to-Haves

- Extend Theorem 4.1 to incorporate gating mechanisms, even in simplified form, since Figure 3 shows they are empirically critical.
- Derive a tighter lower bound that exploits structural properties of M (e.g., row-stochasticity for attention, lower-triangular structure for SSMs) rather than relying solely on worst-case norm bounds.
- Evaluate on standard language modeling benchmarks and correlate μ trajectories with training dynamics (gradient norms, loss curves) to establish the causal chain from rank collapse to training failure to λ-skip as a fix.
- Compare λ-skip against alternative strategies for mitigating rank collapse (different initialization, MLP width, normalization variants) to establish that λ-skip is the most effective intervention.

## Removed Points

- **Harsh Critic point about experiments not isolating λ-skip causally (non-controlled baselines):** While modifying a pre-trained model by removing gating does create an artificial setting, this is standard practice in ablation studies to isolate specific architectural effects. The concern about “out-of-distribution architecture” is legitimate but not fatal—the experiment is designed to show the *mechanism* of λ, not to reproduce deployment conditions.
- **Harsh Critic point about the architecture-agnostic framework “hiding” dependencies on initialization, training, and data:** The paper explicitly acknowledges that C_M depends on weights and sequence length (footnote 2), and the theoretical framework is designed to be architecture-agnostic by design. The criticism that this makes the result “speculative” is too strong—the result is a valid sufficient condition; the issue is its looseness, which is already captured in the “loose bound” weakness above.
- **Neutral Reviewer’s point about the novelty of parametrized skip connections:** The paper itself cites He et al. (2016), Srivastava et al. (2015), and Bachlechner et al. (2020) as prior work on parameterized skip connections. The contribution is the *rank collapse analysis* tied to λ, not the architectural modification itself. The criticism that λ-skip “is not new” misses this distinction.
- **Spark’s suggestion to correlate rank collapse prevention with training dynamics:** While valuable, this is outside the stated scope of the paper, which focuses on establishing the theoretical guarantee and basic empirical validation, not the full training pipeline.
- **Harsh Critic’s point about the μ(Y^(0))² ≥ b condition being problematic for large K:** This is noted as a minor weakness, but the harsh critic’s claim that it “can be extremely large” needs more careful analysis. Since the denominator λ² − a(SC_M + |λ|)² must be positive (this is the main condition), and λ can be chosen freely, b is not independent of λ. The condition is interrelated, not simply impossible.

## Novel Insights

The most novel insight is the empirical connection between gating mechanisms and rank collapse prevention (Figure 3). Gating mechanisms were designed for memory management in recurrent models, but this paper is the first to document their role in structural stability against rank collapse—a finding that could inform future SSM architecture design. The unified analysis revealing that selective SSMs suffer doubly exponential collapse (like transformers) while LTI SSMs only suffer exponential collapse (Theorems A.10 vs. A.7–8) is also a notable structural distinction between these architecture classes.

## Suggestions

1. **Temper the language around “prevention”:** Replace claims of “preventing rank collapse” with “guaranteeing a lower bound on the rank collapse metric” or “slowing the rate of rank collapse.” Be explicit that a < 1 allows exponential decay. This is the single most important revision for intellectual honesty.

2. **Instantiate the theoretical condition concretely:** Compute C_M and S from the standard model configurations used in experiments (Albert, Mamba-2) and report what λ values satisfy Eq. (7) for various a. This would make the theory actionable and reveal the gap between theory and practice quantitatively.

3. **Address the mixed performance results honestly:** The claim that “learning λ does not affect performance” is contradicted by Table 1. Either report this honestly as mixed or provide statistical significance tests and more runs.

4. **Discuss what negative λ means architecturally:** The paper shows negative λ can increase μ, but negative skip connections fundamentally alter the residual structure (subtracting rather than adding). A brief discussion of training stability implications would strengthen the practical guidance.

## Calibration Comparison

- **Residual Connections and Normalization Prevent Oversmoothing (i8vPRlsrYu):** Scores 6,8,6,8, Accept. This paper provided formal proofs that residual connections and normalization prevent oversmoothing in GNNs with *tighter* bounds and more direct practical implications. The current paper has a similar structure but with looser bounds and a larger gap between theory and practice.
- **Mind the Gap: Spectral Analysis of Rank Collapse (X6xzYP2cMk):** Scores 5,5,6,3, Reject. This paper on rank collapse in transformers had similar issues with loose bounds and limited practical guidance. The current paper has stronger theory (lower bound vs. upper bound, more architectures) but similar looseness.
- **Residual Connections Harm Generative Representation Learning (cxKLRlsrYu):** Scores 6,5,5,6, Reject. This paper had a provocative claim (residual connections harm) with limited empirical support. The current paper has a less extreme claim but similar overclaiming relative to what is proven.
- **Simplifying Transformer Blocks (RtDok9eS3s):** Scores 8,8,6, Accept. This paper made architectural modifications with clear practical benefit and thorough evaluation—significantly stronger empirical validation than the current paper.

This paper sits between the rejected rank collapse/oversmoothing papers (which had looser theory and limited practical impact) and the accepted oversmoothing prevention paper (which had tighter bounds). The overclaiming about “prevention” and the loose sufficient condition are significant weaknesses that are not fully compensated by the novelty of extending to SSMs and the gating observation.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>