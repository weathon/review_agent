Now I have a comprehensive understanding of the paper, all three review inputs, and calibration papers. Let me synthesize the final review.

## Summary

This paper extends the theory of rank collapse from transformers to State Space Models (SSMs) using a unifying sequence model framework. The authors introduce "lambda-skip connections," a parameterized version of standard skip connections with strength parameter $\lambda$, and derive a sufficient condition on $\lambda$ (together with LayerNorm) that provides a lower bound on the rank-collapse metric $\mu(Y^{(K)})$ across transformers, LTI SSMs, and selective SSMs. They also show that removing skip connections leads to exponential or doubly exponential rank collapse, and empirically demonstrate the role of both $\lambda$ and gating mechanisms on a pretrained 2B-parameter Mamba-2 model.

## Strengths

- **First theoretical treatment of rank collapse in SSMs.** The extension of rank collapse analysis from transformers to SSM architectures (LTI and selective) is genuinely novel. Prior work on SSMs (e.g., "Demystifying the Token Dynamics of Deep SSMs") studied token dynamics but not rank collapse specifically. This fills a clear gap in the literature and is timely given the prominence of Mamba-family models.

- **Unified framework yielding general results.** By expressing both attention and SSM blocks as $O^{(k)} = M^{(k)}V^{(k)}$, the paper derives results applicable to multiple architectures simultaneously (Theorem 4.1). The exponential and doubly exponential collapse rates shown in Section 4.2 (Theorem 4.3, Theorem A.10) parallel known transformer theory and extend it to selective SSMs, which is technically non-trivial given the input-dependent $M^{(k)}$.

- **Empirical discovery of gating's role in rank collapse.** Figure 3 shows that removing gating mechanisms from Mamba-2 causes rank collapse, even with LayerNorm present. To the authors' knowledge, and confirmed by the literature, this is the first connection made between gating mechanisms (designed for memory) and rank collapse prevention—a finding with practical architectural design implications.

- **Empirical validation on large pretrained models.** Using a 2B-parameter pretrained Mamba-2 model (rather than toy models only) provides practical credibility. Figure 2 showing the relationship between $\lambda$ and the rank collapse measure at the final layer is informative and speaks to the question of how conservative the theoretical bounds are.

## Weaknesses

### Major

- **The "prevention guarantee" language overclaims what Theorem 4.1 delivers.** The abstract states the paper provides "guarantees for rank collapse prevention" and "a general guarantee to prevent rank collapse." However, the theorem shows $\mu(Y^{(K)})^2 \geq a^K \mu(Y^{(0)})^2$. For SSMs (including Mamba), Remark 4.1 explicitly states $a < 1$ is required, meaning the lower bound itself decays exponentially to 0 as $K \to \infty$. This does not constitute "prevention" of rank collapse in the infinite-depth sense defined in Section 3.1; it provides a lower bound on the rate of collapse. For finite practical depths where $\lambda$ is chosen appropriately, the bound can be close to non-trivial (e.g., $a^K \approx 0.993$ for $K=64$), but the paper's central claim uses language inconsistent with the mathematical result. This would be materially strengthened by either qualifying the language (e.g., "bounds the rate of rank collapse" rather than "prevents") or extending the analysis to conditions under which $a \geq 1$ is achievable.

- **The theorem's input condition $\mu(Y^{(0)})^2 \geq b$ is unanalyzed and potentially restrictive.** The quantity $b = \frac{1}{a^K} \frac{2\lambda N d S C_M}{\lambda^2 - a(SC_M + |\lambda|)^2}$ depends on model depth $K$ through the $1/a^K$ factor. For $a < 1$ (the SSM case), this factor grows exponentially with depth, potentially making $b$ larger than any realistic value of $\mu(Y^{(0)})^2$ for deep networks. The paper does not analyze whether this condition is achievable under standard initializations or data distributions, nor do the experiments report $\mu(Y^{(0)})$ versus the implied $b$. This gap makes it difficult to assess when the theorem is applicable in practice.

- **Theory–practice gap due to omitted architectural components.** The theoretical analysis excludes gating mechanisms (acknowledged in the limitations) and uses a simplified LayerNorm (normalization only, no shifting or learnable affine parameters). Yet Figure 3 shows gating is "crucial" for preventing rank collapse in Mamba, and Assumption 4.1 requires $A_t = \alpha I$ (input-independent), which does not hold for standard Mamba. This means the central theorem applies to a simplified version of the target architecture, and the experiments that validate the theory (Figures 1–2) also operate on models with gating removed. The result is that the theoretical framework and the empirical evaluation of it are both on architectures that differ from the most important practical models.

### Minor

- **Experiments do not quantitatively validate Theorem 4.1.** Figures 1–3 demonstrate qualitative trends (small $|\lambda|$ leads to more collapse; removing gating/LN hurts stability), but they do not estimate $C_M$, $S$, or verify that inequality (7) holds for tested $\lambda$ values. The acknowledged conservativeness of the bound (Section 5.1: "our condition on $\lambda$ in Theorem 4.1 is too conservative") deserves deeper analysis—why is it conservative, and how far is it from the empirically observed threshold?

- **The "necessity" analysis is suggestive but not conclusive.** Section 4.2.2 uses a $2 \times 2$ constructed system to show rank collapse can still occur for specific $\lambda$ values. While interesting, these toy examples are far from realistic architectures and do not establish that satisfying inequality (7) is the only way to prevent rank collapse (other mechanisms like gating, MLPs, or specific weight choices could serve). The section title ("Lambda-Skip Connection: Necessary to Prevent Rank Collapse?") and the question mark appropriately hedge this, but the paper could do more to clarify the gap.

- **Table 1 shows variable $\lambda$ degrades performance on LRA Image for Mamba-2** (38.92% vs. 42.28%), contrary to the claim that "learning $\lambda$ does not affect the performance." While the MQAR task improves for Mamba/Mamba-2, the mixed results warrant more discussion.

### Trivial

- The notation $C_V$ is used for both attention ($W_V$) and SSM ($I$) contexts, which could momentarily confuse readers but is clarified in the text.

## Nice-to-Haves

- Analysis of the $b$ threshold in Theorem 4.1: even a brief discussion of typical magnitudes under standard initializations would help readers assess practical relevance.

- Showing the learned $\lambda$ values per layer for the "var. $\lambda$" experiments in Table 1 to understand what the model converges to.

- Comparison of Theorem 4.1's predicted minimum $|\lambda|$ against empirically observed thresholds from Figure 2, to quantify the conservativeness gap.

- An experiment measuring rank collapse during training (not just on a pretrained model with post-hoc $\lambda$ changes) to connect theory to training dynamics.

## Removed Points

- **Experiments should include standard language modeling benchmarks.** The spark reviewer requested perplexity evaluation on WikiText-103 or similar. While this would be informative, the paper's stated contribution is theoretical analysis of rank collapse, not a demonstration of improved task performance. The LRA and MQAR experiments are sufficient to show that learnable $\lambda$ is compatible with training. (Moved: scope beyond paper's stated goal.)

- **The sample size of 32 Wikipedia excerpts is small.** This is a standard setup from Dong et al. (2023) and Wu et al. (2024a), which the paper follows. Increasing sample size is a nice-to-have but not a core flaw; the qualitative trends are clear and standard deviations are reported. (Moved: generic nitpick.)

- **Missing comparison with alternative rank collapse mitigation strategies.** The paper focuses on the skip connection mechanism; demanding comparison with other approaches (different initializations, alternative normalization schemes) is scope creep beyond the paper's contribution. (Moved: scope beyond paper's scope.)

- **No gradient analysis experiment.** While rank collapse is connected to vanishing gradients, the paper's focus is on the expressivity/rank collapse metric $\mu$, not on gradient flow. This would strengthen the paper but is not required for its stated contribution. (Moved: scope beyond paper's stated scope.)

- **Claim that the paper is the "first to provide a general guarantee to prevent rank collapse."** The harsh critic flags this as overclaiming because other works (Dong et al., Wu et al.) showed skip connections can prevent rank collapse in transformers. However, this paper generalizes it to the unifying framework including SSMs, and uses a different mechanism ($\lambda$-skip) with a formal condition—so the "first" claim for this specific framework is defensible. The more serious overclaim is "prevention" vs. "lower-bounding the decay rate," which is addressed as a Major weakness above.

## Novel Insights

The empirical finding that gating mechanisms in Mamba serve a dual purpose—originally designed for selective memory but also crucial for preventing rank collapse—is the most novel and practically actionable observation. This suggests that architecture designers working on next-generation SSMs should be aware that removing or weakening gating has implications for expressivity beyond memory, which has not been previously articulated. The theoretical insight that selective SSMs can exhibit doubly exponential rank collapse (paralleling transformer behavior) due to the quadratic input-dependence of $M^{(k)}$ also provides a meaningful structural understanding of why gating matters.

## Suggestions

- Reframe the claim from "prevention guarantee" to "lower bound on the rank collapse metric" or "bounding the rate of rank collapse," especially for the SSM case where $a < 1$. This would align the claims with the mathematics while preserving the paper's contribution.

- Compute and report $\mu(Y^{(0)})$ and the theoretical bound $b$ for the experimental models to demonstrate when (and whether) Theorem 4.1's preconditions are met, helping readers assess practical applicability.

- Acknowledge the Mamba-2 performance drop (38.92% vs 42.28%) in the text rather than claiming variable $\lambda$ "does not affect performance."

- In the limitations section, briefly discuss the direction of incorporating gating into the theoretical framework, even if this remains future work, to give readers confidence the authors understand the path forward.

## Score and Decision

**Calibration:** Compared against papers with similar profiles:
- "Mind the Gap" (rank collapse theory for simplified transformers, limited experiments): avg score 4.75, rejected
- "Setting the Record Straight on Transformer Oversmoothing" (oversmoothing theory with simplified architecture, some experiments): avg score 5.75, rejected
- "Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs" (provable theory, good experiments, novel normalization): avg score 7.0, accepted poster
- "Demystifying the Token Dynamics of Deep SSMs" (SSM theory with simplifying assumptions, practical improvements): avg score 7.5, accepted spotlight
- "StableSSM" (SSM theory, limited experiments, simplifying assumptions): avg score 5.33, rejected

This paper sits between "Setting the Record Straight" (rejected, oversmoothing theory for simplified transformers) and "Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs" (accepted, provable prevention theory for GNNs). The key differentiators are: (1) the novel extension to SSMs, (2) experiments on large pretrained models, but (3) overclaiming in the "guarantee" language and a significant theory-practice gap (gating excluded, simplified LayerNorm, conservative bounds). It is more novel than "Setting the Record Straight" but weaker in theoretical completeness than the GNN oversmoothing paper. The SSM extension is genuinely valuable, but the core theorem's practical relevance is undermined by the conservative bounds and unanalyzed input conditions.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>