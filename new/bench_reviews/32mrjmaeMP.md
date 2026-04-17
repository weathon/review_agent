## Summary

The paper proposes TAK (Task Arithmetic with KFAC regularization), a method that improves weight disentanglement in Task Arithmetic by reformulating representation drift regularization as a curvature approximation problem. By connecting the Jacobian Gram matrix (which governs representation drift) to the Generalized Gauss-Newton matrix, the authors leverage KFAC to construct a regularizer that does not require external task data during fine-tuning. They further introduce an accumulation heuristic that merges per-task KFAC factors into a single surrogate, achieving constant complexity in the number of tasks, and demonstrate state-of-the-art results on task addition and negation across vision and language benchmarks.

## Strengths

- **Principled theoretical connection (Sec. 3.1–3.2):** The derivation linking representation drift regularization to the GGN/KFAC framework is clean and well-motivated. By identifying the Jacobian Gram matrix (Eq. 3) with the GGN under squared loss, the paper enables the use of well-established second-order optimization machinery as a data-free surrogate, which is a non-trivial and elegant insight.

- **Meaningful practical advantages over τJp:** The fine-tuning stage of TAK does not require accessing external task data at each iteration — only pre-computed KFAC factors are needed. This is a real advantage over τJp, which requires a second forward-backward pass through the linearized model on external task data during every training step. As noted in Sec. 4, TAK requires "roughly one third of the training time of τJp" and avoids repeated data loading into GPU memory.

- **Robustness to scaling coefficient α (Fig. 4):** A practically important result is that TAK with α=1 performs competitively with tuned α, eliminating the need for held-out validation data for coefficient selection. This aligns well with the paper's goal of modularity and is convincingly demonstrated across multiple architectures.

- **Thorough efficiency and compression analysis (Fig. 6–8):** The paper provides detailed measurements of KFAC estimation time, training overhead, memory footprint, and compression strategies (quantization, pruning, block-diagonalization, SVD). This level of engineering detail is unusual and valuable for practitioners.

- **Evaluation breadth:** Results span multiple architectures (ViT-B/32, B/16, L/14, T5-base), two modalities (vision and language), and two task arithmetic operations (addition and negation), providing reasonable evidence of generality.

## Weaknesses

### Major:

- **The "dataless" framing is imprecise and potentially misleading.** The paper repeatedly states TAK "does not use external data" and is "inherently privacy-preserving" (Abstract; Sec. 1; Sec. 4). However, computing KFAC factors for task t *does* require samples from D_t (Eq. 3; Sec. 4 KFAC estimation: "using 128–256 examples per task"). The "dataless" property holds only for the fine-tuning stage of task t', after per-task KFAC factors have been pre-computed and distributed. This is a meaningful distinction from τJp (which needs external task data *during fine-tuning*), and KFAC factors as aggregate statistics are indeed more privacy-compatible than raw data — but the current framing obscures this. The solution envisioned in the Conclusions (releasing KFAC alongside pretrained weights) is only briefly suggested and not the evaluated scenario. The paper should more clearly articulate the data access model: in the evaluated setup, each task trainer has access to its own data and shares only KFAC factors, not that TAK operates with zero data dependencies whatsoever.

- **The accumulation heuristic (Eq. 8) lacks theoretical justification.** Replacing ∑_t λ_t(B^l_t ⊗ A^l_t) with (∑_t λ_t B^l_t) ⊗ (∑_t λ_t A^l_t) is a non-trivial approximation since Kronecker products do not distribute over sums. The paper acknowledges this is heuristic but provides only Tab. 3 as empirical validation on 8 tasks. No analysis of when or why this approximation degrades is offered (e.g., with highly heterogeneous tasks or many tasks). This matters because "constant complexity in T" is presented as a key contribution, yet the very mechanism enabling it is an unprincipled approximation whose failure modes are uncharacterized.

- **Performance gap on language tasks versus data-dependent method.** On T5-base (Fig. 3), τJp achieves 81.3% absolute accuracy vs. TAK's 78.7%, and the paper acknowledges: "leveraging data from other tasks (τJp) yields additional gains, suggesting that textual domains may still benefit from even more accurate curvature estimation." This raises the practical question of when data-free operation is worth the performance cost, and under what conditions TAK is the better choice. The paper does not provide a clear analysis of this tradeoff.

### Minor:

- **Task localization / OOD detection claims are under-supported.** The paper states TAK "enables a clear separation between training and out-of-distribution examples" and "suggests a natural use of our method for out-of-distribution detection" (Sec. 4), but the evidence is limited to qualitative histograms (Fig. 5) without quantitative OOD metrics (AUROC, FPR@95, etc.) or comparison to simple baselines (logit-based OOD scoring). This overreaches relative to the evidence provided.

- **Loss function mismatch.** The KFAC approximation uses the squared loss Hessian (∇²c = I) to connect the Jacobian Gram to the GGN (Sec. 3.2), while actual training uses cross-entropy. This means the curvature used for regularization does not reflect the true loss landscape. The paper does not analyze the impact of this mismatch, though it is noted that this is what makes the approximation data-free.

- **Non-linear regime applicability requires architectural constraints.** In the non-linear setting, TAK must be paired with attention-only fine-tuning to induce approximate linearization (Sec. 4). Without this architectural restriction, TAK's theoretical guarantees do not hold, limiting its direct applicability to standard full fine-tuning scenarios.

### Trivial:

- No standard deviations or confidence intervals are reported for any experiment, making it difficult to assess the significance of small numerical differences.

## Nice-to-Haves

- Evaluation with a larger number of tasks (20+) to stress-test the accumulation heuristic and validate the O(1) scalability claim in a more challenging regime.
- A comparison or combination with post-hoc merging methods (TIES, DARE, etc.) on the same set of TAK-regularized task vectors in main tables, not just in α-sweep analysis.
- Theoretical bounds or error analysis for the Kronecker factor accumulation approximation.
- Evaluation on a larger model (e.g., LLM-scale) where KFAC storage becomes more challenging, to validate the compression strategies in a setting where they matter most.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Harsh Critic's "dataless claim is entirely illusory":** The critic argues that TAK requires external task data in exactly the same way as τJp, making the privacy advantage "largely illusory." This is overstated. KFAC factors are aggregate statistics (covariance matrices of activations and gradients) that can be computed locally by each task's trainer and shared without revealing individual data points. This is analogous to how gradient accumulators are shared in federated learning. The critic's point about framing is valid, but the distinction between sharing aggregate KFAC factors vs. requiring direct data access during fine-tuning is real and meaningful. Kept as a Major weakness about imprecise framing but not as a fatal flaw.

- **Spark's "comparison with dataless post-hoc merging on equal footing":** This is an apples-to-oranges comparison. TAK modifies training to produce better task vectors, while post-hoc methods operate on already-learned vectors. These are complementary approaches, and the paper's Fig. 4 already shows their interaction. Demanding equal-footing competition in main tables misunderstands the different roles of in-training vs. post-hoc methods.

- **Harsh Critic's "disentanglement vs. generic regularization" concern requiring comparison with L2 or random low-rank penalties:** The paper already compares against diagonal GGN (which IS a simpler curvature approximation), showing KFAC's additional intra-layer correlations help. If the concern were merely "generic regularization," diagonal GGN would suffice. The comparison structure already addresses this.

- **Neutral Reviewer's "memory scalability for LLMs":** The paper explicitly addresses this with compression experiments (Fig. 7b) showing 87% memory reduction with only ~1 point accuracy drop. While testing on larger models would be nice, this is a scope-creep request for a paper that already provides practical solutions.

## Novel Insights

The key insight—that representation drift regularization under linearization reduces to a quadratic form involving the Jacobian Gram matrix, which is itself a GGN matrix and therefore amenable to KFAC approximation—is genuinely novel and productive. It transforms a data-dependent regularizer into a data-free one by recognizing that the curvature information can be pre-computed and shared as architectural metadata (KFAC factors) alongside pretrained weights. This reframing opens a practical path: model developers can release KFAC factors as a lightweight companion asset, enabling downstream users to apply disentanglement regularization without any data access.

## Suggestions

- Reframe the "dataless" claim more precisely: TAK is *data-free during fine-tuning* but requires pre-computed per-task curvature factors, which can be distributed without sharing raw data. Clearly articulate the data access model (e.g., each task trainer computes their own KFAC factors locally and publishes them alongside task vectors).
- Provide even a simple error bound or sensitivity analysis for the Kronecker accumulation approximation (Eq. 8), e.g., by characterizing how task similarity affects its quality.
- Add quantitative OOD metrics (AUROC at minimum) if the claim of "natural use for OOD detection" is to be maintained, or soften this claim to "qualitative evidence suggests task localization properties."
- Evaluate with ≥16 tasks to validate the constant-complexity claim under more diverse and numerous task settings.

## Score and Decision

**Calibration anchors:**
- τJp (Yoshida et al., 2025): directly compared, uses data-dependent regularization. Human scores: 5, 6, 5, 8 (avg ~6). This paper addresses τJp's main weakness (data dependency) with a principled approach, though introducing a new approximation (KFAC accumulation heuristic).
- TaLoS (Iurada et al., 2025): task-localized sparse fine-tuning. Human scores: 6, 5, 6, 5 (avg ~5.5). TAK has stronger theoretical grounding than TaLoS's sparsity heuristic.
- Submodule Linearity for Task Arithmetic: Human scores: 6, 6, 6, 6 (avg 6). Similar scope and contribution level.
- CAMEx (curvature-aware merging): Human scores around 5-6. Similar methodological lineage.

This paper makes a solid, well-motivated contribution: the KFAC-based data-free regularizer for task arithmetic is technically sound, empirically effective (especially in vision), and addresses a real limitation of prior work. Its main weaknesses — imprecise "dataless" framing, insufficient theoretical grounding for the accumulation heuristic, and a language-task performance gap — are notable but not fatal. It sits squarely in the 5.5–6.5 range compared to its closest anchors. Slightly above TaLoS (5.5) because of stronger theory, roughly on par with τJp but with a different tradeoff (data-free vs. data-dependent). The accumulation heuristic concern and imprecise framing prevent it from scoring higher.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>