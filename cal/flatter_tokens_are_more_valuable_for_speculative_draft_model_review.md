=== CALIBRATION EXAMPLE 62 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Flatter Tokens Are More Valuable for Speculative Draft Model Training" is accurate but informal. More importantly, the abstract's key claim — "over 2× training speedup using only 50% of the data, while keeping the final model's inference speedup within 4% of the full-dataset baseline" — is essentially a consequence of the trivially true fact that training on less data takes less time (~linearly). The true, non-trivial claim is that SFDD's selected 50% is *better* than a random 50%. The abstract should lead with this distinction more clearly.

---

### Introduction & Motivation

The motivation is well-constructed: the paper correctly identifies the mismatch between KL-divergence training (standard KD) and the L₁-based acceptance rate metric in SD. The observation that simply switching to an L₁ loss doesn't consistently help (citing Zhou et al., 2023) is a good entry point for a data-centric angle.

However, the introduction states flatness "depends only on the target model and can be computed offline, without warming up a draft model." While true and practically useful, this is also a significant limitation: because the metric is entirely target-model-driven, it ignores the draft model's current state. Tokens where the draft model has already converged to the target are equally rated as tokens where the draft is far from the target. This limitation is acknowledged only partially in Section 4.1 ("already closely align... rendering subsequent updates negligible") but is not fully analyzed.

The contributions are stated clearly and are roughly commensurate with what the paper delivers.

---

### Theoretical Analysis (Section 3.2) — Most Critical Section

This is the central theoretical claim, and it has several weaknesses worth scrutinizing carefully.

**The Gaussian toy model.** The entire theoretical derivation (Section 3.2 and Appendix A) restricts both `p` and `q` to univariate Gaussian distributions. Real LLM output distributions are discrete probability vectors over vocabularies of size V ≈ 32,000–128,000, are highly skewed (long-tail), and often contain most mass on 1–3 tokens. The Gaussian assumption is analytically convenient but a significant idealization. The paper argues that "alternative measures such as L_p norm, JS divergence... would yield similar conclusions," but this is stated without proof or even a simulation cross-check. The robustness argument in Appendix F.3 (Exponential and Half-Normal) adds some generality, but these are still continuous parametric families, not the actual discrete distributions found in LLMs.

**The mapping from Gaussian variance to discrete cosine flatness.** Appendix B establishes that, for a *discretized Gaussian*, cos(p, U) ∝ σ^(1/2) as V → ∞. But this derivation requires the assumption `L ≫ σ`, i.e., the distribution must be strongly *concentrated* (small σ relative to the vocabulary window L). This is the regime of **low flatness** — essentially, the theorem is most cleanly proven for precisely the tokens the paper argues contribute *least* to training. For the high-flatness tokens that are the focus of SFDD, the approximation is less clean. The connection between the Gaussian-variance theory and the discrete flatness metric is therefore somewhat circular and needs clarification.

**The budget constraint interpretation.** Equation 3 models one gradient step as:
> r* = argmin D_KL(p‖r) s.t. D_KL(r‖q) ≤ θ

This is a mathematically elegant abstraction, but it presupposes a very specific optimization geometry that does not correspond to how gradient descent actually works (e.g., via AdamW on a cross-entropy loss over a neural network). The paper states "our insights do not depend strongly on the specific choice of budget measurement," but this is not demonstrated — the simulation uses only this one budget measure, and the claim that KKT-solutions here reflect real training dynamics is asserted rather than validated.

**Fixed q assumption.** The derivation fixes `q = N(0,1)` across all tokens in the simulation. In practice, different tokens have different draft model outputs, and these change over training. The finding that "high target variance → higher ΔL₁" is shown for a fixed specific q, but the paper uses this to justify a static, pre-training data selection criterion. No experiment tracks whether the initial flatness ranking (before training) remains predictive of token value throughout training.

---

### Empirical Validation of the Theoretical Analysis (Section 4.1)

The validation (Figure 2) is an important bridge between theory and practice, but it has a critical weakness: **only 10 samples are randomly selected for the training-dynamics analysis.** Conclusions drawn from 10 samples (with token-level smoothing) cannot be considered reliable evidence that flatness generalizes as a token-value indicator across the full training set. Even if the trend holds on average over the full dataset, the variance over such a small sample could easily swamp the effect. The authors should at minimum report results on a larger validation set (e.g., 100–1000 samples) with error bars.

The comparison between flatness and entropy (Figure 2d) demonstrates that flatness filters out more "already-saturated" tokens (lower |ΔL₁| in the bottom 35%). This is the most compelling empirical evidence supporting flatness over entropy. However, the magnitude of the gap `g` (the difference in |ΔL₁| between the two methods' worst tokens) is not contextualized in absolute terms — how large is this gap relative to the mean |ΔL₁|? This context is needed to assess whether the distinction is practically meaningful.

---

### Method (Section 4.2–4.3)

**Token-to-sample aggregation.** The transition from a validated token-level insight to sample-level selection via averaging (Equation 8) is presented without theoretical justification. Averaging flatness across all tokens in a sample introduces a fundamental problem: a sample where half the tokens are very high-flatness and half very low-flatness gets an intermediate score, potentially losing high-value sequences that happen to contain some redundant tokens. Alternative aggregation strategies (e.g., max, percentile-based, or a two-phase selection that first filters tokens then selects samples) are not considered or ablated.

**Relationship to entropy.** Appendix F.2 demonstrates that entropy-based sorting yields nearly identical training-dynamics curves to flatness-based sorting (Figure 5 vs. Figure 2), as both measure distance to the uniform distribution. The paper notes the relationship theoretically (both relate to KL divergence from uniform). Given this similarity, the key question for practitioners is: when does flatness meaningfully outperform entropy? The quantitative advantage in Table 1 (2.41× for SFDD vs. 2.20× for Entropy) is non-trivial, but the mechanism explaining *why* cosine similarity beats entropy for this specific task remains unclear. The paper attributes it to entropy's logarithmic scale being less sensitive to distributional tails, but this is not empirically demonstrated.

---

### Experiments (Section 5) — Limited Scope

**Single model and single framework.** All experiments use exclusively LLaMA3-8B-Instruct with EAGLE-2. There are no experiments on other target models (e.g., Mistral, LLaMA2, Vicuna, Qwen) or other SD frameworks (e.g., MEDUSA, EAGLE-1, DistillSpec). The applicability of SFDD outside EAGLE-2 is entirely unclear, and the EAGLE-2 architecture (a single lightweight transformer layer as the drafter) has particular properties (feature-level conditioning on the target) that may make this selection especially effective in ways that don't generalize.

**Single training dataset.** Training is exclusively on ShareGPT. This is a conversational dataset with specific distributional properties. Whether SFDD transfers to other domains (code, math, scientific text) is unknown and relevant given that SD is used across many task types.

**Statistical significance.** No variance estimates, confidence intervals, or significance tests are reported for any comparison. The differences between SFDD and the second-best method (Top-1 Probability, 2.41× vs 2.23×) are presented as definitive, but without multiple training runs, these differences could be within noise. At minimum, the empirical validation (Figure 2) uses 10 samples, and the main results use one training run.

**Training efficiency framing.** The headline claim of "2× training speedup" at 50% retain ratio is partly driven by the trivial relationship between dataset size and training time. What is genuinely non-trivial — and should be the focus — is that SFDD's 50% outperforms Random's 50% in inference speedup AND requires less training time per epoch due to "enhanced batching efficiency." The latter mechanism (high-flatness samples may be shorter or have different structural properties enabling better batching) is only briefly mentioned and not analyzed.

**At 70% retain ratio, SFDD appears to match or slightly exceed "No Filter"** on several benchmarks (Alpaca: 2.77× vs 2.71×; GSM8K: 2.71× vs 2.71×). The paper attributes this to "removing noisy or redundant data," which is potentially the most interesting finding in the paper (that SFDD acts as a regularizer, not just a speed hack). This deserves more careful investigation rather than a brief mention.

---

### Missing Ablations

1. **What is being filtered?** The paper does not analyze what types of tokens/samples are filtered by SFDD. Are they factual questions, short answers, repeated patterns? Understanding the content of high/low flatness samples would greatly strengthen the narrative.

2. **Alternative aggregation.** No ablation of max vs. mean vs. percentile aggregation for sample-level flatness.

3. **Dynamic selection.** The metric is computed once before training. Selecting data dynamically (e.g., recomputing flatness partway through training as q evolves) is a natural extension that would also be an informative ablation.

4. **Generalization to other SD frameworks.** Even a single comparison on EAGLE-1 or DistillSpec would substantially strengthen the generalizability claim.

---

### Writing & Clarity

Section 3.2 would benefit from a clearer demarcation between what is theorem (the Gaussian KKT solution) and what is conjecture (that this Gaussian analysis reflects behavior in real discrete LLM distributions). Currently these are blended together, which may mislead readers into taking the theoretical results as more directly applicable to practice than they are.

---

### Limitations & Broader Impact

There is **no explicit limitations section**, which is a notable gap for an ICLR submission. Critical limitations not acknowledged:

- **Static selection ignores training dynamics.** Since q changes throughout training, tokens that are uninformative early on may become informative later (or vice versa).
- **Target model dependency.** Flatness scores are specific to the chosen target model. If the target model is updated or swapped, SFDD must be recomputed from scratch.
- **Potential domain mismatch.** If training data after filtering is significantly domain-shifted (e.g., over-representing ambiguous/creative content), the draft model may develop skewed behavior on confident/factual domains.
- **Scope is narrow.** The method applies only to train-based SD with knowledge distillation. It is inapplicable to train-free SD methods, which are widely used.

---

### Overall Assessment

This paper makes a genuine and practically useful contribution: a data-centric approach for efficiently training speculative decoding draft models, grounded in a novel per-token importance criterion. The flatness metric is simple, cheap to compute, and empirically outperforms a reasonable set of baselines across five downstream benchmarks. The public code release and clear experimental setup are commendable.

However, the paper oversells the theoretical depth of its contribution. The Gaussian toy model is far removed from real LLM distributions; the connection to discrete cosine similarity is established only asymptotically under conditions that favor the low-flatness regime; and the sample-level aggregation is an unjustified heuristic. More critically, all experiments are confined to a single target model (LLaMA3-8B-Instruct) and a single training dataset (ShareGPT), leaving the generalizability of SFDD entirely undemonstrated. The absence of statistical significance testing makes it impossible to assess whether the performance differences over strong baselines like entropy-based selection are real or within noise. As an ICLR submission, the paper sits at the boundary: the practical contribution is solid but the theoretical claims are insufficiently rigorous and the empirical scope is too narrow to support a confident conclusion about the method's breadth of applicability. Expanding to at least two target models and adding confidence intervals over multiple runs would substantially strengthen the work.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses the training inefficiency of draft models in Speculative Decoding by proposing a data-centric filtering strategy called Sample-level-flatness-based Dataset Distillation (SFDD). The authors argue that tokens where the target model produces flatter (more uniform) predictive distributions offer greater theoretical potential for reducing the L1 discrepancy governing acceptance rates. Empirically, SFDD achieves over 2× training speedup using 50% of the data while maintaining inference acceptance rates within 4% of the full-dataset baseline on the EAGLE-2 framework.

### Strengths
1.  **Significant Practical Efficiency Gains:** The core contribution yields tangible resource savings. Experiments in Section 5.4 and Table 4 demonstrate a **2.02× reduction in training time** at a 50% retain ratio compared to no filtering, with minimal degradation in inference speedup (Table 1 shows only a 0.08× drop in average speedup compared to the "No Filter" baseline).
2.  **Solid Theoretical Motivation:** Section 3.2 provides a mathematically grounded derivation (using a KL-constrained update on Gaussian distributions) linking the reduction in the L1 norm (acceptance rate proxy) to the variance/flatness of the target distribution. This moves beyond empirical heuristics to provide a theoretical reason *why* specific samples are more valuable.
3.  **Comprehensive Empirical Validation:** The paper evaluates SFDD against seven other common metrics (Entropy, Top-1, Margin, etc.) and across five downstream tasks (GSM8K, Alpaca, MT-Bench, CNN/DM, NQ). The ablation studies on retain ratios (Table 2) confirm the method's robustness, showing consistent superiority over random filtering even at extreme ratios (5%).

### Weaknesses
1.  **Novelty Overlaps with Entropy-Based Filtering:** The proposed "flatness" metric is explicitly recognized in Appendix F.2 (Section F.2) to be highly correlated with Shannon entropy and minimize KL divergence to uniform. In standard active learning or distillation, high-entropy samples are already known to be more informative. The paper must more sharply articulate why cosine similarity offers a computational or theoretical advantage over directly maximizing entropy in the SD context, beyond just the specific derivation provided.
2.  **Reliance on Gaussian Assumption:** The theoretical analysis (Equations 4-5) assumes token distributions can be modeled as Gaussians to derive the relationship between update direction and variance. While Appendix B provides an asymptotic justification for discrete spaces, the main text should more critically discuss the limitations of this proxy. Real LLM logits are often multi-modal or heavy-tailed, and the Gaussian assumption may not capture the true "valley" structure of the loss landscape in discrete token space.
3.  **Framework Specificity:** All experiments are conducted using the **EAGLE-2** framework (Section 5.1). It is unclear if the benefits of SFDD generalize to other SD training methods, such as DistillSpec (Zhou et al., 2023), which minimizes KL divergence differently, or Medusa, which uses different structural heads. Without cross-framework validation, the method appears somewhat tied to EAGLE's specific feature alignment strategy.

### Novelty & Significance
**Novelty:** The paper introduces a novel application of data filtering specifically tailored to the constraints of Speculative Decoding training (maximizing acceptance rate rather than standard loss). While the concept of training on "harder" or more uncertain examples is established in Knowledge Distillation, linking this directly to the L1 norm acceptance metric and proposing a specific cosine-similarity proxy for SD is a distinct contribution. The novelty lies in the application and the specific metric design for this objective, even if the underlying principle of uncertainty-based selection is known.

**Significance:** The significance is high for the community, as training draft models is becoming a bottleneck for scalable SD inference. A method that cuts training costs by 50% without impacting deployment performance is practically impactful. It encourages a shift in how SD training datasets are curated, moving from full-data training to more efficient, selection-based pipelines.

### Suggestions for Improvement
1.  **Clarify Metric Comparison:** Explicitly compare the computational efficiency and performance of SFDD against a pure Entropy-based filter in the main text (not just Appendix F.2). If SFDD outperforms entropy, explain specifically why cosine similarity captures "flatness" better than entropy for *this specific objective*.
2.  **Expand Framework Testing:** Include at least one experiment on a different SD framework or training objective (e.g., vanilla distillation or a different draft architecture) to demonstrate that the "flatness" heuristic is not an accidental fit for EAGLE-2's training dynamics.
3.  **Strengthen Theoretical Discussion:** In the main text (Section 3.2), add a paragraph explicitly discussing the limitations of the Gaussian proxy for discrete vocabularies, acknowledging scenarios where multimodal distributions might invalidate the theoretical insight.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Evaluate on larger target models (e.g., LLaMA-70B) to verify the flatness metric remains predictive and computationally feasible at scale, as ICLR reviewers expect scalability beyond 8B models.
2. Benchmark against other speculative decoding frameworks (e.g., Medusa, Lookahead) to ensure the method is not overfitted to EAGLE's specific draft head architecture.
3. Provide statistical significance testing (multiple seeds) for the performance gap between Flatness and Entropy in Table 1, as current differences may fall within random variance.
4. Measure data selection overhead on larger corpora (e.g., FineWeb) to substantiate the claim that target model forward passes are negligible relative to training costs.

### Deeper Analysis Needed (top 3-5 only)
1. Reconcile the contradiction between Appendix F.2 (entropy and flatness show similar training dynamics) and Table 1 (flatness significantly outperforms entropy); clarify the unique signal flatness provides.
2. Analyze the semantic composition of filtered data (e.g., code, math, reasoning) to ensure high-flatness selection does not inadvertently bias the draft against high-certainty domains.
3. Quantify the direct correlation between token flatness and actual inference acceptance rates to validate the theoretical link between distribution shape and SD performance.

### Visualizations & Case Studies
1. Plot acceptance rate distributions per token type for Full vs. SFDD models to reveal if specific categories (e.g., punctuation, common words) are disproportionately degraded.
2. Provide t-SNE clusters of retained vs. discarded samples to visually expose whether SFDD removes semantically distinct subsets of the data.
3. Show the distribution of flatness scores across different datasets to indicate if the threshold $\tau$ needs dataset-specific tuning or generalizes universally.

### Obvious Next Steps
1. Include an ablation on dynamic thresholding during training to prove static pre-filtering is sufficient and not suboptimal compared to curriculum learning.
2. Add a cross-domain evaluation (e.g., Code vs. Chat) to demonstrate that flatness importance does not vary drastically across task types.
3. Provide a detailed breakdown of why token-level filtering fails to save time on modern hardware, as this claim relies heavily on specific implementation constraints.

# Final Consolidated Review
## Summary

The paper proposes Sample-level-flatness-based Dataset Distillation (SFDD) for efficient training of speculative decoding draft models. The key insight is that tokens where the target model produces flatter (more uniform) predictive distributions yield greater per-step reductions in the L1 discrepancy that governs acceptance rates. The method computes a cosine-similarity-based flatness metric offline using only the target model, then filters training samples to retain high-value data. Experiments on EAGLE-2 with LLaMA3-8B-Instruct demonstrate that training on 50% of data selected by SFDD achieves inference speedups within 4% of full-data training while reducing training time by over 2×.

## Strengths

- **Clear theoretical motivation linking distribution shape to acceptance rate improvement:** The paper correctly identifies that standard knowledge distillation minimizes KL-divergence, while speculative decoding's acceptance rate relates to L1-norm distance. The derivation (Section 3.2, Appendix A) provides a principled starting point by showing that, under Gaussian assumptions, higher target variance correlates with larger ΔL1 improvements per training step.

- **Practical efficiency gains with minimal performance loss:** Tables 1 and 2 demonstrate that at 50% retention, SFDD achieves an average speedup of 2.41× compared to 2.49× for full-data training (a 3.2% gap). The method consistently outperforms random filtering and other selection metrics (Entropy, Top-1 Probability, Margin, Energy Score, PPL) across five downstream tasks.

- **Simple, computationally cheap metric:** The flatness metric requires only a single forward pass of the target model over the training data, which the paper reports takes ~2,242 seconds compared to ~58,227 seconds for full training (Section D). This is genuinely negligible overhead compared to multi-epoch training.

- **Empirical validation includes additional models and datasets:** Appendix G.1 reports results on Vicuna-7B-v1.3 (Table 9) and on GSM8K training split (Table 10), showing consistent benefits beyond the primary LLaMA3-8B-Instruct setup.

## Weaknesses

- **Theoretical analysis relies on Gaussian distributions, which are a significant simplification of real LLM output distributions.** While the paper provides derivations for Gaussian (Appendix A), Exponential, and Half-Normal (Appendix F.3) families, real LLM vocabularies produce discrete, highly skewed distributions over ~32K–128K tokens. The connection between the Gaussian variance analysis and the discrete cosine similarity metric (Appendix B) is established asymptotically, but the assumption L ≫ σ (window large relative to distribution spread) is most accurate precisely for low-flatness tokens—the regime the paper argues contributes least to training. This creates some circularity in the theoretical justification.

- **Token-to-sample aggregation via averaging lacks theoretical grounding.** Equation 8 averages flatness scores across tokens in a sample, but this heuristic could dilute the signal from samples containing a mix of high- and low-flatness tokens. The paper does not ablate alternative aggregation strategies (e.g., max, median, percentile-based). A median-ablation is provided in Table 11 showing similar results, but other aggregation methods are unexplored.

- **Static selection ignores draft model evolution during training.** The flatness metric depends only on the fixed target model and is computed before training begins. As the draft model evolves, tokens that were initially low-value may become informative, or vice versa. The paper notes this limitation partially but does not experiment with dynamic re-selection strategies.

- **Limited experimental diversity in main results.** While Appendix G.1 includes Vicuna-7B and GSM8K experiments, the primary results (Tables 1–3) rely exclusively on LLaMA3-8B-Instruct trained on ShareGPT and evaluated on five downstream tasks. No experiments test generalization to other speculative decoding frameworks (e.g., Medusa, DistillSpec, EAGLE-1), which may have different training dynamics.

- **Figure 2 validation uses only 10 randomly selected samples for training dynamics analysis.** Conclusions about the relationship between flatness and ΔL1 are drawn from a very small sample, which limits confidence in the generality of the token-level trends. While the main experimental results use the full dataset, this core validation step would benefit from larger-scale verification.

- **The mechanism for flatness outperforming entropy is not fully explained.** Appendix F.2 shows that entropy-based sorting yields training dynamics nearly identical to flatness-based sorting (Figure 5 vs Figure 2), both being measures of distance to uniform. Yet Table 1 shows flatness (2.41× average speedup) notably outperforms entropy (2.20×). The paper attributes this to entropy's logarithmic scale being less sensitive to distributional tails, but does not empirically demonstrate this mechanism. This gap between similar dynamics but different final performance warrants deeper analysis.

## Nice-to-Haves

- Experiments on larger target models (e.g., LLaMA-70B) to verify scalability and computational feasibility of the offline flatness computation at scale.

- Experiments on alternative speculative decoding frameworks (Medusa, DistillSpec) to demonstrate the method generalizes beyond EAGLE-2's specific feature-conditioned draft architecture.

- Analysis of the semantic composition of retained vs. filtered samples (e.g., code vs. math vs. dialogue) to understand whether SFDD introduces any systematic domain biases.

- Dynamic re-selection experiments to quantify potential gains from updating flatness scores during training.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that 2× speedup is "trivially true"**: The critic implies the 2× speedup from 50% data is trivial. This mischaracterizes the contribution—the non-trivial claim is that SFDD's selected 50% achieves inference speedup within 4% of full data, while random 50% would yield larger degradation. Table 2 shows SFDD at 50% achieves 2.41× speedup vs. random's 2.20×, demonstrating meaningful data selection quality.

- **Complaint about missing confidence intervals and significance tests**: While additional statistical rigor would strengthen the paper, the differences between SFDD and the strongest baseline (Top-1 Probability: 2.41× vs 2.23×) are consistent across five downstream tasks, and the ablation studies show robust trends across retention ratios. Requesting confidence intervals for a methods paper with consistent multi-benchmark results, while reasonable, is not a critical flaw.

- **Criticisms demanding experiments "outside stated scope"**: The paper explicitly focuses on train-based speculative decoding with knowledge distillation (Section 1). Requesting experiments on train-free methods (e.g., off-the-shelf drafters) or on other frameworks not within this scope is scope creep.

- **Claim that the paper has "no limitations section"**: The paper discusses limitations in multiple places: Section F.6 explains why token-level filtering doesn't yield speedups; Section F.7 discusses the necessity of draft model training; Section F.6 notes that static selection may not capture training dynamics.

## Novel Insights

The key insight is the reframing of token importance for speculative decoding: rather than asking "which tokens does the model get wrong?" (the standard uncertainty sampling view), the paper asks "which tokens have target distributions where even an optimal single-step update yields maximal L1 reduction?" This target-centric view—dependent only on the teacher, not the student's current state—is non-obvious and provides a principled reason why "uncertain" tokens are valuable beyond the standard active learning intuition. The observation that flatness-based filtering can sometimes *match or exceed* full-data performance (70% retention on Alpaca: 2.77× vs 2.71×; GSM8K: 2.71× vs 2.71×) suggests SFDD may act as a regularizer by removing redundant or noisy samples, though this phenomenon is noted only briefly and deserves deeper investigation.

## Suggestions

- Add a concise "Limitations" paragraph in the main text summarizing: (1) static selection ignores draft evolution, (2) the Gaussian proxy is an approximation, (3) experiments are limited to EAGLE-2 and primarily ShareGPT. This would satisfy typical ICLR reviewer expectations without disrupting the paper's flow.

- In the main text, explicitly state the relationship between flatness and entropy (as done in F.2) and provide a clear mechanistic explanation for why cosine similarity outperforms entropy despite their theoretical connection. The current explanation is confined to an appendix and could be more prominent.

- When presenting Figure 2's validation, report results on a larger sample (e.g., 100–500 samples) with standard error bars to strengthen the token-level analysis.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 4.0]
Average score: 5.5
Binary outcome: Accept
