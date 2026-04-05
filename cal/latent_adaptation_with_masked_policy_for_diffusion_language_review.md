=== CALIBRATION EXAMPLE 39 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Accuracy:** The title accurately reflects the core technical components (latent adaptation, masked policy, diffusion LMs).
- **Clarity:** The abstract clearly outlines the problem (dLLM test-time reasoning), method (sparse policy-gradient on token latents with clamp-and-inpaint), and datasets (GSM8K, MATH-500, AIME).
- **Unsupported Claims:** The claim of "modest compute" is asserted without any quantitative backing in the abstract or main text. Given the iterative adaptation loop, readers will expect explicit FLOPs or wall-clock comparisons to standard decoding or majority-vote baselines. Additionally, "consistently improves reasoning accuracy" slightly overstates results on AIME (e.g., 0.0% to 3.3% for Dream-PSRM), where absolute gains are marginal and dataset size is small (30 questions).

### Introduction & Motivation
- **Problem & Gap:** The introduction effectively motivates why AR test-time strategies (CoT, self-consistency) transfer poorly to dLLMs and identifies a clear gap: adapting latent states in a bidirectional, non-sequential decoding regime.
- **Contributions:** The three bullet points accurately map to Sections 2 and 3. 
- **Positioning:** The claim that LAMP "complements existing inference-time scaling methods" is plausible but remains unverified without direct comparison or hybrid experiments. The introduction slightly undersells the relationship to *LatentSeek* (Li et al., 2025b), which proposes a conceptually similar latent policy-gradient approach for AR/conditional generation; the diffusion-specific novelty (clamp-and-inpaint + independent hidden state editing) needs sharper contrast in the intro to justify standalone contribution.

### Method / Approach
- **Clarity & Reproducibility:** The four-step pipeline (Sec 2.2) is logically structured. However, there is a mismatch between the mathematical formulation and implementation: Eq. 4 presents only the pure REINFORCE gradient, yet Section 2.1 and the pseudocode explicitly apply KL and L2 trust-region regularization. The final loss used should be explicitly stated in the main text.
- **Assumptions & Justification:** The method treats token hidden states $h_i$ as *independent* editable latents $z_i$, parameterizing local policies $q_i(z_i) = \text{softmax}(g(z_i))$. Diffusion LMs rely heavily on bidirectional self-attention; modifying a subset of positions in isolation ignores contextual dependencies. While the subsequent `clamp-and-inpaint` pass aims to restore global coherence, the paper lacks a theoretical or empirical justification for why independent latent edits do not immediately destabilize the joint distribution or create adversarial token combinations that confuse the sampler.
- **Logical Gaps & Edge Cases:** Algorithm 1 and the pseudocode indicate that each policy-gradient iteration $t$ requires a full call to `CONSTRAINEDDIFFUSE` to compute the reward $r$. This means 2-3 complete diffusion decodes are run *per instance* before accepting any edits. The paper does not discuss the compounding error or distribution shift that occurs when latents are updated on-the-fly without adjusting the underlying diffusion schedule or noise level.
- **Stability of Sparse Rewards:** Using REINFORCE with a binary PSRM (0/1) and a moving average baseline, especially with a tiny budget of $K=2$ update steps, is notoriously high-variance. The method lacks variance reduction mechanisms (e.g., control variates beyond a simple baseline, multiple rollouts per step, or advantage normalization). How updates stabilize with such sparse, single-sample gradients needs addressing.

### Experiments & Results
- **Claim Verification & Baselines:** Table 1 demonstrates that LAMP+PSRM outperforms vanilla DLMs, but the critical missing comparison is against standard inference-time scaling baselines. For a method claiming efficiency, how does it compare to **Best-of-N** or **Majority Vote** with the same compute budget? Running 2-3 sequential LAMP iterations is roughly equivalent to sampling 3 vanilla trajectories in parallel, which can then be reranked. Without this baseline, the practical advantage of LAMP is unclear. Additionally, related search methods like Particle Gibbs (Dang et al., 2025) or ReMDM are discussed but not benchmarked.
- **Missing Ablations:** The Introduction claims "Ablations confirm that diffusion-specific ingredients… are essential," yet **no ablation tables or curves are present** in the main text. The provided Appendix sections (C, D) contain run configurations and qualitative examples but omit quantitative ablations on: (1) edit budget $k$, (2) number of steps $K$ vs. accuracy/compute, (3) clamp-and-inpaint vs. naive full resampling, and (4) trust-region regularization vs. unconstrained PG.
- **Statistical Rigor:** Results are reported as point estimates without error bars, confidence intervals, or multi-seed variance reporting. Given the stochastic nature of diffusion sampling and PG updates, a single seed per setting (as implied by the config tables) is insufficient for ICLR. Small absolute gains (e.g., +1.3 to +2.2 on MATH-500 with self-reward) cannot be judged significant without variance.
- **Metrics & Datasets:** Benchmarks are appropriate. However, the AIME analysis relies on only 30 problems. Gains of +3.3% correspond to correcting a single question, highlighting high sensitivity to individual samples.

### Writing & Clarity
- **Confusing Sections:** Section 3.4 claims the "True→False rate (fixed at 3% in our construction)." This phrasing is confusing. Is the self-reward artificially constrained to degrade at exactly 3%, or was this an observed empirical rate? If it's a construction choice, the rationale is missing; if empirical, "fixed" is a misnomer.
- **Figures/Tables:** Table 1 is informative but would benefit from computing cost metrics (e.g., average inference time or number of forward passes) alongside accuracy. Figure 2's caption references specific point improvements (+12.8, etc.) that should align explicitly with the iterations shown on the x-axis for reproducibility.
- **Overall:** The narrative flow is strong, but the disconnect between the stated regularization and Eq. 4, along with the unplaced ablation claims, impedes full technical evaluation.

### Limitations & Broader Impact
- **Acknowledged Limitations:** The authors correctly identify that self-reward is weak and that process supervision could help. They note the training-free nature limits compute footprint relative to full RLHF.
- **Missed Limitations:** 
  1. **Compute & Latency:** The serial nature of LAMP (adapt $\to$ decode $\to$ evaluate $\to$ adapt) likely increases wall-clock latency significantly compared to parallel best-of-N sampling. This is a major practical limitation for deployment.
  2. **Oracle Dependency:** The method's success heavily relies on PSRM access. The paper acknowledges this but does not discuss fallback strategies or the performance cliff when only noisy verifiers or human feedback are available.
  3. **Reward Hacking:** Gradient-based optimization on sparse rewards can easily lead to reward hacking (e.g., generating syntactically correct but semantically wrong answers that bypass the extractor/parser). Qualitative Table 2 shows some regressions, but systematic analysis of failure modes (e.g., overconfidence, format gaming) is absent.
- **Broader Impact:** The ethics statement is standard but brief. It does not address the potential for amplifying biases through test-time optimization or the misuse of enhanced reasoning capabilities, though it gestures toward safety filters.

### Overall Assessment
LAMP introduces a compelling concept: leveraging the bidirectional, revisable nature of masked diffusion models to perform sparse, reward-guided latent edits at test time. The motivation is sound, and the integration of clamp-and-inpaint decoding is a clever way to preserve global coherence during local updates. However, the paper currently falls short of the ICLR acceptance bar due to significant empirical gaps. The most critical concern is the lack of strong baselines; without comparing against standard inference-time scaling (Best-of-N/Majority Vote at equal compute), the claimed practical advantage is unsubstantiated. Additionally, the high-variance nature of REINFORCE with binary rewards is not theoretically or empirically mitigated, and key ablation results are claimed but absent from the manuscript. The paper also omits essential compute/latency accounting and multi-seed statistical validation. With rigorous compute benchmarking, proper ablations, comparison to parallel sampling baselines, and clearer stabilization analysis, this work could be a strong addition to dLLM research. In its current form, the contribution is promising but not yet fully validated.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces LAMP, a training-free, instance-level test-time adaptation framework for masked diffusion language models (dLLMs). The method identifies low-confidence token positions, applies sparse policy-gradient updates to their latent hidden states guided by reward signals, and then propagates accepted edits through a constrained "clamp-and-inpaint" diffusion pass. Empirical results across GSM8K, MATH-500, and AIME demonstrate consistent reasoning gains on LLaDA and Dream backbones, particularly when using an oracle reward signal.

### Strengths
1. **Well-Motivated Mechanism for dLLMs:** The paper leverages core properties of masked diffusion (parallel scoring and constrained infilling) to solve a clear gap in test-time adaptation. The "clamp-and-inpaint" decode (Sec. 2.2) is a natural fit for dLLMs, allowing local latent edits to harmonize globally without breaking coherence.
2. **Thorough Analysis of Reward & Scaling Dynamics:** Beyond standard accuracy tables, the paper provides valuable insights into iteration scaling (Fig. 2) and reward transition matrices (Fig. 3). This analysis clearly demonstrates that improvement stems from true reward quality rather than mere compute scaling, a nuance often missed in test-time scaling papers.
3. **Practical & Reproducible Design:** The training-free nature, sparse edit budget (~10%), trust-region regularization, and explicit hyperparameter tables (Appendix D) make the method easy to reproduce. The included PyTorch-style pseudo-code (Appendix C) further lowers implementation barriers.
4. **Substantial Performance Gains with Strong Supervision:** Using the Perfect Sparse Reward Model (PSRM), LAMP achieves double-digit improvements on GSM8K and MATH-500 (+5 to +20 points across models) consistently outperforming vanilla DLM baselines, proving that latent-space optimization is highly effective for diffusion reasoning when guided by precise feedback.

### Weaknesses
1. **Heavy Reliance on Oracle Supervision:** The strongest results depend entirely on PSRM, which requires ground-truth answers at inference time (Sec. 2.3). In realistic deployment scenarios, only noisy self-rewards or external verifiers are available, yet LAMP with self-reward yields marginal (+1–3 pts) or even negative gains (Table 1). The paper acknowledges this but does not offer a concrete pathway to bridge the oracle-to-verifier gap.
2. **Lack of Quantitative Efficiency Analysis:** The claim that LAMP adds "negligible overhead" and is a "favorable compute–performance trade-off" (Sec. 2.2, Sec. 3.3) is not substantiated with concrete metrics. Wall-clock latency, FLOPs, GPU memory overhead, and exact number of additional forward passes per iteration are missing, making it difficult to compare against standard inference scaling baselines like Best-of-N or self-consistency.
3. **Missing Direct Baseline Comparisons:** Related work lists several dLLM-specific inference-time methods (particle Gibbs, ReMDM, classical search), but the experiments do not include empirical comparisons against any of them. Without head-to-head benchmarks, it's unclear whether LAMP offers a superior efficiency/accuracy trade-off or merely an alternative approach.
4. **Ambiguities in Latent Gradient Routing:** The pseudo-code (Appendix C) suggests optimizing hidden states directly via `model.head(z)`, but modern dLLMs typically interleave LayerNorm, residual connections, and positional encodings. The method does not clarify how gradients bypass or respect these components, raising concerns about gradient stability, vanishing signals, or unintended perturbations in non-linear transformer layers.

### Novelty & Significance
The novelty lies in adapting latent policy-gradient optimization specifically to the non-autoregressive, bidirectional decoding paradigm of masked dLLMs. While latent editing has been explored for AR models, the paper successfully reformulates it to respect diffusion's parallel denoising and remasking structure. For the ICLR community, this work is significant as it establishes test-time latent adaptation as a distinct and effective axis for dLLM inference scaling. However, practical significance is currently constrained by the reliance on ground-truth rewards and the absence of rigorous efficiency benchmarking.

### Suggestions for Improvement
1. **Quantify Computational Overhead Rigorously:** Report wall-clock inference time, additional FLOPs, memory peak, and effective tokens-per-second for LAMP vs. vanilla decoding and Best-of-N/self-consistency baselines. ICLR requires explicit compute-performance trade-off analysis for test-time scaling claims.
2. **Include Direct Comparisons to dLLM Inference Baselines:** Add at least one recent inference-time scaling method tailored for diffusion models (e.g., particle Gibbs or remasking schedulers) to Table 1 or a dedicated appendix comparison. This will contextualize LAMP's relative standing in the dLLM ecosystem.
3. **Evaluate with Realistic Verifiers:** Replace or supplement PSRM in at least one experiment with a practical, non-oracle reward (e.g., an open-source LLM math verifier, rule-based equation checker, or process-verifier) to demonstrate that LAMP remains effective without ground-truth access.
4. **Clarify Gradient Flow and Architectural Handling:** Explicitly describe how layer normalization, residual pathways, and positional encodings are frozen or routed during latent updates. Provide an ablation or stability analysis showing that policy-gradient steps do not induce hidden-state divergence or gradient explosion in deeper transformer blocks.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Compute-matched comparisons to Best-of-N, self-consistency, or classical tree search baselines using the same PSRM reward and inference budget. Without this, the claim that LAMP offers a superior or "favorable" compute-performance trade-off over simple sampling-and-reranking is unsupported.
2. Explicit wall-clock time, FLOP counts, and GPU-hour measurements across edit budgets and diffusion steps. Asserting "modest compute overhead" and "efficient test-time scaling" requires rigorous quantitative accounting, not qualitative claims.
3. Ablation replacing the oracle PSRM with a realistic, imperfect verifier or trained outcome reward. The main results rely entirely on ground-truth access; testing performance under noisy/imperfect rewards is necessary to prove the method's practical utility and isolate gains from oracle hacking vs. genuine reasoning improvement.

### Deeper Analysis Needed (top 3-5 only)
1. Clarification of what exact computational graph the policy gradient traverses. The update rule modifies token latents, but it is unclear if gradients flow through the multi-step constrained diffusion denoising or only through the frozen output head; without this, the mathematical validity and stability of the update are ambiguous.
2. Disentanglement of clamp-and-inpaint decoding from latent policy optimization. A control experiment that clamps baseline high-confidence tokens and runs constrained diffusion *without* latent edits is needed to quantify how much of the reported gain comes from LAMP's policy gradient versus the diffusion model's native remasking capabilities.
3. Token-level edit categorization (e.g., reasoning steps, arithmetic operations, answer formatting). Quantifying *what* LAMP changes is essential to validate whether the method genuinely improves multi-step reasoning or merely performs localized token swapping until the answer matcher triggers.

### Visualizations & Case Studies
1. Sequence alignment heatmaps showing accepted edit locations under PSRM vs. self-reward across correct/incorrect transitions. This would reveal whether the method systematically targets genuine reasoning bottlenecks or opportunistically tweaks numerical tokens near the final answer span.
2. Iteration trajectories plotting edit acceptance rate, confidence deltas, and reward changes jointly. Visualizing how many proposed edits are gated out vs. frozen across steps would demonstrate whether the confidence gating mechanism is actively stabilizing learning or arbitrarily suppressing updates.

### Obvious Next Steps
1. Implement and report results against a computationally matched Best-of-K baseline with identical oracle scoring. If Best-of-K achieves equal or higher accuracy at lower complexity, LAMP's core contribution—per-instance latent policy optimization as a superior scaling axis—collapses.
2. Integrate process-level supervision (e.g., step-wise verifiers or symbolic checkers) to move beyond outcome-only oracle rewards. ICLR prioritizes methods that generalize to settings without ground truth; demonstrating viability with process signals is a necessary step toward the paper's stated goal of unlocking diffusion reasoning.

# Final Consolidated Review
## Summary
This paper introduces LAMP, a training-free test-time adaptation framework for masked diffusion language models that applies sparse, reward-guided policy-gradient updates to token-level latents, followed by a constrained `clamp-and-inpaint` diffusion pass to restore global coherence. The method demonstrates consistent accuracy gains on mathematical reasoning benchmarks (GSM8K, MATH-500, AIME) when paired with a ground-truth oracle reward (PSRM), and analyzes iterative scaling dynamics and reward transition patterns across dLLM backbones.

## Strengths
- **Diffusion-Aligned Adaptation Mechanism:** The method correctly identifies and leverages core dLLM properties—parallel scoring and bidirectional infilling—to perform localized latent edits without breaking sequence coherence. The `clamp-and-inpaint` procedure is a theoretically sound and architecturally natural fit for non-autoregressive decoding schedules.
- **Demonstrated Gains with Strong Supervision:** Under oracle PSRM guidance, LAMP yields substantial, consistent improvements (+5 to +20 points across backbones and datasets), establishing that latent-space policy optimization is a viable axis for boosting dLLM reasoning at inference time.
- **Useful Dynamic Analysis of Test-Time Scaling:** The iteration-scaling curves and self-reward transition matrices (Sec. 3.3–3.4) provide meaningful evidence that performance gains are driven by reward quality rather than brute-force compute scaling, offering a nuanced view of how diffusion refinement trajectories evolve during iterative adaptation.

## Weaknesses
- **Missing Compute-Matched Baselines and Efficiency Accounting:** The paper repeatedly claims "modest compute," "negligible overhead," and "favorable trade-offs," but provides zero wall-clock latency, FLOP counts, or explicit forward-pass budgets. More critically, there is no comparison to standard inference scaling baselines (e.g., Best-of-N, majority voting, or self-consistency) under identical compute/PSRM budgets. Without this, it is impossible to determine whether LAMP genuinely outperforms or merely replicates the accuracy of parallel trajectory sampling and reranking at similar cost.
- **Severe Oracle Dependency and Unverified Self-Reward Efficacy:** The method’s practical utility collapses without ground-truth access. With PSRM, gains are substantial; with realistic self-rewards, improvements are marginal (+1–3 pts), inconsistent, and occasionally degenerate (negative AIME gains). The paper acknowledges this gap but provides no experiments with imperfect verifiers, trained outcome models, or LLM-based critics. This heavily restricts the method’s applicability to real-world settings where oracle answers are unavailable.
- **Absent Critical Ablations and Unisolated Mechanisms:** The introduction claims "Ablations confirm that diffusion-specific ingredients… are essential," yet no quantitative ablation tables are present in the main text or appendix. Crucially, there is no control experiment isolating the policy-gradient latent updates from the `clamp-and-inpaint` decoding step itself. Constrained diffusion with naive token clamping may recover similar gains; without proving the gradient updates add independent value, the core mechanism remains under-validated.
- **Gradient Routing Ambiguity and High-Variance Updates:** The mathematical formulation presents a pure REINFORCE gradient (Eq. 4), while the implementation includes KL and L2 trust-region penalties not explicitly formalized. Furthermore, the pseudo-code indicates optimization occurs only at the output head (`model.head(z)`), not through the multi-step diffusion computational graph. Combined with only $K=2$ update steps and a binary reward, the high-variance nature of the updates is inadequately justified, leaving open questions about gradient stability and whether the optimizer meaningfully traverses a useful reward landscape rather than exploiting local logit sensitivity.

## Nice-to-Haves
- Provide token-level edit categorization and alignment heatmaps to clarify whether LAMP genuinely corrects reasoning bottlenecks or opportunistically patches arithmetic/formatting tokens near the answer span.
- Report multi-seed variance and confidence intervals to substantiate gains, particularly on small-sample subsets like AIME (30 questions) where single-question flips drive apparent improvements.
- Clarify the "fixed 3% True→False rate" claim in Sec. 3.4: if artificially constrained by construction, this artificially inflates stability metrics and obscures true self-reward dynamics.

## Novel Insights
The paper highlights a fundamental asymmetry in dLLM test-time scaling: iterative refinement in latent space is highly sensitive to reward density and fidelity, not merely to additional forward passes. By decoupling local proposal generation (independent token-late edits) from global consistency restoration (constrained diffusion inpainting), LAMP exposes a clean separation of concerns that is uniquely enabled by non-autoregressive decoders. However, the stark performance cliff between oracle and self-reward regimes suggests that the primary bottleneck for diffusion reasoning is not the decoding paradigm itself, but the availability of process-aligned supervision that can guide sparse latent updates without ground truth.

## Potentially Missed Related Work
- **Compute-matched inference scaling:** Recent best-of-K and parallel test-time scaling analyses (e.g., Snell et al., 2024; Brown et al., 2024) are essential baselines for validating efficiency claims.
- **Process reward models for diffusion/math:** Works on step-wise verification and implicit reward shaping for non-AR reasoning (e.g., Uesato et al., 2022; Lightman et al., 2023 extensions) could contextualize pathways to replace PSRM in practical deployments.

## Suggestions
1. **Add rigorous compute-matched baselines:** Benchmark LAMP against Best-of-K, majority voting, and recent dLLM-specific inference methods (e.g., particle Gibbs, ReMDM) using identical FLOP/latency budgets and the same PSRM scoring. Explicitly report wall-clock time, GPU memory, and tokens/sec.
2. **Run essential ablations:** Include tables that (a) isolate the `clamp-and-inpaint` step without latent PG updates, (b) vary edit budget $k$ and steps $K$ to show compute-accuracy trade-offs, and (c) compare regularized vs. unregularized PG stability.
3. **Evaluate with realistic, imperfect rewards:** Replace PSRM in at least one main experiment with a practical verifier (e.g., open-source LLM verifier, rule-based math checker, or lightweight trained RM) to demonstrate whether LAMP retains utility without ground truth.
4. **Clarify the computational graph:** Explicitly detail how hidden state updates bypass or interact with layer normalization, residuals, and positional embeddings. Formalize the full loss function (including KL/L2 terms) and discuss why shallow head-only gradients suffice for stable $K=2$ updates on binary rewards.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
