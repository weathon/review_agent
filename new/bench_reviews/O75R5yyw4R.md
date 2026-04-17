The paper content is already provided in the user message. Let me verify the key claims from the reviewers by searching the paper text directly.

Key points to verify from the harsh reviewer:

1. **Reversibility assumption**: The paper states in Proposition 1: "Assume that q and pθ form a reversible Markov kernel." This is indeed stated.

2. **Pool reuse and backward proposal elimination**: Section 3.3 states: "Through an appropriate choice of the balancing function in Equation 2, the acceptance rate can be evaluated without the need for resampled proposals x′′t, while still preserving the theoretical guarantees of the MTM framework. Consequently, the practical implementation eliminates the resampling step and reduces the per-iteration cost by nearly half." And "when a proposal is rejected, we simply reuse the previously generated sampling pool."

3. **Intermediate reward approximation**: The paper says "Intermediate rewards r(xt) can approximate by evaluating the reward function on the diffusion model's prediction of x0" (a single sentence, no further analysis).

4. **NFE metric**: Section 4.1 says they "treat the reward model and the generative model on equal footing" for NFE counting, while Section 3.3 acknowledges that aggregating them "may obscure meaningful differences."

5. **Evaluation scope**: 15 prompts × 3 seeds × 20 samples = 900 total per configuration for MDLM, and 300 for the safety case study.

Now let me synthesize the final review.

## Summary

This paper introduces IterRef, a test-time scaling method for discrete diffusion models that uses Multiple-Try Metropolis (MTM) with noising-denoising transition kernels to iteratively refine intermediate states toward reward-aligned distributions. The method proposes applying reward-guided noising-denoising at selected timesteps during sampling, accepting/rejecting proposals based on reward differences, and provides a theoretical convergence guarantee (Proposition 1) under a reversibility assumption. Empirically, IterRef demonstrates consistent improvements over baselines across text (MDLM, LLaDA-8B) and image (MaskGIT) generation with various reward functions, particularly excelling under low compute budgets.

## Strengths

1. **Addresses a genuine and important gap**: Test-time scaling for discrete diffusion is underexplored compared to continuous diffusion, and the token irreversibility problem is a real challenge that IterRef's noising-denoising mechanism directly targets. The approach of re-masking tokens to enable correction is intuitive and well-motivated.

2. **Strong and consistent empirical results**: IterRef outperforms baselines across three different backbones (MDLM, LLaDA-8B, MaskGIT), four language rewards (Toxicity, Sentiment, CoLA, Perplexity), and one image reward (CLIPScore). The low-NFE performance is particularly notable — achieving with 4T NFEs what FK Steering requires 32T NFEs to match on MDLM Toxicity (~8× speedup).

3. **Insightful analysis on iteration vs. particles and effective timesteps**: Table 3 and Figure 4 demonstrating that iterations (k) yield greater gains than particles (N) is a useful finding. Table 2's observation that later denoising stages are most effective for discrete diffusion — contrasting with continuous diffusion where early steps dominate — is a genuinely novel and valuable insight about the dynamics of discrete diffusion.

4. **Practical engineering choices**: Pool reuse upon rejection, elimination of backward proposals via balancing function design, and selective refinement via timestep set U are well-motivated and reduce computational cost meaningfully.

## Weaknesses

### Major

- **Theoretical guarantee relies on an unsupported assumption and is disconnected from the implemented algorithm**: Proposition 1 requires that "q and pθ form a reversible Markov kernel," which is almost certainly not satisfied by the actual models used (MDLM, LLaDA-8B, MaskGIT), as standard discrete diffusion training does not enforce detailed balance between the forward process q and the reverse model pθ. Furthermore, Section 3.3 modifies the MTM algorithm by eliminating backward proposals and reusing pools upon rejection, stating that this "still preserv[es] the theoretical guarantees." This claim is not substantiated — these modifications alter the transition kernel, and the specific derivation of β = min(1, exp((r(x′t) − r(xt))/α)) in Equation (3) depends on both the reversibility assumption and the specific structure of the MTM backward proposal step. The implemented algorithm and the theoretical framework are therefore misaligned: Proposition 1 does not apply to what is actually run. This significantly weakens the paper's claim of providing a "principled" and theoretically grounded method.

- **Intermediate reward approximation is unanalyzed**: The method depends on r(xt) = α log E[exp(r(x₀)/α)], approximated in practice by evaluating the reward on the model's single-step prediction of x₀. This replaces a potentially complex expectation with a point estimate. No analysis — theoretical or empirical — is provided on how this approximation affects the target distribution, the MTM convergence, or the quality of results. This is the same issue raised for SVDD by its reviewers, and in both cases it creates a gap between the claimed "optimal" distribution and what is actually targeted.

- **Evaluation relies solely on the same reward models used for guidance, with no diversity or external quality metrics**: All language results measure performance by the same classifier/reward used for guidance (Toxicity, Sentiment, CoLA, Perplexity). No human evaluation, no independent fluency/coherence metrics, and critically, no diversity metrics are reported. The authors themselves note (§4.5) that detoxification often works by framing toxic content as quoted speech — a potential instance of reward gaming — yet treat lower toxicity scores as unqualified success. For image generation, CLIPScore is both the guidance signal and evaluation metric, with only ImageReward in the appendix as an alternative. This leaves open the possibility that IterRef is simply better at optimizing the reward proxy rather than producing genuinely better-aligned outputs.

### Minor

- **The effective timestep set U has no principled selection method**: Table 2 shows dramatic variation in which timesteps are most effective across tasks, yet the paper provides no guidance on how to select U without grid search, limiting practical applicability.

- **NFE metric equalizes reward-model and generation-model calls, obscuring real cost**: The paper itself acknowledges this (§3.3) but still uses joint NFE as the primary metric for headline claims like "8× faster scaling." Wall-clock comparisons are deferred to an appendix.

- **The CoLA failure with LLaDA-8B is under-analyzed**: Best-of-N outperforms IterRef on CoLA with LLaDA-8B, suggesting that iterative refinement can be counterproductive when the base model already produces well-formed text. This boundary condition deserves deeper investigation.

### Trivial

- The safety case study uses only 300 generations (15 prompts × 20 samples), which is quite small for claims about "robust safety alignment."

## Nice-to-Haves

- Compare with PG-DLM (Dang et al., 2025), which performs iterative trajectory resampling and is the most directly competing method cited in the paper's related work but omitted from experiments.

- Report acceptance rates β during sampling — this would clarify whether the Metropolis step meaningfully filters proposals or is largely permissive/rejective, and would help assess whether the MCMC chain is mixing in practice.

- Report separate generative-model and reward-model call counts alongside joint NFE, to allow readers to assess cost under different model-scale regimes.

- Measure sample diversity (e.g., self-BLEU, n-gram entropy) to address the concern that reward gains may come at the expense of diversity.

## Removed Points

- **Claim that PG-DLM and DSearch are missing baselines**: While worth including as a nice-to-have comparison, the paper already compares against four baselines (BoN, SoP, SVDD, FK) and PG-DLM/DSearch are concurrent works. This is a reasonable but not mandatory addition rather than a fatal omission.

- **Demand for α sensitivity analysis**: This is a valid suggestion but α is a standard hyperparameter in the reward-guided generation literature (SVDD, FK, etc.) and the paper follows the convention of the field. This falls under nice-to-have rather than a weakness.

- **Claim that the paper's "novelty is incremental from an MCMC perspective"**: While MTM is a classical technique, its adaptation to the discrete diffusion setting with noising-denoising kernels and the specific balancing function design represents a genuine algorithmic contribution that goes beyond straightforward application.

- **Demand for theoretical analysis of intermediate reward approximation error**: This would strengthen the paper but is standard practice in the field (the same approximation is used in SVDD, FK, etc.) and is not a standard requirement.

## Novel Insights

The finding that later denoising timesteps are most effective for reward-guided refinement in discrete diffusion — in contrast to continuous diffusion where early steps are most influential — is a genuinely novel observation. This suggests fundamentally different dynamics between discrete and continuous diffusion during guided generation, which has implications beyond IterRef itself and could inform the design of future discrete diffusion guidance methods.

## Suggestions

1. Be transparent about the gap between theory and implementation: explicitly acknowledge that Proposition 1 assumes reversibility (which does not hold for trained models) and that the pool-reuse simplification breaks the MTM proof. Present the convergence guarantee as a motivating ideal rather than a guarantee of the implemented algorithm.

2. Add at least one external quality or diversity metric alongside reward scores, particularly for language tasks where reward hacking is a known concern.

3. Provide heuristic guidance for selecting the effective timestep set U — even simple rules based on mask ratio or intermediate entropy would improve practical applicability beyond grid search.

## Score and Decision

**Calibration anchors:**

- **SVDD (Scores: 3,5,3,5 → Reject)**: The closest comparable paper. SVDD also claimed principled reward-guided sampling for discrete diffusion but had gaps between its theoretical claims and implementation (especially regarding α=0, intermediate reward approximation, and lack of diversity metrics). IterRef has similar theoretical-implementation gaps but demonstrably stronger empirical results across more backbones and a cleaner algorithmic design.

- **Discrete Guidance (Scores: 6,8,6,6 → Accept Poster)**: Provided a principled guidance framework for discrete diffusion with more rigorous theory. IterRef has interesting but more questionable theory, but provides more extensive empirical analysis including scaling behavior.

- **DAS (Scores: 5,8,8,8 → Accept Spotlight)**: Showed strong empirical results with SMC-based alignment and diversity preservation, with clearer theoretical grounding. IterRef is less mature on the theory-implementation alignment.

- **DNO (Scores: 5,6,5,6 → Reject)**: Similar reward-hacking concerns and evaluation gaps. IterRef has more diverse evaluation settings.

IterRef sits between SVDD (rejected for theoretical/evaluation gaps) and DAS/Discrete Guidance (accepted for cleaner contributions). The empirical contribution is real and substantial — consistent improvements across domains/backbones, strong low-compute performance, and genuine insights about discrete diffusion dynamics. However, the misalignment between the theoretical claims and the implemented algorithm is a significant weakness that undermines the framing of IterRef as a "principled" approach, and the evaluation relies exclusively on internal reward metrics.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>