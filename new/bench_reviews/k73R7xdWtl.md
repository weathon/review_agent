## Summary
The paper proposes using kernelized contextual bandits for online selection of the best text-based generative model for a given input prompt. The key insight is that different models may rank differently across prompt types, so a prompt-conditional selection strategy can outperform always choosing the single best-performing model on average. The authors propose SCK-UCB (Shared-Context Kernel UCB), which maintains per-model kernel ridge regressions on a shared prompt context, and RFF-UCB, which uses random Fourier features to reduce computational cost. Theoretical regret bounds of $\tilde{O}(\sqrt{GT})$ are established for both algorithms, and experiments on text-to-image, image captioning, and text-to-video tasks demonstrate the approach.

## Strengths
- **Well-motivated and timely problem formulation.** The observation that generative model rankings vary across prompt types is practically relevant and cleanly formalized as a contextual bandit problem (Protocol 1). The "outscore-the-best" metric effectively demonstrates the value of prompt-conditional selection over the single-best-model baseline.
- **Theoretical framework is sound.** The $\tilde{O}(\sqrt{GT})$ regret bound for SCK-UCB (Theorem 1) and the analogous result for RFF-UCB (Theorem 2) provide formal guarantees. The computational complexity reduction from $O(t^3/G^2)$ to $O(ts^2)$ per iteration via RFF (Lemma 2) addresses a real scalability concern.
- **Clear algorithmic presentation.** SCK-UCB (Algorithm 2) and its RFF variant (Algorithm 3) are presented in a readable, self-contained manner. The paper includes adaptation experiments (Setup 2) where new models or prompt types are introduced mid-stream, demonstrating practical robustness.

## Weaknesses

### Fatal
None.

### Major

- **The theoretical contribution is incremental over existing kernel-UCB methods.** The "shared context with model-dependent weights" formulation (Assumption 1, Remark 1) is structurally equivalent to running $G$ independent Kernel-UCB instances on the same context stream — there is no shared statistical learning across arms. The resulting $\tilde{O}(\sqrt{GT})$ regret bound is exactly what one would obtain from this parallel construction, and the RFF approximation preserving this rate is a standard result from the kernel-UCB literature. The paper does not articulate what new analytical challenge the shared-context setting poses beyond notational indexing. This is significant because the paper's primary pitch is theoretical ("we adapt kernelized CB… and establish a regret bound"), yet the core analysis appears to be a direct instantiation of known techniques.

- **Experiments rely predominantly on synthetic "expert/noise" setups that are artificially easy.** In Setups 3, 4, and 5, the non-expert models are simulated by adding Gaussian noise to a base model's output, creating perfectly separable score distributions. This makes the problem trivially solvable by any reasonable contextual bandit. The only real-model experiment (Setup 1) uses just two models with CLIPScore differences of ~0.5 points, and the OPR reaches only ~0.68. No experiments test more than 2–3 real models, despite the regret scaling as $\tilde{O}(\sqrt{GT})$ depending on the number of generators $G$. Without experiments with 5+ real models where performance differences are subtle and overlapping, the practical significance of the approach remains unclear.

- **RFF-UCB — the paper's main practical contribution — underperforms significantly on real data and is underspecified.** In the primary real-model experiment (Figure 2), RFF-UCB achieves near-zero O2B and ~0.55 OPR, barely outperforming baselines — while SCK-UCB-poly3 achieves ~0.5 O2B and ~0.68 OPR. This performance gap is never analyzed. Additionally, Algorithm 3 resamples random Fourier features ($\omega_j, b_j$) on every call, whereas standard RFF practice fixes the feature map once. The bonus terms $\mathcal{B}_{g,1}$ and $\mathcal{B}_{g,2}$ and the adaptive feature dimension $s$ are referenced but not defined in the main text, making the algorithm as presented incomplete and its stated regret guarantee unverifiable from the submission alone.

- **The realizability assumption (Assumption 1) is unvalidated and potentially unrealistic.** All theoretical guarantees rest on the assumption that expected scores are exactly linear in the RKHS. Remark 2 motivates this by noting that "kernel methods can approximate any function arbitrarily well with enough training data," but this is an approximation property — not the exact realizability assumption required by the regret bounds. No empirical validation (e.g., held-out prediction error of kernel ridge regression on real model scores) is provided to check whether this assumption is even approximately satisfied for CLIPScore. For a theory-driven paper, the disconnect between the assumption and the application domain is concerning.

### Minor

- **No comparison with modern contextual bandit algorithms.** The baselines include only ablations (Lin-UCB, Naive-KRR) and trivial strategies (Random, Greedy). Comparing against Thompson Sampling, NeuralUCB, or $\varepsilon$-greedy with neural features would help establish whether the kernel-based approach specifically is warranted versus alternatives.

- **No error bars or statistical significance tests are reported.** All results are "averaged over 20 trials" without standard deviations, making it impossible to judge whether the observed differences (especially the small ones in real-model experiments) are statistically meaningful.

- **Small CLIPScore differences raise practical significance questions.** In Figure 1, the best-model alternation across prompts involves CLIPScore differences of merely ~0.5 points (e.g., 36.37 vs. 37.24). Given the noise in CLIPScore from stochastic generation, it is unclear whether these gains are practically meaningful or within measurement noise.

- **The independence condition in Lemma 1 requires clarification.** The lemma requires scores in $\Psi_g$ to be independent, but the adaptive selection of which model to query creates selection bias. The paper defers the proof to the appendix without indicating whether standard martingale-type arguments are used to handle this.

### Trivial
- The abstract states a $\tilde{O}(\sqrt{T})$ regret bound while Theorem 1 specifies $\tilde{O}(\sqrt{GT})$ — the dependence on $G$ should be mentioned in the abstract for precision.

## Nice-to-Haves
- Experiments with 5+ real generative models (e.g., multiple Stable Diffusion variants, DALL-E, IF, PixArt) with naturally varying per-prompt rankings, to validate scalability.
- Sensitivity analysis of RFF-UCB with respect to feature dimension $s$, regularization $\alpha$, and exploration parameter $\eta$.
- Wall-clock time comparisons between SCK-UCB and RFF-UCB to empirically validate the claimed computational savings.
- Validation of Assumption 1 via held-out kernel regression prediction error on real model scores.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **"The O2B metric is ill-posed and unstandardized"** (from the harsh critic): The O2B metric is clearly defined as the difference between the algorithm's CLIPScore and the highest average CLIPScore of any single model. This is a reasonable and interpretable metric. The concern about path-dependency from exploration applies to any bandit algorithm and does not make the metric "ill-posed."
- **"No ground-truth conditional best model is defined"** (from the harsh critic): In the real experiments, the best model per prompt is defined by whichever model achieves higher CLIPScore on that prompt. CLIPScore is a standard evaluation metric for text-to-image generation; calling this "circular" is overly dismissive of a legitimate evaluation protocol.
- **"The One-arm Oracle is a strawman"** (from the harsh critic): The One-arm Oracle is precisely the baseline that the paper sets out to beat — the standard practice of selecting the single best model on average. Beating it validates the core claim that prompt-conditional selection is beneficial. This is an appropriately designed baseline, not a strawman.
- **"Missing per-prompt supervised model selection trained offline"** as a baseline: An offline-trained model selector requires pre-collected data from all models on all prompts, which the online bandit approach is specifically designed to avoid. This is outside the paper's stated scope of online learning with limited queries.
- **"The paper should discuss query cost heterogeneity"** (from the harsh critic): This is scope creep — the paper explicitly focuses on score maximization, not cost minimization. Cost-aware bandits are a different problem formulation.
- **"i.i.d. prompts assumption is unrealistic; non-stationary prompts"**: While true, the i.i.d. assumption is standard in contextual bandit theory and analyzing it is the natural starting point. Non-stationary extensions would be a nice future direction but not a core flaw.

## Novel Insights
The key insight that generative model rankings can vary across prompt categories — making prompt-conditional selection strictly better than selecting the globally best model — is well-demonstrated and practically relevant. However, the paper reveals a tension: the problem formulation is most compelling when performance differences across prompts are large and systematic, yet in the only real-model experiment, these differences are tiny (~0.5 CLIPScore points). The synthetic expert/noise experiments, where differences are large, succeed trivially. This suggests the practical niche for this approach may be narrower than claimed — it works best precisely when the problem is easiest, and struggles when real-world differences are small. Additionally, the observation that RFF-UCB — the computationally efficient variant meant for deployment — dramatically underperforms on real data while the full SCK-UCB works raises questions about whether the theoretical guarantee of "same regret rate" translates to practical reliability.

## Suggestions
- Validate the realizability assumption empirically by computing held-out kernel regression prediction error for each model's scores over prompt embeddings, reporting how well the RKHS assumption holds for CLIPScore.
- Analyze the RFF-UCB performance gap on real data: vary $s$ systematically and report both prediction error and OPR; this would clarify whether the gap is due to approximation quality or exploration degradation.
- Fix the RFF feature sampling to draw features once (not per call) and report whether this changes performance — the current per-call resampling may be the source of RFF-UCB's poor real-data performance.

## Evaluation

**Originality:** The problem formulation is natural and timely but the algorithmic and theoretical contributions are incremental — SCK-UCB is structurally equivalent to $G$ independent Kernel-UCB instances, and the RFF approximation follows standard techniques. The shared-context framing adds notational novelty but not conceptual novelty.

**Importance of research question:** The question of prompt-conditional model selection is relevant and practical, though the paper does not convincingly demonstrate that the magnitude of real-world cross-prompt ranking reversals is large enough to motivate a bandit approach over simpler heuristics.

**Claims supported:** The theoretical claims are formally correct but rest on strong realizability assumptions. The empirical claims overstate the practical impact — the real-model experiments show small gains, and the synthetic experiments are trivially easy.

**Soundness of experiments:** Experiment design is weak — predominantly synthetic expert/noise setups, only 2 real models, no error bars, no sensitivity analysis, and no comparison with modern CB baselines.

**Clarity:** The paper is well-written and the formalization is clear. Algorithm presentation is readable.

**Value to community:** The problem framing may inspire useful follow-up work, but the current technical contribution is insufficient to justify acceptance as-is.

## Score and Decision

**Calibration papers compared against:**
- "Cost-Effective Online Multi-LLM Selection" (JLDAWbzTUg): scores 5/6/5/6, Reject — similar topic (online model selection via bandits), incremental novelty. This paper is comparable but has weaker real experiments.
- "Orthogonal Function Representations for Continuous Armed Bandits" (43WKxTuJxu.md): scores 5/6/5/5, Reject — kernel bandit theory paper with similar concerns about assumptions. This paper has comparable theory but less novelty.
- "Diffusion Models Meet Contextual Bandits" (GGAG3wFEKv.md): scores 5/5/6/6/6, Reject — similar pattern of applying bandits to a new domain with limited novelty. This paper is around this quality.
- "Optimal Mixtures of Generative Models via Mixture-UCB" (2Chkk5Ye2s.md): scores 6/6/5/6/6, Accept-Poster — similar domain with somewhat stronger results and cleaner novelty.

This paper sits below the "Mixture-UCB" paper (which had similar motivation but cleaner novelty) and at or below the "Online Multi-LLM Selection" paper (which also had weak experiments but a more novel problem setting). The incremental theory, weak experiments, and underspecified RFF-UCB algorithm place it below acceptance threshold.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>