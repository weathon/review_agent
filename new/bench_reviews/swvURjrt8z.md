Now I have a thorough understanding of the calibration landscape. Let me synthesize the final review.

## Summary

TabDiff proposes a mixed-type diffusion framework for tabular data generation that combines Gaussian variance-exploding (VE) diffusion for numerical features with masked (absorbing) diffusion for categorical features, both indexed by a shared continuous time variable and coupled through a shared denoiser network. The paper introduces feature-wise learnable noise schedules to address column-wise distribution heterogeneity, a stochastic sampler to correct accumulated decoding errors during sampling, and extends classifier-free guidance to the mixed-type setting for conditional generation. Experiments across seven datasets and eight metrics show improvements over prior baselines, most notably a 22.6% average improvement on the Trend (correlation) metric versus the previous state-of-the-art TabSyn.

## Strengths

- **Strong empirical results across a comprehensive evaluation**: TabDiff consistently outperforms baselines on Shape (marginal distribution) and Trend (pairwise correlation) metrics across virtually all datasets, with particularly notable gains on Diabetes (a larger, categorical-heavy dataset). The evaluation spans fidelity, downstream utility (MLE, imputation), and privacy (DCR), providing a holistic picture. The ablation in Table 5 cleanly demonstrates the contribution of each component (learnable schedules and stochastic sampler).

- **Principled continuous-time formulation avoids latent-space encoding overhead**: Unlike TabSyn which maps all features into a latent VAE space before diffusing, TabDiff operates directly in the original data space. This avoids potential information loss from encoding categorical variables into continuous latencies and eliminates the need for a separate VAE training stage. The continuous-time ELBO for the masked diffusion component (Eq. 9) avoids the looseness of discrete-time ELBO bounds, which is a genuine theoretical advantage over TabDDPM and CoDi.

- **Feature-wise learnable schedules are practically effective**: The ablation (Table 5, Figure 2) shows clear improvements from learning per-column noise schedules, particularly on the Trend metric (2.33→1.80 average error). The training loss curves confirm that learnable schedules accelerate convergence for both numerical and categorical losses.

- **Stochastic sampler for error correction in categorical generation is a sensible design**: The observation that decoded categorical features become "frozen" in the masking reverse process (Eq. 8) and the proposed solution of adding forward-noise perturbations to re-open correction opportunities is well-motivated and supported by ablation results.

## Weaknesses

### Fatal
None.

### Major

- **Limited conceptual novelty relative to concurrent work**: The core components—VE diffusion for numerical features, masked diffusion for categorical features, per-feature learnable noise schedules—are all individually established techniques. The concurrent CDTD paper (Mueller et al., 2024) proposes essentially the same key idea of feature-wise adaptive schedules in a continuous diffusion model for mixed-type tabular data. The paper mentions this concurrence in one sentence (Section 3) but provides no experimental comparison or detailed differentiation. Without clearly articulating what TabDiff offers beyond CDTD (e.g., operating in original data space vs. latent space, different schedule parameterizations), the conceptual contribution is significantly diminished. — This matters because novelty is a key criterion, and the central claimed innovation ("feature-wise learnable diffusion processes") substantially overlaps with published concurrent work.

- **Experimental comparison fairness concerns**: TabSyn results are "obtained via our reproduction" while most other baselines are taken directly from Zhang et al. (2024) without re-running. Several baselines show OOM entries on larger datasets (News, Diabetes), and anomalous results (e.g., TabDDPM achieves 78.75% Shape error on News and 4.86 RMSE for MLE) are included in averages without discussion. It is unclear whether TabDiff uses comparable model sizes, training budgets, or hyperparameter tuning effort to all baselines. The "superior average performance across all eight metrics" claim is weakened when some baselines are not run on equal footing or when they fail to produce results on certain datasets. — This matters because the paper's core claim is empirical superiority, which must be established through fair and reproducible comparisons.

- **No computational cost or efficiency analysis**: The paper does not report training time, sampling time, number of parameters, or memory usage for TabDiff or any baseline. The stochastic sampler doubles the computational cost per sampling step (Algorithm 2 adds a forward perturbation at each step). Diffusion models are already known to be slower than GAN/VAE baselines; without efficiency analysis, the practical value of marginal quality improvements is unclear. — This matters because practical deployment of generative models for tabular data requires understanding the efficiency-quality tradeoff, and omitting this makes it impossible for practitioners to assess whether TabDiff's quality gains justify its computational cost.

### Minor

- **The "joint continuous-time diffusion" framing somewhat overstates the conceptual unification**: The paper frames TabDiff as a "joint continuous-time diffusion process" (abstract, Sec. 1), but the forward process factorizes by type (Eq. 1): *q*(**x**_t | **x**_0) = *q*(**x**_t^num | ·) · *q*(**x**_t^cat | ·). The interaction between numerical and categorical features occurs only through the shared denoiser network, which is standard in multi-modal models. The categorical process is a continuous-time Markov chain (CTMC) on a finite state space, not a diffusion in the SDE sense. The paper should acknowledge this distinction more clearly rather than presenting it as a single unified diffusion.

- **Ablation studies lack per-dataset granularity**: Table 5 reports only averaged Shape/Trend metrics across all datasets. Given that learnable schedules are the core contribution, per-dataset breakdowns (which features benefit most, and why) would substantially strengthen the analysis and help readers understand when adaptive schedules matter most.

- **Missing value imputation comparison is limited to TabSyn and XGBoost**: Table 4 only compares TabDiff against TabSyn for generative imputation, while other baselines (TabDDPM, CoDi, GReaT) are excluded. The claim of "superior conditional generation capacity" would be more convincing with broader baselines.

- **The paper claims "seven datasets" (Section 4.1) but the paragraph lists eight dataset names (including Faults), and Faults does not appear in any table**: This discrepancy should be clarified.

### Trivial

- **Equation (11) terminology**: The text calls α_{k_j}(t) = 1 - t^{k_j} a "log-linear schedule," but this is a power-law schedule (log-linear would imply the log of α is linear in t). The terminology is slightly misleading.

## Nice-to-Haves

- Visualize the learned noise schedules ρ_i and k_j across features to demonstrate that the model actually learns meaningfully different schedules for features with different distribution properties (e.g., skewness, multimodality). This would directly validate the core design motivation.

- Include a formal complexity analysis comparing TabDiff's sampling cost (with the stochastic sampler) against TabDDPM, TabSyn, and non-diffusion baselines.

- Evaluate on datasets with higher feature counts (50+) or high-cardinality categorical features to demonstrate scalability of the per-feature schedule approach and one-hot encoding.

## Removed Points

- **Harsh Critic Point #1 about "joint diffusion being overstated to the point of fatal flaw"**: While the framing could be more precise, the paper does explicitly show the factorization in Eq. (1) and the interaction through the shared denoiser. In the context of multi-modal generation, coupling different modalities through a shared denoiser with a shared time index is a standard and accepted approach (e.g., in image-text generation). This is a presentation/framing issue, not a methodological error, and is properly handled as a minor weakness above.

- **Harsh Critic Point #4 about "metrics under-specified in main text"**: The metrics (Shape, Trend, α-Precision, β-Recall, CS2T, DCR, MLE, Imputation) are standard in the tabular generation literature and follow TabSyn's evaluation protocol. Deferring details to an appendix is acceptable for a methods-focused paper. The paper explicitly acknowledges MLE's limitations. This is a presentation preference, not a substantive weakness.

- **Harsh Critic Point #5 about "stochastic sampler is incremental and not fully justified"**: The stochastic sampler is clearly adapted from Karras et al. (2022) and the paper cites this. The ablation (Table 5) demonstrates its empirical value. While the theoretical justification could be stronger, this is standard for practical diffusion improvements. Downgraded from the harsh critic's characterization.

- **Spark Point about "OOM entries making average comparisons unfair"**: Baselines that OOM on certain datasets are excluded from averages for those datasets, not included with zero scores. This is standard practice. The paper should discuss why certain models OOM, but this doesn't invalidate the comparisons on datasets where all methods can run.

- **Spark Point about "formal privacy analysis / differential privacy"**: The paper does evaluate DCR ( Distance to Closest Records) as a privacy metric, which is standard in tabular generation. Demanding formal differential privacy guarantees is scope creep for a methods paper focused on generation quality.

- **Human Finder Point about "high-cardinality categorical features"**: This is a valid concern but it applies to all one-hot-based tabular generation methods. No baseline in the comparison handles this fundamentally differently. Listed as a nice-to-have rather than a major weakness.

## Novel Insights

The stochastic sampler's design insight—that masked diffusion's categorical decoding is "frozen" once unmasked (Eq. 8), preventing self-correction—is genuinely important for mixed-type diffusion generation. This asymmetry between continuous features (which can be iteratively refined) and categorical features (which cannot, under the standard absorbing-state reverse process) is a structural property that prior work on mixed-type discrete-time diffusion (TabDDPM, CoDi) did not explicitly address. The proposed fix of injecting forward noise to "re-open" categorical features for correction is conceptually simple but addresses a real mechanistic limitation, and the ablation confirms it matters primarily for correlation metrics (Trend). This observation may be more impactful than the learnable schedules themselves.

## Suggestions

- Add a direct experimental comparison with CDTD (Mueller et al., 2024) and clearly articulate what TabDiff's original-space approach provides that CDTD's latent-space approach with adaptive schedules does not.

- Provide per-dataset ablation breakdowns for the learnable schedules, and visualize the learned ρ_i and k_j values to demonstrate that feature heterogeneity genuinely drives schedule divergence.

- Include training time, sampling time, parameter count, and memory usage comparisons with TabDDPM, TabSyn, and at least one GAN/VAE baseline to enable practical efficiency assessment.

- Clarify the Faults dataset discrepancy (listed in text but absent from results tables).

## Score and Decision

Comparing to calibration anchors:
- **CDTD** (very similar concurrent work: continuous diffusion + feature-wise adaptive schedules): Accept (Poster), scores 5.5 average. TabDiff has stronger empirical results and operates in original data space (avoiding latent encoding overhead), but CDTD was accepted despite similar novelty concerns.
- **TabSyn** (key baseline, scored 6.75 average, Accept oral): Incremental contribution (VAE + latent diffusion for tabular data) but very strong results and clean execution. TabDiff has comparable empirical improvements over prior baselines but less architectural novelty.
- **MULAN** (learned noise schedules for diffusion): Reject, scores 5.5. Novelty of learnable schedules alone was judged insufficient.
- **TabDAR** (combination of diffusion + autoregressive for tabular): Reject, scores ~4.75. Incremental combination of existing techniques.

TabDiff sits between CDTD and TabDAR: it has stronger empirical results than CDTD and more comprehensive evaluation, but faces the same novelty ceiling (combining existing diffusion components with adaptive schedules). The concurrent work overlap, absent experimental comparison with CDTD, and missing computational cost analysis are meaningful gaps. However, the results are genuinely strong and the method is clearly the current state-of-the-art on these benchmarks.

MY FINAL SCORE: 5.5
MY FINAL DECISION: <orange>Reject</orange>