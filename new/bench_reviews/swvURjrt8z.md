## Summary

TABDIFF proposes a continuous-time diffusion model for mixed-type tabular data that operates in the original data space, avoiding latent encodings used by prior work. It combines a variance-exploding SDE for numerical features with a masked absorbing diffusion for categorical features, trained end-to-end via a unified ELBO. The paper introduces feature-wise learnable noise schedules and a mixed-type stochastic sampler, and reports state-of-the-art average performance across seven datasets and eight metrics.

## Strengths

- **Native-space continuous-time formulation for mixed-type data.** The paper avoids latent-space encodings by directly diffusion-modeling normalized numerical features and one-hot categorical features (with a [MASK] state) in their original representations, and trains both modalities with a single joint ELBO (Eq. 12). This is a clear practical advantage over latent-space methods such as TabSyn.
- **Feature-wise learnable schedules improve fidelity.** Table 5 shows that enabling learnable schedules reduces average Shape error from 1.39 (fixed, deterministic) to 1.24 (learnable, deterministic) and reduces training losses (Figure 2). This demonstrates that per-column schedule flexibility provides real empirical benefit.
- **Stochastic sampler improves generation quality.** Table 5 isolates the stochastic sampler’s impact: with fixed schedules, switching from deterministic to stochastic sampling improves Shape from 1.39 to 1.20 and Trend from 2.29 to 1.93. This is evidence that the proposed corrector-like mechanism helps.
- **Strong average empirical results.** Across seven datasets, TABDIFF achieves the best average Shape (1.17 vs. TabSyn 1.35) and Trend (1.80 vs. TabSyn 2.33), and competitive or best results on MLE and missing-value imputation (Tables 1–4).

## Weaknesses

### Fatal
None.

### Major
- **Critical inconsistency between numerical training objective and sampling algorithm.** Equation (5) trains the numerical denoising network $\boldsymbol{\mu}_\theta^{\text{num}}$ to predict the noise $\boldsymbol{\epsilon}$, yet Algorithm 2 (line 12) computes the probability-flow direction as $(\mathbf{x}^{\text{num}} - \boldsymbol{\mu}_\theta^{\text{num}})/\boldsymbol{\sigma}^{\text{num}}$, which is the correct ODE update only if $\boldsymbol{\mu}_\theta^{\text{num}}$ predicts the clean data $\mathbf{x}_0$. For the variance-exploding formulation (Eq. 3–4), a noise-predicting model implies the ODE direction is $\boldsymbol{\mu}_\theta^{\text{num}}$ itself (since $d\mathbf{x} = \boldsymbol{\epsilon}\, d\sigma$), whereas an $\mathbf{x}_0$-predicting model implies the direction $(\mathbf{x} - \boldsymbol{\mu}_\theta^{\text{num}})/\sigma$. The paper cannot have it both ways as written. Because the sampler and the loss refer to two incompatible parameterizations, the core generative procedure is ambiguous and irreproducible from the text alone. This is a serious flaw for a methods paper.
- **Evaluation relies on externally sourced baselines without statistical validation.** With the exception of TabSyn (reproduced by the authors) and the Diabetes subset, all baseline numbers are imported directly from Zhang et al. (2024). This raises fair-comparison concerns: differences in preprocessing, data splits, or hyperparameter tuning can easily affect the small absolute gaps reported (e.g., Magic Trend: 0.88 vs. 0.76; Beijing Trend: 3.13 vs. 2.59). The paper reports no statistical significance tests, and the headline claim of “superior average performance … across all eight metrics” is therefore not firmly established.

### Minor
- **Central motivation for feature-wise schedules is not validated.** The paper claims the schedules “counteract the high heterogeneity across different feature distributions” (Sec. 2.3), but the ablation (Table 5) only shows that learnable schedules outperform fixed ones. It does not analyze the learned values of $\rho_i$ or $k_j$, nor show any correlation between a feature’s marginal statistics (variance, skewness, category count) and its learned schedule. Without this, the schedules may simply act as per-feature capacity knobs rather than adaptive allocations that respect distributional disparity.
- **Classifier-free guidance implementation deviates from its theoretical derivation.** Equations (13)–(16) derive CFG assuming a single model $\theta$ that can be evaluated either conditionally or unconditionally. The actual implementation uses the full joint model (with $\mathbf{y}$ fixed) as the conditional model and a separate, smaller model trained only on the missing columns as the unconditional model. Because the two models differ in architecture and training distribution, the interpolation in Eq. (15)–(16) is not theoretically grounded for this setup, and the paper does not discuss this approximation.
- **Categorical stochastic sampler lacks theoretical justification.** The approximate reverse kernel in Eq. (8) assumes that once a categorical feature is unmasked it stays fixed. The stochastic sampler in Sec. 2.4 relaxes this by re-perturbing decoded features, but the paper provides no argument that the resulting Markov chain preserves the correct marginal distribution. Citing continuous-diffusion samplers (Karras et al., 2022) does not automatically justify the categorical case.
- **MLE is acknowledged to be unreliable yet counted in the overall superiority claim.** The paper notes that “the MLE score evaluated under the current setting may not be a reliable indicator of data quality” (Sec. 4.3), yet MLE is still one of the eight metrics used to claim overall superiority. This weakens the strength of that aggregate claim.

### Trivial
- The loss weights $\lambda_{\text{num}}$ and $\lambda_{\text{cat}}$ in Eq. (12) are introduced without any discussion of how they are set or whether they are tuned.
- The relationship between the two categorical parameterizations—$\alpha_t=\exp(-\sigma^{\text{cat}}(t))$ in Sec. 2.2 and $\alpha_{k_j}^{\text{cat}}(t)=1-t^{k_j}$ in Eq. (11)—is left implicit rather than explicitly reconciled.
- The ablation in Table 5 averages results across all datasets, masking per-dataset variance.

## Nice-to-Have
- Correlation analysis plotting learned $\rho_i$ and $k_j$ against feature statistics (standard deviation, skewness, category count) to validate the heterogeneity-adaptation claim.
- Self-contained baseline re-runs or paired statistical significance tests for the small gaps against TabSyn.
- Architecture specification (input dimensions, embedding strategy, number of layers, attention details) in the main text to aid reproducibility.
- Evaluation on true missing-value imputation with random missingness patterns (MCAR/MAR) and multiple simultaneous missing columns, rather than only the single target column of each dataset’s built-in task.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Joint” forward process criticism.** The reviewer argues that Eq. (1) factorizes independently across numerical and categorical features, so calling it a “joint continuous-time diffusion process” overstates novelty. This misunderstands diffusion models: even standard image diffusion has a factorized forward process $q(\mathbf{x}_t|\mathbf{x}_0)=\prod_i q(x_t^i|x_0^i)$; the coupling arises in the learned reverse process. The paper’s usage is conventional and correct.
- **Claim that “no existing method explores mixed-type diffusion framework in the continuous-time limit” is too strong.** The reviewer says this is false because the forward process is separable. But the paper’s claim is about operating in the original data space (not latent) with continuous-time diffusion for mixed types. Prior work either uses latent encodings (TabSyn) or discrete-time processes (CoDi, TabDDPM). The claim is accurate in context.
- **Characterization of TabSyn as suffering from “encoding overhead” and “low model capacity.”** The reviewer finds this questionable because TabSyn wins on two datasets. However, the paper never singles out TabSyn for these criticisms; it describes latent-space methods generally, and TabDiff does outperform TabSyn on average. This criticism is a misreading.
- **Evaluation metrics deferred to Appendix A.2.** Under space constraints, deferring metric definitions to the appendix is standard practice and not a meaningful weakness.
- **Missing appendix / proofs.** The parser strips appendix sections; they exist in the original submission.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. **Clarify the training–sampling parameterization.** The authors must explicitly state whether $\boldsymbol{\mu}_\theta^{\text{num}}$ predicts noise $\boldsymbol{\epsilon}$ or clean data $\mathbf{x}_0$, and ensure Eq. (5) and Algorithm 2 agree. If the network predicts $\boldsymbol{\epsilon}$, the sampler should use $d\mathbf{x}^{\text{num}} = \boldsymbol{\mu}_\theta^{\text{num}}\, d\sigma$ (or the equivalent discretization). If it predicts $\mathbf{x}_0$, Eq. (5) should be rewritten as a loss on $\mathbf{x}_0$ or the appropriate preconditioning should be shown.
2. **Add significance tests or self-contained baselines.** Either re-run all baselines with identical preprocessing and splits, or report confidence intervals and paired statistical tests (e.g., bootstrap or t-test on the 20 random samples) for the small average gaps, especially on Trend where several per-dataset margins are narrow.
3. **Validate the adaptive-schedule hypothesis.** Plot the learned schedule parameters against simple feature statistics. If they do not correlate, the paper should soften its claim from “counteract heterogeneity” to “increase per-feature capacity.”

## Score and Decision

**Calibration anchors:**
- **TabSyn** (`/home/wg25r/review_agent/human_reviews/4Ay23yeuz0.md`, avg 6.75, Oral): Clean methodology, strong results, available code, well-written. TABDIFF has comparable empirical strength but is marred by the Eq. 5/Algorithm 2 inconsistency and weaker experimental controls, placing it clearly below TabSyn.
- **CDTD** (`/home/wg25r/review_agent/human_reviews/QPtoBPn4lZ.md`, avg 5.50, Poster): Similar topic (continuous diffusion for mixed-type tabular data). Reviewers found it incremental but methodologically sound. TABDIFF introduces more novel components (native-space modeling, stochastic categorical sampler) and stronger average results, but pays a price in methodological clarity because of the training–sampling mismatch. Overall comparable to or slightly below CDTD.
- **TabDAR** (`/home/wg25r/review_agent/human_reviews/kkGIbmpCHU.md`, avg 4.75, Reject): Strong empirical results but messy notation, correctness questions, and unclear methodology. TABDIFF is better written and its contributions are clearer, but it shares the liability of a concrete algorithmic inconsistency. TABDIFF sits above TabDAR.
- **CATDM** (`/home/wg25r/review_agent/human_reviews/JD6j7XSluo.md`, avg 3.50, Withdrawn): Serious baseline discrepancies and weak overall performance. TABDIFF is well above this anchor.

**Reasoning:** TABDIFF presents genuinely useful ideas—native-space continuous diffusion, feature-wise schedules, and a stochastic categorical sampler—and backs them with strong average empirical results. However, the inconsistency between Eq. (5) and Algorithm 2 is a severe methodological flaw for a methods paper: a reader cannot implement the core algorithm from the text alone because the training objective and sampler assume incompatible network outputs. Combined with the reliance on externally sourced baseline numbers and the lack of statistical validation for small gaps, the paper falls below the standard of a clean accept. If the authors resolve the training–sampling mismatch and provide stronger experimental validation, the work would rise to the level of a strong poster or better.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>