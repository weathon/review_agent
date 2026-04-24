Now let me do calibration searches.Now I have enough information to write the final review. Let me synthesize.---

## Summary
TabDiff introduces a joint continuous-time diffusion framework for mixed-type tabular data generation that directly operates on numerical features (via Gaussian VE-SDE) and categorical features (via masked absorbing diffusion) in their native formats, without encoding to a latent space. The paper's key innovations are feature-wise learnable noise schedules (parameterized per-column), a mixed-type stochastic sampler that enables error correction during sampling, and classifier-free guidance extended to the mixed-type setting for imputation. Extensive experiments on seven datasets across eight metrics show that TabDiff achieves consistent improvements over the prior state-of-the-art (TabSyn), with the strongest result being a 22.6% average improvement on pairwise column correlation (Trend) across all seven datasets.

---

## Strengths

- **Feature-wise learnable noise schedules backed by ablation (Table 5, Figure 2):** The power-mean schedule for numerical features (Eq. 10, parameter ρᵢ) and log-linear schedule for categorical features (Eq. 11, parameter kⱼ) are jointly learned end-to-end. Table 5 shows they reduce Shape from 1.24→1.17 and Trend from 1.92→1.80 (paired with stochastic sampler), and Figure 2 confirms substantially lower training losses for both categorical and numerical components compared to fixed-schedule baselines.

- **Joint continuous-time diffusion in original data space:** Unlike TabSyn (which uses a VAE encoder to convert all features to a continuous latent space) and CDTD (which uses score interpolation in an embedding space), TabDiff applies principled Gaussian VE-SDE to numerical features and masked absorbing diffusion to categorical features directly in their native types. The continuous-time ELBO (Eqs. 9, 12) avoids the discretization-induced bound looseness of discrete-time methods like TabDDPM/CoDi.

- **Mixed-type stochastic sampler with demonstrated effectiveness (Table 5):** Algorithm 2 introduces an intermediate forward perturbation step at each backward iteration, allowing self-correction of already-decoded categorical features that would otherwise be frozen. The ablation is clean: under fixed schedules, Shape improves from 1.39→1.20 and Trend from 2.29→1.93; under learnable schedules, improvements are 1.24→1.17 and 1.92→1.80. The stochastic sampler alone provides a larger Shape improvement than the learnable schedule alone (0.19 vs 0.15), confirming its practical value.

- **Consistent and large margin on Trend metric (Table 2):** The 22.6% average improvement over TabSyn on pairwise column correlation is the paper's most convincing result — it holds across all seven datasets, with improvements ranging from 4.4% (Beijing) to 37.3% (Diabetes). This is the paper's core empirical claim and it is fully supported.

- **CFG extension to mixed-type conditional generation (Table 4):** The unified CFG framework handles both numerical (Eq. 15) and categorical (Eq. 16) guided sampling without requiring a separate classifier. With ω=0.6, TabDiff achieves 5.60% average imputation improvement vs. the non-generative XGBoost baseline, outperforming TabSyn's 4.99%, and confirming CFG's efficacy on 5/7 datasets.

---

## Weaknesses

### Fatal
None.

### Major

- **Asymmetric baseline evaluation: TabSyn reproduced by the authors, all other baselines taken from the original TabSyn paper (Zhang et al., 2024).** As stated in footnote 1 of Table 1: "TaBSYN's performance is obtained via our reproduction. The results of other baselines except on Diabetes are taken from Zhang et al. (2024)." This creates a structural asymmetry where the principal competitor is evaluated under the TabDiff authors' training environment, while all other baselines are evaluated under TabSyn's authors' (likely tuned) environment. If the reproduction of TabSyn is even slightly sub-optimal in any hyperparameter, all improvements over it are inflated. The paper provides no analysis of where the reproduction matches or diverges from original TabSyn numbers, and does not report the original TabSyn published numbers as a cross-check. This is the most methodologically significant concern.

### Minor

- **MLE "15% improvement" framing is misleading relative to the paper's own self-qualification.** Section 4.3 reports TABDIFF "outperforms TaBSYN by 15.0%," derived from the Average Gap column in Table 3 (gap from real-data performance: 5.76% vs. 6.78%). In absolute AUC/RMSE values, the differences are trivially small: Adult .912 vs. .909, Default .763 vs. .763 (tied), Magic .936 vs. .937 (TabSyn marginally better), News .866 vs. .862. Critically, the paper itself immediately concedes: "methods with varying performance on data fidelity metrics might have very close MLE scores. This suggests that the MLE score evaluated under the current setting may not be a reliable indicator of data quality." Foregrounding a 15% claim from a metric the authors themselves call unreliable, when absolute differences are within standard error, is a real presentation problem even though the paper partially self-corrects.

- **Shape metric is not consistently superior to TabSyn at the per-dataset level.** The abstract claims "superior average performance over existing competitive baselines across all eight metrics," but on Shape (Table 1), TabDiff loses to TabSyn on two of seven datasets: Default (1.24 vs. 1.01, a meaningful regression) and News (2.35 vs. 2.06). For the Trend metric, TabDiff's wins are consistent, but the Shape claim requires qualification — the paper only delivers consistent superiority in aggregate, not uniformly.

- **CFG is load-bearing for imputation claims, but this is underemphasized.** Table 4 shows that without CFG (ω=0.0), TabDiff achieves only 3.76% average improvement vs. XGBoost, compared to TabSyn's 4.99% — TabDiff without guidance underperforms TabSyn on 5/7 datasets. The stated improvement of 5.60% requires CFG. The reliance of the conditional generation result on CFG should be stated more prominently.

- **No sensitivity analysis for λ_num and λ_cat loss weights.** Eq. 12 introduces two balancing weights for the joint loss. Because the numerical loss (MSE of ε) and categorical loss (cross-entropy) have fundamentally different scales, this balance is a real design choice. Neither the main text nor a visible ablation reports the chosen values or sensitivity to them.

### Trivial

- The fixed schedule baselines in the ablation use ρ_i = 7 and k_j = 1 as defaults. The motivation for these specific values is not discussed; reporting the distribution of learned ρ_i and k_j values for representative datasets would strengthen the argument that learning meaningfully adapts schedules rather than simply recovering reasonable fixed values.

---

## Nice-to-Haves

- A runtime comparison (training + sampling time) against TabSyn would be valuable for practitioners, since TabSyn's VAE-based latent approach may be faster at inference and the paper makes no claim about computational efficiency.
- Formal statistical significance tests (e.g., paired t-tests across 20 samples) on key comparisons, particularly where margins are small (Shape on Default, MLE scores), to solidify the empirical claims.
- A visualization of the distribution of learned noise schedule parameters (ρ_i and k_j) across features on a representative dataset, which would confirm that learning produces meaningful per-feature adaptation rather than near-uniform convergence.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Ablation reveals the backbone alone is competitive with TabSyn"** (Harsh Critic): Factually incorrect. TABDIFF-Fix.+Det. achieves Shape = 1.39, which is actually *worse* than TabSyn's 1.35. The backbone without the key innovations *underperforms* TabSyn, meaning both innovations genuinely contribute. The critic misread the table direction.

- **"Stochastic sampler contributes more than learnable schedule as the primary innovation"** (Harsh Critic): This is a framing preference, not a flaw. Both components contribute independently and are clearly supported by the 2×2 ablation in Table 5. Per Shape metric: stochastic sampler adds 0.19, learnable schedule adds 0.15; for Trend: both contribute ~0.37. The relative magnitudes are close enough that calling either "primary" is reasonable. Not a weakness.

- **Strength: "Consistent state-of-the-art results across all eight metrics"** (Strength Finder): Partially dropped. The claim of *consistent* superiority is not supported at per-dataset resolution for Shape (losses on Default and News). Retained as a qualified strength for Trend only, which is genuinely consistent across all 7 datasets.

- **Strength: "End-to-end training with no special encoding/decoding pipelines"** (Strength Finder): Dropped as generic and insufficiently grounded in a specific contribution beyond architecture design. The main novelty is what is modeled in that pipeline, not that training is end-to-end.

---

## Novel Insights

The stochastic sampler design reveals a fundamental tension specific to masked absorbing diffusion for categorical features: once a token is "unmasked" during the backward process, standard DDPM-style sampling leaves it frozen for the remainder of the trajectory (Eq. 8), making it impossible to correct early decoding errors in the presence of inter-column correlation. TabDiff's sampler directly addresses this by re-masking intermediate decoded tokens before each denoising step, which is a natural but non-obvious extension of continuous stochastic sampling to mixed-type settings. The ablation confirms this is the *individually* stronger contribution for Shape correction (0.19 Shape improvement from stochastic sampler alone vs. 0.15 from learnable schedules alone), suggesting that the categorical frozen-state pathology is a bigger bottleneck than schedule heterogeneity for marginal distribution matching.

---

## Suggestions

1. Add comparison against TabSyn's published (original) numbers as a sanity check alongside the reproduction, or explain why the reproduction differs, to address the asymmetric baseline concern.
2. Revise the MLE claim in the abstract and Section 4.3 to reflect that the 15.0% is a gap-reduction figure on a metric the authors themselves qualify, not an absolute accuracy improvement.
3. Qualify the "consistent superiority" claim to note that the paper loses to TabSyn on the Shape metric for two specific datasets (Default and News), and note this in the conclusion as a limitation.
4. Report learned ρ_i and k_j distributions for at least one dataset to validate the claim that the model meaningfully specializes schedules per feature.
5. Add a brief analysis of λ_num, λ_cat sensitivity (even a 3-point sweep) to establish robustness.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Decision | Comparison |
|---|---|---|---|---|
| CDTD (mixed-type tabular diffusion, adaptive schedules, embedding space) | QPtoBPn4lZ.md | 5.50 | Accept Poster | Very close topic; TabDiff is clearly superior (native space, stronger results on Trend, stochastic sampler) |
| TabSyn (tabular diffusion in VAE latent space) | 4Ay23yeuz0.md | 6.75 | Accept Oral | TabDiff's direct SOTA baseline; its own contributions received similar "mostly incremental but solid" characterization |
| CardiCat (VAE for high-cardinality tabular) | vW6rsXAGrz.md | 4.00 | Reject | Low anchor; weaker methodology and thinner empirical evidence |
| TabDAR (diffusion-nested autoregressive) | kkGIbmpCHU.md | 4.75 | Reject | Low anchor; mixed results and methodological issues |

**Reasoning:** TabDiff sits clearly above CDTD (5.50): it is more principled (native data space vs. embedding interpolation), has cleaner theoretical grounding, and consistently outperforms TabSyn on the Trend metric across all seven datasets, which CDTD could not demonstrate. The key weaknesses (MLE overclaiming, Shape losses on 2/7 datasets, TabSyn reproduction asymmetry) are real but do not invalidate the core contribution. The 22.6% consistent Trend improvement over TabSyn is the paper's crown result and it holds up to scrutiny. At the same time, TabDiff does not reach TabSyn's oral quality (6.75): TabSyn's contribution was paradigmatic for the subfield (introduced the latent-space approach), while TabDiff's improvements are incremental over it, and the presentation issues are non-trivial. A score of **6.5** positions TabDiff above CDTD and below TabSyn, which matches the qualitative comparison well.

**Score: 6.5 — Accept (Poster)**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>