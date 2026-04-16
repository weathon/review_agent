Now I have enough context for calibration. Let me summarize my understanding:

**Calibration anchors:**
- **CDTD** (QPtoBPn4lZ): Very similar to TabDiff - continuous diffusion for mixed-type tabular data with adaptive noise schedules. Accepted as poster, scores 6/5/5/6 (avg ~5.5). Reviews noted incremental novelty but solid results.
- **TabSyn** (4Ay23yeuz0): Direct baseline, latent diffusion for tabular data. Accepted as oral, scores 8/6/8/5 (avg ~6.75). Strong results, incremental but well-executed.
- **TabDAR** (kkGIbmpCHU): Diffusion+autoregressive tabular generation. Rejected, scores 5/3/5/6 (avg ~4.75). Weak methodology.
- **TabUnite** (Zoli4UAQVZ): Encoding schemes for tabular diffusion. Withdrawn/Rejected, scores 3/5/3/3/5 (avg ~3.8). Lacks novelty.

Now let me consolidate the review for TabDiff.

**Key assessment of TabDiff:**

The paper proposes a joint continuous-time diffusion framework that directly models numerical (VE-SDE) and categorical (absorbing/masked) features in data space, with feature-wise learnable noise schedules and a mixed-type stochastic sampler. This is technically more principled than CDTD (which maps everything to continuous embeddings first) and more direct than TabSyn (which uses a VAE latent space).

**Strengths:**
1. Principled joint formulation in continuous-time, operating directly in data space
2. Feature-wise learnable schedules are well-motivated and effective (ablation shows improvement)
3. Stochastic sampler is a nice practical contribution for mixed-type data
4. Strong empirical results across 7 datasets and 8 metrics
5. Clean extension to CFG for conditional generation

**Weaknesses I need to evaluate:**

From harsh critic:
1. **Evaluation protocol reuses baselines**: TabDiff is re-trained by authors while baselines come from prior paper. This is partially valid - but note the paper explicitly reproduces TabSyn (footnote 1), and this is standard practice in the field (TabSyn itself reuses baselines from TabDDPM). This is a common concern but the authors have been transparent about it. I'll include it as a moderate concern.
2. **Limited ablation**: Only on Adult dataset, averaged metrics only. This is valid.
3. **Ambiguity in metrics**: Definitions deferred to appendix. This is common practice, and the Shape/Trend metrics are from prior work (TabSyn).
4. **CFG derivation/implementation**: The implementation involves two models (not standard CFG with condition dropout). This is a valid concern about clarity, but the approach works.
5. **Complexity vs dataset scale**: Valid that no efficiency comparison is provided.

From neutral reviewer:
1. CFG implementation clarity: Same as above, valid.
2. Scalability concerns: Valid but speculative.
3. Forward process independence: This is a valid theoretical point but standard in hybrid models.
4. Privacy results deferred: Valid but common in the field.

From spark's suggestions:
1. No computational cost comparison: Valid.
2. Ablations on only one dataset: Valid.
3. No continuous-time vs discrete-time ablation: Reasonable but would be a lot of work.
4. Imputation lacks standard baselines beyond TabSyn: Valid.
5. No visualization of learned schedules: Valid - this would strengthen the paper.
6. No failure mode analysis: Valid.

From human finder:
1. Incremental technical novelty: Some validity - composing known diffusion types with noise schedules. But the integration is non-trivial and the specific design choices matter.
2. Computational cost: Same concern as above.
3. High-cardinality categorical features: Valid but speculative concern.
4. Ablation scope: Same as above.
5. Heuristic design choices: Valid - the functional forms of schedules are not theoretically justified.
6. Privacy results in appendix: Same concern.

Now let me synthesize the final review. The paper is solid but has real weaknesses. Compared to CDTD (accepted poster, avg 5.5) and TabSyn (accepted oral, avg 6.75), TabDiff is at a similar or slightly better level. It has stronger empirical results than CDTD and a more principled framework, but shares some of CDTD's concerns about incremental novelty and limited ablations. I'd place it around 6.</think>Now let me synthesize the final review based on careful cross-checking of all reviewer claims against the actual paper.

## Summary
TabDiff introduces a joint continuous-time diffusion framework for mixed-type tabular data generation. It models numerical features via a variance-exploding (VE) SDE and categorical features via an absorbing-mask process, both operating directly in data space (avoiding latent encodings). The key innovations are: (1) feature-wise learnable noise schedules that adapt to per-column distribution heterogeneity, (2) a mixed-type stochastic sampler that allows self-correction during decoding, and (3) a classifier-free guidance (CFG) extension for conditional missing-value imputation. Experiments across seven datasets and eight metrics demonstrate consistent improvements over prior state-of-the-art, with up to 22.6% average improvement on pairwise column correlation estimation (Trend) over TabSyn.

## Strengths
- **Principled joint continuous-time formulation**: TabDiff directly operates on numerical and categorical data in their native formats within a unified continuous-time diffusion framework, avoiding the information loss of latent-space encodings (TabSyn) and the looser ELBO bounds of discrete-time approaches (TabDDPM, CoDi). This is a technically cleaner foundation than prior work.
- **Feature-wise learnable noise schedules are well-motivated and effective**: The power-mean schedule (Eq. 10) for numerical features and log-linear schedule (Eq. 11) for categorical features directly address a core challenge of tabular data—column-wise distribution heterogeneity. The ablation (Table 5) shows a clear improvement from fixed to learnable schedules on both Shape (1.24→1.17 avg) and Trend (2.29→1.80 avg), and Figure 2 demonstrates reduced training loss.
- **Mixed-type stochastic sampler addresses a real failure mode**: The "once unmasked, stays fixed" property of the absorbing process in categorical features is a genuine problem for tabular data with strong inter-column correlations. The stochastic sampler (Algorithm 2) introduces self-correction, and the ablation confirms consistent gains over deterministic sampling (Table 5).
- **Comprehensive empirical evaluation**: Seven datasets, eight metrics (including fidelity, downstream tasks, and privacy), and eight baselines spanning GANs, VAEs, autoregressive models, and diffusion methods. Consistent improvements on Shape and Trend are convincing.
- **Clean extension to conditional generation**: The CFG extension to mixed-type data for missing-value imputation is a meaningful practical contribution, and Table 4 shows it yields consistent improvements (ω=0.6 vs ω=0.0).

## Weaknesses

### Major:
- **Limited ablation depth for core methodological claims**: The two main innovations—feature-wise learnable noise schedules and the stochastic sampler—are ablated only on the Adult dataset using averaged Shape/Trend metrics (Table 5). There is no per-dataset breakdown, no isolation of per-feature schedules vs a single global learnable schedule, and no analysis of numerical-only vs categorical-only schedule effects. Given that these are the paper's primary claimed contributions, this is a meaningful evidential gap. Additionally, no visualization or interpretation of the learned schedule parameters (ρ_i, k_j) is provided—without this, the claim that schedules "adapt to feature-specific distributions" remains asserted rather than demonstrated.

- **Fairness of experimental comparisons**: The paper reuses baseline results from Zhang et al. (2024) (explicitly stated in Table 1 footnote) for all baselines except TabSyn (which the authors reproduce) and the Diabetes dataset. This means TabDiff benefits from the authors' own hyperparameter tuning and potential implementation advantages (e.g., different preprocessing, early stopping, or hardware), while baselines are evaluated under potentially different conditions. The claimed margins (13.3% Shape, 22.6% Trend average) could partially reflect these discrepancies. The authors do note this in the footnote, but it remains a structural concern.

- **No computational cost or efficiency analysis**: The stochastic sampler adds a forward perturbation step at every reverse step (Algorithm 2), roughly doubling per-step cost. The continuous-time formulation typically requires many discretization steps. Feature-wise schedule parameters add overhead. Neither training time, sampling time, number of function evaluations, nor memory usage are reported. This omission is significant for practical adoption, especially since TabSyn explicitly touts faster synthesis as an advantage.

- **Heuristic design choices lack theoretical justification**: The specific functional forms of the noise schedules (power-mean for numerical: Eq. 10, log-linear for categorical: Eq. 11) are presented without theoretical grounding. The choice to fix initial/final noise levels across all features (Section 2.3) and the joint loss weighting terms λ_num and λ_cat (Eq. 12) are also heuristic. While empirically effective, these are not derived from first principles, and no sensitivity analysis is provided for λ_num, λ_cat, or the schedule parameterization families.

### Minor:
- **Forward process independence assumption**: The hybrid forward process factorizes as q(x_t|x_0) = q(x_t^num|...) · q(x_t^cat|...) (Eq. 1), meaning cross-modal correlations in the forward process are not explicitly modeled—only learned through the reverse denoising network. While standard for hybrid models, this is a modeling choice whose consequences are not analyzed.

- **CFG implementation clarity**: Section 2.5 describes the conditional model as the original unconditional TabDiff and the unconditional model as "an additional small model" trained only on target columns—this deviates from standard CFG (which uses condition dropout in a single model). The rationale for this design choice is not thoroughly discussed, and the appendix reference for implementation details is not available in the review material.

- **MLE metric unreliability acknowledged but unresolved**: The authors explicitly note in Section 4.3 that "methods with varying performance on data fidelity metrics might have very close MLE scores" and that MLE "may not be a reliable indicator of data quality." Including it as a primary metric group despite this acknowledgment is internally contradictory, though the fidelity metrics (Shape/Trend) carry the weight of the argument.

- **Privacy results fully deferred to appendix**: Privacy protection is a core motivation stated in the Introduction, and DCR is listed as one of eight evaluation metrics. However, all DCR results are in the appendix, making it impossible to assess the privacy–utility tradeoff from the main paper.

### Trivial:
- The notation regarding α vs σ in the categorical schedule (Eq. 11 defines α_{k_j}^{cat}(t) = 1 - t^{k_j}, while earlier α_t = exp(-σ^{cat}(t))) creates some ambiguity but is resolvable with careful reading.

## Nice-to-Haves
- Visualization of learned per-feature noise schedules (σ_{ρ_i}(t) curves for numerical, α_{k_j}(t) curves for categorical) to substantiate the claim that they adapt to feature heterogeneity.
- Ablations on at least 2-3 more datasets (especially Diabetes, which is more categorical-heavy and where TabDiff shows the largest gains) to establish robustness of the core components.
- Wall-clock training/sampling time and memory comparisons against TabSyn and TabDDPM.
- Analysis of failure cases—TabDiff underperforms TabSyn on Shape for the Default dataset (1.24 vs. 1.01) and ties on several MLE datasets; discussing when and why the method struggles would strengthen the paper.
- A per-feature schedule ablation (global learnable vs. per-feature learnable) to isolate the benefit of feature-specific adaptation versus just having a learnable schedule at all.

## Removed Points
These points are flagged to be removed; treat them with caution.
- **"Incremental technical novelty"—composing known diffusion types:** The harsh critic and human finder question whether combining VE-SDE and absorbing diffusion is "just" composition. However, the integration involves non-trivial design: continuous-time formulation for both, a joint training objective, feature-wise schedules, and a stochastic sampler that handles the mixed-type interaction. While individual components build on prior work, the system contribution is meaningful and the empirical gains are substantial. Downgraded from "major" to "noted."
- **High-cardinality categorical scalability is speculative concern:** The human finder raised that one-hot encoding creates high-dimensional sparse representations for high-cardinality features. This is a valid design consideration but the paper does not claim scalability to such settings, and all standard benchmarks in this field use modest-cardinality features. Scope creep.
- **Missing standard imputation baselines (MICE, missForest):** The imputation task (Table 4) is designed to showcase CFG within the diffusion framework, not to compete with general-purpose imputation methods. Including them would be informative but is not the paper's main claim.
- **Continuous-time vs. discrete-time ablation:** Demanding a discrete-time variant of TabDiff as an ablation is a substantial engineering effort. The theoretical motivation for continuous-time (tighter ELBO) is established in prior work (Kingma et al., 2021; Song et al., 2021).
- **Reproducibility concerns about unreported hyperparameters:** The λ_num, λ_cat, and other implementation details are referenced as being in the appendix, which is standard. Nitpicking undisclosed hyperparameters is outside reasonable review scope.

## Novel Insights
The stochastic sampler's design specifically addresses the irreversibility of the absorbing ("once masked, stays masked") process in categorical features—the categorical tokens, once decoded, cannot be revised even if they prove inconsistent with later numerical denoising. This is a genuine problem unique to hybrid mixed-type models that does not arise in pure continuous diffusion (where all dimensions can be revised continuously). The paper's solution—adding a small forward perturbation step to re-mask likely-incorrect categorical tokens—is a conceptually simple but operationally important fix. However, the ablation evidence for this is thin (one dataset, one metric pair), and the community would benefit from a more thorough analysis of when and how much re-masking matters.

## Suggestions
- Visualize the learned per-feature noise schedule parameters (ρ_i, k_j) across columns of different types and distributions. This directly validates the paper's central design claim.
- Report training time, sampling time, and GPU memory alongside all baselines on the same hardware, as computational efficiency is crucial for practical adoption.
- Run ablations on at least 2-3 additional datasets (especially Diabetes and one with many numerical columns) to establish robustness of the learnable schedules and stochastic sampler.
- Include a brief sensitivity analysis for λ_num, λ_cat to help practitioners set these hyperparameters.

## Score and Decision

**Calibration:**
- **CDTD** (QPtoBPn4lZ): Very similar methodological niche (continuous diffusion + adaptive schedules for mixed-type tabular data). Accepted as poster, human scores 6/5/5/6. CDTD uses continuous embeddings for categoricals (losing discrete structure) and lacks a stochastic sampler. TabDiff is technically more principled and achieves better empirical results.
- **TabSyn** (4Ay23yeuz0): Direct baseline, accepted as oral, human scores 8/6/8/5. TabSyn uses a VAE+latent diffusion pipeline, which is conceptually simpler but introduces encoding overhead. TabDiff operates directly in data space, has comparable-or-better results, but has weaker ablation depth than TabSyn had.
- **TabDAR** (kkGIbmpCHU): Rejected, scores 5/3/5/6. Weaker methodology, mixed autoregressive+diffusion approach. TabDiff is clearly stronger.
- **TabUnite** (Zoli4UAQVZ): Withdrawn/Rejected, scores 3/5/3/3/5. Largely a recombination of encoding tricks. TabDiff is substantially stronger.

TabDiff sits above CDTD (which was accepted as poster) due to a more principled formulation (data space rather than embedding space), effective stochastic sampler, and stronger empirical results. It sits below TabSyn (which received oral) due to TabSyn's overall cleanliness, faster sampling speed claims, and more polished execution. TabDiff's main limitations—limited ablation depth, no efficiency analysis, heuristic design choices—prevent it from reaching the same level but do not undermine the core contribution.

**Score: 6** — A solid, above-threshold paper with a meaningful system contribution and strong empirical results, but with enough gaps in analysis (especially ablations and efficiency) that it falls short of a strong accept.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>