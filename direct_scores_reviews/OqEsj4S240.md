## Summary
This paper introduces a framework based on zigzag persistence — a tool from Topological Data Analysis (TDA) designed for dynamic, time-varying data — to analyze the internal representations of Large Language Models. The key contribution is **persistence similarity**, a descriptor that quantifies how many topological features (p-cycles) in one layer survive to another, tracking the full evolutionary trajectory rather than comparing isolated snapshots. Applying this framework to seven decoder-only transformer models, the authors identify a consistent three-phase structure (increasing → plateau → decreasing average similarity with depth) and leverage the metric as a principled layer-pruning criterion, achieving performance comparable to state-of-the-art methods.

---

## Strengths

- **Methodologically creative adaptation of zigzag persistence**: The decision to interpret transformer layers as time snapshots in a discrete dynamical system, and to apply zigzag persistence accordingly, is technically well-motivated. Unlike per-layer persistent homology, this captures *trajectories* of topological features rather than static snapshots. The construction of intersection layers (Eq. 2) is mathematically principled and the use of existing tools (DIONYSUS2, FASTZIGZAG) is appropriate.

- **Three-phase structure is a concrete, non-trivial finding**: The observation that $\bar{S}_1$ exhibits a consistent increasing–plateau–decreasing profile across seven models spanning 7B to 70B parameters, two datasets (Pile and SST), and multiple $k_{\text{NN}}$ values (Figure 4) is a genuinely distinctive empirical result. The fact that the *rate of increase* in the early phase appears approximately universal across model families and sizes is notable and would not be predicted a priori.

- **Robustness checks across hyperparameters and datasets**: The paper systematically varies $k_{\text{NN}} \in [1,15]$ and compares two datasets, showing qualitative stability of the similarity curves. The consistency across dataset choices (Pile vs. SST) lends credibility to the finding.

- **Honest framing of pruning as a "showcase"**: The paper correctly frames layer pruning as a downstream application of the framework rather than a primary contribution, and Table 1 accurately reports mixed results. The pruning criterion (Algorithm 2) is simple and transparent.

---

## Weaknesses

### Fatal
None.

### Major

- **No ablation of zigzag persistence vs. static per-layer persistent homology**: The central claim of the paper is that tracking the *trajectory* of topological features across layers — rather than computing static homology at each layer separately — provides additional insight. This is never tested empirically. Without a baseline that computes per-layer Betti numbers or persistence diagrams independently and derives a similarity metric from them, there is no evidence that the zigzag trajectory information is strictly more informative than static snapshots. This directly undermines the core methodological claim.

- **No random pruning baseline in Table 1**: The paper claims comparable performance to state-of-the-art pruning methods. However, without a random-layer-removal baseline at the same pruning budget, it is impossible to determine whether the method performs better than uninformed selection. For aggressive 20% cuts, where performance degrades substantially (e.g., Llama 2 7B MMLU: 45.74 → 23.16 for "This work" at 20% cut), the value of a topologically-motivated criterion over random removal needs to be established.

- **Universality claim is not supported by the model selection**: The abstract and Section 4.2 claim "a universal structure in LLM internal representations." However, all seven tested models are decoder-only autoregressive transformers (Llama2/3, Mistral, Pythia), trained on broadly similar data distributions. Architectural outliers — encoder-only models (BERT, RoBERTa), encoder-decoder models (T5), or models with fundamentally different training objectives — are absent. The paper's own contribution list uses the more careful phrase "Similarity of Results Across Models and Hyperparameters," which is appropriate; "universality" in the abstract and Section 4.2 overstates what the evidence supports.

- **No comparison against non-topological similarity measures**: The paper does not demonstrate that the topological phase structure reveals something that simpler, computationally cheap measures (e.g., CKA, cosine similarity of representations, angular distance) would not. Since the pruning literature already establishes that middle layers are structurally redundant, the additional explanatory power of topology over geometry is undemonstrated.

### Minor

- **Pruning comparison protocol**: The number of layers pruned by other methods is constrained to match the output of the proposed method, meaning baselines cannot be evaluated at their individually optimal pruning budgets. While this is one reasonable comparison protocol, a sweep over pruning ratios for all methods jointly would give a clearer picture of the accuracy-vs-compression trade-off. Additionally, no confidence intervals or multiple-evaluation statistics are reported for Table 1 — though single-run evaluation at 5-shot is standard for large LLMs, differences of 1–2 percentage points (e.g., on Winogrande) may be within evaluation variance.

- **Computational cost not contextualized**: The 2-hour computation time for 10K points (footnote 2) vs. the near-instantaneous cost of angular distance or Bi-score similarity is a non-trivial barrier. The paper does not discuss approximation strategies or whether subsampling could reduce this cost while maintaining the phase structure, which limits the practical uptake of the framework.

- **Self-inclusion in average persistence similarity (Eq. 6)**: The average $\bar{S}_p(\ell) = \frac{1}{N_\text{layers}}\sum_{\ell_i} S_p(\ell, \ell_i)$ includes the trivially maximal self-similarity $S_p(\ell, \ell) = 1$. The effect of this self-inclusion on the shape and amplitude of the curves in Figure 4 is not discussed.

- **$k_{\text{NN}}$ selection criterion ad hoc**: The criterion of choosing $k_{\text{NN}}$ to maximize total cycle count (Section 4.2) is unusual and unmotivated. While the paper shows qualitative robustness across $k_{\text{NN}} \in [1,15]$, a principled or data-driven selection criterion would strengthen reproducibility.

- **Untrained model control absent**: The claim that the three-phase structure reflects learned representations rather than architectural inductive biases (e.g., residual connections) cannot be distinguished without a comparison to untrained (randomly initialized) models of the same architecture. This is important for interpreting whether topology captures training-induced semantics or simply structural priors.

### Tiny

- The self-similarity term is trivially 1 but is included in the average without comment; the impact on figure amplitudes should be noted.
- The condition $b,d > 0$ in Eq. 4 silently excludes edge cases at the first and last layers; a brief comment on whether this affects interpretation would add clarity.
- The priority claim — "it has not yet been recognized that the internal representations of LLMs can essentially be viewed as dynamic point clouds evolving in time" — is too strong given prior work on layer-wise intrinsic dimension evolution (cited in the paper's own Related Work). The novelty should be scoped specifically to zigzag persistence, which is fair.

---

## Nice-to-Haves

- Include instruction-tuned/RLHF-finetuned model variants to test whether topological universality survives post-alignment.
- Plot performance degradation vs. pruning percentage continuously (rather than two snapshots at 10% and 20%) to reveal whether methods fail gracefully or catastrophically at intermediate thresholds.
- Explore whether specific persistent 1-cycles correlate with token clusters or probing task performance (e.g., syntactic vs. semantic probes), which would connect the topological framework to linguistic interpretability.
- Apply the framework to model checkpoints during training to validate the dynamic system interpretation over the training trajectory as well as the depth trajectory.
- Discuss or briefly experiment with whether the topological phase structure is robust to using non-last-token representations (e.g., mean pooling), even if a full ablation is not feasible.

---

## Removed Points
*These points are flagged for removal; treat them with caution.*

- **Critic's claim that Eq. 4 is a lossy operation**: Footnote 3 explicitly states the operation "does not modify the information about the model layers contained in the original $\text{Pers}_p(\Phi)$, as it redefines consistently all the births and deaths." This is a well-defined coordinate projection following Kim & Mémoli (2017), and the critic's concern appears to misread the footnote. **Removed.**

- **Critic's concern that kNN graphs "lack a natural distance-based filtration parameter"**: This is the point of the paper — the filtration parameter is layer index (time), not a geometric scale. The design is intentional and the critic misunderstands the construction. **Removed.**

- **Critic's claim that the pruning comparison is unfair to baselines**: The opposite is true — fixing the pruning budget to match this work's output and then letting baselines choose *which* layers to prune freely is actually favorable to the baselines, since they can optimize layer selection within that budget. The asymmetry, if anything, favors baselines. **Removed.**

- **Critic's concern about no standard deviations in Table 1 as a methodological failure**: Single-run evaluation without confidence intervals is the norm for 5-shot LLM benchmarking on large models. This is not a methodological flaw relative to community standards. Retained only as a minor note above. **Weakened, not removed.**

- **Critic's claim that the 20% cut's severe performance degradation undermines practical utility**: The paper explicitly frames pruning as a "showcase" application, not a production method. Criticizing the pruning regime's practical utility misreads the paper's stated scope. **Removed.**

---

## Novel Insights

The most genuinely novel observation in this paper — which the reviewers partially surface but do not fully develop — is the *rate universality* of the increasing phase: not only does average persistence similarity peak in the middle layers across diverse model sizes and families, but the *slope* of the increase in early layers appears approximately constant regardless of model architecture, size, or $k_{\text{NN}}$ choice. If this rate universality survives in architecturally diverse models (including non-autoregressive ones), it would constitute a strong constraint on how transformer-like architectures build up topological structure during forward propagation — potentially linking to residual stream geometry or attention head aggregation behavior. This deserves a dedicated quantitative analysis (fitting the slope, testing its dependence on training data, depth normalization, etc.) rather than being left as a visual observation.

---

## Suggestions

1. **Add a static-homology baseline**: Compute per-layer Betti numbers or pairwise similarity using conventional persistent homology (one diagram per layer) and show that zigzag-derived persistence similarity contains strictly more information. This is the single most important missing experiment.

2. **Add a random pruning baseline**: Include random layer selection at the same pruning budget in Table 1 to calibrate whether the topological criterion provides genuine signal over chance.

3. **Quantify the rate of the increasing phase**: Fit a linear slope to the early-phase increase in $\bar{S}_1$ for each model and report the slope values in a small table. This would turn a visual claim of universality into a quantitative one.

4. **Run the framework on a randomly initialized (untrained) model**: This control experiment, which is computationally cheap (no benchmark evaluation needed), would clarify whether the phase structure is architecturally or training-induced.

5. **Temper the "universality" language**: Replace "universal structure in LLM internal representations" with "consistent topological structure across decoder-only transformers of varying sizes" in abstract and Section 4.2 until broader architectural coverage is achieved.

6. **Discuss approximation strategies for computational scaling**: Even a brief paragraph on approximate zigzag algorithms or subsampling strategies would substantially improve the framework's practical accessibility.

---

**Evaluation summary:**
- **Novelty**: High — the application of zigzag persistence to LLMs and the persistence similarity descriptor are genuinely original.
- **Technical soundness**: Moderate — the mathematical construction is correct and well-specified, but key design choices (kNN criterion, m=4, last-token restriction) lack ablations, and the central methodological advantage over static homology is unproven.
- **Empirical support**: Moderate — the three-phase pattern is convincing across models and hyperparameters, but the missing zigzag-vs-static and random-pruning baselines leave significant gaps.
- **Significance**: Moderate — the interpretability finding (three-phase structure) is interesting; the pruning application is a proof of concept, not a performance advance.
- **Clarity**: Good — the paper is well-structured and the mathematical framework is explained clearly, with appropriate figures.

MY FINAL SCORE: <pineapple>5.4</pineapple>