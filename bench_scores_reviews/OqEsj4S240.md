## Summary

This paper introduces a framework based on **zigzag persistence** — a tool from Topological Data Analysis — to characterize the dynamic evolution of internal representations across the layers of pre-trained LLMs. Treating layers as time snapshots of an evolving point cloud, the authors define a novel descriptor called **persistence similarity**, which tracks the fraction of topological features (p-cycles) that survive the entire trajectory between two layers, rather than comparing only endpoint representations. The empirical study reveals a consistent three-phase pattern (increasing → plateau → decreasing) in average similarity across multiple decoder-only LLMs, and persistence similarity is demonstrated as a viable pruning criterion, achieving results broadly comparable to state-of-the-art similarity-based methods.

---

## Strengths

- **Genuinely novel machinery.** The adaptation of zigzag persistence to the specific setting of LLM layer sequences — with layers as discrete time steps and kNN clique complexes at each step — is a non-trivial methodological contribution. The persistence similarity metric of Eq. (5) is qualitatively different from prior static TDA descriptors or pointwise activation similarity: it accumulates evidence over the entire trajectory between two layers, rather than comparing endpoint states. This trajectory-sensitivity property is unique among published layer-similarity measures.

- **Empirical universality finding is robust and specific.** The three-phase pattern in $\bar{\mathcal{S}}_1$ peaks at the same *relative depth* across Llama2 (7B, 13B, 70B), Llama3 (8B, 70B), Mistral 7B, and Pythia 6.9B, and is stable under variation of $k_{\rm NN}$ across a wide range $[1, 15]$. While qualitatively similar to observations from intrinsic-dimension analyses (Valeriani et al., 2023), the quantitative stability of the curve shape across model families and sizes, as a function of rescaled depth, goes beyond a restatement of prior results. The paper correctly positions this as a topological validation of a geometric observation.

- **Pruning competitive on multiple benchmarks with a novel criterion.** Despite selecting fundamentally different layers than both Gromov et al. (2024) and Men et al. (2024) in many cases (notably Mistral 7B and Llama 3 8B), persistence similarity achieves comparable or superior benchmark performance on HellaSwag, Winogrande, and MMLU for most model-task pairs (Table 1). This demonstrates that topological trajectory information can substitute for activation-based similarity without a systematic degradation.

---

## Weaknesses

### Fatal
None.

### Major

- **The core claimed advantage — trajectory-sensitivity — is never empirically validated.** The central selling point of persistence similarity over angular similarity or Block-Influence score is that it captures the *path* between layers, not just the endpoints. Yet no experiment constructs or identifies a case where two layer pairs have similar endpoint representations but different intermediate paths, and shows that the methods disagree or that one assignment is more informative. Without this, the trajectory-sensitivity advantage remains entirely asserted. All downstream pruning results are consistent with the possibility that endpoint-based and path-based metrics happen to select similar functional information, which would mean the added TDA complexity is unnecessary.

- **Non-trivial performance gaps undermine the "comparable performance" claim.** Table 1 contains several cases where the performance gap is substantive:
  - MMLU, Llama2-7B, 10% cut: **37.38** (this work) vs. **43.95** (other works) — a 6.6-point gap
  - HellaSwag, Pythia, 10% cut: **31.43** vs. **34.96** — a 3.5-point gap
  
  While the paper outperforms baselines in other cases (e.g., Mistral 7B MMLU: 53.17 vs. 38.20), the existence of large underperformance cases is inconsistent with the abstract's blanket claim of "comparable performance." A more nuanced characterization of when the method succeeds and when it does not is needed.

- **No ablation separating zigzag trajectory-sensitivity from standard per-layer TDA.** The paper compares against activation-similarity-based pruning, but not against applying standard (non-zigzag) persistent homology at each layer independently and using Betti numbers or bottleneck distance for pruning. Without this baseline, the marginal value of the zigzag formulation over simpler static TDA is entirely unestablished. This is particularly important given the large computational overhead of the zigzag computation.

- **Computational overhead is unaddressed for the practical application.** Footnote 2 states ~2 hours for 10K points with $d = 4096$ (at $k_{\rm NN} = 10, m = 10$, which is more expensive than the settings actually used). Even if actual runtime is faster, it is orders of magnitude more expensive than cosine-similarity-based approaches. For a method presented as a practical pruning tool, the absence of any runtime comparison or discussion of cost-benefit trade-off is a significant omission, particularly when benchmark results are sometimes worse.

### Minor

- **The theoretical justification for the pruning heuristic is weak.** Algorithm 2 prunes layers with *high* persistence similarity on the grounds that they are redundant. In TDA, long-lived (persistent) features typically represent stable signal, not noise. The paper does not provide a theoretical argument for why topologically stable layers are functionally expendable — this is the opposite of the usual TDA intuition. The argument presumably is that "stable" means "not changing," which implies redundancy in a sequential chain — but this reasoning needs to be stated and justified explicitly.

- **The asymmetry of $\mathcal{S}_p$ (Eq. 5) is acknowledged but not quantified.** The paper notes approximate symmetry empirically, but provides no quantification (e.g., mean absolute asymmetry, or plots of $|\mathcal{S}_p(\ell_1, \ell_2) - \mathcal{S}_p(\ell_2, \ell_1)|$). This matters because the interpretation of persistence similarity as a "fraction of cycles that persist" changes depending on direction, and a non-trivial asymmetry would require justifying which direction is used in the pruning criterion.

- **No error bars on $\bar{\mathcal{S}}_p$ curves (Figure 4).** Results are computed on a 10K-sentence subset of Pile. The variability of the three-phase pattern under different random subsets of the same size is unknown. Given that the universality claim rests on visual alignment of curves across models, quantifying sample-to-sample variability is important.

- **$k_{\rm NN}$ selection criterion is internally circular.** The paper selects $k_{\rm NN} = 5$ because "it gives the highest values of $\bar{S}_1$" (Section 4.2). Since $\bar{S}_1$ is the metric being analyzed (and used for pruning), selecting the hyperparameter that maximizes it is circular. The selection should be justified by a downstream criterion (e.g., pruning performance) or by a principled connectivity argument.

### Tiny

- The choice of $m = 4$ (maximum simplex dimension) is stated but never justified. The paper notes 0- and 3-cycles are sparse, but does not show whether varying $m$ affects results.
- The "universality" claim in the title and abstract is stronger than the evidence supports: all tested models are decoder-only transformers in the 7B–70B range. Calling this "universal structure in LLM representations" overstates the coverage. The paper should hedge this more carefully.

---

## Nice-to-Haves

- **Direct comparison with CKA or standard representation similarity for pruning.** This would clarify whether topological complexity offers any advantage over simpler geometric descriptors at equal computational budget.
- **Full Pareto curve (accuracy vs. % layers pruned)** rather than two fixed thresholds (10%, 20%). This is standard for structured pruning papers and allows fairer comparison across all budgets.
- **kNN sensitivity analysis on layer *ranking*.** Figure 4 (left) shows that $k_{\rm NN}$ affects the magnitude but apparently not the shape of $\bar{S}_1$. Explicitly showing that the *ranking* of layers by similarity is stable across $k_{\rm NN}$ values would directly validate the pruning criterion's robustness.
- **Ablation on sample size** (1K vs. 10K vs. 50K sentences) to show that the three-phase structure is not an artifact of undersampling.
- **Code release** with documented dependencies on DIONYSUS2 and FASTZIGZAG, given the specialized TDA pipeline.

---

## Removed Points

*These points were raised in sub-reviews but are removed or substantially weakened on closer reading of the paper. Treat with caution.*

- **"Unfair comparison protocol" in pruning (Harsh Critic).** The protocol is explicitly disclosed: the paper feeds its own $N_{\rm prune}$ to baselines to test whether the *criterion* matters at fixed budget. This is a reasonable, if constrained, comparison design. It is not systematically biased against baselines — indeed, at 20% cut, baselines are free to optimize within the given count. The protocol limits the comparison but does not invalidate it.
- **kNN-based filtration being "non-standard"** for zigzag (Harsh Critic). The paper cites Le & Taylor (2024) as a prior use of kNN-based filtrations, and the intersection construction in Eq. (2) is a natural consequence.
- **Demanding encoder-only (BERT) or encoder-decoder (T5) architectures** (Harsh Critic, Spark Finder). The paper is explicitly scoped to autoregressive decoder-only LLMs (Section 3). Evaluating on BERT or T5 is out of scope; noting it as a future direction is sufficient.
- **Eq. (4) "double-counting"** concern (Harsh Critic). Footnote 3 states that the operation "does not modify the information about the model layers contained in the original ${\rm Pers}_p(\Phi)$, as it redefines consistently all the births and deaths." This is an information-preserving reparametrization, not a summation that double-counts contributions.
- **Fourth contribution being "just an empirical observation"** (Harsh Critic). Empirical universality observations are legitimate contributions in representation learning; this criticism is editorial.
- **The framing that "it has not yet been recognized that LLM representations can be viewed as dynamic point clouds"** (Harsh Critic calling this "overstated"). Intrinsic-dimension analyses do track changes across layers, but they do not frame the problem as a time-varying point cloud amenable to zigzag persistence. The framing is novel even if the qualitative conclusions have precedent.

---

## Novel Insights

The most interesting observation from this paper — underappreciated in the sub-reviews — is the stability of the *rate of increase* in $\bar{S}_1$ during the early-layer phase across architectures of different sizes. The left panel of Figure 4 shows that varying $k_{\rm NN}$ changes only the normalization (amplitude) of the curve, not the shape; the right panel shows the shape is nearly identical across models when depth is rescaled. This implies that the *topology of the progression* is a feature of the transformer training objective or architecture family, not of any individual model. If this holds more broadly, it would be a non-trivial structural constraint on how transformer-trained representations evolve — a constraint invisible to static geometric methods. The present paper does not fully develop this observation (it is partially addressed in Appendix D), but it is the most scientifically interesting finding here and warrants a more central treatment.

---

## Suggestions

1. **Design a targeted trajectory-sensitivity experiment.** Identify pairs of adjacent layers $(A, B)$ and $(A', B')$ such that $\mathcal{S}_p(A, B) \approx \mathcal{S}_p(A', B')$ but where the intermediate paths differ, or alternatively use two models where the endpoint representations are similar but the paths differ. Show that persistence similarity distinguishes cases that endpoint-based metrics cannot. This is the single most impactful addition to the paper.

2. **Add a static TDA baseline for pruning.** Compute Betti numbers or bottleneck distances between *adjacent* layer complexes (no zigzag), use these to rank layer redundancy, and compare pruning performance. This directly isolates the contribution of the zigzag (trajectory) formulation.

3. **Rewrite the abstract and conclusion to accurately characterize pruning results.** Rather than "comparable performance," use something like: "competitive performance in most settings, with notable underperformance on MMLU for Llama2-7B" — and explain the failure case in the main body.

4. **Add a runtime comparison table.** Report wall-clock time for the full pipeline (representation extraction + zigzag computation + pruning) versus angular similarity and Block-Influence score at each model scale. Discuss whether approximations (subsampling, lower-dimensional projections, approximate kNN) could close the gap.

5. **Quantify the asymmetry of $\mathcal{S}_p$.** Add a heatmap or summary statistic of $|\mathcal{S}_p(\ell_1, \ell_2) - \mathcal{S}_p(\ell_2, \ell_1)|$ and clarify which direction is used in Eq. (6) and Algorithm 2.

---

**Overall assessment:** The paper makes a **genuine methodological contribution** by adapting zigzag persistence to the LLM interpretability setting and introducing the persistence similarity descriptor. The three-phase universality finding is well-supported and interesting. However, the central claimed advantage of the method — trajectory-sensitivity — is never directly validated, the pruning application has a notable failure case that is understated, the computational cost is not justified, and the comparison with non-zigzag TDA is absent. The paper is **above the novelty bar** for ICLR but requires substantive revision to meet the empirical rigor and clarity of claims expected at that venue.