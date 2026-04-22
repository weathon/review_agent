# Perturbed Dynamic Time Warping: A Probabilistic Framework and Generalized Variants

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
Dynamic Time Warping (DTW) is a classical method for measuring similarity between time series, but its non-differentiability hinders integration into end-to-end learning frameworks. To address this, soft-DTW replaces the minimum operator with a smooth soft-min, enabling differentiability and efficient computation. Motivated by soft-DTW, we propose perturbed-DTW, a differentiable framework of DTW obtained by adding random perturbations to warping costs and taking the expected minimum. Under Gumbel noise, perturbed-DTW exactly recovers soft-DTW, providing a natural probabilistic interpretation of soft-DTW. We further generalize this framework by extending the Gumbel noise to the broader family of generalized extreme value (GEV) distributions, leading to a new class of soft-DTW variants. Building on this insight, we introduce nested-soft-DTW (ns-DTW), which integrates GEV perturbations into the dynamic programming formulation of perturbed-DTW. This extension induces alignments with tunable skewness, offering greater flexibility in modeling diverse alignment structures. We validate ns-DTW on barycenter computation, clustering, and classification, demonstrating its effectiveness over existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a new probabilistic perturbation-based interpretation of soft-DTW, formulated using a concept similar to the Gumbel-Max (softmax) trick—specifically, its softmin counterpart (Lemma 1).
Under this formulation, the authors define perturbed-DTW and show that when the perturbation noise follows a $\mathrm{Gumbel}(-c, 1)$ distribution (where $c$ is the Euler–Mascheroni constant, the perturbed-DTW becomes exactly equivalent to soft-DTW.

Furthermore, the paper introduces nested-soft-DTW (ns-DTW), which employs the generalized extreme value (GEV) distribution—a generalization of the Gumbel distribution—as the perturbation noise model.
Theoretical analysis (Theorem 1) derives a closed-form solution for this generalized case.
In ns-DTW, the set of alignment matrices is partitioned into $J$ groups, each associated with a parameter $\tau_\ell \in (0,1]$. Groups with smaller $\tau_\ell$ values correspond to higher correlations among the perturbation samples within that group.
Like soft-DTW, ns-DTW can be efficiently solved via dynamic programming (Algorithm 1).

Experiments using the UCR Time Series Classification Archive compare ns-DTW with existing methods (e.g., DBA, soft-DTW) on barycenter estimation, classification, and clustering tasks.
Results show that ns-DTW achieves comparable or slightly superior performance to soft-DTW in certain parameter settings.

### Strengths
* The paper provides a probabilistic perturbation-based interpretation of soft-DTW and extends it to a more general formulation, ns-DTW.
* The proposed ns-DTW demonstrates improved performance over soft-DTW under some experimental conditions.

### Weaknesses
* The method introduces $\tau_1$ as a hyperparameter, but it is unclear how this value was selected in the experiments. Since setting $\tau_1 = 1$ makes ns-DTW equivalent to soft-DTW, this choice is crucial for a fair comparison.
* The performance improvement over soft-DTW appears limited (e.g., Fig. 5(c)), and in some settings (e.g., Table 1 with $\gamma = 0.1$), ns-DTW performs worse than soft-DTW.

### Questions
* ll. 045–046: The paper refers to “through the lens of perturbed optimizer (Blondel et al., 2021)”, but according to Blondel et al. (2021), no discussion of perturbed optimizers appears. Is this citation incorrect?
* Eq. (8): Since $\langle A, C \rangle$ is a scalar while $\epsilon$ is described as a vector, the subtraction seems ill-defined.
* In the introduction of ns-DTW, $\mathcal{A}_{m,n}$ is said to be divided into $J$ groups, but the description around line 312 (and Algorithm 1) divides the transition vectors into two groups. The correspondence between these explanations is unclear.
* In Fig. 3, it is not clearly explained how varying $\tau_1$ affects the skewness of the resulting distribution.
* Eq. (16): The text states that this equation assumes two groups “to ease the illustration,” yet the same form is used in Algorithm 1. Shouldn’t the algorithm adopt the general formulation instead? The same issue applies to Eq. (17).
* As shown in Table 1, ns-DTW can produce smaller DTW losses by adjusting the skewness of the distribution, but the underlying reason for this effect is not well explained.
* l. 374: “nd-DTW” -> “ns-DTW”?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Perturbed-DTW, a probabilistic and differentiable extension of traditional Dynamic Time Warping (DTW) that injects random perturbations into the alignment cost and takes the expectation of the minimum cost. Under Gumbel noise the formulation recovers the well-known soft‑DTW, providing a probabilistic interpretation of its smoothing effect. The authors further generalize to the generalized extreme value (GEV) family of distributions, giving rise to a nested-soft-DTW (ns-DTW) variant with tunable skewness in the alignment structure. They validate the method across barycenter computation, clustering, and classification on several UCR time-series datasets, showing improved performance relative to standard DTW and existing differentiable DTW variants.

### Strengths
1. Presents a clear probabilistic interpretation of soft-DTW through perturbed optimization, which is conceptually elegant.

2. Extends the formulation to the generalized extreme-value (GEV) family, yielding a tunable nested-soft-DTW variant with adjustable skewness — a novel theoretical contribution.

3. The mathematical exposition is generally sound and the experimental evaluation spans several UCR datasets.

### Weaknesses
1. The supplementary archive hides the core implementation and scripts, making the experiments effectively non-reproducible. This lack of transparency raises serious doubts about the validity and reproducibility of the reported results.

2. Empirical analysis is largely numerical and does not provide insight into runtime cost or robustness; ablations and qualitative visualizations are limited.

3. Presentation could be tightened: notation is occasionally heavy, and experimental details (parameter settings, initialization, convergence) are insufficient for independent verification.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper reframes soft-DTW as the expected DTW under i.i.d. Gumbel perturbations and generalizes to GEV-perturbed alignments, yielding a nested/“skewable” variant (ns-DTW). It preserves dynamic-programming structure and complexity, with a DP recursion that mirrors soft-DTW (noting the nested DP is not identical to the closed-form nested objective). On UCR subsets, ns-DTW improves barycenters and clustering and is competitive for nearest-centroid classification. Overall, it clarifies theory and adds a useful control knob for shaping alignments.

### Strengths
- Clean probabilistic interpretation of soft-DTW (soft-DTW = E[DTW with Gumbel-perturbed costs]). I believe  many practitioners will find such an intuitive interpretation of soft-DTW clarifying.
- Principled GEV-based extension that induces tunable skewness/correlation across alignment moves.
- Same order of computation as soft-DTW, makes it easy drop-in to existing pipelines.
- Consistent gains for barycenters and clustering across many datasets.

### Weaknesses
- Little guidance on choosing γ and τ; limited sensitivity and compute profiling.
- Classification baselines could be stronger (e.g., 1-NN DTW and 1-NN soft-DTW).
- Limited sensitivity and runtime/memory analysis, 
- The link between the nested objective and its DP instantiation could be spelled out more clearly.

### Questions
The paper is well-motivated and, I believe, will stimulate further study and applications. I was left wondering 
1. how forward/backward runtimes and memory compare to soft-DTW for fixed (m, n) on the same hardware, and 
2. how sensitive the results are to γ and τ, including any simple rule of thumb for selecting them, an ablation over grouping choices would also be helpful.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The author proposes perturbed-DTW, that adds random noise to the alignmetn costs, and then taking average over the noise distribution. With this modification, the DTW optimal solution became a distribution over paths, rather than a single-path one. This also naturally makes the DTW differentiable like soft-DTW. The author also found, under Gumbel noise distribution, the perturbed-DTW reduces to soft-DTW. Therefore, the perturbed-DTW is a differentiable relaxation of the soft-DTW. In the experiments, the authors demonstrate the effectiveness of ns-DTW on diverse time-series tasks, showing that it captures meaningful alignment structures while remaining computationally tractable.

### Strengths
1. The paper has strong theoretical contibution. It provides a new probabilistic interpretation for soft-DTW, and extends it to a new family with GEV.
2. The proposed method also remains practical and computationally tractable.
3. Experiments shows good representation of time series can be captured using the proposed ns-DTW.

### Weaknesses
1. Some hyper-params might needs tunning.
2. The ns-DTW might exhibit high computation cost.

### Questions
How much computation cost in terms of big O notation, would be invovled when using perturbaed-DTW ?

### Soundness
4

### Presentation
3

### Contribution
3
