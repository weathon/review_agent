## Human Reviewer 1

### Summary
This paper proposes a two-stage algorithm for offline change point localization in dynamic multilayer random dot product graphs. Stage I uses seeded binary segmentation with CUSUM statistics; Stage II refines estimates via tensor heteroskedastic PCA. The authors establish consistency for change point detection/localization, derive limiting distributions under vanishing and non-vanishing jump regimes, and provide a data-driven confidence interval procedure. Experiments on synthetic and real networks demonstrate strong performance.

### Strengths
1. The theoretical contributions are novel, deriving limiting distributions for change point estimators in network data. The localization rates match minimax optimality for single-layer networks while extending to multilayer settings.

2. The paper provides consistency results, limiting distributions, a data-driven confidence interval procedure, and extensive robustness checks.

### Weaknesses
While online change point detection in multilayer networks has been recently studied (Wang et al., 2025), the paper does not clearly articulate what specific technical challenges arise in the offline setting or why such an extension from online methods is non-trivial.

### Questions
Please refer to the question in the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 2

### Summary
The paper proposes and analyzes a new moethod for offline change point localization and inference in dynamic multilayer networks (following the D-MRDPGs model). They prove consistency of their estimator, and derive limiting distributions of the refined estimators in general regimes. They provide methods for computing confidence intervals, and demonstrate the favourable performance of their estimator on both synthetic and real data.

### Strengths
The paper is original in that they consider a model that has not been studied before. This model is well motivated from the literature and appears in practice. To the best of their (and my) knowledge, this is the first result of its kind in the context of dynamic network data. 

The paper is well written, they state the scalings very clearly which is important in the area of change points. Their theory is sound, and they use standard techniques/sub-methods from the change point literature such as CUSUM statistics. They also acknowledge potential improvements in the conclusion section which are sound. 

The contribution is significant to some extent, as the experiments indicate an experimental improvement over previous models that seems to come from the fact that they are looking at a more specific problem than competing methods (some competing methods are nonparametric). However, I must mention that the techniques are not in themselves novel (at least at first glance), but it would be good if the authors could comment on this.

### Weaknesses
From my understanding, competing methods only work change point detection in single layer dynamic networks, or they are inherently nonparametric methods. Can these be adapted to the multilayer setting? Can you comment on how fair this comparison is? It is not clear to me at this point. 

Also, implications from the experimental results could be made more clear, for example, by including figures in the main body of the text. You state that “across all scenarios, our method achieves best performance” but that does not seem to be true, as if I dig into Table 3 in Appendix F.1, it seems that in the n=50 case you do not clearly outperform kerSeg (nets.). Could you comment on this, and at least make it clear in the paper that you do not outperform in all cases, and maybe some intuition as to why? Showing some of these results graphically also would help for clarity. 

The comparison with other methods on real world data should be made more clear (in Appendix F.2 for example), and brought into the main paper. This is very important, and can help better clarify the benefit of your method, which is currently not so clear.

### Questions
- Middle of page 3: “We allow all model parameters … to diverge with T”. Please clarify this, how do they relate to T exactly?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper develops a two-stage CUSUM pipeline for change-point detection in dynamic multilayer random dot product graphs. Stage-1 performs seeded/binary segmentation with a tensor CUSUM; Stage-2 applies a low-rank TH-PCA/matched-filter refinement that exploits shared latent positions across layers, yielding a $\sqrt{L}$ SNR gain and $O(\kappa^{-2})$ localization rate under large minimal spacing and global layer-weight changes. Proofs establish consistency and limit laws; simulations and small real data are provided.

### Strengths
The paper is clearly written; the model, assumptions, and the two-stage algorithm are easy to follow.

Clean multilayer insight: the Stage-2 projection/matched filter makes precise how low rank and layer pooling provide $\sqrt{L}$ SNR gains.

Technical development is careful (ensuring consistency and limiting distribution for the refined estimator); a standardized pipeline (screen then refine) is appealing.

### Weaknesses
1. $\textbf{Scope (extension vs.~novelty):}$ The contribution appears closely aligned with prior multilayer RDPG pipelines [1] that (i) \emph{screen} for candidate changes via CUSUM/binary segmentation–type statistics and then (ii) refine via a low-rank projection step (and the target is similar). In the current draft, the primary difference seems to be a regime shift (online $\rightarrow$ offline) while keeping the same modeling assumptions (multilayer RDPG/GRDPG) and the same screen–then–refine structure and target. As written, this reads more like an extension to the offline setting than a fully new methodological framework.


2. $\textbf{Narrow regime.}$ The methodology focuses on time-invariant latent positions with only changes in global layer weights. This means that the communities of nodes remain consistent over time; in this scenario, the change points appear to be narrowly defined, as they cannot disrupt the community structure of the networks. Many practical settings involve node-localized changes alongside shifts in community structure, where the latent positions $X(t)$ should also be able to vary (See [2] for an example). The current scope does not account for these regimes. 


3. $\textbf{Spacing dependence.}$ The paper assumes a strong gap at the order $T$--with a gap at the order of $T$ and the number of change points is $O(1)$, spacing-free $\ell_0$ approach can already achieve $\textbf{exact selection}$ and the same $O(\kappa^{-2})$ localization approach without a minimal spacing condition. In contrast, using binary segmentation requires setting up the minimal spacing window, which is not necessary and could be misleading. For example. The paper effectively must generate evenly spaced change points. If, instead, change times are drawn i.i.d.\ from a distribution—e.g.,
$
\eta_1,\ldots,\eta_K \sim \mathrm{Unif} \{ 2,\ldots,T \},
$
the minimal spacing
$
\Delta_{\min}:=\min_{1\le k\le K+1}\bigl(\eta_k-\eta_{k-1}\bigr)
$
typically collapses. This destroys the large-spacing assumption required by a CUSUM screen and makes any fixed spacing window ill-posed. By contrast, spacing-free $\(\ell_0\)$ approaches remain optimal/consistent in this setting and do not require enforcing even spacing.


For the $\ell_0$ approach to estimation of dynamic (possible multilayer) networks, I refer to [2], but I believe more literature can be found.

For comparison between CUSUM and $\ell_0$ approach, I refer to [3]. See Section 4, 'However, compared to the $l_0$-penalization methods, WBS is computationally more expensive and involves more tuning parameters.''.

4. Simulations use (nearly) evenly spaced changes, reinforcing the strong-spacing assumption; real-data experiments also use very small $n,T,L$, This leaves the reader with the impression that the method hinges on large spacing and may not scale or adapt to realistic, short-gap or node–localized scenarios.

[1] Wang, Fan, et al. "Multilayer random dot product graphs: Estimation and online change point detection." Journal of the Royal Statistical Society Series B: Statistical Methodology (2025): qkaf051.

[2] Zhao, P., Bhattacharya, A., Pati, D., & Mallick, B. K. (2022). Factorized fusion shrinkage for dynamic relational data. arXiv preprint arXiv:2210.00091.

[3] ''Wang, Daren, Yi Yu, and Alessandro Rinaldo. "Univariate mean change point detection: Penalization, CUSUM and optimality." Electronic Journal of Statistics 14 (2020): 1917-1961.

### Questions
1. Comparison with $\ell_0$ approaches. Please include a discussion comparing against spacing-free $\ell_0$ methods. Proper references should be cited and compared; I list some, but I think more can likely be found.

2. I think including a vectorized version of multiplayer networks and then using binary segmentation is worth being included as a baseline to see how much the improvement in stage 2 is. 

3. To demonstrate robustness beyond the current (near-)even spacing and global-weight-jump setting, please add stress tests that randomly change times i.i.d. from a distribution. 


The paper has very strong assumptions, and the model is not very applicable given those assumptions. However, the paper is technically solid, and the TH-PCA refinement plus distributional results are valuable. I would not oppose acceptance if proper questions are addressed or at least discussed.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4