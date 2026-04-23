Now I have enough information to write the final review. Let me compile my assessment.

## Summary

The paper develops deep α-stable kernel processes as the infinite-width limit of deep Bayesian neural networks with elliptical infinite-variance prior weights. By exploiting the conditionally Gaussian representation of elliptical α-stable vectors, the authors obtain recursive formulas for the conditional covariance kernels (extending Cho & Saul 2009) and provide a tractable MCMC posterior inference procedure that avoids the exponential-in-dimension computational bottleneck of the prior shallow-network work by Loria & Bhadra (2024).

## Strengths

- **Conditionally Gaussian representation with recursive kernel formula (Theorem 1):** The insight that deep BNNs under elliptical infinite-variance priors converge to processes with α-stable marginals that admit a conditionally Gaussian representation is genuinely useful. The recursive formula for Σ^{(ℓ)} naturally extends Cho & Saul (2009) to the infinite-variance regime, and the observation that kernels become stochastic for α < 2 (enabling feature learning) while degenerating to deterministic for α = 2 is a clean theoretical result.

- **Kernel-space computation resolves the exponential-in-dimension bottleneck:** By working in kernel space rather than feature space, the method avoids the O(n^{I+2}) complexity of Loria & Bhadra (2024). This is concretely demonstrated in Tables 1 and 3, where the Stable method is infeasible for I > 2 while Dα-KP runs on I = 10 and I = 13 datasets.

- **Formal feature learning characterization (Proposition 3):** The proposition formally establishes that features depend on observations when α < 2 but are independent of observations when α = 2, making the comparison with deep GPs precise and connecting stochastic kernels to representation learning.

- **Complete posterior inference framework:** Proposition 2 provides closed-form posterior predictive distributions conditional on the scale parameters, and Algorithm 1 implements full MCMC with uncertainty quantification. Figure 2b demonstrates the practical benefit: Dα-KP posterior intervals appropriately capture jump discontinuities where GP intervals oversmooth.

- **Honest presentation of negative depth results:** Table 2 openly shows no empirical benefit from increasing depth, and the paper does not hide this finding.

## Weaknesses

### Fatal
None.

### Major

- **Depth provides no empirical benefit, undermining the "deep" contribution:** The paper's primary claimed advance over Loria & Bhadra (2024) is extending from shallow (single hidden layer) to deep networks. However, Table 2 shows that increasing depth from L=2 to L=16 yields essentially no improvement in any simulation setting. All experiments in Sections 3.2 and 4 use L=2. The authors' explanation that "the resultant stable process can capture a function class rich enough with even one hidden layer" (Section 3.3) is conjectural and unsupported by analysis. This leaves the practical contribution as a kernel-space reformulation for the shallow case—which is genuinely valuable but substantially narrower than the "deep" framing suggests. The paper should either demonstrate settings where depth matters or honestly reframe the contribution as primarily theoretical (deep extension) plus a practical computational improvement (kernel-space) that is most impactful for shallow architectures.

- **The "naturalness" framing overstates the generality of the construction:** The paper repeatedly contrasts its "natural" infinite-width limit with the "artificial noise injection" of Aitchison et al. (2021). However, the construction in Theorem 1 uses a specific hierarchical prior: a single shared scale parameter s_+^{(ℓ)} per layer for ℓ ≥ 2, meaning all neurons within a layer share the same mixing variable. This is one particular tractable construction—not the generic infinite-variance limit. If the weights were truly i.i.d. with infinite variance (e.g., i.i.d. Cauchy), the limit would be different and likely intractable. The shared-scale-per-layer structure is as much a design choice as Aitchison et al.'s noise injection; the paper should acknowledge this rather than presenting its construction as uniquely natural. (See the abstract: "this artificial noise injection could be critiqued in that it would not naturally emerge in a classic BNN architecture"; and Section 1: "these processes do not result naturally as an infinite-width limit.")

### Minor

- **Narrow experimental evaluation:** All simulation truths are step functions or sign functions, specifically designed to favor heavy-tailed methods. No experiments test performance on smooth functions (where GP methods might be preferred) or mixed settings, making it hard to assess when the method is recommended versus when a GP would be more appropriate. The UCI datasets are small (n ≤ 769, I ≤ 13). On Boston, GP Bayes actually outperforms Dα-KP (RMSE 2.58 vs. 2.59), and on Energy the margins over DIWP (0.46 vs. 0.48) and GP MLE (0.46 vs. 0.49) fall within overlapping standard deviations.

- **CMI analysis limited to L=2, δ=1:** The conditional mutual information analysis in Section 3.1 is presented only for a single hidden layer with ReLU activation. It remains unclear how CMI behaves with depth or other activations, limiting the theoretical insight into the effect of depth.

- **Cannot directly compare with the most relevant predecessor in higher dimensions:** The Stable method of Loria & Bhadra (2024) is unavailable for I > 2, so the proposed method's advantage over its direct predecessor cannot be assessed in the higher-dimensional settings where it supposedly matters most. In the dimensions where Stable is available (I = 1, 2), Stable clearly outperforms Dα-KP (Table 1).

### Trivial
None.

## Nice-to-Haves

- Experiments on smooth or mixed data-generating functions to clarify the method's practical scope and when a GP prior is preferable.
- Comparison with modern variational deep GP methods (e.g., Damianou et al.) or other scalable approaches to better position the method in the broader landscape.
- Deeper analysis of why depth provides no benefit—whether this is fundamental to α-stable processes or an artifact of the shared-scale-per-layer parameterization.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic claim: "Theorem 1 should clarify that the recursive kernel formula requires Gaussian w̃."** The paper already distinguishes this clearly: the general convergence result states w̃ needs only "zero mean and unit variance" (line 83), while the recursive formula section explicitly adds the condition "w̃_{ij}^{(ℓ)} i.i.d. standard Gaussian" (line 91). The paper is already clear on this point.

- **Harsh critic claim: "Comparing MCMC-based inference against optimization-based methods conflates model and inference contributions."** While technically true, this is standard practice in the Bayesian nonparametrics literature. The paper's contribution is the model plus inference procedure as a package; separating them would require implementing MCMC for the competing models or VI for the proposed model, which is outside the paper's scope.

- **Harsh critic demand: "Move MCMC details to the main paper."** This is a nitpick about what goes in the main text vs. appendix; the details exist and are available. The main paper presents the key algorithm (Algorithm 1) and references the appendix for implementation specifics, which is standard.

- **Harsh critic concern about missing comparison with modern variational deep GP methods.** Moved to Nice-to-Haves. The paper compares against the most directly relevant baselines (DIWP, NNGP from Aitchison et al. 2021; Stable from Loria & Bhadra 2024; standard GPs). Additional comparisons would strengthen but are not a core flaw.

- **Strength Finder claim: "Empirically outperforms competing methods on both simulations and UCI benchmarks."** This is partially weakened by the verified observations that GP Bayes matches/beats Dα-KP on Boston, and margins on Energy are within SDs. The strength is kept but qualified.

## Novel Insights

The paper reveals an interesting structural asymmetry between the first layer (per-input-dimension scales s_{m,+}^{(1)}) and subsequent layers (single shared scale s_+^{(ℓ)} per layer) in the construction. This asymmetry may partially explain why depth provides no empirical benefit: the rich per-dimension scaling in the first layer already captures substantial heterogeneity, while the shared scale in deeper layers may be too restrictive to allow depth to express meaningful additional structure. This possibility—neither confirmed nor explored by the authors—deserves investigation, as it could either motivate richer per-neuron scaling schemes or confirm that α-stable processes are fundamentally "rich enough at depth 2" in a way that Gaussian processes are not.

## Suggestions

- Reframe the contribution honestly: present the deep extension as a theoretical contribution, and the kernel-space reformulation (which works at any depth but is most impactful at L=2) as the primary practical advance. This is still a valuable contribution and avoids the overclaim that currently weakens the paper.
- Add at least one experiment on a smooth or mixed data-generating function to clarify the practical scope of the method and when a standard GP is preferable.
- Investigate whether per-neuron or per-path scaling (rather than shared per-layer) could allow depth to express meaningful structure, and discuss whether the depth-invariance result is fundamental to α-stable processes or an artifact of the specific parameterization.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Generalization Error of Spectral Algorithms | 3SJE1WLB4M.md | 8.0 | Much deeper theoretical contribution with power-law spectral framework; purely theoretical. Our paper has less deep theory but adds experiments. |
| Residual Deep GPs on Manifolds | JWtrk7mprJ.md | 7.6 | Deep GP with strong theory AND demonstrated empirical benefits of depth. Our paper's depth extension shows no empirical benefit, a key difference. |
| Bayesian nonparametric GPFA | 2iCIHgE8KG.md | 7.5 | Conditionally Gaussian model with theoretical framework, similar spirit to our paper but with stronger empirical validation. |
| Pretraining Label Granularity | PhnGhO4VfF.md | 5.67 | Theory + experiments but experiments don't fully validate claims. Similar profile to our paper. |
| Implicit Functional BDL | ohHtdp3jDi.md | 4.0 | BNN → GP connection but shaky theoretical justification and weak performance. Our paper has much better theory. |
| MinMax BNN | WoJzHQIIUk.md | 1.5 | BNN with stable process reference, poorly written with no baselines. Our paper is clearly far above this. |

The paper sits between the medium-scoring anchors (4.0–5.67) and high-scoring anchors (7.0+). It has genuine theoretical and computational contributions (the conditionally Gaussian representation, recursive kernel formula, and resolution of the exponential bottleneck), but the two Major weaknesses—the lack of empirical benefit from depth and the overclaimed "naturalness"—significantly temper the impact. The paper is clearly above the low anchors and the weaker medium anchors (4.0), but below the strong medium anchors (5.67) because the Pretraining Label Granularity paper at least has comprehensive experiments supporting its (more modest) claims. Our paper's main claimed advance (deep extension) is unvalidated empirically, and the actual practical advance (kernel-space reformulation for shallow architectures) is narrower than presented.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>