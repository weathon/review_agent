## Summary
This paper introduces FedMPDD, a federated learning algorithm that compresses client gradients via multi-projected directional derivatives along random vectors. It reduces uplink communication from O(d) to O(m) (m ≪ d) while providing empirical protection against gradient inversion attacks through the rank deficiency of the projection mechanism. The method achieves a convergence rate of O(1/√K), matching FedSGD, with m growing logarithmically in dimension d.

## Strengths
- **Novel joint design for efficiency and privacy.** The core mechanism of encoding gradients via multiple directional derivatives is a fresh approach that simultaneously tackles communication overhead (transmitting only m scalars) and provides privacy via the projection's nullspace, distinct from separate compression or additive-noise methods.
- **Rigorous theoretical analysis.** The paper provides a complete convergence analysis for both the single-projection (FedPDD) and multi-projection (FedMPDD) variants, establishing an O(1/√K) rate for FedMPDD under standard assumptions and a Johnson-Lindenstrauss condition on m. Privacy is analytically characterized via gradient reconstruction error (Lemma 1) and a lower bound on data recovery (Lemma 2).
- **Comprehensive empirical validation.** Experiments span multiple datasets (MNIST, FMNIST, CIFAR-10), models, and data distributions (IID/non-IID). Evaluation under both fixed communication budgets and fixed accuracy targets convincingly demonstrates FedMPDD's advantages in communication savings and empirical resistance to gradient inversion attacks (using SSIM and visual reconstructions) compared to baselines like FedSGD, QSGD, Top-k, and LDP.

## Weaknesses
- **Incorrect convergence rate stated in the abstract.** The abstract claims a convergence rate of O(1/K), but Theorem 2 correctly states O(1/√K). This is a significant error that misrepresents the theoretical guarantee and must be corrected.
- **Computational overhead on clients is not empirically quantified.** While Remark 1 discusses the O(dm) encoding cost and mentions projected-forward methods (Jacobian-vector products) as a potential optimization, the paper does not measure the actual client-side runtime or energy consumption compared to baselines. This is important for assessing practical suitability in resource-constrained settings, as the increased computation could offset communication savings.
- **Privacy claims are heuristic and lack formal comparison to state-of-the-art private FL.** The privacy analysis (Lemmas 1 & 2) establishes a reconstruction error bound, providing a computational security argument against gradient inversion attacks. However, the language ("inherent privacy," "privacy guarantee") risks overstatement, as the method does not provide a formal, composable guarantee like differential privacy. Furthermore, the comparison is primarily against simple LDP and non-private compression; benchmarking against strong private FL baselines (e.g., DP-FedAvg, DP with secure aggregation) would better contextualize the privacy-utility trade-off.
- **Experimental results lack statistical robustness.** The paper mentions using five random seeds but reports results as single numbers (e.g., test accuracy). Reporting means and standard deviations (or confidence intervals) across seeds is essential for ICLR to assess the significance of the reported advantages, especially in the main tables (1, 2, and Appendix tables).

## Nice-to-Haves
- Include wall-clock time measurements (computation + communication) to provide a holistic efficiency assessment.
- Extend evaluation to larger-scale models (e.g., ResNet) to better validate the logarithmic scaling claim for m.
- Provide a more detailed discussion of the multi-round privacy bound (Remark 2) in the main text, emphasizing that privacy degrades with more observations and discussing how evolving gradients affect this in practice.

## Removed Points 
*These points are flagged to be removed, treat them with caution*
- **Weakness about missing FedAvg comparison.** The paper's scope is a gradient compression and privacy mechanism built on the FedSGD framework. While comparing to FedAvg could be interesting, it is not a core requirement for evaluating the proposed encoding scheme.
- **Weakness about requiring formal DP guarantees.** The paper explicitly positions its privacy mechanism as an alternative to DP, analyzing protection against gradient inversion via reconstruction error. Demanding a formal DP proof is scope creep for this contribution.
- **Weakness about the Lipschitz constant in Lemma 2 weakening the bound.** The bound's dependence on the model-specific Lipschitz constant is a standard feature of such reconstruction error analyses; it does not invalidate the derived relationship between m and reconstruction error.
- **Nitpicks about formatting artifacts in the extracted PDF.** These are parser issues, not paper problems.
- **Suggestion to evaluate against membership inference attacks.** The paper focuses on gradient inversion attacks, which is appropriate for its stated privacy threat model.

## Novel Insights
The paper's key novel insight is that averaging multiple independent, low-rank (rank-1) gradient projections can overcome the dimension-dependent convergence penalty of a single projection while preserving the inherent privacy afforded by each projection's nullspace. This multi-projection mechanism creates a tunable trade-off: the number of projections m controls the bias-variance trade-off of the gradient estimator (affecting convergence speed and communication cost) and simultaneously governs the dimensionality of the remaining nullspace (affecting privacy against reconstruction). This unified perspective on the communication-privacy-accuracy trade-off via a single parameter m is a distinct and valuable contribution.

## Suggestions
- Correct the convergence rate in the abstract from O(1/K) to O(1/√K) to align with Theorem 2.
- Augment key experimental results (e.g., Tables 1, 2) with means and standard deviations across multiple runs to establish statistical significance.
- In the main text, more precisely frame the privacy property (e.g., "empirical protection against gradient inversion via obfuscation" or "computational privacy based on reconstruction hardness") to avoid potential misinterpretation as a formal DP guarantee. Expand the limitations section to explicitly note the heuristic nature of the guarantee and the multi-round erosion bound.
- Include a simple empirical evaluation of client-side encoding time (even for the naive O(dm) method) versus baseline gradient computation to ground the discussion of computational overhead.