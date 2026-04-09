## Summary

FedMPDD introduces a federated learning algorithm that encodes each client's gradient via multi-projected directional derivatives—computing inner products along m random Rademacher vectors and transmitting only those m scalars plus a seed. The server reconstructs gradient estimators from these compressed messages, achieving O(m) uplink communication (m ≪ d) while providing privacy against gradient inversion attacks through the rank-deficiency of the projection. The paper proves O(1/√K) convergence (matching FedSGD) and provides reconstruction error bounds as a privacy metric, supported by experiments on MNIST, FMNIST, and CIFAR-10.

## Strengths

- **Unified mechanism for compression and privacy**: Unlike methods that bolt privacy (e.g., noise injection) onto compression, FedMPDD derives both from a single principled mechanism—the nullspace of a low-rank projection. The observation that the (d−m)-dimensional nullspace simultaneously provides compression and gradient obfuscation is a genuine insight, and the paper cleanly characterizes the relative reconstruction error (d−1)/m in Lemma 1.

- **Seed-based communication protocol**: Transmitting only m scalars and a random seed (from which the server regenerates the projection vectors) is an elegant design that avoids sending the projection vectors themselves, keeping uplink cost strictly O(m). This is a practical engineering contribution over methods that must share projection matrices.

- **Honest analysis of single-projection failure and principled fix**: The paper explicitly shows that FedPDD (single projection) suffers O(√d/√K) convergence due to variance scaling (Eq. 3), then proposes the multi-projection averaging mechanism to recover O(1/√K). This build-then-fix structure strengthens the technical narrative and the theoretical contribution.

- **Empirical privacy evaluation against state-of-the-art attacks**: The paper tests against both DLG (Zhu et al., 2019) and a recent attack (Yu et al., 2025), reporting SSIM scores and visual reconstructions (Figure 2). The consistently low SSIM across training epochs (Figure 1) provides concrete evidence that the nullspace protection holds in practice, not just theory.

## Weaknesses

### Major:

- **Privacy claims are not Differential Privacy; framing is misleading**: The paper repeatedly uses "privacy preservation" and positions FedMPDD against LDP, but its guarantees are reconstruction error bounds (Lemmas 1–2), not ε-DP. LDP provides worst-case guarantees against arbitrary adversaries; FedMPDD provides obscurity against specific gradient-inversion-style attacks. The paper's claim of "uniform privacy" (Abstract, Section 2) conflates consistent reconstruction error with formal privacy. The distinction matters: an adversary who knows the projection distribution could potentially extract label information, class membership, or semantic features from the m-dimensional projection even if full gradient reconstruction fails. The paper does not analyze what partial information *is* recoverable. This should be foregrounded, not buried in a remark contrasting with LDP.

- **Convergence rate stated incorrectly as O(1/K) in the Abstract and Theorem 2**: The actual bound in Theorem 2 (Eq. 5) with step size η = 1/(L√K) yields O(1/√K) for the average squared gradient norm, which is the standard non-convex rate. The Abstract's claim of "O(1/K)" is incorrect. While the "matching FedSGD" conclusion is still valid (FedSGD is also O(1/√K) for non-convex objectives), the notational error undermines confidence in the theoretical presentation and should be corrected.

- **Evaluated implementation does not reduce client-side computation; JVP optimization is future work**: Algorithm 2 (Line 6) computes the full stochastic gradient before projecting, incurring the same O(d) computational cost as FedSGD plus the additional O(dm) projection cost. Remark 1 suggests using Jacobian-vector products to avoid computing g_i explicitly, but Section F states this is planned future work ("we plan to implement a fully optimized version"). The paper's framing of suitability for "resource-constrained scenarios" (Abstract) thus applies only to bandwidth-constrained settings, not compute-constrained ones—a qualification that should be made explicit rather than implied.

- **Claim that "smaller m values sometimes yielded faster convergence" is misleading**: The Conclusion states this, and Section 3 references Fig. A.9, but Table A.9 clearly shows accuracy *increasing* with m (30.44% at m=50 vs. 79.02% at m=600 on LeNet-MNIST). The claim is only defensible if "faster convergence" means fewer total bits transferred (Table 2: m=600 uses 1.32 GB vs. m=2000 uses 3.26 GB for 60% target accuracy). Without this qualification, the statement contradicts the experimental data and confuses the reader about the m-accuracy relationship.

### Minor:

- **Multi-round privacy bound T < d/m is acknowledged but its practical implications are underexplored**: Remark 2 notes this worst-case constraint. For d ≈ 300,000 and m = 600, privacy erodes after ~500 rounds. While often sufficient, this is a hard limit for long-running FL tasks (e.g., continual learning). The paper mentions that gradient evolution provides additional protection but does not formalize this, leaving the bound as the only compositional guarantee.

- **No analysis of what partial/semantic information is extractable from the m-dimensional projection**: Lemma 1 bounds the full gradient reconstruction error, but does not address whether an adversary can recover sensitive attributes (e.g., class labels, membership) from the projected information alone. For m ≈ 600 out of d ≈ 300,000, the projection preserves some signal; characterizing what semantic content survives would strengthen or appropriately bound the privacy claims.

- **Experimental scale is limited to small models**: The largest model tested has ~300K parameters (CNN on CIFAR-10). The communication advantage of O(m) vs O(d) is most impactful for large models (ResNet-18 with ~11M parameters, as cited in the Introduction). The paper's motivating example of ResNet-18 is never actually tested, leaving the practical scalability unvalidated.

### Trivial:

- The Rademacher vs. Gaussian variance comparison (Lemma 3) is a nice theoretical addition but its practical impact on convergence is never isolated experimentally.

## Nice-to-Haves

- Comparison with a formally private baseline (e.g., DP-SGD or clipped Gaussian mechanism with ε-accounting) under equivalent communication budgets, to contextualize what FedMPDD's reconstruction-resistance buys relative to standard privacy definitions.
- Wall-clock benchmarks for total client-side time (gradient computation + encoding) vs. FedSGD, to quantify the real computational overhead.
- Experiments on a modern large-scale model (e.g., ResNet-18 or a fine-tuning task) where d is large enough for the communication savings to be practically decisive.
- Integration of error feedback with FedMPDD to investigate whether the compressed gradient estimator can benefit from accumulated error correction.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Convergence to a neighborhood, not exact optimum"** (from transferable weaknesses): Incorrect for this paper. Theorem 2 shows the bound goes to 0 as K → ∞; there is no non-vanishing neighborhood term. The convergence is to a stationary point in the standard non-convex sense.

- **"Missing privacy analysis"** (from transferable weaknesses): The paper has extensive privacy analysis including Lemmas 1–2, Remarks 2–6, and Appendix C–D. Whether the analysis is *sufficient* is debatable, but claiming it is *missing* is factually wrong.

- **"Only tested on two datasets"** (from transferable weaknesses): The paper tests on MNIST, FMNIST, and CIFAR-10 across multiple models and both IID/non-IID settings.

- **"Baselines are 2–3 years old"** (from transferable weaknesses): The paper compares against Yu et al. (2025), a very recent attack method, and includes SA-FedLora and lp-proj as compression baselines.

- **"Memory consumption footprint not reported"** (from transferable weaknesses): The paper discusses memory complexity in Remark 1 and provides Table F.1 comparing time/memory complexities of different gradient computation methods.

- **"Unfair comparison with QSGD because FedMPDD sends 32-bit floats"**: QSGD with 8-bit quantization on d=300K sends ~2.4M bits; FedMPDD with m=600 sends ~19.2K bits. The asymmetry favors FedMPDD, which makes the comparison *more* convincing, not less. Per the hard rules, this is not a valid criticism.

- **"Missing related works"**: Per the rules, I do not have external sources to confirm existence of specific missing references.

- **Formatting and reproducibility nitpicks** (undisclosed hyperparameters, etc.): The paper provides detailed hyperparameter tables (Tables H.4–H.7) and random seeds. Removed per hard rules.

## Novel Insights

The core insight—that averaging multiple rank-deficient projections can simultaneously recover FedSGD-level convergence while preserving a tunable nullspace for privacy—is sound and distinguishes FedMPDD from both fixed-subspace sketching methods (which lack per-round privacy randomness) and additive-noise DP (which hurts convergence direction). However, the paper's most underappreciated tension is that m serves three masters—convergence (wants large m), communication (wants small m), and privacy (wants small m)—and the convergence requirement (m = O(ln d)) is actually quite modest, meaning the practical bottleneck is the privacy–accuracy trade-off at small m, not the convergence–communication trade-off. This suggests the method's sweet spot may be in regimes where moderate privacy is acceptable and communication is the binding constraint, rather than as a replacement for formal DP.

## Suggestions

- Correct the convergence rate from O(1/K) to O(1/√K) throughout the Abstract, Introduction, and Theorem 2 statement. This is a straightforward fix that would resolve the inconsistency with the actual proof.
- Qualify the "smaller m yields faster convergence" claim explicitly as "faster convergence in terms of total communication bits" rather than optimization rounds, or remove it if the data doesn't robustly support even this interpretation across all settings.
- Add a paragraph in Section 2 or Appendix C explicitly discussing what the nullspace guarantee does *not* protect against (e.g., partial information leakage, attribute inference) and frame the contribution as "gradient inversion resistance" rather than "privacy preservation" to align terminology with the actual guarantees provided.
- Disclose upfront (in the Introduction or Method section, not just in Future Work) that the current implementation computes full gradients and the JVP-based computational savings are a prospective optimization, so the method's current benefit is communication reduction only.

## Axis Evaluations

- **Novelty**: Moderate-to-high. The projected directional derivative formulation and the specific nullspace-privacy argument in the FL context are novel, though the underlying random-projection machinery is well-established. The dynamic per-round sampling and seed protocol are practical contributions.

- **Technical soundness**: Mixed. The convergence proof is correct (modulo the O(1/K) vs O(1/√K) notation error), and the JL-based analysis is appropriately applied. However, the privacy analysis conflates reconstruction error with formal privacy, and the multi-round composition bound (T < d/m) leaves a significant gap for long-running training. The claim about smaller m yielding faster convergence contradicts the experimental data without careful qualification.

- **Empirical support**: Adequate for the communication and privacy-against-GIA claims on small models, but limited in scale (no models >300K parameters despite motivating with ResNet-18). The privacy evaluation against two attack families is a strength, though restricted to pixel-level SSIM rather than attribute-level leakage.

- **Significance**: Moderate. The joint communication-privacy mechanism addresses a real need in bandwidth-constrained FL, and the nullspace insight is valuable. However, the lack of formal DP guarantees limits applicability in regulated settings where privacy auditing requires ε-accounting, and the computational cost issue tempers the "resource-constrained" narrative.

- **Clarity**: Generally good, with clear algorithmic presentation and logical flow from FedPDD to FedMPDD. The main clarity issues are the incorrect convergence rate notation and the ambiguous "faster convergence" claim in the conclusion.