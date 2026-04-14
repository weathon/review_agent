=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
Now I have enough to write the consolidated review. Let me synthesize carefully.

---

## Summary

This paper proposes four collaborative compression schemes for Distributed Mean Estimation (DME)—NoisySign, HadamardMultiDim, SparseReg, and OneBit—that exploit vector similarity agnostically (without requiring prior knowledge of the correlation structure). The central theoretical contribution is achieving exponential decay of estimation error in the number of clients *m* (versus the polynomial *1/m* of independent compressors), quantified by novel dissimilarity measures (Δ_Hadamard, Δ_reg, Δ_corr) that characterize graceful degradation as vectors become less similar. The paper extends collaborative DME theory beyond ℓ₂ error to ℓ∞ and cosine distance metrics, and validates the schemes on both synthetic DME tasks and downstream distributed learning applications.

---

## Strengths

- **Exponential-in-*m* error decay is a genuine and significant theoretical advance.** HadamardMultiDim achieves ℓ∞ error O(B·2^{-(m−1)}) + Δ_Hadamard and SparseReg achieves ℓ₂ error O(B²·exp(−m log L / d)) + Δ_reg. The contrast with all prior schemes (whose non-dissimilarity term is poly(1/m)) is striking and shown rigorously. The mechanism—having each client contribute at a different level of a collaborative binary search / sparse regression code—is conceptually elegant.

- **Novel reduction of cosine DME to halfspace learning with malicious noise (Section 3).** The equivalence between dissimilar unit-vector estimation and learning a halfspace under malicious noise (Lemma 1) is a genuinely insightful cross-domain connection. It allows the paper to directly leverage Shen (2023)'s optimal halfspace learner and yields the first cosine-distance guarantee in the collaborative DME literature.

- **Application of Sparse Regression Codes (SRC) to collaborative DME is a non-trivial cross-domain contribution.** SRCs are known to achieve the Gaussian rate-distortion function; extending them to the collaborative DME framework through a client-level permutation mechanism to split binary search levels across clients is novel. The paper correctly credits and adapts (Venkataramanan et al., 2014b) while adding the collaborative layer.

- **Coverage of ℓ∞ and cosine metrics fills an explicit gap.** The paper correctly notes that all existing collaborative compressors provide only ℓ₂ guarantees (except the single-coordinate result of Suresh et al., 2022). ℓ∞ guarantees matter for coordinate-wise tasks, and cosine guarantees matter for direction-critical applications (gradient descent, power iteration). Providing both, with distinct dissimilarity measures that tightly characterize when each scheme helps, is a substantive contribution.

- **Experiments cover both regimes (low and high dissimilarity) and downstream tasks.** Testing on MNIST (low dissimilarity) and FEMNIST (high dissimilarity) for KMeans and power iteration, plus UJIIndoorLoc and synthetic mixture for linear regression, directly illustrates both the benefit and the limitations of the proposed schemes, which is honest and informative.

---

## Weaknesses

### Fatal
None.

### Major

- **Notation error in Algorithm 3 (HadamardMultiDim) that is genuinely confusing and potentially incorrect.** The Init declares ρ as "a random permutation on [m]," which means ρ is defined on indices {1, …, m}. However, the Encode step uses ρ^(j) indexed by j ∈ [d], and the Decode uses B/2^{ρ^(j)−1}—a weight that does not depend on client index i at all, meaning every client for a given coordinate gets the same weight, contrary to the entire design principle. Section 2.1 (text) correctly states the intent: "the *i*th client can perform binary search until level ρ^(i)." The pseudocode should use ρ^(i) (client index) throughout, not ρ^(j) (dimension index). As written, the Decode applies one weight to all m clients for each coordinate j, which makes the algorithm nonsensical. If d > m, ρ^(j) is also undefined for j > m. The authors must correct the pseudocode and confirm there is no implicit d ≤ m assumption.

- **OneBit's error bound (Theorem 3) is vacuous in the natural operating regime.** Theorem 3 states cos(⟨g̃, g⟩) ≥ cos(π(Δ_corr + d/(mt))). For t = 1 bit per client, the second term is d/m. When d > m (the common case in federated learning), this term exceeds 1, making the angle bound π·(anything > 1), so arccos is undefined and the guarantee is vacuous. The paper never states the necessary condition t ≥ d/m for the bound to be non-trivial. Since t is tunable, the bound can be made non-vacuous by increasing t, but the paper should explicitly characterize the regime of applicability and the communication cost at which OneBit becomes useful.

- **Theory–implementation gap for OneBit experiments.** Theorem 3 (the main theoretical result) applies to Technique I (Shen, 2023), described as "harder to implement." The experiments almost certainly use Technique II (OneBitAvg, the simple average decoder), which is stated in the paper to be suboptimal and whose guarantees are deferred to Appendix B. The paper does not clarify which decoder was used in Figure 2c and 2g–2i. This gap between the proven guarantee and the implemented variant needs to be explicitly stated; without it, readers cannot assess whether the experimental results are consistent with the theory.

- **Unspecified constants δ₁, δ₂ in Theorem 2 (SparseReg) limit the bound's utility.** The theorem asserts existence of constants δ₁, δ₂ > 0 such that a Gaussian A satisfies the bound with high probability, but never pins down these constants. Without knowing δ₁, δ₂, the simplified bound O(B·exp(−m/d)) presented in the abstract conceals potentially large pre-exponential factors. It is not possible to verify from the paper alone whether the exponential term dominates or is swamped by Δ_reg for practical (m, d, L) combinations.

### Minor

- **Optimality claims are unsubstantiated.** The paper claims "optimal dependence on *m* and *B*" for ℓ∞ and ℓ₂ errors (contributions 2 and 3; introduction). The conclusion itself acknowledges: "Lower bounds for collaborative compressors in terms of their dissimilarity metrics will allow us to assess the optimality of our schemes." This is a contradiction: one cannot claim optimality while admitting the necessary lower bounds do not exist. The claim should be softened to "best known dependence" until lower bounds are established.

- **No bias/convergence analysis despite bias being acknowledged.** The conclusion briefly notes all four schemes are biased and refers to Beznosikov et al. (2022) for unbiased conversion. However, the paper's primary motivation is distributed optimization (gradient descent), and biased compressors can cause gradient descent to diverge without error-feedback mechanisms. The paper provides no analysis—even informal—of how the bias interacts with SGD convergence, nor does it empirically validate that the schemes remain useful when plugged into an optimizer rather than a single-round DME problem. The unbiased variant mentioned in the conclusion is not implemented or tested.

- **Selective experiment reporting.** The paper states it "compare[s] against all baselines in Table 2 for 3 random seeds and report[s] the methods which perform the best in Fig 2." Reporting only the best-performing subset of baselines, with no error bars and without disclosing which baselines were excluded and why, makes it difficult to assess robustness of the reported improvements.

- **NoisySign parameter σ is untransparent.** The paper notes "NoisySign obtains competitive performance to other baselines as we use a large σ" without specifying the value of σ used or how it was selected. If σ was tuned per experiment, the comparison may not be on equal footing with methods without free parameters.

### Tiny

- **Δ_Hadamard and Δ_reg are defined in terms of encoding decisions, not input-space quantities.** The paper provides a lower bound (Eqs. 3 and 5) in terms of Δ∞ and Δ₂ but no upper bound. Thus Theorem 1 and 2 do not directly give guarantees in terms of input-space dissimilarity. The motivating example in Section 2.3 provides intuition but is restricted to a very special case (d = 1, two-point distribution, all coordinates equal). A broader upper bound on Δ_Hadamard / Δ_reg in terms of interpretable input-space quantities would strengthen the theory.

- **The NoisySign error expression in Table 1 is extremely difficult to parse**, and no simplified or intuitive form is provided anywhere in the main paper. Since NoisySign's analysis is entirely in Appendix A, the main paper gives readers no way to assess this contribution beyond the raw formula.

---

## Nice-to-Haves

- **Rate-distortion (error vs. bits/client) curves** would allow readers to compare methods across varying communication budgets, rather than just at fixed budgets. This is the most natural way to assess communication efficiency.

- **Visualization of error vs. number of clients m on log-linear scale** would directly verify the claimed exponential decay vs. polynomial baselines—the central theoretical claim.

- **Discussion of straggler/partial participation robustness.** The collaborative schemes require all m clients to complete their encoding. A brief discussion of degradation under client dropout would improve practical relevance without being in scope as a full contribution.

- **Online dissimilarity estimation / adaptive switching heuristic.** Since performance degrades sharply at high dissimilarity (Figure 2), even a simple heuristic for estimating Δ online and switching between collaborative and independent modes would substantially improve practical deployability.

- **Computational complexity table.** The encoding cost of SparseReg (O(mLd) per client in the worst case) and HadamardMultiDim (O(md) per client) vs. O(d) or O(K) for baselines should be quantified in a table and measured empirically, not just mentioned in prose.

- **Explicit treatment of the *d* ≫ *m* regime.** For SparseReg, the exponent −m log L/d is negligible when d ≫ m. Explicitly characterizing when the exponential improvement is meaningful (i.e., when m = Ω(d)) would help practitioners understand where the schemes apply.

---

## Removed Points

*These points were flagged for removal; treat them with caution.*

- **[R1-Concern 4 as "fatal d ≤ m constraint"]: REMOVED (kept as notation issue in Major weaknesses).** The text of Section 2.1 clearly intends ρ to be client-indexed (ρ^(i)), consistent with the theory. The issue is a pseudocode notation error (ρ^(j) instead of ρ^(i)), not a hidden constraint that restricts to d ≤ m. The algorithm works for any d if the pseudocode is corrected.

- **[R1-Concern 5 as material communication unfairness]: REMOVED.** In collaborative DME, clients and server sharing randomness via a shared seed is standard practice and carries zero communication cost. Matrix A and vectors {z_i} can be generated deterministically from a shared seed, negating the initialization-overhead objection.

- **[R1-Concern on Lemma 1 edge case (Δ_corr → 0)]: REMOVED.** When Δ_corr = 0 all vectors g_i = g, so the fraction of corrupted labels is exactly 0 and the lemma holds trivially with probability 1. The probability bound 1 − O(exp(−mΔ_corr)) → 0 as Δ_corr → 0, but this is just the bound loosening at a trivially-satisfied edge case.

- **[Missing related works criticisms]: REMOVED per instructions.** No external verification possible.

- **[R1: "Practical regime of exponential decay" (d ≫ m in introduction)]: WEAKENED and absorbed into Major weakness on Theorem 2 constants / nice-to-have on d ≫ m regime.** The concern is legitimate but the paper does implicitly address it through the error expression; the failure to highlight the regime is a presentation gap, not a fundamental error.

- **[Generic "larger dataset", "more models" type requests]: Not raised by reviewers in this paper, no action needed.**

- **[R3: Deep network training (ResNet/CIFAR-10)]: REMOVED as scope creep.** The paper's stated focus is DME as a subroutine. Evaluating on deep FL training is outside the paper's scope and not a standard requirement for a theory-oriented DME paper.

- **[R3: Straggler-tolerant design, non-convex convergence theory, unbiased variant implementation]: REMOVED as out-of-scope demands.** These are legitimate future work but go beyond the paper's stated contributions and are not expected of a DME theory paper.

---

## Novel Insights

The most intellectually surprising contribution is the reduction of collaborative cosine-distance DME to the halfspace learning under malicious noise problem (Section 3 / Lemma 1). This is not a superficial analogy: the corruption fraction ζ is precisely Θ(Δ_corr) — the natural angular dissimilarity between clients — and the paper shows this holds with probability 1 − O(exp(−mΔ_corr)). This reduction immediately unlocks state-of-the-art halfspace learners (Shen, 2023) with optimal noise tolerance and linear runtime, yielding the first rigorous cosine-error guarantees for collaborative DME. The insight that client dissimilarity maps onto adversarial noise in a halfspace learning model could generalize to other estimation problems where clients have similar but not identical models.

---

## Suggestions

1. **Fix the Algorithm 3 pseudocode**: Replace ρ^(j) with ρ^(i) in both Encode and Decode. Verify that the stated theorems correspond to the corrected algorithm, not the written one.

2. **State the non-vacuousness condition for Theorem 3 explicitly**: Add a corollary or remark stating that OneBit's cosine guarantee is non-trivial when t ≥ d/m bits per client, and specify the error–budget tradeoff explicitly.

3. **Disclose OneBit decoder choice in experiments**: State clearly which technique (I or II) was used in each figure, and discuss the theoretical–empirical gap if Technique II was used.

4. **Pin down constants δ₁, δ₂ in Theorem 2**: Even lower-bounding them numerically for standard parameter ranges (e.g., δ₁ = 0.1, δ₂ = 0.1) would make the bound interpretable in practice.

5. **Soften or qualify optimality claims** until lower bounds are established. "Best known" is accurate; "optimal" requires a lower bound.

6. **Add error bars to Figure 2** and either show all baselines or provide a per-panel table in the appendix documenting all excluded methods and their performance.

7. **Add a log-linear plot of error vs. m** (Figure suggestion from Spark Finder) to empirically demonstrate exponential vs. polynomial decay—this is the paper's central claim and the most compelling visualization it is missing.

8. **Include a brief convergence discussion or experiment**: Even a single experiment showing that an unbiased variant (via dithering, e.g., Tang et al., 2023) plugged into gradient descent converges would substantially address the optimization motivation.

---

**Summary assessment**: The paper contains genuine theoretical novelty—particularly the exponential-in-*m* error bounds, the cross-domain reduction to halfspace learning, and the new ℓ∞/cosine dissimilarity frameworks. These advance the state of the art. However, the paper's claims must be tempered: Algorithm 3 has a notation error that must be corrected; Theorem 3 provides vacuous guarantees in the very regime (d > m) where the paper is most relevant unless explicitly conditioned on t; and the claim of "optimal dependence" is unsubstantiated without lower bounds. With corrections to the algorithm pseudocode, explicit regime characterization for OneBit, and more transparent experiments, this is a strong paper. In its current state, the theoretical contribution is solid but the presentation requires revision to be trustworthy.

# Actual Human Scores
Individual reviewer scores: [6.0, 5.0, 5.0, 5.0]
Average score: 5.2
Binary outcome: Reject
