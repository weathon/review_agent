## Summary

This paper analyzes the Random Feature Model (RFM) under spiked covariance data to explain why RFMs often empirically outperform linear models, contrary to what isotropic Gaussian equivalence theory predicts. The authors extend the universality theorem of Hu & Lu (2023) to the spiked covariance setting and show that when input-label correlation is strong (high alignment parameter α and/or spike magnitude θ), the RFM behaves equivalently to a high-order polynomial model rather than a noisy linear model. The degree of the equivalent polynomial is governed by a scalar η that encodes this correlation, with the full characterization given in Theorem 2 and Corollary 3.

---

## Strengths

- **Concrete polynomial equivalence with Hermite-coefficient characterization (Theorem 2 + Remark 4):** The result that the effective polynomial degree of the RFM is determined by η—which scales with (ξ + θαγ)/√(1+θα²), i.e., the first component of the input-label covariance—is a specific and non-trivial insight. Remark 4 further shows that *which* Hermite coefficients of σ and σ∗ vanish precisely controls whether the RFM reduces to a polynomial or collapses back to a linear model (e.g., for tanh/ReLU pairs, the relevant Hermite coefficients determine the outcome). This level of mechanistic specificity is a genuine contribution beyond generic "nonlinearity helps" narratives.

- **Extension of universality to spiked covariance is non-trivial and well-motivated:** Prior universality results (e.g., Hu & Lu 2023) required isotropic data, which is inconsistent with empirical observations. By requiring the feature matrix scaling 1/(n+θ) and deriving new moment-matching conditions (eqs. 10–11) under the spiked structure, the paper fills a concrete gap. The modified Lindeberg exchange under the anisotropic setting requires that moment conditions (16)–(18) be re-derived for the spiked case.

- **Figure 2 provides clean empirical validation of the polynomial equivalence:** Under aligned settings (α = 1, θ = n^{1/2}), the RFM generalization error matches the noisy polynomial model but diverges from the noisy linear model—and this reversal occurs in a functionally interpretable way depending on the Hermite structure of both σ and σ∗ (Figures 2a, 2b). This directly validates the core theoretical claim.

- **Phase-transition boundary (Corollary 3 + Figure 1a):** The characterization of when η = O(n^{-1/2}), demarcating the linear vs. nonlinear regime as a function of (α, θ), is a precise quantitative result that matches the numerically observed boundary well.

---

## Weaknesses

- **Assumption A.2 inconsistency—β ∈ [0, 1/2] vs β < 1/2:** The formal assumption (A.2) lists β ∈ [0, 1/2] (closed interval), yet the Discussion of Assumptions states "our current proofs necessitate that β < 1/2," and the abstract states β ∈ [0, 1/2). It is unclear whether β = 1/2 is actually covered. This inconsistency should be resolved; if β = 1/2 is not covered, the set notation in A.2 should be corrected to an open interval.

- **ReLU excluded by Assumption A.6 with only informal empirical justification:** The odd-function requirement in A.6 excludes ReLU, the dominant activation function in practice. The paper acknowledges this but only notes that "empirical evidence suggests our findings remain valid even when using ReLU." For a theory paper, this gap is significant: no formal bound is provided on the error incurred by applying the theory to non-odd activations, and no quantitative analysis (even an approximation argument) is offered. This limits how confidently the theoretical framework can be applied to modern architectures.

- **η is treated as a deterministic condition in Theorem 2, but is random:** The condition η ≤ C/n^{1/l} (eq. 15) depends on the random feature rows f_i via |(ξ + θαγ)^T f_i|. The claim that η = O(n^{-1/4}) holds with high probability when β < 1/2 is stated informally in Section 5 without a formal proposition or proof. As stated, Theorem 2 requires the practitioner to verify a condition on a random quantity before applying the result. This should be formalized as a lemma.

- **Single-spike model is restrictive, and multi-spike generalization is not discussed:** The entire analysis rests on rank-one spiked covariance (I_n + θγγ^T). Real data covariances typically have a decaying spectrum with multiple significant eigenvalues. The paper provides no discussion of whether the qualitative conclusions (polynomial degree governed by input-label correlation) persist with multiple spikes, or whether new phenomena arise. This limits the theoretical scope of the paper's claims.

- **Figure 3c extends to β ≥ 0.5 without adequate flagging:** The theoretical results are confined to β < 1/2, yet Figure 3c plots β up to 1.0. While the text mentions "our analysis is confined to β < 0.5," the figure presents empirical results beyond this range without any visual demarcation, potentially misleading readers about the reach of the theory.

- **Figure 4 legend conflates model types and activation functions:** The CIFAR-10 figure (Figure 4) lists "ReLU (blue circles)" and "Softplus (red circles)" alongside "Random Feature Model (blue diamonds)" and "Noisy Linear Model (red diamonds)"—but ReLU and Softplus *are* random feature models. This typological inconsistency makes the figure difficult to interpret without careful reading of the appendix. A clearer labeling scheme is needed.

- **CIFAR-10 experiment does not verify spiked covariance structure:** The theoretical framework assumes a rank-one spiked Gaussian covariance, but no analysis is provided to verify that CIFAR-10 images (non-Gaussian, high-order structure) exhibit this structure. The experiment shows that the polynomial model tracks the RFM as input-label correlation increases, which is suggestive, but the connection to the theoretical assumptions is not formally justified.

- **Technical novelty over Hu & Lu (2023) is not clearly articulated in the main text:** The paper states it follows "the proof technique used by Hu & Lu (2023)" via Lindeberg's method. The specific proof challenges introduced by the spiked covariance—e.g., controlling spectral norms under rank-one perturbations, modified moment estimates in eqs. (16)–(18)—are not described in the main body. Without this, it is difficult to assess the depth of the technical contribution relative to the baseline work.

---

## Nice-to-Haves

- **Eigenspectrum validation for CIFAR-10:** Plotting the top-K eigenvalues of the CIFAR-10 sample covariance matrix would help justify the spiked covariance approximation for this dataset.

- **Formal verification/plot of the η condition:** Adding a figure showing η measured in simulations as a function of n and β would corroborate the informal claim that η = O(n^{-1/4}) with high probability, giving Theorem 2 an empirically grounded condition.

- **Discussion of the strong-spike regime (β ≥ 1/2):** Even a heuristic discussion of how the results might change when the spike dominates the spectrum would help practitioners understand the boundary of applicability.

- **Non-odd activation ablation:** Comparing a properly centered/symmetrized ReLU (e.g., ReLU(x) − ReLU(−x) ∝ x) against standard ReLU would help isolate the impact of violating A.6, providing empirical guidance on when the odd-function assumption is material.

- **Hermite coefficient table for ReLU and tanh:** The insight in Remark 4 that Hermite coefficients μ₂μ̃₂ and μ₃μ̃₃ determine equivalence class is non-trivial; a table with numerical values of μ_j and μ̃_j for common activation pairs would make this immediately accessible.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Title is misleading:** The harsh critic argues the title should acknowledge conditionality. The subtitle "Effect of Strong Input-Label Correlation in Spiked Covariance Data" already explicitly scopes the setting. The title as written is acceptable and consistent with how RFM superiority is usually discussed in the literature.

- **Comparison between optimized polynomial/linear and fixed-form ReLU/Softplus is unfair:** The harsh critic argues the optimized coefficients in (21)–(22) give polynomial/linear activations an advantage. However, this is by design: the paper constructs an "optimal" polynomial to characterize the theoretical upper bound achievable by the equivalent model class. The comparison illustrates how close practical activations come to this bound, not that practical activations are inferior because they are unoptimized. The asymmetry is transparent and intentional.

- **Contribution 3 is just an enabling ingredient, not independent:** This is an organizational critique with no bearing on correctness or value. The universality extension to spiked data is a technically non-trivial prerequisite that is likely of independent interest to the RMT community.

- **No comparison to kernel SVMs or fully trained networks:** This is scope creep. The paper explicitly situates itself within the fixed random features literature and does not claim to surpass trained networks.

- **Unknown parameters θ and α in practice:** As a theoretical paper establishing asymptotic equivalences, the practical unknowability of θ and α is a standard limitation of the entire proportional-asymptotics program (including Hu & Lu 2023). This is not a weakness specific to this paper.

---

## Novel Insights

The most genuinely novel insight emerging from this synthesis is the *Hermite-coefficient decoder* view of the linear-to-polynomial transition: Theorem 2 + Remark 4 together imply that for a given activation pair (σ, σ∗), the minimum polynomial degree needed for RFM-equivalent performance is determined by the lowest j such that μ_j(σ) · μ̃_j(σ∗) ≠ 0, modulated by whether η exceeds the corresponding n^{-1/j} threshold. This means that the question "when does nonlinearity help?" reduces to a structured factorization problem in Hermite space, with the answer depending jointly on the activation function, the label function, and the data geometry (α, θ)—not on any single factor in isolation. This three-way interaction is a sharper characterization than what exists in the isotropic theory.

---

## Suggestions

- **Fix the A.2 notation:** Change β ∈ [0, 1/2] to β ∈ [0, 1/2) everywhere to match the actual proof requirement, or formally prove coverage of β = 1/2 if it is genuinely included.

- **Add a formal lemma for the η bound:** State and prove (or provide a proof sketch in the appendix) that η ≤ C/n^{1/4} with high probability when β < 1/2 and F is independent of γ and ξ. Reference this lemma when invoking Theorem 2, so it is clear the theorem's condition is satisfied in the main experimental regime.

- **Clarify Figure 4 legend:** Rename "Random Feature Model (blue diamonds)" to "Noisy Polynomial Model (blue diamonds)" if that is what is being shown, and confirm that ReLU and Softplus are labeled as RFM variants in the caption.

- **Add a vertical β = 0.5 boundary marker in Figure 3c** to visually distinguish the proven regime from the extrapolated regime.

- **Expand the proof approach section** to explicitly describe the technical difficulties introduced by the spiked covariance structure—specifically, how the rank-one perturbation θγγ^T affects the Lindeberg exchange step and moment estimates—to make the contribution over Hu & Lu (2023) transparent to readers.

---

**Evaluation summary:**
- *Novelty:* Moderate-to-good. The extension of universality to spiked covariance data and the polynomial equivalence characterization via Hermite coefficients are genuine contributions, though the techniques build directly on Hu & Lu (2023).
- *Technical soundness:* Good within the stated assumptions, but with a formal gap around the η condition and a genuine limitation from the odd-activation requirement.
- *Empirical support:* Strong for synthetic experiments; the CIFAR-10 results are suggestive but methodologically under-justified.
- *Significance:* Moderate. The paper offers a principled explanation for a widely observed empirical phenomenon, making it meaningful for the theoretical ML community.
- *Clarity:* Good overall, with specific issues in Figure 4 labeling and the informal treatment of the η distributional claim.