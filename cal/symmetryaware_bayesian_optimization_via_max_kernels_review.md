=== CALIBRATION EXAMPLE 75 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract

The title accurately reflects the core contribution: using max-type kernels to enforce group invariance in Bayesian optimization. The abstract correctly summarizes the idea—projecting the non-PSD max kernel onto the PSD cone—and the claims about lower regret are backed by Table 2 and Figure 3. However, calling the final object a "max kernel" throughout the paper is slightly imprecise: what is actually deployed in BO is *k*⁺(*D*), the PSD-projected, Nyström-extended surrogate, not *k*_max itself. This distinction could mislead readers about what is being used.

---

### Introduction & Motivation

The motivation is well-constructed. The intuition that "only the best-aligning group element matters" is sensible, and the contrast with orbit averaging is conveyed clearly through the cat-rotation analogy. The authors also correctly identify the non-PSD nature of *k*_max as the core obstacle and describe the PSD-projection approach to address it.

One gap is that the introduction presents *k*_max as a generically applicable tool, but Proposition 1 (formalized as Proposition 5) relies on the existence of a map ϕ_G satisfying minimal-distance representativity (condition (ii)). The introduction does not caution readers that this map may not exist for all group actions, and its existence is quietly assumed throughout. This has downstream consequences for the GP motivation of *k*_max.

The claim in the "Relation with spectral-based theory" paragraph—that *k*⁺(*D*) has *slower* eigendecay yet lower regret—is one of the paper's most interesting empirical findings, and it is introduced up-front. That honesty is appreciated. However, framing this as "challenging the usual spectral intuition" risks overstating the finding, since the regret bounds in question are upper bounds; a method could easily outperform a looser bound without the bound being vacuous.

---

### Section 3: The Max Kernel

**Existence of ϕ_G (critical gap).** Proposition 5 assumes the existence of a map ϕ_G : S → S_h satisfying (i) G-invariance and (ii) ∥ϕ_G(x) − ϕ_G(x')∥₂ = min_{g,g'} ∥gx − g'x'∥₂. No discussion of when this map exists is provided. For general compact groups acting on R^d (e.g., SO(3) acting on R^3), the quotient metric min_{g,g'} ∥gx − g'x'∥₂ is not always embeddable isometrically into a Euclidean space; isometric embeddability is a non-trivial condition. Without verifying this for the examples used in the paper (hyperoctahedral group, SO(2), rescaling), the GP motivation of *k*_max rests on an unverified premise. The paper would be substantially strengthened by either (a) proving ϕ_G exists for all groups used, or (b) acknowledging this as a condition on the group/action and discussing whether it holds in the examples.

**Lemmas 2 and 6.** These results are essentially tautological: averaging reaches the maximum only if all terms are equal. The extension to different base kernels in Lemma 6 is similarly elementary. While not incorrect, the claims are presented as meaningful lemmas, which over-inflates their significance.

**Data-dependence of *k*⁺(*D*).** The kernel *k*⁺(*D*) depends on the entire past design set *D*, which changes at every BO iteration. This creates several conceptual and practical issues that the paper underplays:
1. The kernel, and therefore the associated RKHS, changes at each step of the BO loop. The standard theoretical guarantees for BO (e.g., GP-UCB regret bounds) require a fixed kernel and a fixed RKHS. It is unclear whether these guarantees transfer.
2. When fitting hyperparameters by maximizing the log-likelihood, the likelihood itself depends on *K*⁺ (the PSD-projected Gram matrix). Since *K*⁺ is a non-differentiable function of the data (eigenvalue clipping introduces kinks), gradient-based hyperparameter optimization may behave unexpectedly. The paper does not address this.
3. The Nyström approximation means that *k*⁺(*D*)(x,x') is not a proper RKHS kernel in the usual fixed sense; it is better thought of as an adaptive feature map. While Appendix C.4 shows spectral convergence to the data-independent *k*⁺, this is an asymptotic consistency result, not a finite-sample BO guarantee.

The paper acknowledges the data-dependence in Section 3.2 and mentions *k*⁺ (the intrinsic, data-independent version) as an object for future theoretical work. However, the implications for the validity of the BO procedure within the current paper deserve more prominent and honest treatment.

**Per-query computational cost.** Table 1 states that the per-candidate acquisition evaluation cost is O(n|G|*) for *k*⁺(*D*), compared to O(|G|*) for *k*_avg. The claim that "this difference is negligible as long as we keep m ≲ n" is asserted but not verified. In modern BO implementations (e.g., BOTorch with multi-start gradient ascent), the number of candidate evaluations m can easily be 512–2048 even for modest n (e.g., n = 50 after 50 iterations). In this regime, m ≫ n and the factor-of-n overhead could be practically significant, especially for large |G|. Table 3 shows a roughly 2× overhead for Ackley2d and Ackley3d, which suggests the overhead is manageable in these small cases, but the regime where it might become problematic (large group, many candidates) is not systematically explored.

---

### Section 4: Experiments

**Only one acquisition function.** All experiments use GP-UCB. Expected Improvement (EI) and Thompson sampling are also common in practice, and the relative performance of *k*⁺(*D*) vs *k*_avg might differ under different acquisition functions—especially since the behavior of UCB depends heavily on the posterior variance, which *k*⁺(*D*) models differently. This restriction limits the generalizability of the conclusions.

**Number of seeds.** Ten seeds are used throughout. Given the high variance in some benchmarks (e.g., Griewank6d has stderr ≈ 841 for *k*_avg out of a mean of 3067; Scaling2d has std ≈ 1135 for *k*_b), 10 seeds provides insufficient statistical power to conclusively differentiate some methods. Reporting 95% confidence intervals based on 10 seeds (as the paper does) is reasonable, but the confidence intervals for some benchmarks are wide enough that the ranking might change with more seeds.

**Ackley, Rastrigin as benchmarks.** These are standard benchmark functions, but they were specifically chosen because they are invariant under the hyperoctahedral group. The authors should clarify whether the *exact* symmetry group was used or whether a subgroup was used. More importantly, the fact that these synthetic objectives are highly structured (multimodal but globally symmetric) may favor *k*⁺(*D*) specifically. Benchmarks where the assumed symmetry is only approximate or where the function has additional structure not captured by the group would better test robustness.

**Scaling2d benchmark.** The function *f*_Scaling(x) = -(x₁/x₂ - 1)² on [0.1, 10]² is a simple 2D function with the global optimum along the line x₁ = x₂. Yet *k*_b has extremely high variance (σ_err ≈ 1135), much larger than for other benchmarks. The paper provides no explanation for this. If this reflects poor optimization of the acquisition function on this domain (due to the large scale range [0.1, 10]), then the baseline may be underperforming for implementation reasons unrelated to kernel choice.

**Brown et al. (2024) comparison.** The paper compares against *k*_avg from Brown et al. (2024) but uses *different* benchmark functions (not the GP-draw benchmarks used in that paper). This makes it difficult to directly assess whether the improvements are due to the kernel itself or due to differences in the test suite. The authors speculate in Section 6 that Brown et al.'s benchmarks were specifically structured (synthetic objectives are linear combinations of few kernel atoms), making *k*_avg artificially competitive there. This is a reasonable hypothesis, but it remains speculative. A direct comparison on Brown et al.'s benchmarks would be more convincing.

**Missing baselines.** The paper does not compare against:
- Eigenvalue *flipping* (setting negative eigenvalues to their absolute value) as an alternative PSD correction.
- Kreĭn-space formulations of indefinite kernels, which are cited in the related work.
- The data-independent *k*⁺ kernel (approximated on a dense grid), which the paper proposes as a cleaner theoretical object—its empirical performance would be informative.

**Real-world experiments.** The WLAN and particle packing results are genuinely interesting. However, for PartPack6d, only 30 BO iterations are reported (constrained by the ~4-hour simulation cost per seed). 30 iterations in 6D is very limited, and it is unclear whether the observed advantage of *k*⁺(*D*) would persist or diminish with more observations. The authors should discuss this limitation explicitly.

---

### Section 5: Spectral Analysis and Regret Bounds

The observation that *k*_avg has faster eigendecay but worse empirical regret than *k*⁺(*D*) is empirically interesting. However, the analysis remains at the level of description and hypothesis—no theorem or formal bound is derived for *k*⁺(*D*). This is a significant gap for an ICLR paper:

1. **No regret bound for *k*⁺(*D*).** The paper identifies that existing bounds don't explain the results, but proposes no alternative bound. Given that the intrinsic *k*⁺ is defined and shown to be PSD and G-invariant, even a regret bound for BO with *k*⁺ (the data-independent version) would be a meaningful contribution.

2. **The geometry hypothesis.** The claim that *k*_avg induces "similarity reversals" is illustrated in Figures 1–2 for the specific case of radial invariance with RBF. Whether this generalizes to other settings (e.g., finite groups, Matérn base kernels) is not analyzed.

3. **The approximation hardness hypothesis.** The argument that *f** may be harder to approximate in H_{k_avg} than in H_{k+(D)} is plausible, but the claim specifically that Brown et al.'s benchmarks are "linear combinations of few *k*_avg atoms" is asserted without evidence (citing their Appendix B.1). This deserves formal verification.

The Schatten norm inequalities in Appendix D.2 are technically correct but only bound *k*⁺ against *k*_avg in terms of operator norm growth with |G|. They do not shed light on why *k*⁺(*D*) achieves lower regret despite slower eigendecay.

---

### Appendix C: Intrinsic PSD Projection

The mathematical development in Appendix C is careful and appropriate. The convergence of *k*⁺(*Dn*)/n to the intrinsic integral operator *Tk*⁺ in the spectral *ℓ*₂ sense (Proposition 9) and the O(n^{-1/2}) expected HS rate (Proposition 10) are correctly established by leveraging Koltchinskii & Giné (2000). Lemma 14 (Borel functional calculus preserves invariance) is correctly proved and the key insight (U_g fixes nonzero eigenspaces of T when U_g T = T) is valid. These results are technically sound but are primarily convergence statements about spectral approximation quality, not about BO regret.

The assumption in Lemma 13 that µ is G-invariant is worth noting—this may not hold for the empirical measure arising from BO queries, since BO is not random and tends to cluster observations near the optimum.

---

### Writing & Clarity

The paper is generally well-written. Section 3.2 is somewhat dense, covering PSD construction, regularity, cost, and convergence in close succession; a cleaner delineation would help. The discussion of eigendecay in Section 5 is honest but abrupt—the section essentially concludes "theory doesn't explain practice; here are two hypotheses." More developed analysis would be expected from an ICLR submission at this stage.

---

### Limitations & Broader Impact

The paper provides a short limitations acknowledgment embedded in the conclusion, but several limitations are underemphasized:
- **Data-dependence of the kernel** and its implications for BO theory and implementation.
- **Scalability**: the O(n³) PSD projection and O(mn|G|) per-candidate cost grow rapidly with both n and |G|. For high-dimensional problems with large symmetry groups (|G| = 3840 in Rastrigin5d), the experiments run on only 50 iterations, which may be too few for practical problems.
- **Requiring exact symmetry knowledge**: the method requires the user to specify the symmetry group G exactly. Robustness to approximate or misspecified symmetry groups is not evaluated.
- **Single acquisition function**, as noted above.

---

### Overall Assessment

This paper presents a conceptually natural idea—using max-alignment rather than orbit-averaging for symmetry-invariant GP kernels in BO—and executes it with care, combining PSD projection and Nyström extension to obtain a practically usable kernel. The experimental results are broadly convincing, with *k*⁺(*D*) consistently outperforming *k*_avg across diverse benchmarks, and the analysis correctly identifies a gap between spectral theory and empirical BO performance.

The principal weaknesses are: **(1)** the existence of the map ϕ_G motivating *k*_max as a valid covariance is unverified for the experiments; **(2)** the data-dependence of *k*⁺(*D*) is not adequately reconciled with the standard BO theoretical framework, and its implications for hyperparameter optimization and regret analysis are glossed over; **(3)** no theoretical regret bound is provided for the proposed method, leaving the spectral analysis section at the level of observation and speculation; **(4)** the experimental setup is narrow (GP-UCB only, 10 seeds, 50 iterations, limited baselines). Despite these gaps, the empirical contribution is solid and the overall narrative is coherent and honest about what remains open. At ICLR, where the bar for theoretical completeness is high and novel empirical methods benefit from stronger guarantees or broader evaluation, this paper sits at the borderline—likely a weak accept if the data-dependence concern and missing existence argument are adequately addressed, but currently not at the level expected for a full ICLR contribution without revisions.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses symmetry in Bayesian Optimization (BO) by proposing a "max-kernel" ($k_{max}$) that aligns points based on the best group transformation rather than averaging over all symmetries. Since $k_{max}$ is typically not positive semi-definite (PSD), the authors introduce a PSD surrogate ($k_+(D)$) constructed via eigenvalue clipping and Nystrom extension, demonstrating empirically that this leads to lower regret than standard orbit-averaged kernels without significantly increasing computational complexity. The work includes a detailed analysis of why standard spectral regret bounds fail to predict these gains, attributing the improvement to geometric alignment over eigendecay rates.

### Strengths
1.  **Convincing Empirical Gains:** The method consistently outperforms orbit-averaged baselines ($k_{avg}$) and base kernels across a diverse set of synthetic and real-world benchmarks. For example, in Table 2, $k_{+}(D)$ achieves a cumulative regret of 18.4 on the Rastrigin5d benchmark compared to 1583.5 for $k_{avg}$, a substantial improvement.
2.  **Geometric Intuition and Analysis:** The paper provides strong motivation for why averaging dilutes signal information in symmetric settings. Example 3 and Figure 1 explicitly visualize the structural mismatch of $k_{avg}$ versus the geometric fidelity of $k_{max}$, which is a compelling argument for the max-alignment approach independent of regret bounds.
3.  **Strong Theoretical Grounding:** Beyond the practical PSD trick, the authors develop a rigorous theoretical framework (Appendix C) showing that the data-dependent finite-sample kernel converges to an intrinsic data-independent PSD operator $k_+$ under i.i.d. sampling. The spectral analysis in Section 5 is particularly valuable for the community by highlighting a mismatch between spectral decay rates and empirical BO performance.
4.  **Reproducibility:** The authors provide a GitHub repository link and detail experimental setups, including hardware specifications and hyperparameters, in the Appendix, meeting ICLR's standards for reproducibility.

### Weaknesses
1.  **Acquisition Function Cost Overhead:** While the asymptotic complexity per iteration is claimed to be comparable to $k_{avg}$ (Table 1), the per-candidate evaluation cost for $k_{+}(D)$ scales linearly with dataset size $n$ ($O(n|G|)$), whereas $k_{avg}$ scales independently of $n$ ($O(|G|)$). Table 3 confirms this in wall-clock time, showing $k_{+}(D)$ takes roughly 2x as long per iteration (0.924s vs 0.451s on Ackley2d). This limits scalability for long runs where $n$ becomes large.
2.  **Theoretical Bound Gap:** The paper concludes that standard eigendecay-based theory fails to explain the performance gains but does not propose a theoretical justification that *does* explain them (Section 6). While honest about the limitation, the lack of a new regret bound or approximation error analysis leaves a gap between the strong empirical results and theory.
3.  **Limited Baseline Comparison:** The comparison focuses primarily on $k_{avg}$ and the base kernel $k_b$. While Appendix G discusses fundamental domain approaches, the paper dismisses them without a direct quantitative comparison in the main experiments, which might be of interest to other ICLR readers working on symmetry.
4.  **Data-Dependence Concerns:** The kernel $k_{+}(D)$ depends on the specific design set history. While Appendix C provides convergence guarantees, the implications of this non-stationarity on acquisition function optimization (e.g., potential bias in uncertainty estimation depending on $D$) are briefly discussed and could be expanded in the main text.

### Novelty & Significance
The novelty lies in adapting the "max-alignment" principle, common in structure learning, to the BO domain and resolving the PSD violation specifically for BO requirements. The significance is high for applications where symmetry groups are large (e.g., molecular biology, physics), as shown in the WLAN and Particle Packing experiments. The finding that averaging symmetries can be detrimental compared to maximizing alignment is counter-intuitive and contributes to a deeper understanding of symmetric kernels beyond just invariance enforcement.

### Suggestions for Improvement
1.  **Clarify Computational Overhead:** Explicitly discuss the implications of the $O(n|G|)$ acquisition evaluation cost in the main text. While claimed to be negligible for $m \lesssim n$, this should be contextualized against the typical growth of $n$ in BO and offer strategies (e.g., sparse approximation) if $n$ becomes large.
2.  **Expand on the Theory/Practice Gap:** In Section 6, briefly suggest potential directions for future theoretical work that could bridge this gap (e.g., approximation hardness in the RKHS or effective dimension arguments), even if the immediate derivation is out of scope.
3.  **Add Fundamental Domain Comparison:** If space permits, add a small ablation study comparing $k_{+}(D)$ against a fundamental domain approach (as implemented in Baird et al. 2023a) on one of the more expensive benchmarks to solidify the choice between kernel design vs. search space restriction.
4.  **Refine Main Text Theory:** Summarize the convergence results of Appendix C slightly more in Section 3 to make the validity of the data-dependent kernel more accessible to readers before the detailed proofs.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Add "Fundamental Domain + Base Kernel" baseline.** You discuss domain restriction in Appendix G but exclude it from Table 2; without beating this primary alternative symmetry-handling strategy, the claim of superior efficiency is incomplete.
2. **Estimate RKHS norms $\|f^*\|_{\mathcal{H}_k}$ for both kernels.** Your "approximation hardness" hypothesis in Section 6 relies on $f^*$ being easier to approximate in $\mathcal{H}_{k_+}$, but you provide no empirical evidence quantifying this distance.
3. **Extend benchmarks to $d \geq 10$.** Current experiments stop at $d=6$, which is insufficient to validate scalability claims for Bayesian Optimization methods where the curse of dimensionality is the primary hurdle.
4. **Ablate the PSD projection aggressiveness.** Report the fraction of negative eigenvalues clipped per iteration; without this, it is unclear whether performance gains correlate with how much the original $k_{max}$ geometry is distorted.

### Deeper Analysis Needed (top 3-5 only)
1. **Correlate Information Gain $\gamma_T$ with Regret.** Since eigendecay fails to explain performance (Section 5), you must show if empirical information gain captures the advantage of $k_+$ to validate the theoretical disconnect.
2. **Analyze acquisition optimization convergence stability.** The kernel is only "almost everywhere differentiable"; quantify how often gradient-based acquisition optimizers (L-BFGS) fail or stall compared to the smooth $k_{avg}$.
3. **Hyperparameter sensitivity analysis.** BO performance is highly sensitive to lengthscale and noise; demonstrate that $k_+$ does not require significantly more tuning stability than $k_{avg}$ to achieve its gains.
4. **Spectral distribution of $k_{max}$ before projection.** Analyze why $k_{max}$ is indefinite across different group sizes; understanding the source of indefiniteness is crucial to justify why projection preserves the useful signal.

### Visualizations & Case Studies
1. **Posterior variance contour maps.** Figure 2 shows means, but BO relies on uncertainty reduction; visualize variance to prove $k_+$ reduces uncertainty faster in symmetric regions than $k_{avg}$.
2. **Acquisition function surface plots.** Plot the UCB surface at an intermediate iteration to expose whether $k_{avg}$ creates misleading local optima due to symmetry dilution while $k_+$ does not.
3. **Gram matrix heatmaps (before/after projection).** Visualize the $n \times n$ kernel matrix to直观 show how much structural similarity information is discarded by the eigenvalue clipping step.

### Obvious Next Steps
1. **Derive a regret bound using geometric alignment.** Admitting standard spectral theory fails (Section 5) requires a new theoretical bound incorporating your proposed "geometry vs. rates" metric to support the core contribution.
2. **Benchmark against Krein-space GPs.** You dismiss indefinite kernels via projection but do not compare against native indefinite kernel methods; this is needed to justify projection as the superior remedy.
3. **Integrate with Sparse GP approximations.** The $O(n^3)$ projection cost limits scalability; extending $k_+$ to variational or sparse frameworks is essential for practical utility beyond small sample sizes.

# Final Consolidated Review
## Summary
This paper proposes a max-alignment approach for symmetry-invariant kernels in Bayesian Optimization. Rather than averaging over all group transformations (the standard approach), the authors define a kernel that takes the maximum similarity between orbits. Since this max kernel is not positive semidefinite (PSD), they project it onto the PSD cone using eigenvalue clipping and extend it via Nyström approximation. Empirically, the resulting kernel consistently outperforms orbit-averaged alternatives across synthetic and real-world benchmarks, with gains increasing as the symmetry group size grows.

## Strengths
- **Strong empirical performance:** The proposed k₊⁽ᴰ⁾ kernel achieves substantially lower cumulative regret than orbit-averaged kernels k_avg across all tested benchmarks (Table 2). On Rastrigin5d (|G|=3,840), k₊⁽ᴰ⁾ achieves cumulative regret of 18.4±70.6 versus 1583.5±341.9 for k_avg—a dramatic improvement. Even on real-world tasks (WLAN, particle packing), k₊⁽ᴰ⁾ consistently finds better parameter combinations.
- **Geometric clarity:** The paper provides excellent motivation for why averaging dilutes signal. Example 3 and Figure 1 visually demonstrate that k_max correctly captures the similarity structure for rotation-invariant functions (k_max(x,x')=1 when ||x||₂=||x'||₂), while k_avg produces distorted iso-similarity contours. Lemma 2 formally establishes that averaging cannot reproduce maximization geometry.
- **Rigorous theoretical scaffolding:** Appendix C develops a data-independent PSD kernel k₊ via spectral projection of the integral operator and proves that the finite-sample k₊⁽ᴰ⁾ converges to k₊ in the spectral ℓ₂ sense (Propositions 9-10). The Schatten norm inequalities (Appendix D.2) bound the relationship between k_avg and k_max operators.
- **Identifies theory-practice gap:** Section 5 honestly documents that k_avg often has faster empirical eigendecay than k₊⁽ᴰ⁾, yet achieves worse regret—a finding that challenges conventional spectral regret bounds and motivates deeper investigation.

## Weaknesses
- **Data-dependent kernel with no regret bound:** The practical kernel k₊⁽ᴰ⁾ depends on the design set D, which changes at each BO iteration. Standard GP-UCB regret bounds assume a fixed kernel and RKHS; whether these guarantees transfer to data-dependent kernels remains unproven. While Appendix C.4 shows spectral convergence to an intrinsic k₊, this is an asymptotic result, not a finite-sample regret guarantee. The paper acknowledges this gap but does not resolve it.
- **Existence of the embedding φ_G is assumed, not verified for all experiments:** Proposition 5 (motivating k_max as a valid GP covariance) requires a map φ_G satisfying minimal-distance representativity. The paper shows this exists for rotation groups (Example 3), but does not explicitly verify it for the hyperoctahedral group or rescaling symmetries used in experiments. This is likely satisfied but should be confirmed or noted as an assumption.
- **Single acquisition function tested:** All experiments use GP-UCB. Whether the advantages of k₊⁽ᴰ⁾ transfer to Expected Improvement, Thompson sampling, or other acquisition functions is not evaluated. Since UCB relies heavily on posterior variance modeling (where k₊⁽ᴰ⁾ and k_avg differ significantly), testing alternative acquisitions would strengthen generality claims.
- **Limited scalability testing in higher dimensions:** Experiments stop at d=6 for synthetic benchmarks. The curse of dimensionality is the primary challenge in BO, and the O(n³) PSD projection cost combined with O(mn|G|) per-candidate evaluation raises scalability questions for larger problems. The paper addresses computational cost but only for small-scale experiments.
- **Moderate number of seeds for high-variance benchmarks:** Ten seeds are used throughout, but some benchmarks show high variance (e.g., k_b on Scaling2d has stderr ≈ 1135). While confidence intervals generally separate k₊⁽ᴰ⁾ from competitors, more seeds would strengthen conclusions on noisy benchmarks.

## Nice-to-Haves
- **Direct comparison with fundamental domain approaches:** Appendix G discusses restricting the search space but does not include quantitative comparison. A small ablation against fundamental domain methods (Baird et al., 2023a) would clarify when kernel design outperforms domain restriction.
- **Empirical validation of approximation hardness hypothesis:** Section 6 hypothesizes that f* is easier to approximate in H_{k₊} than in H_{k_avg}, but provides no quantitative evidence. Computing empirical RKHS distances or approximation errors would strengthen this argument.
- **Analysis of acquisition optimization stability:** The Nyström extension of k₊⁽ᴰ⁾ is only "almost everywhere differentiable." A brief comparison of gradient-based acquisition optimization convergence rates would be informative.
- **Testing alternative PSD corrections:** Eigenvalue flipping (rather than clipping) and Krein-space formulations are mentioned in related work but not empirically compared. Understanding whether projection is the optimal PSD remedy would be valuable.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Claim that "max kernel" terminology is misleading:** The paper consistently distinguishes k_max (the non-PSD max-alignment kernel) from k₊⁽ᴰ⁾ (the PSD-projected practical kernel). The abstract and introduction clearly state what is deployed. This criticism overstates a non-issue.
- **Lemmas 2 and 6 dismissed as tautological:** While mathematically straightforward, these lemmas formalize the key insight that averaging cannot achieve maximization geometry—essential for positioning the contribution. Formal statements serve technical completeness.
- **Demand for Brown et al.'s exact benchmarks:** The paper provides reasonable justification (Section 6) that Brown et al.'s synthetic objectives are structured as linear combinations of few kernel atoms, making k_avg artificially competitive. Demanding identical benchmarks is scope creep; the current benchmarks better represent practical BO scenarios.
- **Hyperparameter optimization concerns from non-differentiability:** The PSD projection involves eigenvalue clipping, which introduces kinks. However, the empirical results show the method works with standard likelihood optimization. This is a practical concern without demonstrated impact.
- **Criticism of missing missing baselines:** Requests for comparisons with Krein-space GPs, eigenvalue flipping, and data-independent k₊ are reasonable suggestions but not flaws—this is a methods paper with finite space, and the primary comparison against k_avg (the standard approach) is appropriate.
- **Missing missing related work claim:** Removed per instructions—no external verification available.

## Novel Insights
Beyond the paper's own contributions, the reviews reveal an underappreciated tension in kernel design for BO: invariance enforcement alone does not guarantee good optimization geometry. The paper shows that two G-invariant kernels (k_avg and k₊⁽ᴰ⁾) can induce fundamentally different similarity structures on orbit space, with the "max-alignment" approach better preserving the high-contrast regions that matter for optimization. The finding that faster eigendecay correlates with worse regret challenges the community to look beyond spectral rates toward geometric alignment metrics.

## Suggestions
1. **Add explicit verification of φ_G existence for all experiment groups** (hyperoctahedral, rescaling) or clearly state this as an assumption.
2. **Include one experiment with a different acquisition function** (EI or Thompson sampling) to demonstrate robustness beyond GP-UCB.
3. **Scale up to at least d=10** on one benchmark to address scalability concerns more directly.
4. **Report the fraction of negative eigenvalues clipped** during PSD projection across iterations—this would clarify how much distortion k_max undergoes.
5. **Briefly discuss hyperparameter sensitivity:** Whether k₊⁽ᴰ⁾ requires more careful lengthscale/noise tuning than k_avg is relevant for practitioners.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0]
Average score: 7.3
Binary outcome: Accept
