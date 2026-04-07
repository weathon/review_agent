## Summary

This paper addresses the problem of **Exploratory Causal Inference (ECI)**: discovering which outcomes are affected by a treatment in randomized controlled trials when outcomes are not pre-specified but are measured indirectly through high-dimensional observations (e.g., video, images). The authors combine foundation model representations with Sparse Autoencoders (SAE) to create interpretable neural codes, identify a "Paradox of ECI" where standard multiple testing fails due to neural entanglement, and propose **Neural Effect Search (NES)**, a recursive stratification algorithm that recovers principal effect directions while controlling false discoveries. The method is validated on semi-synthetic CelebA experiments and a real-world ant behavioral ecology trial.

## Strengths

- **Clear problem formulation and theoretical contribution:** The paper formalizes a novel and practically important problem—discovering unknown treatment effects from high-dimensional RCT data—and correctly identifies the statistical challenge that standard multiple testing corrections fail when neural representations are entangled (Theorems 3.1, 3.2). The ECI paradox formulation is technically sound and the mathematical development is rigorous.

- **Principled algorithmic solution:** The NES algorithm (Algorithms 1-2) addresses entanglement via recursive stratification over discovered neurons, with explicit theoretical justification. The consistency proof in Appendix A.3 is detailed and includes explicit assumptions (A.1-A.3), making the theoretical contributions transparent and evaluable.

- **Extensive empirical validation:** The paper provides thorough ablation studies (Appendix E) varying the foundation model (SigLIP, DINOv2), SAE architecture (dimensions, nonlinearities), and data-generating process parameters. The method's robustness across these variations strengthens confidence in the approach.

- **Honest treatment of limitations:** The limitations section explicitly acknowledges the untestable sufficiency assumption, SAE identifiability concerns, and properly frames the method as a "rescue system for hypotheses that may have been missed" rather than standalone inference. The inclusion of a "background marking artifact" discovery as an example of finding experiment design biases is a useful demonstration of what the method actually produces.

## Weaknesses

- **Empirical evidence is thin:** The real-world experiment has only n=44 videos, and the authors explicitly disabled Bonferroni correction for this experiment. The theoretical guarantees (Theorem 4.1) require asymptotic behavior that cannot be tested at this sample size. While this is transparently disclosed, it means the strongest empirical claim ("first successful application") rests on a setting where the method's safeguards are disabled. The semi-synthetic experiments only test r=2 effects, leaving scalability to more complex outcome structures unclear.

- **Assumption A.2 (principal alignment) may be fragile:** The consistency proof requires each true effect to have a distinct "principal neuron" that strictly dominates others (Equation 25). Appendix E.1 shows that for DINOv2 SAEs, the Wearing_Hat concept has F1≈0.43 with its best neuron, and the authors note it "could possibly be captured by all the top three most predictive neurons"—suggesting potential violation. The paper provides no diagnostic for detecting assumption violations in practice, and no failure-mode analysis showing how performance degrades gracefully when assumptions are violated.

- **FM sufficiency is fundamentally untestable:** The assumption that foundation model representations preserve all outcome information (I(X,Y) = I(ϕ(X),Y)) cannot be verified in exploratory settings where Y is unknown. The paper acknowledges this but offers no practical guidance for practitioners to assess whether their foundation model is adequate for a given domain.

- **Finite-sample collider bias is uncharacterized:** Conditioning on post-treatment SAE codes during stratification introduces potential collider bias. Assumption A.3 bounds this with an ε term that vanishes asymptotically, but no empirical or theoretical characterization of this bias in finite samples is provided—particularly relevant since the real-world experiment operates precisely in a small-n regime.

- **Relationship to sequential/selective inference literature is unaddressed:** The recursive selection procedure bears conceptual similarity to forward stepwise selection and post-selection inference problems. The paper does not engage with this literature (e.g., knockoff filters, selective inference) or explain how NES differs from or improves upon these approaches for this specific setting.

## Nice-to-Haves

- **Negative control experiment:** Applying NES to a real dataset with no treatment effect (or a shuffled treatment label) would empirically validate false positive rates.

- **Failure mode characterization:** Systematic experiments violating Assumption A.2 (e.g., using SAEs known to have distributed representations) would establish robustness boundaries.

- **Computational complexity analysis:** Runtime and memory as functions of SAE dimension m and sample size n would help practitioners assess feasibility for larger studies.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Typo complaint about Y(1) notation:** The review notes a minor text error but this is purely formatting-related and irrelevant to evaluation.

- **Claim that the ECI paradox isn't novel:** The review argues this is "standard" in hypothesis testing literature, but this misreads the contribution—the paper correctly applies this phenomenon to the specific context of entangled neural representations and causal effect discovery, which IS novel. The theoretical formalization in Theorems 3.1-3.2 provides new insight into the specific failure mode.

- **Demand for comparison with continuous outcomes:** The paper explicitly scopes to binary outcomes and notes continuous extensions as future work. This is scope creep—evaluating the paper on what it claims to do, not on a broader agenda.

- **Demand for end-to-end causal SAE training:** Suggesting a fundamentally different representation learning approach is outside the paper's scope; the contribution is the inference procedure given SAE representations.

- **Complaint about "rationalist vs empiricist" framing:** The terminology provides useful conceptual structure and is clearly motivated. This is stylistic preference, not a substantive weakness.

## Novel Insights

The "Paradox of Exploratory Causal Inference" formulation is the paper's key conceptual contribution: as sample size or effect magnitude increases, ANY neuron with non-zero leakage (entanglement) with true effects becomes significant, overwhelming multiplicity corrections. This insight specifically for *entangled representations in discovery settings* is not apparent from standard multiple testing theory and explains why simply running Bonferroni-corrected t-tests on SAE codes fails catastrophically with sufficient power. The recursive stratification solution—iteratively conditioning on discovered effects to "peel off" their leakage—directly addresses this structural property of the problem.

## Suggestions

- Provide an empirical diagnostic (even heuristic) for assessing Assumption A.2 violations before running NES, such as inspecting the concentration of F1-scores among top neurons for known attributes in a labeled validation subset.

- For small-sample applications (like the n=44 ecology experiment), report sensitivity analyses showing how results change under different multiple testing thresholds, rather than simply disabling Bonferroni.

- Include at least one experiment with r>2 effects of varying magnitudes to demonstrate that NES can recover multiple effects beyond the toy setting.

- Explicitly discuss the finite-sample implications of Assumption A.3's ε bound—either provide theoretical characterization of collider bias magnitude or empirical measurements of stratification-induced bias in simulated settings.