Let me check some specific claims against the paper content I have. The paper is provided in the user message, so I can directly reference it.

Key points to verify:

1. **The "cannot solve" framing**: The paper says "compositionally structured kernel models can only solve tasks that can be expressed in conjunction-wise additive terms" and "partitioning the compositional task space into tasks that can and cannot be solved by kernel models (Fig. 2d)." However, the paper also says "we analyzed compositionally structured inputs, as we consider them the best-case scenario" and the theoretical setup makes clear this is within the compositionally structured kernel + min-norm regime. The abstract states "we find that these models are limited to functions that assign values to each combination of components seen during training, and then sum up these values" — this is somewhat more carefully stated, but still lacks explicit qualifiers.

2. **Theorem 4.2 proof**: It's in Appendix A.4, not in the main text. The main text doesn't give a proof sketch.

3. **Empirical validation**: The paper states "we empirically validate our theory, showing that it captures the behavior of deep neural networks" — this is the claim to evaluate. The experiments show qualitative alignment (increasing slopes, etc.) but not quantitative fits to the specific formula.

Now let me write the final review.

## Summary

This paper develops a theory of compositional generalization for kernel models with compositionally structured representations—inputs whose pairwise similarities depend only on the number of overlapping discrete components. The main theoretical result (Theorem 4.2) establishes that such models are constrained to "conjunction-wise additive" computations: they sum values assigned to conjunctions of components seen during training, and on test inputs with novel conjunctions, they can only use the component-wise additive portion. This partitioning of the task space implies that kernel models with compositionally structured representations cannot solve tasks like transitive equivalence. The paper further identifies two failure modes—memorization leak and shortcut bias—arising from the interaction of representational geometry and training data statistics, and provides qualitative empirical validation on ConvNets, ResNets, and ViTs.

## Strengths

- **Clean formalization of compositional generalization for kernel models.** The paper provides a precise definition of compositionally structured representations (Def. 3.1) and derives a non-trivial structural constraint (conjunction-wise additivity, Theorem 4.2) that cleanly partitions the compositional task space. The result that even compositionally structured representations—supposedly the best case for compositional generalization—are insufficient for tasks like transitive equivalence (while sufficing for transitive ordering) is a genuinely novel and informative insight.

- **Analytical characterization of failure modes.** Proposition 5.1 provides a closed-form expression for the memorization leak factor *m* as a function of representational salience and training set coverage. This is a crisp, quantitative result that connects representational geometry to generalization performance in an interpretable way. The shortcut bias analysis on context dependence (showing that accuracy is either 100% or 0% depending on salience) is also compelling and well-motivated.

- **Informative analysis of representational salience.** The metric S(k;C) provides a parsimonious summary of compositionally structured kernels, and the analysis of how salience evolves with network depth (Fig. 3, Proposition B.2) yields an interesting and practically relevant finding: very deep random networks become pure lookup tables (S(C;C)→1), which impairs compositional generalization.

- **Well-designed experiments bridging theory and practice.** The spatial distance manipulation in ConvNets (Fig. 5a,c) that varies S(1;2) is a creative way to test theoretical predictions in real networks. The random assignment of categories to components controls for idiosyncratic correlations. The observation that deep networks exhibit conjunction-wise additive behavior (even though they can in principle express non-additive functions) is a non-trivial empirical finding.

- **Clarifies an important debate.** The paper provides a principled explanation for why empirical results on disentangled representations and compositional generalization have been inconsistent: even on tasks that are in principle solvable by conjunction-wise additive models, the specific representational geometry and training data critically determine success or failure.

## Weaknesses

### Fatal
None.

### Major

- **Overclaimed generality of impossibility results.** The central negative claim—that kernel models "cannot solve" non-conjunction-wise-additive tasks like transitive equivalence—is established only for the specific setting of compositionally structured kernels (Def. 3.1, where the kernel depends *only* on the number of overlapping components) with minimum ℓ₂-norm solutions. However, throughout the paper (abstract, Section 4.3, Fig. 2d), these results are presented as general impossibility statements about "kernel models" or "the compositional task space," dropping these critical qualifiers. For instance, the abstract states "these models are limited to functions that assign values to each combination of components seen during training" and Fig. 2d shows a partition of "the compositional task space" without noting this is conditional on the representation being compositionally structured. The paper does say "we analyzed compositionally structured inputs, as we consider them the best-case scenario," but this framing positions the strong symmetry assumption as if it were obviously favorable, when in fact many reasonable representations (including disentangled VAEs with continuous factors) have similarities that depend on *which* components overlap, not just *how many*. The implications for general kernel models are more speculative than the presentation suggests. This matters because the impossibility of transitive equivalence is a core selling point, and it only holds under assumptions that may be narrower than the reader infers.

- **Empirical validation is qualitative rather than quantitative.** The theory makes precise quantitative predictions—e.g., the memorization leak slope *m = p·S(1;2)/(1+(p−2)S(1;2))*—yet the deep network experiments only confirm qualitative trends ("higher distance increases the slope," "larger training sets increase the slope"). There is at least one explicit counterexample (for p=4, extrapolation slope deviates from the interpolation-invariance prediction). Furthermore, the claim that conjunction-wise additive models are "highly predictive" of deep network responses is stated without quantitative metrics in the main text (delegated to Appendix D.3). Without showing that the *functional form* matches within error bars—and without quantifying how closely real deep networks approximate compositionally structured representations—claiming the theory "captures the behavior of deep neural networks" (abstract) overstates the evidence. The correct evidential status is: qualitative patterns predicted by the kernel theory appear in these specific deep networks and tasks.

### Minor

- **The transition from compositionally structured representations to real learned representations is under-supported.** The paper acknowledges this limitation but the supplementary evidence (Appendix A.5: conjunction-wise additive "in expectation" over random tasks; Appendix C: disentangled representations on dSprites) does not establish that real networks satisfy Def. 3.1 to any approximation quality. An analysis of how much deviation from compositionally structured kernels degrades the theoretical predictions would significantly strengthen the practical relevance of the theory.

- **Shortcut bias analysis lacks a theorem analogous to Proposition 5.1.** While memorization leak is characterized in closed form (Proposition 5.1), the shortcut bias phenomenon on context dependence is analyzed empirically (Fig. 4c,d) without a formal result stating precisely when and why the model falls into the shortcut solution. This makes shortcut bias less firmly grounded than memorization leak.

- **Limited scope of tasks and architectures tested.** The experiments cover only two task families (symbolic addition and context dependence, with transitive equivalence noted as unsolvable) and three vision architectures on MNIST/CIFAR-10. Whether the theory's predictions extend beyond these settings—especially to sequence models, language tasks, or tasks with continuous components—remains open.

- **Theorem 4.2's proof is entirely delegated to the appendix** without a proof sketch or explanation of the key algebraic insight in the main text. Given that this is the paper's central structural result and all subsequent analysis builds on it, the reader is asked to take the result largely on faith. A paragraph explaining the role of the symmetries in Def. 3.1 and how they yield the additive decomposition would improve verifiability.

### Trivial
- Fig. 2d labels the task space partition without noting the conditional dependence on representation type, which could mislead readers skimming the figures.

## Nice-to-Haves

- Quantitative comparison between predicted and observed slopes for symbolic addition (even if approximate) would substantially strengthen the empirical claim that the kernel theory captures DNN behavior.
- Testing the theory's negative predictions by training deep networks on non-conjunction-wise-additive tasks (e.g., transitive equivalence) and showing they fail in predictable ways.
- An experiment comparing kernel-regime networks (large initialization) vs. feature-learning-regime networks (small initialization) on the same compositional tasks to assess whether feature learning can overcome conjunction-wise additivity.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"The paper overstates novelty relative to known kernel properties."** (From human finder, point 4.) The conjunction-wise additivity result is not merely a repackaging of known kernel spectral properties—it provides a *task-level* constraint on what compositional computations these models can and cannot perform, which is a novel structural insight. The prior work mentioned (Canatar et al., 2021) analyzes spectral bias in kernel regression generally, but does not derive the conjunction-wise additive decomposition or its implications for compositional generalization.

- **"Missing experiments testing deep networks on non-additive tasks."** (From Spark, point 1.) The paper explicitly proves that kernel models with compositionally structured representations *cannot* solve transitive equivalence (Section 4.3). Testing this negative prediction in deep networks is interesting but not necessary for the paper's claims—the theory's value lies in characterizing both what these models can and cannot do, and the empirical validation focuses on verifying the additive predictions (which are the more testable part). Demanding validation of impossibility is a high bar not typically required.

- **"Compare kernel-regime vs. feature-learning-regime directly."** (From Spark, point 2.) The paper explicitly scopes its theory to the kernel regime and acknowledges this limitation in Section 7. While such comparisons would strengthen the paper, they fall outside the stated scope and constitute future work.

- **"The best-case rhetoric understates how unrealistic the assumption is."** (From harsh critic, point 4.) This is partially valid but overstates the issue. The paper provides justification for focusing on compositionally structured representations (they are the natural "best case" that disentangled representation learning aims for), and the appendix provides preliminary evidence that the limitations extend beyond this setting. The "best case" framing is defensible as a theoretical strategy—to identify fundamental limitations even in the most favorable setting—but the caveat that real representations may differ qualitatively could be stated more prominently.

- **"Theorem 4.2 relies on symmetry assumptions that may be flawed."** (From harsh critic, point 2.) This concern about proof correctness is legitimate in that the proof is in the appendix, but claiming the argument "may not be airtight" without verifying the appendix is speculative. The symmetry argument is standard in representation theory: when the kernel matrix is invariant under a group action (permutations of items within each component type), the dual solution can be decomposed via the corresponding irreducible representations, which naturally yields the conjunction-wise additive structure. This is not a "single point of failure" whose correctness is in doubt—it is a well-understood technique. The legitimate concern is about the *strength* of the symmetry assumption, not about whether the proof correctly derives the additive decomposition from that assumption.

- **"The paper lacks actionable architectural guidance."** (From human finder, point 6.) Providing architectural guidance for overcoming the identified failure modes is beyond the paper's stated scope. The paper's contribution is diagnostic (identifying *when* and *why* compositional generalization fails), not prescriptive. This is a reasonable direction for future work.

## Novel Insights

The paper reveals an intriguing asymmetry between transitive ordering (which kernel models *can* solve) and transitive equivalence (which they *cannot*). Both are relational reasoning tasks, and at first glance they seem similarly structured—yet conjunction-wise additivity permits one and forbids the other. This distinction is not obvious and highlights the value of formal analysis over intuition. The memorization leak—where the model "leaks" training-set-specific conjunction information into its test predictions, systematically shrinking generalization—also provides a precise mechanistic account for a phenomenon (training set memorization corrupting systematic generalization) that had been observed empirically but not analytically characterized with closed-form dependencies on salience and training coverage.

## Suggestions

- Add a brief proof sketch of Theorem 4.2 in the main text (2-3 sentences describing the role of permutation symmetries and the decomposition into irreducible subspaces). This would greatly improve verifiability without requiring readers to navigate the appendix.
- Temper the framing throughout: replace "tasks that cannot be solved by kernel models" with "tasks that cannot be solved by kernel models with compositionally structured representations" in all instances, especially the abstract and Fig. 2d.
- Report quantitative fit metrics (e.g., R² between predicted and observed slopes) for the symbolic addition experiments, even if approximate, to clarify whether the theory makes accurate numerical predictions or merely captures directional trends.

## Score and Decision

**Calibration anchors:**
- "Interaction Asymmetry" (cCl10IU836, scores 6-8, accept poster): similarly strong theory on compositional generalization with formal definitions and proofs, but with comparable gaps between theory and experiments.
- "On Provable Length and Compositional Generalization" (Hxm0hOxph2, scores 5-6, reject): theoretical results on compositional generalization but with idealized/limited settings and qualitative validation.
- "Breaking Neural Network Scaling Laws with Modularity" (5Qxx5KpFms, scores 3-8, accept poster): strong theoretical contribution but with concerns about overclaiming generality and limited experimental validation.
- "Understanding and addressing spurious correlation via NTK" (89AOrk05uy, scores 3-6, withdrawn/treated as reject): NTK-based analysis of shortcut learning with qualitative validation, concerns about gap between kernel regime and real networks.
- "Generalization error of spectral algorithms" (3SJE1WLB4M, scores 8, accept spotlight): gold standard for kernel generalization theory with precise quantitative results.

This paper sits above the "Provable Length" paper (which was rejected) because it provides a broader task-space characterization and identifies concrete failure modes with some empirical support. It sits below the "Spectral Algorithms" paper because its quantitative predictions are less precisely validated and the theory's scope is more restricted than claimed. It is comparable to "Interaction Asymmetry" (accepted as poster) in having a solid theoretical contribution with meaningful but imperfect experimental validation, and a significant overclaiming issue that tempers but does not eliminate the contribution. The core theory is genuinely insightful, but the overclaiming of generality and the qualitative-only validation are meaningful weaknesses.

Given the substantial theoretical contribution (conjunction-wise additivity, memorization leak, shortcut bias), the meaningful empirical findings, but the notable overclaiming of generality and lack of quantitative validation, I rate this paper somewhat above the acceptance boundary.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>