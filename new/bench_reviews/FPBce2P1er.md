Good, I've read the full main body. Now let me retrieve the human review files for calibration.Now I have enough to produce the final consolidated review. Let me synthesize everything carefully.## Summary
This paper presents a theory of compositional generalization in kernel models with compositionally structured representations (Definition 3.1). The central result (Theorem 4.2) shows that such models are constrained to "conjunction-wise additive" computations — summing values assigned to each overlap-conjunction seen during training. Building on this, the paper identifies two failure modes (memorization leak, Proposition 5.1, and shortcut bias), and empirically tests whether these phenomena extend to deep networks (ConvNets, ResNets, ViTs) on MNIST and CIFAR-10 compositional tasks.

---

## Strengths

- **Theorem 4.2 is a sharp and novel characterization.** The conjunction-wise additivity result cleanly partitions the compositional task space (Fig. 2d) and yields immediately non-obvious implications — notably that transitive *equivalence* fails while transitive *ordering* succeeds. This is a precise, non-trivial distinction.
- **Proposition 5.1 produces a concrete, falsifiable prediction.** The closed-form slope formula `m = p·S(1;2) / [1 + (p-2)·S(1;2)]` links representational geometry and training set size to a specific quantitative distortion. This is rare clarity for a theory paper.
- **Two practically-named failure modes.** Memorization leak and shortcut bias are identified with mechanistic explanations grounded in the theory. This moves beyond purely formal characterization to recognizable pathological behaviors.
- **Explains contradictory literature on disentangled representations.** The finding that context-dependence generalization is highly sensitive to representational salience (Fig. 4c) provides a principled account of why empirical results on disentangled representations have been inconsistent. This is a directly useful insight for the community.
- **Multi-architecture empirical support.** The qualitative predictions are confirmed across ConvNets, ResNets, and ViTs — three structurally different architecture families — increasing confidence that the mechanisms are not architecture-specific artifacts.
- **The spatial-distance manipulation (Fig. 5a, 5c)** is a particularly compelling demonstration: manipulating the physical distance between image components predictably shifts S(1;2) and consequently the generalization slope, linking the theory's parameters directly to architectural choices.

---

## Weaknesses

### Fatal
*(none — the core theoretical results stand as stated and proofs are provided in the appendix)*

### Major

- **Qualitative-only empirical validation; no quantitative verification of Proposition 5.1 in deep networks.** The paper's most elegant contribution is the exact slope formula in Eq. (3), but Section 6 never compares the formula's numerical prediction against observed slopes from deep networks. The authors acknowledge this gap in the Discussion ("we do not provide any quantitative bounds"), but it remains the most significant empirical shortcoming: readers cannot assess how accurate the kernel approximation is for these architectures. Measuring S(1;2) from intermediate representations (which the paper does, Fig. 5a) and plugging it into Eq. (3) to produce a predicted slope for comparison would be a direct and feasible test.

- **Transitive equivalence — the centerpiece theoretical result — is never empirically tested.** The key claim that kernel models *cannot* solve transitive equivalence (the main impossibility consequence of Theorem 4.2) is never tested in deep networks. Without this, the boundary between what kernel models and deep networks can solve is theoretically asserted but empirically invisible. A test showing whether deep networks succeed or fail at transitive equivalence (and whether they succeed in the feature-learning regime) would substantially strengthen the paper.

- **Deep networks are empirically evaluated on only two tasks.** Symbolic addition and context dependence are both conjunction-wise additive, meaning only one side of Fig. 2d is empirically explored with deep networks. The claim that the conjunction-wise additive partition is real and predictive for modern architectures would be more compelling with at least one task from the non-additive side.

### Minor

- **Definition 3.1's symmetry assumption is stronger than the prose suggests.** "Compositionally structured" in this paper means the kernel depends only on the *count* of overlapping components, not their *identity*. This is a non-trivial exchangeability constraint that many real learned representations may violate. The paper provides preliminary evidence in Appendix A.5 and C that the results generalize somewhat, but the main text does not adequately foreground this scope restriction, occasionally making claims that read more broadly than the theorem supports.

- **The kernel regime assumption is not empirically verified for the tested architectures.** The paper notes in passing that "these networks did not perfectly predict the training split, which may affect the results" — a direct signal that the architectures are not strictly in the kernel regime. There is no check on whether the NTK approximation holds for the ConvNets, ResNets, and ViTs under the reported training conditions, which affects how much confidence to place on the bridge between theory and Section 6.

- **The shortcut bias for context dependence lacks a closed-form characterization.** Unlike memorization leak, where Proposition 5.1 gives an exact formula, shortcut bias is characterized qualitatively (Fig. 4c/d). The paper describes the mechanism ("context shortcut plus full conjunction memorization") but does not provide a theorem-level result analogous to Proposition 5.1. A theoretical condition on when shortcut bias occurs would substantially strengthen this section.

### Trivial

- The paper reports n=10 random category assignments for controlling image-category correlation; some experiments show fairly wide error bars (Fig. 5c). This is noted but it would help to explicitly clarify whether the main trend holds under all 10 instantiations.

---

## Nice-to-Haves

- Estimate S(1;2) from deep network representations and directly compare the Proposition 5.1 slope prediction against empirical slopes. This is feasible given the infrastructure already built in Fig. 5a.
- Analyze representational salience S(k;C) evolution during training (not just at initialization / final weights) to understand whether feature learning shifts the effective conjunction distribution.
- A brief theoretical or empirical characterization of how robust the results are to perturbations of Definition 3.1's symmetry (e.g., when different components contribute asymmetrically to kernel similarity). The Appendix A.5 result is a start.

---

## Removed Points

> These points are flagged to be removed; treat them with caution.

**[Harsh Critic W1 — partially]**: The strongest version of the claim that "Section 6 provides no evidence that deep networks operate via the conjunction-wise additive mechanism" is overstated. The paper explicitly cites Appendix D.3 for the conjunction-wise additive decomposition fit, and the Discussion openly acknowledges the qualitative nature. The narrower and accurate concern — that the quantitative comparison is missing — is kept as a Major weakness.

**[Spark — No experiments in feature-learning regime]**: The request for experiments deliberately designed to be in the feature-learning regime (e.g., narrow networks, small initialization) is removed as scope creep. The paper is explicitly a kernel-regime theory paper; extending to the feature-learning regime is an acknowledged future direction, not a current omission.

**[All reviewers — kernel regime NTK conditions]**: The Human Finder's concern ("NTK requires not only infinite width but also small step size and large initialization scale") is technically correct but applied too bluntly here. The paper is careful to present kernel models as a *tractable approximation* that captures some behaviors of deep networks, not as an exact description. The limitation is real and kept as a Minor weakness, but reviewer framing that treats this as a fatal oversight is not carried forward.

**[Spark — training dynamics / convergence analysis]**: The request for a formal analysis of how salience changes over training is removed as a nice-to-have; demanding convergence analysis of training dynamics is not standard for this type of theoretical/empirical learning theory paper.

---

## Novel Insights

The most genuinely novel synthesis across the reviews is the following: the representational salience metric S(k;C) serves as the single scalar summary of representational geometry that connects the abstract kernel theory to actionable architectural predictions. Specifically, S(1;2) can be measured from intermediate-layer activations (Fig. 5a) and controls both *whether* generalization succeeds (via the conjunction-wise partition in Theorem 4.2) and *how much* distortion it introduces (via the slope formula m in Proposition 5.1). This creates a diagnostic pipeline — measure S(k;C) from a trained encoder, compute predicted m, compare to empirical slope — that neither the authors nor the reviewers fully spell out as a practical tool but which is implicit in the paper's contribution and could make the kernel theory directly operational for practitioners designing compositional tasks and architectures.

---

## Suggestions

1. **Add a single quantitative comparison panel**: Take the ConvNet (where S(1;2) is already measured from Fig. 5a), plug measured S(1;2) values and training-set sizes p into Eq. (3), and overlay predicted slopes on Fig. 5c/d. This would transform the qualitative validation into a genuine quantitative test of the theory.

2. **Test transitive equivalence empirically**: Train ConvNets and ResNets on the transitive equivalence task and compare performance to the kernel model prediction (failure). A single experiment filling this gap would strongly validate the conjunction-wise partition and its consequences.

3. **Foreground the symmetry assumption in Definition 3.1**: In the main text, add a sentence when first applying Theorem 4.2 to emphasize that "compositionally structured" means overlap-count-symmetric, and note the preliminary appendix evidence for robustness. This would pre-empt the reasonable concern about generality without requiring additional experiments.

---

## Score and Decision

**Calibration:**

- *Tj3xLVuE9f* ("Foundations of Shortcut Learning," accepted spotlight, avg score ~6.8): That paper also uses NTK theory for shortcut bias analysis with empirical support. The paper under review has a comparably clean theoretical contribution but narrower empirical validation.
- *H98CVcX1eh* ("Discovering modular solutions that generalize compositionally," accepted poster, avg ~6.5): Theory + empirics on compositional generalization with acknowledged limitations. Comparable structure and strength to the paper under review.
- *Hxm0hOxph2* ("Provable length and compositional generalization," rejected, avg ~5.3): Rejected theory paper with similar kernel-regime limitations but weaker main theorem and less actionable empirical predictions. The paper under review is clearly stronger on both axes.
- *vt5mnLVIVo* ("Grokking as lazy-to-rich transition," accepted poster, avg ~6): Theory paper bridging kernel/feature-learning regimes qualitatively. Comparable scope and validation depth.

**Assessment**: This is a well-executed theory paper with a genuine, provable central result (Theorem 4.2), a clean quantitative prediction (Proposition 5.1), and meaningful qualitative validation across modern architectures. Its weaknesses — the absence of quantitative deep-network comparison and the empirical gap on non-additive tasks — are real but do not undermine the theoretical contributions. The paper sits comfortably above the *Hxm0hOxph2* rejection baseline and is comparable to the accepted poster papers in this area. The missing quantitative bridge and the untested transitive-equivalence prediction keep it below spotlight territory.

**Originality:** Good — conjunction-wise additivity as a framework is novel and the salience parameterization is a new and useful lens.  
**Importance of research question:** High — compositional generalization is a central open problem in ML and cognitive science.  
**Claim support:** Moderate-to-strong — theoretical claims are well-supported; empirical claims are qualitatively supported but quantitatively unverified.  
**Soundness of experiments:** Moderate — qualitatively confirmatory across three architectures, but two tasks only, and no quantitative fit.  
**Clarity of writing:** Good — the paper is clearly structured and the kernel-to-deep-network connection is transparently bounded.  
**Value to community:** Good — named failure modes, falsifiable predictions, and an explanation for inconsistent disentanglement results are all directly useful.

**Score: 6.5 → Accept (poster)**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>