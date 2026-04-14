## Summary

TTVD proposes a geometric framework for test-time adaptation (TTA) by revealing that neighbor-based TTA methods have an underlying Voronoi Diagram (VD) structure. Building on this insight, the paper introduces two extensions: (I) Cluster-induced Voronoi Diagram (CIVD), which replaces single-point sites with self-supervised rotation-augmented clusters to create a multi-influence partitioning, and (II) Power Diagram/CIPD, which uses weighted sites to produce flexible boundaries for filtering noisy samples. Experiments on CIFAR-10/100-C, ImageNet-C, and ImageNet-R under the standardized TTAB testbed show consistent improvements in both classification error and expected calibration error over strong baselines.

---

## Strengths

- **Principled geometric reframing with cross-domain connections.** The paper makes a concrete and non-trivial connection between neighbor-based TTA methods and Voronoi geometry, grounded in prior formal results (Ma et al., 2022; 2023; Chen et al., 2013; 2017) that link logistic regression to power diagrams and prototype methods to VD. This unification is distinct from the dominant entropy-minimization or self-supervision paradigms and gives the paper a clear conceptual identity.

- **Substantially improved calibration on large-scale datasets.** The ECE improvements — 4.1% on ImageNet-C and 4.3% on ImageNet-R — are large in absolute terms and exceed what competing methods achieve. Given that calibration directly affects practical trustworthiness, this is a specific, concrete advantage over the baseline field, not merely a marginal improvement.

- **Robustness to prototype quality.** Table 4 demonstrates that TTVD performs nearly identically when Voronoi sites are computed from 10%, 5%, or even 1% of ImageNet training data (59.8/59.8/59.9). This is a meaningful practical finding, as it relaxes the data availability assumption significantly and adds confidence that the method is not brittle to prototype estimation error.

- **Systematic ablation with full per-corruption breakdown.** Table 2 shows the monotonic improvement VD → CIVD → CIPD across all 15 corruption types on CIFAR-10-C, which cleanly supports the paper's structural progression. Reporting all individual corruption types rather than only averages is a notable positive for reproducibility.

---

## Weaknesses

### Fatal
None identified.

### Major

- **The VD-based loss (Eq. 3) reduces to entropy minimization on distance-based logits — novelty requires sharper justification.** The VD loss is entropy of a softmax over negative distances to class means. This is operationally equivalent to prototype entropy minimization with distance-based logits, a well-explored formulation. The paper frames this as "applying the geometric framework," but if the VD loss is not algorithmically new, the novelty of the first contribution rests entirely on the conceptual reframing, which is valuable but should be stated honestly rather than implied as a mechanistic advance.

- **The foundational claim — that neighbor-based methods "align with the Voronoi Diagram" — is asserted without formal mapping.** The paper states that "the underlying geometric structure of neighbor-based methods is Voronoi Diagram" but does not formally show how T3A, TAST, or AdaNPC correspond to specific VD constructions. A theorem or even an informal proposition demonstrating equivalence or approximation for at least one method is necessary to validate the paper's starting premise.

- **The claim that CIVD "avoids negative transfer since the objective is now unified" is unsubstantiated.** Section 3.2 asserts that expanding each class site into rotation-augmented clusters "integrates self-supervision and entropy minimization" and avoids conflicting gradients. No gradient analysis, ablation comparing CIVD against naive multi-task combination, or formal argument is provided. Since avoiding negative transfer is one of the two core motivations in the introduction, the absence of supporting evidence for this specific claim is a significant gap.

- **The core adaptation algorithm (Algorithm 3) is deferred to Appendix H.** Section 3.3 states "we infer and adapt the model accordingly by CIPD (Algorithm 3 in Appendix H)." The full operational procedure of the proposed method — including how samples are filtered, what the exact loss function is, and what parameters are updated — must appear in the main paper to meet basic reproducibility standards. Critical algorithmic details should not be pushed to an appendix.

### Minor

- **Ablation study (Table 2) is limited to CIFAR-10-C.** Since ImageNet-C and ImageNet-R are larger-scale and arguably more meaningful benchmarks, and since the paper claims broad generality, at least a partial ablation on ImageNet would be needed to confirm that CIVD and CIPD contribute similarly on harder distributions. It is unclear whether the same +5.7% and +2.2% improvements hold at scale.

- **The exponent 7 in Eq. 4 and γ in Eq. 6 are unexplained and unablated.** Eq. 4 fixes the exponent as 7 (from prior geometry literature), while Eq. 6 treats γ as a tunable exponent for CIPD. Neither the rationale for 7 as the fixed choice nor the sensitivity of the method to γ in Eq. 6 is analyzed. For a parameter that governs the entire influence function and boundary shaping, this is a notable omission.

- **The numbers in parentheses in Table 2 (e.g., 22.7_(1.57), 20.5_(1.23)) are never explained.** Whether these are standard deviations, confidence intervals, or something else is not stated in the caption or main text. This must be clarified.

- **Source statistics assumption is not prominently discussed.** TTVD requires precomputed class means from training data, which goes beyond the strictest "source-free" TTA setting (where only the pretrained model is available). While this is shared with other neighbor-based methods like T3A, and the paper is transparent in Section 4.1, the introduction's framing around "source-free" adaptation should be explicit about this assumption.

- **Rotation augmentation for cluster construction may not generalize uniformly across datasets.** Using Rot_α ∈ {0, 90, 180, 270} works well for rotation-invariant data, but on ImageNet-R (non-photorealistic renditions) or fine-grained datasets, rotation semantics may be less consistent across classes. No discussion or sensitivity analysis for this design choice is provided.

### Tiny

- **Figure 4 adaptation curves show only four noise types.** The claim of robustness to overfitting over a long test stream would be more convincing if shown across more corruption types, including weather and digital distortions.

- **The contradiction in Section 4.2 on Tent/SAR dynamics.** The text first says "Tent and SAR do not show signs of overfitting" and then says their early stagnation "may indicate potential overfitting." These statements are inconsistent and should be reconciled; stagnation is not the same as overfitting.

---

## Nice-to-Haves

- **Benchmark on ViT architectures.** The geometric Voronoi structure depends on the feature space geometry of the backbone. Since ViTs have different inductive biases from ResNets, verifying that TTVD transfers to ViT-B/16 on ImageNet-C/R would substantially broaden the paper's impact.

- **Test across multiple severity levels.** Results are reported only at corruption severity level 5 (the highest). Reporting across all five severity levels, or at least levels 1 and 3, would clarify whether TTVD's gains hold under milder distribution shifts, which are arguably more common in deployment.

- **Compute overhead comparison.** A wall-clock or FLOP comparison between TTVD and lightweight baselines like TENT (which updates only BN parameters) would help practitioners assess the tradeoff. The offline precomputation time is given, but per-batch online overhead is not.

- **Ablation isolating PD filtering from CIVD structure.** A condition of CIVD + entropy-based filtering (without PD boundary shifting) would directly isolate the contribution of the Power Diagram, one of the two main technical contributions. This is currently confounded in the VD→CIVD→CIPD progression.

- **Visualization of filtered samples on high-dimensional benchmark data.** Figure 2 illustrates the filtering concept on 2D MNIST. A t-SNE/UMAP projection of ImageNet features showing which samples are filtered by CIPD (and whether they are truly noisy relative to ground truth) would make the filtering mechanism more convincing at scale.

---

## Removed Points

*These points were flagged for removal — treat with caution.*

- **"Moderate gains call into question whether the framework is justified" (Harsh Critic):** The harsh reviewer argued that modest classification error improvements (~0.7–1.6 points) are insufficient given the conceptual complexity. This framing is too absolute — ECE gains of 4+ points are large, and classification gains are consistent across all datasets. The improvement level is not a fatal weakness, and the reviewer's implicit standard (large classification gains required) is not universal.

- **No variance/statistical significance on large-scale benchmarks (Harsh Critic):** The reviewer demanded repeated runs and uncertainty estimates. Single-run evaluation is the standard for large-scale TTA benchmarks (TTAB, ImageNet-scale). This is not a real weakness given community norms.

- **"The abstract slightly overstates what is established" (Harsh Critic on "remarkable improvements"):** Minor rhetorical criticism with no factual basis. Not substantive.

- **Requests for confidence intervals on Table 4 (single proportions):** With only three data points (10%/5%/1%), asking for mean±std over random subsets is reasonable in principle but standard practice in this setting does not require it given the stability already demonstrated.

- **Missing related works (Harsh Critic, Spark Finder):** Per review instructions, missing related works are not included, as there are no external sources to confirm their existence.

- **Criticism that TTVD comparisons to AdaNPC (Table 3) are unfair due to cross-paper numbers:** The harsh reviewer notes this as a fairness concern. However, the asymmetry (not rerunning AdaNPC under TTAB) is unfair to AdaNPC relative to TTVD — i.e., the asymmetry benefits the baseline, not TTVD. Per the review instructions, such comparisons are intentionally conservative toward the author's method and should not be penalized.

- **"Insufficient engagement with classifier-geometry literature" (Harsh Critic):** Lemma 3.1 explicitly cites Ma et al. (2022; 2023) and the related work covers DeepVoro and iVoro. The concern that the paper underplays prior connections is partially addressed and overstated by the reviewer.

- **Comparison to EATA, CoTTA, and other recent TTA methods (Spark Finder):** The paper explicitly uses TTAB as its evaluation framework for fairness. If these methods are not in TTAB, the omission is a framework constraint, not a paper flaw. This criticism is weakened; TTAB does include a reasonable range of methods spanning different paradigms.

---

## Novel Insights

The most genuinely novel observation synthesized across reviews is the following: the paper's geometric framing is not merely metaphorical — the existing formal correspondence between logistic regression and Power Diagrams (Lemma 3.1, from Ma et al. 2022/2023) means that if TTVD's class means are thought of as classifier site estimates, then the transition from VD to CIVD is structurally equivalent to moving from a linear to a multi-prototype non-linear boundary. This implies that CIVD's gain over VD may be interpretable as implicit kernel approximation in feature space, not just a richer sampling of prototypes. This connection is underdeveloped in the paper and, if formalized, could provide a much stronger theoretical anchor for why CIVD specifically reduces calibration error rather than just classification error — since power-diagram-based partitions are known to produce well-calibrated posteriors when cluster covariances are accounted for.

---

## Suggestions

1. **Move Algorithm 3 to the main paper.** The complete CIPD adaptation procedure including the filtering rule, loss function, and update target must appear in the main text, not in an appendix. This is a prerequisite for the paper to meet reproducibility standards.

2. **Formally establish or bound the correspondence between at least one existing TTA method and VD.** Even a Proposition showing that T3A's prototype updates approximate VD-boundary descent would substantially validate the paper's foundational claim.

3. **Add an ablation on CIVD vs. CIVD + entropy filtering (without PD)** on ImageNet-C to isolate the contribution of Power Diagram boundary shifting from the multi-site CIVD structure alone.

4. **Clarify the parenthetical values in Table 2** (e.g., 22.7_(1.57)) — state explicitly in the caption what they represent.

5. **Add a paragraph in the methodology explicitly acknowledging the source statistics assumption**, comparing it to other neighbor-based methods that also use stored prototypes (e.g., T3A), and distinguishing it from strictly model-only TTA.

6. **Provide an ablation on γ in Eq. 6.** Since γ controls the boundary geometry of CIPD, its sensitivity should be characterized — even a three-point ablation (γ = 1, 2, 4) would add credibility.

---

## Paper Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | Moderately high. Applying CIVD and PD to TTA is a fresh angle grounded in real computational geometry results. The VD loss formulation overlaps with existing prototype entropy minimization, which the paper does not sufficiently acknowledge. |
| **Importance of research question** | High. TTA under distribution shift is a practically significant and active research area; improving both accuracy and calibration simultaneously is a meaningful target. |
| **Claims well supported** | Partially. Empirical results are consistent and use a standardized testbed. However, the "negative transfer avoidance" claim and the "underlying structure of neighbor-based methods is VD" claim lack formal support. |
| **Soundness of experiments** | Good for what is shown. TTAB provides fairness. The ablation is limited to CIFAR-10-C, which weakens generalization claims. Single severity level is a gap. |
| **Clarity of writing** | Adequate at the high level; the geometric progression VD→CIVD→CIPD is clear. However, the actual adaptation procedure is deferred to an appendix, and several equations have unexplained components (exponent 7, parenthetical values in Table 2). |
| **Value to research community** | Moderate-to-high. The geometric perspective offers a reusable conceptual lens. The strong ECE improvements are practically valuable. The incomplete formal justifications limit immediate uptake. |
| **Contextualization relative to prior work** | Adequate. TTAB baseline coverage is reasonable. The connection to prior geometric ML work (Ma et al., DeepVoro, iVoro) is acknowledged. The overlap with prototype entropy minimization should be discussed more directly. |