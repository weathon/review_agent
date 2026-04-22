Now I have sufficient calibration context. Let me compile the final review.

## Summary

This paper identifies that the quality of post-hoc explanations for pre-trained models depends heavily on how the classification probe is trained, rather than on the pre-training paradigm itself. The key finding is that switching the probe loss from cross-entropy (CE) to binary cross-entropy (BCE) consistently improves explanation localization across 5 pre-training paradigms, multiple architectures, and 10+ attribution methods, supported by a theoretical analysis of softmax shift-invariance that makes CE probes unreliable for attribution.

## Strengths

- **Softmax shift-invariance insight (Section 3.2, Equations 2–3, Figure 4)** is a genuine and important contribution. The demonstration that two functionally equivalent CE probes can yield attributions with 65.7% vs. 11.9% GridPG scores is a powerful illustration of a fundamental reliability problem for post-hoc explanations applied to CE-trained models. This provides a clear mechanistic reason for BCE's superiority.

- **Impressive experimental breadth.** The paper evaluates across 5 pre-training paradigms (supervised, MoCov2, BYOL, DINO, CLIP), 10+ attribution methods, 2 architectures (ResNet-50, ViT-B/16), 3 datasets (ImageNet, VOC, COCO), and multiple metrics (GridPG, EPG, pixel deletion, compactness, complexity). The consistency of the BCE improvements across this entire matrix makes the main empirical observation very robust.

- **Practical significance and ease of adoption.** The proposed fix—switching from CE to BCE loss for probe training—is a minimal-implementation-change solution with no computational overhead, making it immediately actionable for practitioners. The finding that a probe-training choice matters more than the entire pre-training paradigm for explanation quality has clear implications for the XAI community.

- **Multiple evaluation metrics beyond localization.** The paper evaluates not just localization (GridPG, EPG) but also pixel deletion stability (Section 5.1, Figure 2b), compactness (Gini index), and complexity (entropy), showing improvements across these diverse metrics (Table E1).

## Weaknesses

### Fatal
None.

### Major

- **Equating "more localized" with "better explanations" without establishing faithfulness.** The paper frames its contribution as enhancing "the quality of model explanations" (abstract, conclusion), but all primary metrics (GridPG, EPG, pixel deletion) favor localized attributions. If a CE-trained model genuinely uses distributed features across the image, then scattered attributions could be more *faithful* (reflecting what the model actually computes) even though they score worse on localization. The pixel deletion metric partially addresses this (BCE probes show more stable predictions under pixel deletion), but pixel deletion also conflates localization with faithfulness—removing the highest-attributed pixels will disproportionately hurt models with concentrated true features. The paper never provides a faithfulness evaluation that can distinguish "the model uses localized features" from "the explanation method produces localized maps." This matters because the practical recommendation (use BCE probes) optimizes for localization rather than necessarily for correctness of the explanation. The claim in the abstract and conclusion should be qualified accordingly (e.g., "as measured by localization and compactness metrics").

- **The "more dependent on probe training than pre-training" claim lacks controlled statistical comparison.** The conclusion states that explanation quality "is more dependent on how the classification layer is trained...than on the choice of pre-training paradigm itself." While the empirical data qualitatively supports this (BCE vs CE produces large improvements across all pre-training paradigms, while pre-training paradigm variations produce smaller differences), the paper never quantifies both effects in a comparable framework. No variance estimates, confidence intervals, or statistical tests (e.g., two-factor ANOVA) are reported. The pre-training methods are tested at different accuracy levels with different off-the-shelf checkpoints, making the comparison apples-to-oranges. This claim should be moderated to acknowledge it is supported by qualitative observation rather than controlled statistical analysis.

### Minor

- **B-cos MLP recommendation conflates depth and architectural bias effects.** Section 5.2 recommends B-cos MLP probes for improved localization, and the paper transparently notes that conventional MLPs show a *decrease* in GridPG on ImageNet (line 305). However, this directly contradicts the broader claim that "more complex probes" improve explanations—the improvement is specifically tied to the B-cos architectural property of weight-input alignment (Equation 6), not to probe depth per se. The paper should more clearly separate these two factors (depth vs. architectural inductive bias) in its claims, noting that the B-cos recommendation is specifically about architectural alignment, not merely about adding layers.

- **Some accuracy drops from CE to BCE switching are not analyzed.** Table 1 shows accuracy drops for supervised ViT-B/16 (73.2→72.8) and MoCov3 ViT-B/16 (76.3→75.1) when switching to BCE. While these are small, the paper should note this tradeoff and discuss whether the localization gains come at any meaningful accuracy cost in practice.

### Trivial
None.

## Nice-to-Haves

- A random-seed variation analysis for CE probes would clarify whether the shift-invariance problem (Figure 4) manifests in standard training or only in contrived examples, strengthening the practical impact argument.
- Including a faithfulness metric that can disentangle "the model uses localized features" from "the explanation method produces localized maps" (e.g., on a synthetic dataset where ground-truth feature importance is known, or agreement between different explanation methods on the same probe) would substantially strengthen the paper's framing.
- Evaluating BCE probes on end-to-end trained models (rather than just linear/MLP probes on frozen backbones) would broaden the practical applicability of the findings.

## Removed Points

- **No error bars / confidence intervals**: The harsh critic flags the lack of statistical significance tests. While some improvements are modest (Table 1: +0.4 Rollout for supervised ViT-B/16), most improvements are very large (30-40 p.p. for DINO and MoCov2). For single-run evaluations standard in this field and given the scale of the improvements reported, this is a nice-to-have rather than a substantive weakness. Removed as minor nitpick per rules on reproducibility concerns.

- **Random seed variation analysis for CE probes**: The harsh critic requests this to assess practical impact of shift-invariance. This is a reasonable future experiment but goes beyond what's needed for the paper's claims—the theoretical argument (Equations 2-3) establishes the invariance exists, and the empirical results show BCE consistently improves localization regardless. Removed as beyond scope.

- **B-cos MLP results for conventional MLPs relegated to appendix**: The paper does discuss conventional MLP results in the main text (line 305 explicitly states their GridPG *decrease* on ImageNet), and references Appendix E.2. This is not an omission; it's a transparent acknowledgment. Removed as factually incorrect criticism.

- **Missing related work**: Per rules, I cannot confirm the existence of unspecified references. Removed.

- **Formatting and typos**: Removed per rules about parser artifacts.

## Novel Insights

The softmax shift-invariance argument (Equations 2-3) provides a previously underappreciated mechanistic explanation for why CE-trained probes produce unreliable attributions—a problem that is inherent to the softmax normalization rather than being an artifact of any particular explanation method. This insight generalizes beyond any single attribution method, which is why the BCE improvement is observed consistently across all 10+ methods tested. The practical implication is important: practitioners using post-hoc explanations should be aware that how they train their probe matters as much as or more than which explanation method they choose, a finding that challenges the field's implicit assumption of method-independence.

## Suggestions

- Qualify the "better explanations" claims (abstract, introduction, conclusion) to specify "as measured by localization, compactness, and deletion stability metrics," and add a brief discussion of the localization-vs-faithfulness distinction.
- Moderate the "more dependent on probe training than pre-training" claim to acknowledge it is supported by qualitative and quantitative comparison but without formal statistical effect-size analysis.
- Add a sentence or two analyzing the accuracy-localization tradeoff observed in Table 1 for ViT models.

## Score and Decision

**Calibration anchors:**

- **High-scoring papers (>7):** UNI (avg 8.0, oral) — strong mechanistic insight + empirical validation for attribution improvements; CLIP text decomposition (avg 8.0, oral) — novel decomposition method with multiple evaluation metrics; Inherent Interpretability Score (avg 7.33, spotlight) — discovers positive correlation between classifiability and interpretability with a new metric; Cross-Entropy Is All You Need (avg 8.0, oral) — theoretical identifiability results for CE training with empirical validation.
- **Medium-scoring papers (4-6):** Sum-of-Parts (avg 4.8, reject) — faithful-by-construction attributions but overclaimed scope and methodology concerns; various localization/faithfulness conflation papers (avg 4.7-5.75).
- **Low-scoring papers (<3):** Support Spectrums (avg 3.0, reject) — empirically extremely weak with anecdotal evaluation and no baselines.

**Comparison:** This paper identifies a genuinely important and overlooked problem (probe training affects explanation quality), provides a clear mechanistic explanation (softmax shift-invariance), and demonstrates it with exceptional empirical breadth. It is stronger than Sum-of-Parts and other medium-scoring papers because it has a solid theoretical mechanism and large-scale, consistent empirical evidence across many settings. It is somewhat weaker than the high-scoring papers because: (1) the faithfulness-vs-localization issue is a meaningful gap in the claims, (2) the "more dependent on probe than pre-training" claim overreaches the evidence, and (3) the B-cos recommendation conflates two factors. The core insight (softmax shift-invariance → unreliable explanations) is novel and valuable, similar in spirit to the UNI paper's insight about baseline choices. But the overclaiming on faithfulness and the comparative importance claim keep it below the top tier.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>