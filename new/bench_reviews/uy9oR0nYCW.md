## Summary

This paper adapts two explainability methods—occlusion (from vision) and attention rollout (from NLP)—to transformer-based audio deepfake detectors (AST, Wav2Vec) and evaluates their cross-dataset generalizability by training on ASVspoof 5 and testing on FakeAVCeleb. The key findings are (1) occlusion highlights padded regions rather than content, suggesting the method is unreliable for this setting, (2) attention rollout produces time-localized, sample-specific explanations, though without faithfulness validation, and (3) a stark generalizability gap exists, with GBDT collapsing to ~51% accuracy cross-dataset while AST maintains 85% and Wav2Vec 81%. The paper positions itself as preliminary, noting that the explainability methods are "still in their infancy," but the claims in the abstract about "closing the explainability gap" overstate what is actually an exploratory adaptation study with important unresolved methodological issues.

## Strengths

- **The cross-dataset benchmark reveals a dramatic generalizability gap.** Table 1 shows the GBDT drops from strong in-distribution performance to near-random 51% accuracy when trained on ASVspoof 5 and tested on FakeAVCeleb, while AST maintains 85% and Wav2Vec 81%. This provides compelling evidence that traditional feature-based methods lack real-world robustness—a concrete motivation for why transformer explainability matters beyond pure methodology.

- **The occlusion failure is a transparent and informative negative result.** Figure 4 shows importance concentrated in zero-padded spectrogram regions rather than meaningful audio content. The paper correctly connects this to the phenomenon described by Wu et al. (2021) about transformers storing global information at consistent locations. This negative finding usefully rules out naive occlusion as a reliable explainability approach for audio spectrogram transformers.

- **Attention rollout adaptation to audio tokens produces time-localized, sample-specific explanations.** Figure 5 demonstrates that the rollout mechanism can identify specific ~20ms Wav2Vec tokens instrumental in classification, with influential tokens appearing in groups. This satisfies the paper's own time-specific criterion from Section 3.3.

- **The GBDT feature importance analysis reveals a spurious correlation: RMS (loudness) ranks among the top features** (Figure 2), which the authors correctly flag as troubling since loudness should not inherently distinguish deepfakes. This demonstrates how interpretability methods can expose problematic model behaviors.

## Weaknesses

### Fatal
None.

### Major

- **Attention rollout is presented as an explanation method without any validation of faithfulness.** The paper applies attention rollout to audio transformers (Section 4.2) and presents the resulting weight distributions (Figure 5) as explanations, but never validates whether high-attention tokens actually correspond to deepfake-specific artifacts or causally drive the classification decision. Section 3.3 defines explainability criteria (sample-specific, time-specific, feature-specific), but provides no quantitative evaluation metric for whether attention rollout actually "explains" model behavior. Without faithfulness validation—such as perturbation/deletion tests, gradient-based attribution comparison, or correlation with known forensic artifacts—the attention visualizations remain descriptive of information routing rather than verified explanations. This is a fundamental gap for a paper claiming to "close the explainability gap." (Sections 4.2, 5.2)

- **The occlusion experiment is under-diagnosed and the failure mode is not properly analyzed.** While the paper correctly identifies that occlusion highlights padded regions (Figure 4, Section 5.2), it does not investigate why: whether this is due to distributional shift from zero-masking, architectural properties of AST, or the specific patching scheme. The paper reports the negative result but does not diagnose the root cause, and the conclusion that occlusion "is suggestive of a phenomenon posited by Wu et al." is speculative without controlled ablation (e.g., comparing zero-masking vs. noise-masking vs. mean-replacement). This weakens the scientific value of the negative finding, as there is no clear guidance to the community on what *should* be done differently for audio occlusion. (Section 5.2)

- **The cross-dataset generalizability benchmark lacks statistical rigor and potential pipeline alignment controls.** Table 1 reports single-point precision/recall/F1 metrics without variance across seeds, confidence intervals, or multiple runs, making it impossible to assess the reliability of the reported 85% (AST) vs. 81% (Wav2Vec) split. The GBDT's collapse to 51% accuracy is dramatic enough to suggest potential preprocessing or feature-extraction mismatches between ASVspoof 5 and FakeAVCeleb (e.g., different sampling rates, frame lengths, or normalization schemes) rather than genuine algorithmic failure. Without reporting matched preprocessing details or variance, the benchmark results are difficult to reproduce or trust. (Section 6, Table 1)

### Minor

- **The paper frames itself as a "conceptual framework" and claims to "close the explainability gap" in the abstract, but delivers an exploratory evaluation of two adapted off-the-shelf XAI techniques rather than a novel framework.** The body presents attention rollout and occlusion as applications of existing methods to a new domain, with the occlusion result being a failure and the attention result being unvalidated. This gap between framing (constructive solution) and execution (preliminary exploration) overstates the contribution. The authors' own conclusion that methods are "in their infancy" partially acknowledges this but does not fully resolve the framing mismatch. (Sections 1, 7)

- **The paper does not report the in-distribution performance on the same FakeAVCeleb split used for evaluation.** While Appendix D reportedly contains in-distribution results on both datasets separately, Section 6 would benefit from reporting in-distribution FakeAVCeleb performance as a reference point, to distinguish between "cross-dataset generalization degradation" and "absolute performance on a new dataset." (Section 6)

### Trivial

- **The distinction between interpreting attention visualization and providing "explanations" is not clearly operationalized.** The paper draws a careful distinction between interpretability and explainability in Section 3.3 but applies attention visualization as an explainability method without defining how it meets the stated criteria beyond qualitative observation.

## Nice-to-Haves

- **Faithfulness/perturbation validation for attention rollout** (e.g., insertion-deletion metrics, AUC on feature removal) would strengthen the central claim that attention weights serve as meaningful explanations.

- **Controlled masking ablations for occlusion** (zero vs. noise vs. mean spectrogram masking) would help diagnose whether the failure is architectural or implementation-specific, providing actionable guidance for the community.

- **Explicit reporting of preprocessing pipeline details** (sampling rates, frame lengths, normalization) across datasets would rule out configuration mismatch as a confound in the cross-dataset benchmark.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Structural: Attention weights are treated as faithful explanations without validation, invalidating the core methodological contribution."** This criticism has **real merit** but was **not invalidated entirely**—it was downgraded from "fatal/invalidating" to a **Major weakness**. The explanation literature (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019) does indeed document that attention weights ≠ explanations. However, the paper does not claim to have solved faithfulness; it adapts the method and shows what it produces. This is a gap, not a complete invalidation.

- **"Structural: Occlusion implementation introduces severe distribution-shift artifacts, rendering the experiment uninformative."** **Partially incorrect.** The occlusion failure *is* informative—it reveals that naive occlusion does not work for audio spectrogram transformers, which is valuable knowledge. The paper actually reports this and connects it to literature on transformers storing information at consistent locations. The harsh critic overstates the severity; the experiment is not "uninformative," it is "under-diagnosed." This was moved to a Major weakness with the diagnosis caveat.

- **Missing statistical reporting (variance, CIs)** was **kept** but classified as **Major** since it affects the reliability of the benchmark's central claim.

- **"The abstract contribution overpromises relative to technical execution."** **Kept but classified as Minor.** The paper's own conclusion acknowledges methods are "in their infancy," which partially resolves the tension.

- **Strength: "GBDT feature importance analysis reveals spurious correlation (RMS)"** — **Kept.** This is a genuine finding that demonstrates the value of interpretability analysis.

- **Strength: "Hierarchical clustering approach to address multicollinearity"** — **Kept.** This is a practical and valid methodological contribution, though limited in scope.

## Novel Insights

The paper's most genuinely novel contribution is **empirical evidence of the generalizability gap between traditional feature-based and transformer-based audio deepfake detectors in a cross-domain setting**. The finding that GBDT drops to near-random performance while AST maintains 85% accuracy is striking and practically important—it demonstrates that the community's reliance on traditional methods for explainable detection comes at a severe cost in real-world robustness. This finding has actionable implications: it suggests that the focus should shift from "keeping methods interpretable" to "making generalizable methods explainable." The attention rollout adaptation to audio tokens is a straightforward translation from NLP, but the documentation that influential tokens cluster in groups of ~20ms segments is a useful observation for forensic analysis workflows. However, neither finding represents methodological innovation; both represent careful empirical reporting within a well-framed problem space.

## Suggestions

1. **Add faithfulness validation for attention rollout** using at least one standard metric (insertion-deletion, correlation with gradient-based attribution, or perturbation sensitivity) to verify that high-attention tokens actually drive model predictions. This would elevate the paper from exploratory adaptation to substantive contribution.

2. **Conduct controlled occlusion ablations** to diagnose why importance concentrates on padding: compare zero-masking vs. Gaussian noise masking vs. mean-spectrogram replacement. This would clarify whether occlusion is fundamentally unreliable for audio transformers or whether the implementation can be improved.

3. **Report matched preprocessing details and statistical variance** for the cross-dataset benchmark. Include standard deviations across multiple random seeds, and explicitly document sampling rates, frame lengths, and feature extraction parameters to rule out pipeline mismatches.

4. **Reframe the contribution claims** to accurately reflect the exploratory nature of the work. The abstract's claim of "closing the explainability gap" should be replaced with language about initiating a systematic evaluation of explainability methods for audio deepfake detection.

5. **Consider a hybrid presentation**: combine attention rollout heatmaps with audio spectrograms annotated with known forensic indicators (e.g., phase discontinuities, vocoder artifacts) to provide concrete evidence that attention highlights deepfake-relevant features.

## Score and Decision

I calibrated against several anchor papers:
- **EoTIlDT0Tr** (deepfake detection explainability with MLLMs, scores 3,8,6,5, rejected): similar thematic area but more methodology; this paper is less technically substantial.
- **rGGwXo0Fo0** (SONAR, audio deepfake benchmark, scores 3,6,5,3, withdrawn): similar domain and comparable weaknesses (limited novelty, methodological gaps); this paper is roughly comparable in quality but slightly less comprehensive.
- **50UzaXh0gC** (XAI faithfulness issues, scores 5,5,5,5, rejected): similar concern about unvalidated explainability claims; this paper shares that gap.
- **2GcR9bO620** (robust audio deepfake detection, scores 6,6,8,8, accepted): much more technically substantive with novel training method and extensive dataset; this paper is clearly weaker.

The paper has genuine value—especially the generalizability benchmark results and the transparent reporting of negative findings—but the unvalidated attention explanations and under-controlled benchmark are significant methodological weaknesses that the authors cannot fully address in a rebuttal (they require new experiments). The paper is an **exploratory adaptation study** rather than a complete contribution. Given the anchor range, I position this slightly below the borderline papers that received 5s and slightly better than the rejected benchmarks at 3-4, due to the importance and honesty of the negative findings.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>