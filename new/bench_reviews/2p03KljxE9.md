Now I have thoroughly verified all claims. Let me compose the final review.

---

## Summary

LAFT (Language-Assisted Feature Transformation) is a training-free method that leverages CLIP's shared embedding space to transform visual features according to user-specified language prompts. It constructs "concept axes" by computing PCA on pairwise differences of CLIP text embeddings, then projects image features onto (guide mode) or orthogonal to (ignore mode) these axes, enabling users to control which attributes define the normality boundary in anomaly detection. The method is evaluated on semantic AD benchmarks (Colored MNIST, Waterbirds, CelebA) with a kNN classifier (LAFT AD) and on industrial AD benchmarks (MVTec AD, VisA) integrated with WinCLIP (WinCLIP+LAFT).

## Strengths

- **Clean and well-motivated problem formulation.** The distinction between "guide" and "ignore" modes for controlling normality boundaries is a useful conceptual contribution (Sections 3, 4.3), formalized via information-theoretic desiderata (Eqs. 1–2). Prior work has not clearly formalized this user-controllable framing of AD.

- **Genuinely training-free and easy to apply.** Unlike InCTRL or APRIL-GAN which require adapter pre-training, LAFT operates purely through geometric projection in CLIP space. This makes it agnostic to downstream AD methods, as demonstrated by successful integration with both kNN (LAFT AD) and WinCLIP (WinCLIP+LAFT) (Section 4, Table 3).

- **Strong guide-mode results on semantic AD.** LAFT AD achieves the best results across all semantic AD benchmarks in the guide setting — 98.5 AUROC on Colored MNIST Number, 95.6 on Waterbirds Bird, 98.1 on CelebA Eyeglasses (Tables 1–2) — substantially outperforming prior language-based methods like ZOE, CLIPN, and WinCLIP.

- **Robustness to prompt quality (Table 4).** The ablation demonstrates that LAFT performs well even with incomplete concept value specification. On Colored MNIST, using only "seen" normal values for guiding still achieves 96.2 AUROC (vs. 98.5 with all candidates), and partial unseen values jump to 98.4. This is practically important for deployment.

- **Effective t-SNE visualization of concept subspace (Figure 3).** The Colored MNIST visualization clearly shows how LAFT aligns CLIP features with intended attribute axes, making the mechanism intuitively understandable.

- **Meaningful zero-shot improvements on IAD.** On MVTec AD K=0, LAFT-G improves WinCLIP from 90.4 to 92.6 (+2.2); on VisA K=0, from 75.5 to 80.0/80.6 (+4.5–5.1). These are non-trivial gains without any reference images (Table 3).

## Weaknesses

### Fatal
None.

### Major

- **The ignore mode — presented as a core dual-mode contribution — fails on the harder, more realistic benchmark.** On Waterbirds, the canonical test for nuisance attribute removal, LAFT AD (ignore) achieves only 84.8 AUROC, barely above the no-guidance kNN baseline of 82.3 and far below the guide mode's 95.6 (Table 1). The paper acknowledges this in one sentence: "ignoring one attribute (background) does not directly improve performance on the other attribute (bird)" (Section 5.1, p.270), but does not explain why orthogonal projection onto text-derived concept axes cannot suppress the entangled background attribute. The Colored MNIST ignore result (97.4) is uninformative because color is a simple, decorrelated attribute that CLIP trivially separates. Since the paper's framing prominently features dual-mode control (guide *and* ignore), and the ignore mode has not been demonstrated on any scenario where attribute entanglement is non-trivial, this claimed contribution is unsupported by the evidence.

- **The guide mode's strong semantic AD results are less impressive upon closer inspection.** When the user guides on the exact attribute defining anomalies (e.g., "number" for Colored MNIST Number) and the method projects onto that attribute's subspace, the subsequent kNN is essentially performing attribute classification in a projected space. The kNN baseline with all normal images (i.e., simple data augmentation covering both colors) already achieves 98.0 on Colored MNIST Number, compared to LAFT's 98.5 — a gap of only 0.5 AUROC (Table 1). LAFT's genuine advantage is that it achieves this *without* needing augmented data, replacing image augmentation with language. This is a real but narrower contribution than the headline numbers suggest. The paper does not discuss this proximity to the augmented-data baseline or contextualize the practical significance of the improvement.

### Minor

- **Industrial AD improvements diminish rapidly with more reference images and are within noise at higher K values.** On VisA K=8, WinCLIP+LAFT-C improves by only 0.1% (88.2→88.3) over WinCLIP+ (Table 3). On MVTec K=4, the improvement is 0.7% (95.6→96.3) with overlapping standard deviations (±0.5 vs ±0.4). The claim that WinCLIP+LAFT "consistently outperforms WinCLIP" (Section 5.2, p.302) is technically true but overstates the practical significance for few-shot settings where improvements are marginal. The zero-shot improvements are more meaningful (+2.2% on MVTec, +5.1% on VisA).

- **LAFT-C uses category-specific anomaly knowledge that creates an information asymmetry with WinCLIP+.** LAFT-C employs prompts like "bottle with large breakage" (Section 5.2, p.282), which encode knowledge about specific anomaly types. While WinCLIP also uses state-word prompts, LAFT-C's prompts are more specific. LAFT-G, which uses only general state words comparable to WinCLIP, shows smaller improvements (e.g., MVTec K=4: 96.2 vs 96.3 for LAFT-C, Table 3). The paper does not clearly separate these when claiming improvement over WinCLIP+.

- **The orthogonality assumption underlying the ignore mode is asserted without verification.** Section 4.3 (p.172) states that "irrelevant attributes are nearly orthogonal to the concept axes," which is the key assumption enabling the ignore projection (Eq. 6). This claim receives neither theoretical justification nor empirical measurement. In a 512-dimensional CLIP space with 8–384 concept axes, orthogonality to all irrelevant attributes is a strong assumption. The Waterbirds ignore failure (see Major weakness above) suggests this assumption does not hold when attributes are entangled.

### Trivial
None.

## Nice-to-Haves

- **Comparison with a simpler projection baseline** (e.g., project onto a single text-derived direction $E_\text{text}(\text{normal}) - E_\text{text}(\text{abnormal})$ and apply kNN) would clarify whether the PCA-on-pairwise-differences step provides benefit over simpler alternatives.

- **Analysis of what the concept subspace captures** — e.g., after projection, train linear probes to predict each attribute. If ignore mode works, the irrelevant attribute should be unpredictable from projected features. This would also help explain the Waterbirds failure.

- **Per-category breakdown for MVTec AD and VisA** to reveal whether LAFT helps consistently or is driven by a few categories.

- **Statistical significance testing** (e.g., paired bootstrap) for the few-shot IAD comparisons, where improvements are within standard deviations.

- **Demonstration of ignore mode on a scenario where it plausibly helps**, or an honest scaling-back of the ignore claim with analysis of when/why it fails.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Unfair comparison with Language-only methods that don't use training images"** (Harsh Critic): The paper clearly separates methods into sub-categories — "Language" (MCM, ZOE, CLIPN, WinCLIP zero-shot) vs "Image + Language" (WinCLIP+, InCTRL, LAFT AD) — and the tables visually distinguish these groups. The comparison structure is transparent; readers can see which methods use images. This is not an unfair comparison.

- **"k=30 not optimized, potentially biased comparison"** (Harsh Critic): The same k=30 is used for both the kNN baseline and LAFT AD, so the comparison is fair. Using an unoptimized k for both methods is actually more conservative.

- **"No comparison with PCA on raw text embeddings vs. PCA on differences"** (Harsh Critic): This is a reasonable methodological question but is more of a nice-to-have ablation than a weakness. The word2vec analogy provides motivation, and the empirical results validate the choice. Moved to Nice-to-Haves.

- **"Dual-mode operation provides complementary control"** (Strength Finder): This strength conflicts with the verified Major weakness that the ignore mode fails on the harder benchmark. Moved here because the "complementary" claim is not well-supported.

- **"Effective extension to industrial AD without additional training"** as a standalone top-tier strength (Strength Finder): While true that LAFT is training-free, the IAD improvements are marginal for K≥4. The zero-shot improvements are meaningful but the few-shot story is weak. Downgraded to a supporting detail rather than a core strength.

- **Formatting and presentation nitpicks** from the Harsh Critic: Removed per rules.

## Novel Insights

The paper reveals an interesting asymmetry in the difficulty of attribute manipulation via CLIP subspaces: projecting *onto* a known attribute axis (guide mode) is relatively easy because CLIP already encodes attribute differences along separable directions; projecting *away from* an entangled attribute (ignore mode) is much harder because orthogonality in the CLIP embedding space does not hold for correlated attributes. This asymmetry — easily demonstrable on synthetic data (Colored MNIST) but breaking down on realistic entangled data (Waterbirds) — suggests that the practical utility of text-guided feature transformation is currently limited to the "amplify known signal" regime rather than the "suppress nuisance signal" regime. This is an important finding for the community, though the paper undersells it by not analyzing the failure mode.

## Suggestions

- **Honest characterization of ignore mode limitations**: Add a dedicated analysis section explaining why ignore mode fails on Waterbirds (likely because background and bird features are entangled in CLIP space, violating the orthogonality assumption of Eq. 6). This would turn a weakness into a meaningful contribution about the limits of text-guided projection.

- **Reframe the IAD contribution around zero-shot**: The clearest practical value of LAFT for industrial AD is in the zero-shot and K=1 settings. Consider presenting the story around this rather than claiming consistent improvements across all K values.

- **Add a direct comparison with kNN + augmented data**: Showing how LAFT's language-based guidance compares to simple data augmentation (e.g., Table 1's "kNN + all normal images" row) more prominently would clarify when language guidance is genuinely preferable to just collecting more diverse training data.

## Score and Decision

**Calibration anchors:**

- **High-scoring anchors (>7):** DOHC (orthogonal projection for AD, avg 8.0) — stronger theoretical grounding and comprehensive experiments across data types; NegLabel (VLM OOD detection, avg 7.5) — novel method with theoretical analysis and comprehensive experiments. LAFT is weaker than both: it lacks the theoretical depth of DOHC and the empirical breadth of NegLabel.

- **Medium-scoring anchors (4–6):** MuSc (zero-shot industrial AD, avg 5.2) — good results but fairness concerns; DOS (language-guided OOD, avg 5.0) — novelty concerns, inconsistent evaluation. LAFT is comparable to these: it has a clean idea with real but narrow empirical support, similar to how DOS uses LLM labels but with information leakage concerns.

- **Low-scoring anchors (<3):** CLAD (avg 2.2) — no ablation, overclaimed SOTA, weak baselines; Generalized AD (avg 2.5) — self-serving evaluation. LAFT is clearly above these: it has a sound method, real experiments, and honest (if incomplete) evaluation. It does not have fundamental methodological flaws.

LAFT sits in the 4.5–5.5 range. It is above the low-scoring papers because the method is sound and the guide mode genuinely works, but below the medium-good papers because the ignore mode (a key claimed contribution) is unsupported on hard benchmarks and the IAD improvements are marginal where they matter most. The contribution is real but thinner than presented.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>