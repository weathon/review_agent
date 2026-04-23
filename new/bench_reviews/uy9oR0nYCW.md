Now let me run calibration searches in parallel.Now I have enough to write the final review.

---

## Summary
The paper tackles explainability and real-world generalizability of audio deepfake detectors. It adapts three existing methods—permutation feature importance (with multicollinearity correction) for a GBDT baseline, occlusion, and attention rollout—from vision/NLP domains to audio transformer classifiers, and evaluates all three models cross-dataset (trained on ASVspoof 5, tested on FakeAVCeleb). The headline empirical finding is that AST generalizes substantially better than GBDT (85% vs. 51% accuracy) and offers more balanced per-class performance than Wav2Vec on out-of-distribution data.

---

## Strengths

- **Cross-dataset generalizability finding (Table 1):** Training on ASVspoof 5 and evaluating on FakeAVCeleb reveals a dramatic gap: GBDT collapses to ~51% (random chance), Wav2Vec reaches 81% overall but with severe class imbalance (F1=0.84 bonafide vs. 0.77 spoof), while AST achieves 85% with balanced F1 for both classes. The counter-intuitive result that AST—not the commonly preferred Wav2Vec—offers the best generalization is a practically significant finding for deployment decisions.

- **Multicollinearity-corrected feature importance analysis (Section 5.1, Figure 3):** The authors go beyond naive permutation importances by performing hierarchical clustering on Spearman rank-order correlations, preventing correlated features from masking each other's true importance. This yields the honest observation that MFCC2, MFCC4, and RMS dominate—and that the prominence of RMS (loudness) is suspicious as a genuine deepfake signature.

- **Intellectual honesty about failures:** The paper clearly documents where each method falls short—GBDT not providing sample-specific explanations, occlusion highlighting padded regions, attention visualization requiring rollout aggregation—which increases trust in reported findings and provides concrete future-work directions.

- **Conceptual explainability framework (Section 3.3):** The three-criterion framework (sample-specific, time-specific, feature-specific) provides an actionable rubric for evaluating explanation methods, and the paper does evaluate each proposed method against it, even when the verdict is negative.

---

## Weaknesses

### Fatal
None. The paper makes no claims that are simply false; the problems are primarily insufficient novelty and unevaluated claims.

### Major

- **Overclaimed novelty in the central framing.** The abstract claims "we introduce novel explainability methods," and the contribution list (Section 1) re-asserts this. However, Section 4 itself concedes: *"We appropriate methods for vision and natural language explainability and translate them to the audio domain."* Occlusion is a decades-old technique; attention rollout is directly attributed to Abnar & Zuidema (2020). The actual contribution is *domain adaptation* of known methods, not invention of new ones. For a top-tier venue, this mischaracterization is significant because the degree of methodological novelty is the primary claimed advance—and it is absent.

- **No faithfulness evaluation of attention rollout.** The central explainability result (Figure 5) is entirely qualitative: a line plot of token attention weights with the observation that "influential tokens typically appear in groups." There is no quantitative demonstration that high-attention tokens causally influence the model's prediction (e.g., via deletion/insertion experiments or AOPC scores). Without this, Figure 5 shows that attention rollout *generates* visualizations, not that those visualizations are *faithful* explanations. This is the core missing experiment.

- **No human evaluation to support "citizen intelligence" and "trust" claims.** The abstract states the results "build trust with human experts" and "pave the way for unlocking the potential of citizen intelligence." Section 3.3 frames the entire explainability desiderata around non-technical users. Yet there is no user study, no annotation task, no perceptual experiment, and no trust calibration measure anywhere in the paper. These are the paper's broadest claimed impacts, and they are entirely without evidence.

- **One of three explainability methods fails by the paper's own admission, without resolution.** Section 5.2 states directly: *"the importance is greatest for the padded regions—regions that theoretically contain no predictive information"* and that this is *"obviously unhelpful in explaining the model's decision-making."* The paper attributes this to a phenomenon from Wu et al. (2021) but does not investigate whether the cause is a model artifact (the model exploits padding position as a spurious cue), an occlusion artifact (masking already-zero regions confounds the baseline), or both. Including a failed method as a listed contribution without analysis of the failure weakens the paper's coherence.

### Minor

- **RMS importance finding (Section 5.1) is flagged but not investigated.** The authors correctly call it "troubling" that loudness is a top feature, which strongly implies the GBDT is exploiting a recording-condition artifact rather than a genuine deepfake signature. A simple RMS distribution comparison between ASVspoof 5 and FakeAVCeleb training splits would confirm or refute this hypothesis and would directly explain the GBDT's 51% cross-dataset performance. This conversion from suspicion to finding is within the paper's scope.

- **Comparative analysis of rollout distributions between bonafide and spoof is absent.** Figure 5 presents two rollout plots side by side but provides no statistical characterization of how they differ, or whether attention patterns differ systematically between correctly and incorrectly classified samples. The qualitative observation that "influential tokens appear in groups" is present in both plots and is indistinguishable by visual inspection.

- **The "novel benchmark" claim is overclaimed.** Cross-dataset evaluation (train on A, test on B) is standard practice in the anti-spoofing literature. The paper uses two existing published datasets, no new data collection, no new annotation protocol, and no new evaluation metric. The results are valuable but the framing as a "novel benchmark" inflates the contribution.

### Trivial

- **Sections 3.1–3.2 are extended tutorial content** (signal processing features, GBDT formulation, transformer equations) that occupies several pages better devoted to deeper analysis of the explainability results.

---

## Nice-to-Haves

- **Overlay attention rollout onto the spectrogram.** Figure 5 shows attention by token ID on a 1-D line plot. Mapping high-attention tokens back to time-frequency space on the Mel-spectrogram (as in ViT attention overlays) would directly test whether influential tokens correspond to perceptually meaningful acoustic events and would make the visualization actionable for a human reviewer.

- **Ablation of the occlusion baseline value.** Replacing occluded patches with zero is confounded by the zero-padding already present in the spectrogram. Testing occlusion with a mean-spectrogram baseline or a random noise baseline might reveal whether the padding-importance artifact is an artifact of the masking strategy rather than the model.

- **Comparison to gradient-based attribution (e.g., GradCAM on the spectrogram).** Given the paper's stated desideratum of feature-specific explanations, gradient saliency on the Mel-spectrogram is a natural and well-understood baseline for assessing whether attention rollout adds value beyond simpler methods.

- **Population-level attention analysis.** Beyond individual examples, a class-level aggregate of rollout distributions (bonafide vs. spoof) would support or refute the claim that attention rollout provides meaningful, class-discriminative explanations.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh critic's claim that the "novel benchmark" has no value**: The cross-dataset evaluation is standard but the results (AST > Wav2Vec > GBDT on OOD data) are genuinely informative. Kept as a minor overclaim rather than a fatal flaw.
- **Missing appendix/proofs/hyperparameter details**: Per review rules, the parser strips appendices. The paper explicitly notes hyperparameters are in Appendix C, and detailed per-dataset performance is in Appendix D. These are not missing from the paper.
- **The "state-of-the-art for FakeAVCeleb" claim in Appendix D**: Per rules, appendix-referenced content exists in the original submission; this is not a verifiable absence.
- **"Cannot be verified" regarding Wav2Vec, AST, or ASVspoof dataset availability**: All cited systems exist. No availability critique is appropriate.
- **Formatting/notation criticisms**: Treated as parser artifacts.

---

## Novel Insights

None beyond the paper's own contributions. The most genuinely useful insight the paper surfaces—that AST, despite being less popular than Wav2Vec in the deepfake detection literature, generalizes more robustly to out-of-distribution data (85% balanced F1 vs. Wav2Vec's 81% with class imbalance)—is the paper's own empirical finding. The observation that the occlusion importance artifact may reflect transformer global information storage (citing Wu et al. 2021) is an interesting connection but is speculative and unevaluated in the paper.

---

## Suggestions

1. **Replace the "novel methods" framing with accurate scoping.** Reframe contributions as "first systematic application and evaluation of vision/NLP explainability methods to audio transformer deepfake detectors," which is accurate and still valuable.
2. **Add a faithfulness experiment for attention rollout.** Token erasure or insertion/deletion curves on the AST model would transform the qualitative Figure 5 into a verifiable claim.
3. **Investigate the RMS dataset artifact.** Plot the RMS distribution for bonafide/spoof samples in ASVspoof 5 and FakeAVCeleb separately. If they differ, this explains both the GBDT's high in-distribution performance and its collapse on FakeAVCeleb.
4. **Analyze the occlusion padding failure mode.** Re-run occlusion with a non-zero baseline and check whether the padding-importance artifact disappears, which would clarify whether the problem lies in the model or the method.
5. **Conduct even a small pilot user study** (e.g., 10–20 annotators shown rollout visualizations vs. raw spectrograms on a binary labeling task) to provide at least preliminary evidence for the "citizen intelligence" claim.

---

## Score and Decision

**Calibration anchors consulted:**

| Path | Avg Human Score | Comparison to paper under review |
|------|----------------|----------------------------------|
| `/home/wg25r/review_agent/human_reviews/2GcR9bO620.md` | 7.0 | Deepfake audio detection—accepted poster. Much stronger: novel dataset of 1.3M samples, novel F-SAT robust training method, rigorous quantitative evaluation. This paper has neither a novel dataset nor a novel method. |
| `/home/wg25r/review_agent/human_reviews/St7k6NJKn1.md` | 3.5 | Adversarial attacks on deepfake speech detectors—withdrawn. Comparable topic; reviewers found it lacked novelty and rigor. Somewhat stronger than current paper in systematic experimental design. |
| `/home/wg25r/review_agent/human_reviews/50UzaXh0gC.md` | 5.0 | Wavelet-based XAI for audio/images—rejected. Most topically similar. Had genuinely novel methodology (WAM), extensive quantitative faithfulness evaluation, multi-modal coverage. Even with those contributions, reviewers rejected it for insufficient differentiation from baselines. The current paper has neither novel methodology nor faithfulness evaluation. |
| `/home/wg25r/review_agent/human_reviews/3ZdGSTxKuy.md` | 2.0 | OOD detection via existing outlier exposure—withdrawn. Applied existing method to new domain without novel contribution. Parallel pattern to this paper; reviewers identified minimal contribution and overclaimed scope. The current paper has somewhat more substance (three models, honest failure reporting, cross-dataset finding) but the pattern is similar. |
| `/home/wg25r/review_agent/human_reviews/P49gSPmrvN.md` | 1.0 | Straightforward UMAP/word-embedding visualization—rejected. Weaker than the current paper (no real experimental design); not a close anchor but sets the low end. |

**Scoring rationale:** The paper is more substantive than the 1.0–2.0 anchors (it has real training runs, honest failure analysis, and the cross-dataset finding is practically useful). However, it falls clearly below the 5.0 anchor (`50UzaXh0gC.md`), which had a genuinely novel methodology and quantitative faithfulness evaluation—and was still rejected. The current paper claims novelty it does not have, supports its broadest claims (citizen intelligence, trust-building) with no evidence, and presents one failed method among three without resolving the failure. The strongest finding (AST generalizes better than Wav2Vec on FakeAVCeleb) is valuable but is a single table in Section 6 without statistical significance testing. The center of the relevant cluster (3.5–2.0 range, leaning toward 3.0 given the additional substance over pure application papers) is where this paper sits.

**Final score: 3.0 — Reject**

The paper addresses a real and important problem—explainability and generalizability of audio deepfake detectors are genuine open challenges—but the execution does not meet the bar for ICLR. The methodological novelty is essentially absent (all three methods are imported from prior literature and applied with modest adaptation), the primary broadest claims (trust with humans, citizen intelligence) have no empirical support, one of three contributions fails by the paper's own admission without adequate analysis, and the cross-dataset evaluation, while producing a useful finding, is a standard procedure framed as a "novel benchmark." The contribution is more appropriate for a workshop venue or a short paper with accurate framing, a faithfulness evaluation, and either an investigation of the occlusion failure mode or a user study.

MY FINAL SCORE: <pineapple>3.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>