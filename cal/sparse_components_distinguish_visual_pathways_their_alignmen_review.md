=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
---

## Summary

This paper investigates the apparent contradiction that DNNs trained on object recognition model all three human visual pathways (ventral, dorsal, lateral) comparably well under standard metrics, despite those pathways having clearly distinct functions. The authors apply Bayesian NMF to fMRI responses in the Natural Scenes Dataset (NSD) to extract dominant functional components per stream, then introduce Sparse Component Alignment (SCA)—a novel metric sensitive to neural tuning axes unlike rotation-invariant metrics (RSA, linear encoding). Using SCA, they find that standard vision DNNs align strongly with the ventral stream but only weakly with dorsal and lateral streams, revealing a distinction that RSA and linear encoding fail to adequately surface.

---

## Strengths

- **SCA as a genuinely novel rotation-sensitive alignment metric.** Most existing neuroAI evaluations use RSA or linear encoding, both of which are explicitly invariant to rotations of the representational space. SCA's formulation around sparse component membership directly targets this blind spot, and the simulation in Figure 2c concretely demonstrates that SCA degrades under axis rotations while RSA remains near-ceiling—a property that follows from design but is empirically verified.

- **Novel functional characterization of dorsal and lateral streams.** While ventral stream components (faces, scenes, bodies, food, text) replicate known results, the lateral stream components (group interactions, implied motion, hand actions, reachspaces) and the dorsal stream components (scenes, implied motion) are presented as wholly novel hypothesis-free findings. Quantitative saliency correlations (r = 0.30–0.66) and behavioral ratings provide external validation beyond purely qualitative inspection. This fills a real gap—lateral and dorsal stream functional organization is far less characterized than the ventral.

- **A principled corrective to an over-optimistic literature claim.** The finding that standard object-recognition DNNs score near-zero on SCA for dorsal and lateral streams (r≈0.05–0.06), despite respectable RSA scores (r≈0.20–0.22), is a substantive, specific challenge to recent claims that DNNs universally model high-level visual cortex. This nuance has direct implications for how the field should design future models of non-ventral streams.

- **Behavioral consistency of ICMs (Figure 6).** The connectivity matrices derived by SCA align with the Meadows behavioral RDM at levels comparable to RSA-derived RDMs, despite operating on a much sparser coding structure. This shows the ICMs capture behaviorally relevant visual similarity and are not merely introducing noise via binarization.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Algorithm 1 contains a clear indexing error that impedes reproducibility.** The connectivity matrix is initialized as `0^{S,S}` (S = stimuli) on line 3, but the inner loop on line 5 iterates `i,j ← 1:C` (C = components), and the assignment on line 8 sets `C^n_{i,j}`, implying a C×C structure. The conceptual intent is clear from Equation 2 and Figure 1: `c_{ij}` should compare pairs of *stimuli* (i,j ∈ 1:S) based on which *component* they maximally activate. The loop variable should be `1:S`, not `1:C`. While the method is recoverable from the prose and figures, this pseudocode error prevents exact reimplementation without additional clarification, which is a serious reproducibility issue at a venue like ICLR. The authors should publish corrected pseudocode and ideally release code.

- **Absence of noise ceiling normalization.** The three visual streams almost certainly differ in fMRI signal-to-noise ratio (SNR)—the ventral stream contains large, well-characterized category-selective ROIs, while dorsal and lateral stream regions are noisier and more spatially diffuse. Without noise ceiling normalization, the lower SCA alignment scores for dorsal/lateral streams could partly or entirely reflect lower SNR rather than genuinely weaker DNN alignment. This is a fundamental confound for the paper's central empirical claim, and it is not addressed in the main text or limitations.

- **No formal statistical testing.** With only N=4 subjects, the paper reports alignment values without any significance tests, confidence intervals, or permutation-based null distributions. This is especially critical for the SCA dorsal (r=0.058) and lateral (r=0.047) values—these are so close to zero that they may be indistinguishable from noise. The claim of "markedly higher alignment to the ventral stream" under SCA requires demonstrating that the between-stream differences are statistically reliable across subjects, not just visually apparent in bar plots.

- **Unexplained reversal in linear encoding.** Under linear encoding, the dorsal stream shows the *highest* alignment (r=0.232), exceeding both lateral (r=0.179) and ventral (r=0.180). This directly contradicts the paper's narrative that DNNs align best with the ventral stream and is not discussed anywhere in the paper. Possible explanations include dorsal ROI geometry being more amenable to linear readout, or higher effective dimensionality of dorsal representations enabling better ridge regression fits—but none are explored. Without explanation, this finding undermines confidence in the paper's interpretive framework.

### Minor

- **The claim that RSA "fails" to distinguish streams is overstated.** RSA reports ventral r=0.347 vs. dorsal r=0.199 and lateral r=0.222—a 50–70% gap that already favors the ventral stream. The paper's framing that RSA gives "similar" alignment across streams mischaracterizes the RSA results. The accurate and stronger story is that SCA *dramatically amplifies* a distinction that RSA already partially captures, not that RSA is blind to it.

- **Sensitivity of C=20 is only verbally claimed.** The paper states "similar results also arise when deriving between 10 to 30 components," but provides no figure or quantitative evidence in the main text. The dorsal stream yields only 2 consistent components even with C=20, suggesting the choice of C matters differentially across streams. A brief sensitivity figure should be in the main paper, not deferred to the appendix.

- **Behavioral validation is limited.** The Meadows behavioral dataset involves only 4 participants arranging stimuli along 2 dimensions. While the patterns are consistent with neural findings, this is a narrow assay, and near-zero lateral/dorsal behavioral alignment could reflect either genuine behavioral irrelevance or a mismatch between the task and what those streams compute.

- **Code availability not stated.** Given the complexity of Bayesian NMF with MCMC sampling, verification of the main results requires code release. The paper does not commit to this.

### Tiny

- **Inconsistent correlation metrics.** RSA uses Spearman's ρ while SCA uses Pearson's r. When the two methods are compared head-to-head in Figure 5, this difference in metric confounds direct numerical comparison of their magnitudes. Using the same correlation measure for both would remove an interpretive ambiguity.

- **Saliency rating methodology in appendix.** Basic information about the behavioral saliency ratings (number of raters, inter-rater reliability) should appear in the main text given they are the primary quantitative validation of the component interpretations.

---

## Nice-to-Haves

- **Evaluate at least one video-trained or motion-sensitive DNN (e.g., a ViViT or 3D-CNN).** The Discussion speculates that video-trained models may better capture dorsal/lateral streams, but no exploratory test is provided. Even a single model comparison would begin to validate the mechanistic claim rather than leaving it purely speculative.

- **Component–component correlation heatmap between brain streams and DNN layers.** Visualizing which brain components have matching DNN counterparts (and which have none) would make the alignment story more mechanistically transparent.

- **Validate on an independent fMRI dataset.** Replicating the component structure on a different stimulus set (e.g., one less dominated by COCO-style object images) would help demonstrate that the dorsal/lateral components are not artifacts of the NSD stimulus distribution.

- **Add CCA or PWCCA as additional baselines.** These are established rotation-accounting alternatives to linear encoding; comparing SCA against them would situate the contribution more precisely in the alignment methods landscape.

- **Ablate the argmax (binarization) choice.** An empirical comparison of ICMs constructed with argmax vs. soft top-k assignments would quantify how much the binarization step costs in sensitivity, addressing the concern that multi-selective stimuli are treated too coarsely.

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"SCA's simulation results are circular."** (Harsh Critic) — While simulations are designed to confirm by construction, this is standard methodological validation practice. Generating data with known sparse latent structure and verifying recovery is a legitimate proof-of-concept, not circular reasoning. The value of Figure 2 is in contrasting NMF/SCA behavior against PCA/RSA on the same controlled data.

- **"RSA and SCA use different correlation metrics (Spearman vs. Pearson), invalidating head-to-head comparison."** (Harsh Critic, promoted to Tiny above) — This is a minor issue, but not a fatal one; moved to Tiny.

- **"Unfair comparison: COCO-dominated stimuli inflate ventral alignment."** (Harsh Critic, framed as a fatal confound) — The paper explicitly acknowledges this stimulus-set limitation in Section 4.1 and frames it as a scope limitation. The broader concern about the NSD stimulus distribution is real (addressed as Minor above as lack of noise ceiling) but the specific "unfair comparison" framing suggesting the authors are making an asymmetric argument in their own favor is removed.

- **"Only 1,000 shared images is an arbitrary choice."** (Harsh Critic) — This is the standard NSD "shared" stimulus set used by all four subjects; it is not arbitrary but rather the dataset's primary shared image pool. This is a standard design choice in the NSD community.

- **"The dorsal 'scenes' and ventral 'scenes' components are not distinguished."** (Harsh Critic) — The paper does not claim these are the same component; both streams having a scene-responsive component is consistent with neuroscience (scenes activate many areas). The criticism assumes an interpretive failure that is not present.

- **"Computational scalability of Bayesian NMF is not discussed."** (Review 2) — While runtime complexity is worth mentioning, demanding a scalability analysis is not a standard requirement for a neuroscience-at-ICLR methods contribution of this type; moved to Nice-to-Have omission.

---

## Novel Insights

The most genuinely novel conceptual insight synthesized across these reviews concerns what the SCA results actually reveal about the *type* of alignment DNNs have to the brain. The finding is not merely that DNNs are better models of the ventral stream (this was already known). Rather, it is that the *axes* of neural tuning—not just the geometry of the representational space—are shared between object-trained DNNs and ventral neurons. This is a mechanistically stronger claim: it implies that ventral-DNN convergence is not coincidental geometric overlap, but reflects genuinely aligned computational primitives (what images drive specific sub-populations). By the same token, the near-zero SCA scores for dorsal and lateral streams, despite moderate RSA scores, suggest that these pathways have learned *different* sparse coding bases that overlap with DNN representations only in aggregate geometry—a subtle but consequential distinction for understanding what object-recognition training actually learns and what it misses.

---

## Suggestions

1. **Fix Algorithm 1**: Correct the inner loop to iterate over stimulus pairs (`1:S`) rather than component pairs (`1:C`), or provide an explicit reconciliation with Equation 2. Release code.

2. **Add noise ceiling normalization**: Compute the noise ceiling (expected alignment given within-subject split-half reliability) for each stream and report SCA alignment as a fraction of that ceiling. This would allow fair cross-stream comparison controlling for SNR.

3. **Add permutation-based significance testing**: Shuffle stimulus labels to build a null distribution for SCA scores, and report which stream differences survive correction for multiple comparisons. This is especially urgent for the dorsal and lateral near-zero values.

4. **Discuss and explain the linear encoding dorsal > ventral reversal**: Provide at least a post-hoc analysis (e.g., comparing effective noise ceilings, ROI sizes, or voxel count distributions) to explain why linear encoding favors the dorsal stream despite the paper's narrative.

5. **Soften the claim that RSA "fails"**: Reframe SCA as *dramatically amplifying* a distinction that RSA partially captures, rather than implying RSA is uninformative. This is both more accurate and a stronger argument for SCA's value.

6. **Include a C-sensitivity figure in the main paper**: Plot SCA alignment scores for C ∈ {10, 15, 20, 25, 30} to substantiate the verbal claim that results are robust.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 5.0, 8.0]
Average score: 7.2
Binary outcome: Accept
