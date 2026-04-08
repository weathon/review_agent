=== CALIBRATION EXAMPLE 18 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "DenseFace: Bias Mitigation in Face Recognition via Density-Aware Probabilistic Matching" accurately reflects the method. The abstract makes three headline claims: (1) demographic bias is reduced, (2) accuracy is preserved, and (3) no retraining is required. All three are at least partially supported by the experiments. However, the abstract's blanket "no retraining" claim becomes misleading once the reader discovers the learning-based variant (DenseFace†, Section 4.5), which explicitly trains a regression network. This should be qualified. The abstract also references "demographic biases" broadly, but experiments cover racial groups almost exclusively; gender is shown in one figure (Figure 3) and not evaluated quantitatively.

---

### Introduction & Motivation

The problem motivation is clear and socially important. The post-training calibration paradigm is well-positioned — no retraining, applicable to any existing model, no inference-time demographic labels needed. These are genuine practical advantages over the majority of prior work.

However, two claims in the introduction deserve scrutiny:

1. **"DenseFace does not downgrade verification performance while significantly reducing bias"** (end of Section 2.3). This is a very strong claim. The experiments in Table 4 only show TPR@FPR for a few models/datasets, and the calibration procedure sets the Caucasian FPR to exactly 10⁻³ by construction. This means any accuracy impact on the Caucasian group is hidden by design. A more honest framing would acknowledge that the calibrated threshold trades off precision on the reference group.

2. The observation that "training on large-scale unbalanced datasets possess lesser or comparable bias" (Section 4.4) compared to methods trained on BUPT-Balancedface is interesting, but the comparison in Table 1 is under vanilla RFW accuracy (Std), which the paper itself argues is an *insufficient* metric. Using the very metric the paper critiques to make a positive claim about its own experimental setting is internally inconsistent.

---

### Related Work

The related work is thorough and well-organized across data-based, architecture-based, loss-based, and post-processing approaches. The positioning of DenseFace relative to Terhorst et al. (2020a,b), Conti et al. (2022), and Linghu et al. (2024) is clear.

**Critical gap:** The paper's methodology is closely related to SCF (Li et al., 2021, "Spherical Confidence Learning for Face Recognition"), which uses the exact same MF mutual likelihood score (Equation 9) and was designed specifically for hyperspherical embeddings. The related work section does not describe SCF in detail, even though Equation 9 is directly adopted from it. SCF also involves density/uncertainty estimation for face verification. The difference — that DenseFace uses *inter-class* density from an external anchor set rather than *per-image* uncertainty — is important and real, but it needs to be articulated explicitly and upfront as the core technical novelty, rather than being mentioned only obliquely in Section 3.

---

### Methodology (Section 3)

**Overall structure:** The method has three components — (1) MF representation, (2) local distortion/density estimation, (3) density-aware matching. This is a sensible decomposition.

**Component 1 — MF representation (Section 3.1):** The choice of MF for unit-hypersphere embeddings is standard and well-motivated. Setting µ_i = z_i is reasonable. The κ estimation formula (Equations 3–4) follows Banerjee et al. (2005). No issues here.

**Component 2 — Local distortion (Section 3.2):** This is the most novel technical element. The authors observe that in high-dimensional embedding spaces, most anchor set pairs are nearly orthogonal (cos θ ≈ 0), which makes the K-NN sum degenerate (all embeddings contribute ~equally to r_i). The margin m in Equation 7 artificially amplifies angular differences so that the K nearest neighbors actually contribute more than the rest.

This is a reasonable heuristic but raises several concerns:
- **How is m selected?** The paper mentions K=128 and provides Figure 4 to show the distribution shift, but there is no ablation or principled justification for the specific value of m used. The sensitivity of the resulting bias metrics to m is not studied.
- **What value of m is actually used?** I cannot find the numerical value of m in the text — it seems to be relegated to an appendix (referenced as Appendix G, which is not available in the parsed text). For the methodology to be reproducible, m must be stated in the main paper.
- **Implicit threshold behavior:** When θ_kl ≤ m, Equation 7 returns 1 (i.e., the pair is treated as perfectly aligned). This hard thresholding could cause instability if many pairs have θ slightly above m.

**Component 3 — Density-aware matching (Section 3.3):** Equation 9 is the MF mutual likelihood score from Li et al. (2021). The derivation of Equation 8 (the double integral over S^{d−1} × S^{d−1}) is cut off by the parser mid-derivation (between lines 648–700 in the parsed text), but the final formula follows from SCF with density-based κ substituted. This is a reasonable substitution, but the paper should formally verify that using *inter-class* density estimates in a formula derived for *uncertainty* estimates is theoretically sound. The two quantities measure different things. SCF's κ_i reflects how spread out a single identity's embedding is; DenseFace's κ_i^{(m)} reflects how crowded the surrounding identity-space is. Why should the same matching formula apply?

**Anchor set design:** The anchor set is taken from Glint360K (54,000 balanced identities). The models evaluated in Table 3 on RB-WebFace include CosFace-R50-Glink360K and AdaFace-R50-MS1MV2. The CosFace model is *trained on the same dataset* used to construct the anchor set. This is a potential data leakage concern that the paper does not address. The authors do note (Section 4.6) they evaluate Glint360K-trained models only on RFW and MS1MV2-trained models only on RB-WebFace to avoid train/test leakage for the face *verification* pairs, but this does not address the anchor set issue for Glint360K-trained models on RFW.

**Learning-based variant (Section 4.5):** DenseFace† replaces the K-NN anchor set lookup at inference with a small MLP regressor for κ prediction. This is practical and the paper reports it achieves even better bias reduction (Table 5). However, training this MLP requires labels (i.e., knowing the density values from the anchor set process), and the MLP is trained on Glint360K. This is indeed a form of retraining that uses demographic information encoded in the anchor set. The abstract claim of "no retraining" is not accurate for this variant.

---

### Bias Evaluation Protocol (Section 4.3)

The critique of the RFW Std metric is valid: a model can have low Std while having systematically shifted cosine similarity distributions across demographic groups. The proposed NIST-style metric (FPR at a threshold calibrated to Caucasian FPR = 10⁻³) is a genuine improvement over vanilla accuracy-based metrics, and the expanded negative pair set (≈50M vs. 3,000 in vanilla RFW) makes the estimate more reliable.

**However, a significant concern:** The chosen reference group is *Caucasian*. The threshold is set such that the Caucasian false positive rate equals 10⁻³, and then all other groups are evaluated at *that same threshold*. This design makes the Caucasian group the "neutral" or "privileged" reference, which is a non-neutral design choice. A model that over-accepts Caucasians (FPR > 10⁻³ for Caucasians at some natural threshold) would be *masked* by this calibration. The authors should at minimum discuss this asymmetry and justify why Caucasian is the appropriate reference cohort (it matches NIST's "MW" = Majority/White convention, but that convention itself has been criticized).

Furthermore, by calibrating the threshold to Caucasian FPR = 10⁻³, DenseFace's "success" on minority groups (reducing their FPR toward 1.0) may partly come from simply *raising the threshold* globally rather than genuinely improving feature discrimination for minority groups. This confound is not disentangled.

---

### Experiments & Results (Section 4)

**Strengths:**
- Multiple backbone architectures (R50, R100), training datasets (MS1MV2, WebFace4M, WebFace12M, Glint360K), and loss functions (ArcFace, AdaFace, CosFace) are evaluated.
- Both RFW and RB-WebFace benchmarks are used.
- FPR reductions are large (e.g., CosFace-R50: African 3.74→0.40, Asian 9.96→0.71 in Table 3).

**Weaknesses:**

1. **No comparison to prior post-processing methods under the NIST protocol.** Tables 2 and 3 only compare DenseFace to the cosine baseline, not to Terhorst et al. (2020a,b), Linghu et al. (2024), or Conti et al. (2022) — the most directly comparable prior work. The authors instead compare to training-based methods under the vanilla RFW Std metric (Table 1). This makes it impossible to assess whether DenseFace outperforms existing post-processing baselines under a fair comparison.

2. **No comparison to SCF (Li et al., 2021).** Since DenseFace uses Equation 9 from SCF, a natural baseline is: what if you applied SCF's *original* per-image uncertainty estimation (which requires training), and compared it to DenseFace's density-based κ estimation? Without this baseline, it is unclear whether the improvements come from the probabilistic matching formula or from the density estimation specifically.

3. **Missing ablations:**
   - Effect of margin m
   - Effect of K (number of neighbors; only K=128 reported)
   - Effect of anchor set size and demographic balance
   - Balanced vs. unbalanced anchor set (a key claimed advantage is the balanced construction — this should be ablated explicitly)

4. **Table 4 (TPR@FPR) results** are presented incompletely. The DenseFace rows appear to show very small differences from the cosine baseline (e.g., "preserving or improving"), but the numbers are not clearly readable due to parsing artifacts. The claim "no accuracy loss" needs cleaner support.

5. **Cross-racial matching** is mentioned in the introduction as a motivation ("the RFW protocol does not account for cross-racial matching as typically required in real-world scenarios") but then is not evaluated experimentally. This is a gap given it is cited as a motivation.

6. **Only racial groups** are quantitatively evaluated. The abstract and introduction mention "demographic biases" broadly; Figure 3 shows gender distributions; Section 4.7 notes that "female identities have higher local density values." But no gender bias results are reported in any table.

---

### Writing & Clarity

The method pipeline involves several interacting components (MF representation, inter-class density, angular margin, mutual likelihood, anchor set), and the paper would benefit from a cleaner, unified description. As is, the description is partially in Section 3 and partially in Algorithm 1, with Figure 2 serving as the primary reference point. The key formula (Equation 9) appears after the implementation details section (Section 4.2) due to what appears to be layout fragmentation — the derivation of Equation 8 is cut in the parsed text, though this may be a parser artifact.

The positioning of Section 4.5 (learning-based approach) before Section 4.6 (quantitative analysis) creates confusion — the learning-based variant's results appear in Table 5 before the main quantitative results of Tables 2 and 3 are discussed.

The claim at the end of Section 2.3 that DenseFace "does not downgrade verification performance" is very strong and appears in the related work section before any results are presented, which is inappropriate.

---

### Limitations & Broader Impact

The paper includes a brief limitations discussion (Section 4.8) focused on computational cost, which is a minor limitation. More substantive limitations that go unacknowledged include:

1. **Reliance on pre-trained race/gender classifiers** for constructing the balanced anchor set. Errors in these classifiers (which are themselves known to exhibit bias) could affect the anchor set composition and propagate through the method.

2. **The method implicitly encodes demographic information.** The anchor set is deliberately balanced by race/gender, and density estimation implicitly acts as a race proxy (as the authors acknowledge in Section 4.7 that the anchor set functions as an "implicit race classifier"). This raises regulatory concerns in jurisdictions where any use of race in automated decision systems is restricted — even post-hoc calibration. The paper does not discuss this.

3. **No evaluation on underrepresented or intersectional groups.** The evaluation is limited to four racial groups (Caucasian, African, Asian, Indian). Performance on intersectional groups (e.g., elderly Asian females) or groups not covered by the anchor set is entirely unknown.

4. **Threshold calibration artifact.** If the method primarily works by raising the global threshold (rather than modifying feature geometry), it may reduce false positives at the cost of increased false negatives for minority groups — but this tradeoff is not reported.

---

### Overall Assessment

DenseFace offers a genuinely practical contribution: a post-training, anchor-set-based approach that reduces racial bias in face recognition without modifying the underlying model. The key idea — that inter-class embedding density correlates with demographic bias and can be used to calibrate the matching score — is intuitive and supported empirically. The proposed NIST-style FPR evaluation protocol is a useful methodological contribution to the community.

However, the paper has significant gaps that need to be addressed before it meets ICLR's bar. Most critically: (1) the core matching formula (Equation 9) is adopted directly from SCF (Li et al., 2021) without a principled justification for why an uncertainty-based formula is appropriate for density-based inputs, and without an SCF baseline comparison; (2) the main experimental tables compare only to the cosine baseline rather than to directly comparable post-processing methods under the proposed NIST protocol; (3) key hyperparameters (notably margin m) are never ablated or even reported in the main text; (4) the "no retraining" claim is not accurate for DenseFace†; and (5) the potential for data leakage through the anchor set is unaddressed. The paper's empirical results are promising, but in its current form the contribution reads more as an engineering pipeline with good numbers than a technically rigorous method with well-understood properties. A revision that adds the missing ablations, includes direct comparisons to prior post-processing baselines, and tightens the theoretical justification would substantially strengthen the submission.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes DenseFace, a post-training bias mitigation technique for face recognition that models embeddings as von Mises-Fisher (MF) distributions and adjusts similarity scores based on estimated local inter-class densities. The authors observe that demographic biases correlate with embedding density variations and introduce a probabilistic matching procedure that normalizes scores across dense and sparse regions, alongside a metric critique advocating for NIST-style false positive rate evaluation. Extensive experiments demonstrate significant bias reduction across multiple architectures and datasets without model retraining or accuracy degradation.

### Strengths
1. **Practical Post-Hoc Design:** The method operates entirely at inference, avoiding the computational cost and potential performance degradation associated with fine-tuning or retraining. As shown in Tables 2 and 3, the approach consistently reduces False Positive Rates (FPR) for non-Caucasian and Indian demographics while preserving or improving verification accuracy (TPR) across diverse backbones (AdaFace, CosFace) and training datasets (WebFace, MS-Celeb).
2. **Geometrically Sound Distributional Modeling:** Modeling face embeddings on the unit hypersphere using MF distributions is well-aligned with the inductive biases of modern margin-based losses. Deriving the matching score from the mutual likelihood of two MF distributions (Eq. 8-9) provides a principled probabilistic alternative to raw cosine similarity, and the "local distortion" trick (Eq. 7) creatively addresses numerical instability when estimating density in sparse, near-orthogonal regions.
3. **Rigorous and Industry-Aligned Evaluation:** The critique of standard academic bias metrics (e.g., RFW accuracy standard deviation) is well-founded. By adopting the NIST FRVT protocol (FPR fixed across demographic cohorts), the evaluation better reflects real-world biometric system deployment. The authors provide clear implementation details, including anchor set construction (balanced across 4 races × 2 genders), $K=128$, and full training specifications for the learned regressor (Sec. 4.2), supporting strong reproducibility.

### Weaknesses
1. **Heuristic Margin Selection and Lack of Ablation:** The angular margin $m$ introduced in Eq. 7 is critical for stabilizing density estimation, yet the paper provides no guidance on how $m$ is selected, optimized, or validated. There is no ablation study analyzing sensitivity to $m$ or $K$, making it difficult to determine how robust the method is across different embedding spaces or anchor set sizes.
2. **Computational Overhead and Scalability Concerns:** While the learning-based variant mitigates this, the baseline DenseFace requires a $K$-nearest neighbor search across a 54K anchor set for *every* query during verification. Section 4.8 notes a 1.5×–2.5× slowdown, but the analysis lacks discussion on the memory footprint of storing the full anchor embedding set and its scalability for large-scale 1:N identification pipelines, which is common in deployment scenarios.
3. **Incomplete Fairness Characterization (Missing FNMR):** The evaluation heavily emphasizes FPR parity (impostor bias). However, demographic bias also manifests as disparities in False Non-Match Rates (FNMR) for genuine users. The paper does not report TPR/FPR curves per demographic or analyze whether the density adjustment inadvertently compresses genuine match scores for underrepresented groups, which limits the holistic assessment of fairness.
4. **Potential Data Leakage / Generalization Risks:** The anchor set and regressor are trained on Glint360K. Given that WebFace4M/12M and MS1MV2 share significant identity overlap with Glint360K, the density estimator may inadvertently memorize dataset-specific distributions rather than learning transferable demographic calibration. The paper does not explicitly quantify or mitigate this overlap, raising questions about out-of-distribution generalization.

### Novelty & Significance
**Novelty:** The work offers a novel synthesis of probabilistic representation learning and post-hoc fairness calibration. While MF distributions have been used for face uncertainty (e.g., Shi & Jain, 2019), leveraging inter-class local density as a demographic bias signal and dynamically warping the matching function is a fresh contribution. The introduction of a local margin to stabilize density estimation in sparse hypersphere regions adds a useful heuristic to the geometric learning literature.
**Clarity:** The paper is generally well-structured, with clear motivation and logical progression from distributional modeling to matching. Some mathematical derivations (particularly the transition from Eq. 5 to Eq. 7) are slightly dense and could benefit from tighter exposition, but the core algorithm (Algorithm 1) is straightforward.
**Reproducibility:** Strong. The authors provide explicit hyperparameters, network architectures, optimizer settings, and clear instructions for anchor set construction. The learning-based variant is fully specified. Providing code would further strengthen reproducibility, but the textual description is sufficient for independent implementation.
**Significance:** Highly relevant to ICLR. The paper addresses a critical gap between academic fairness benchmarks and industry standards, pushing the community toward threshold-consistent evaluation (NIST FPR). The post-training calibration approach is highly practical for deployed systems, and the geometric calibration aligns well with ICLR's focus on representation learning and uncertainty quantification.

### Suggestions for Improvement
1. **Provide Hyperparameter Sensitivity Analysis:** Include an ablation table sweeping the margin $m$ and neighbor count $K$. Explain whether $m$ is fixed globally or computed adaptively, and justify the chosen values.
2. **Report FNMR/TPR Disparities:** Extend the evaluation to include False Non-Match Rate (FNMR) analysis at fixed FPR. Plot ROC curves per demographic to show whether the method improves parity for both impostor rejection and genuine acceptance.
3. **Address Dataset Overlap and OOD Generalization:** Explicitly report the estimated identity overlap between Glint360K (anchors) and the test/training datasets. If possible, run an experiment with a completely disjoint anchor set (e.g., constructed from LAION or a curated subset) to demonstrate that the calibration generalizes beyond dataset memorization.
4. **Clarify Computational Trade-offs for Production:** Provide a detailed breakdown of memory requirements (storing 54K×512D anchors vs. MLP weights) and inference latency for 1:1 vs. 1:N search modes. Discuss strategies for dynamic anchor pruning or product quantization to accelerate the baseline method.
5. **Theoretical Justification for Density-Bias Correlation:** Briefly discuss *why* non-Caucasian groups cluster in denser regions. Is this a direct consequence of skewed training data priors, a byproduct of loss function optimization, or a reflection of intra-class diversity differences? A short theoretical or empirical analysis would strengthen the motivation.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Compare against state-of-the-art post-training calibration baselines (e.g., Terhorst et al., Linghu et al.) rather than only raw Cosine, as current tables do not prove superiority over existing mitigation techniques.
2. Include quantitative gender bias results (equivalent to Tables 2-5) to substantiate the abstract's claim of mitigating "gender, race and other attributes," as current results focus almost exclusively on race.
3. Evaluate verification performance on cross-racial pairs (e.g., Asian probe vs. Caucasian gallery) to address the stated limitation of current protocols and prove real-world applicability.
4. Explicitly verify zero identity overlap between the Glint360K anchor set and RFW/RB-WebFace test subjects to rule out data leakage in the density estimation process.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze sensitivity to anchor set composition and size, as the method's fairness depends entirely on an anchor set constructed using potentially biased demographic classifiers.
2. Provide full ROC curves instead of single-point FPR metrics to ensure bias reduction is not achieved by collapsing score distributions and degrading overall separability.
3. Explain why the learning-based approach (Table 5) outperforms the anchor-based method, as this suggests the proposed density mechanism may not be the primary driver of improvement.

### Visualizations & Case Studies
1. Visualize the embedding space distortion (e.g., via t-SNE) before and after DenseFace to confirm the claimed mechanistic effect of "expansion and contraction" in dense/sparse areas.
2. Present failure cases where DenseFace incorrectly adjusts scores (e.g., false accepts in dense Caucasian regions) to expose the method's limits and error modes.
3. Plot the actual demographic distribution of the constructed anchor set to verify it is truly balanced and not merely balanced according to the predictions of biased classifiers.

### Obvious Next Steps
1. Develop a clustering-based anchor construction method to remove the dependency on external demographic classifiers, which introduces a circular dependency on biased models.
2. Evaluate latency and accuracy in a 1:N identification scenario, as the reported 1.5x matching slowdown is critical for large-scale deployment feasibility.
3. Provide a theoretical justification for the correlation between embedding density and demographic bias, moving beyond the current empirical observation to ground the method mathematically.

# Final Consolidated Review
## Summary

DenseFace proposes a post-training bias mitigation method for face recognition that models embeddings as von Mises-Fisher distributions and adjusts similarity scores based on local inter-class embedding density. The key insight is that demographic bias correlates with embedding space density—minority groups tend to occupy denser regions, leading to higher false positive rates. DenseFace calibrates matching scores by estimating density from a balanced anchor set, reducing bias without retraining the underlying face recognition model. The paper also advocates for NIST-style FPR evaluation at fixed thresholds rather than accuracy-based bias metrics.

## Strengths

- **Practical post-hoc design**: The method operates entirely at inference time without modifying or retraining the face recognition backbone. This is genuinely valuable for deployed systems where retraining is costly or infeasible. The learning-based variant (DenseFace†) requires only training a small MLP regressor (132k parameters), not the full model.

- **Novel density-bias observation**: The empirical finding that inter-class embedding density correlates with demographic bias (Figure 3) is meaningful and well-motivated. The local distortion trick (Equation 7) to handle near-orthogonal embeddings in high-dimensional spaces is a creative heuristic that addresses a real numerical stability issue.

- **Methodological contribution to bias evaluation**: The critique of RFW's standard deviation metric and the adoption of NIST's FPR-at-fixed-threshold protocol is a genuine improvement. Figure 5 effectively demonstrates that Std can be misleading—CosFace-R50 has lower Std but larger cross-group similarity gaps than AdaFace-R50-WebFace4M.

- **Strong empirical coverage**: Experiments span multiple architectures (ResNet-50/100), training datasets (MS1MV2, Glint360K, WebFace4M, WebFace12M), and loss functions (ArcFace, AdaFace, CosFace). FPR reductions are substantial (e.g., Table 3: CosFace African FPR drops from 3.74 to 0.40, Asian from 9.96 to 0.71).

## Weaknesses

- **Missing comparison to post-processing baselines under NIST protocol**: Tables 2–5 compare DenseFace only to raw cosine similarity. The paper claims superiority over prior work (Terhörst et al., 2020a,b; Conti et al., 2022; Linghu et al., 2024) but does not evaluate any of these methods under the proposed NIST FPR protocol. Without this comparison, it is impossible to assess whether DenseFace outperforms existing post-hoc bias mitigation techniques on the proposed metric.

- **Core equation adopted from SCF without comparison**: Equation 9 (mutual likelihood score) is taken directly from Li et al. (2021), which the paper acknowledges. However, SCF estimates κ from per-image uncertainty (multiple augmented views), while DenseFace estimates κ from inter-class density. This is a meaningful difference, but no comparison to SCF is provided to show whether the density-based estimation is actually better than SCF's original uncertainty-based approach.

- **Margin hyperparameter m not specified or ablated**: The angular margin m in Equation 7 is critical to the density estimation procedure, but its numerical value is not reported in the main paper. There is no ablation study on m, K (number of neighbors), or anchor set size. This limits reproducibility and makes it difficult to assess robustness.

- **Potential data leakage for Glint360K-trained models**: The anchor set is constructed from Glint360K (Section 4.2). For models trained on Glint360K (CosFace-R50-Glink360K in Table 3), the anchor set identities are from the same training data. While the test pairs use different identities, the anchor set encodes demographic structure learned from Glint360K, which could advantage Glint360K-trained models in ways that don't generalize.

- **Scope of demographic claims exceeds evaluation**: The abstract and introduction mention "gender, race and other attributes," but quantitative experiments (Tables 2–5) focus exclusively on racial groups. Figure 3 shows gender density distributions but no gender bias results are reported. Cross-racial matching is mentioned as a motivation ("RFW protocol does not account for cross-racial matching") but never evaluated.

- **Caucasian FPR calibration may mask privileged-group effects**: The NIST-style protocol calibrates thresholds to Caucasian FPR = 10⁻³, then evaluates other groups at that threshold. This makes Caucasian the implicit reference, and the paper does not discuss whether this choice affects the reported bias reduction. A model that systematically over-accepts Caucasians would have its FPR set to 10⁻³ by construction, potentially masking an issue.

## Nice-to-Haves

- Analysis of FNMR (false non-match rate) per demographic group to provide a complete fairness picture beyond FPR
- Ablation study on anchor set composition: balanced vs. natural distribution, effect of using different demographic classifiers
- Visualization of embedding space distortion (t-SNE) before/after DenseFace to confirm the claimed expansion/contraction mechanism

## Removed Points

- **"Abstract misleading about retraining for DenseFace†"**: The paper distinguishes between DenseFace (anchor-based, no training) and DenseFace† (learned κ predictor). The abstract's "no retraining" refers to the face recognition model, which holds for both variants. Training a small regressor is not equivalent to retraining the backbone. *Removed as the criticism mischaracterizes what "retraining" means in this context.*

- **"Vague theoretical justification for density-bias correlation"**: While a deeper theoretical analysis would strengthen the paper, this is empirical systems work. The empirical correlation is clearly demonstrated, and demanding mathematical proof is scope creep for this venue.

- **"Writing quality issues"**: Formatting and exposition clarity are not substantive technical concerns for ICLR review.

## Novel Insights

The paper surfaces an intriguing observation: models trained on larger, unbalanced datasets can exhibit *less* bias than models trained on smaller balanced datasets (Table 1). CosFace-R50-Glint360K (trained on unbalanced data) achieves lower Std than AdaFace-R50-WebFace4M trained on BUPT-Balancedface. This challenges the conventional wisdom that balanced training data is necessary for fairness, suggesting that scale may matter more than balance—a hypothesis that merits further investigation. The discovery that inter-class embedding density (not intra-class variance) encodes demographic information is also noteworthy: identities in denser regions (typically minority groups) have higher pairwise similarities with impostors, directly causing higher FPR.

## Suggestions

1. **Add a direct comparison to at least one post-processing baseline** (e.g., Terhörst's score normalization or Linghu's score calibration) under the NIST FPR protocol to establish DenseFace's relative effectiveness.

2. **Specify the margin m value in the main text** and provide an ablation showing sensitivity to m and K. This is essential for reproducibility.

3. **Evaluate gender bias quantitatively** or scale back the abstract's claims to focus specifically on racial bias, which the experiments support.

4. **Construct an ablation anchor set from a disjoint dataset** (e.g., for Glint360K-trained models, use an anchor set from a different source) to rule out memorization concerns and demonstrate generalization.

# Actual Human Scores
Individual reviewer scores: [2.0, 6.0, 2.0]
Average score: 3.3
Binary outcome: Reject
