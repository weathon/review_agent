---
job_id: 5b4103c0-50e1-45aa-92e4-ad37b321c95c
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: c7OsKOOZo8.pdf
paper: Expert-Guided Cross-View Fusion with Self-Derived Lesion Proposals for Multi-View Diabetic Retinopathy Grading
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length  
Pass ✅.

## Topic Compatibility  
Pass ✅. The paper proposes a new multi-view deep learning architecture (GALP + LGRF) for DR grading, clearly within representation learning and computer vision for healthcare.

## Minimum Quality  
Pass ✅. The paper is in English and contains all required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion). The method is technically coherent, experiments are reasonably substantial on two benchmarks with ablations, and no obvious fatal methodological, theoretical, or evaluation flaws are present.

## Prompt Injection and Hidden Manipulation Detection  
Pass ✅. I did not find any instructions targeting AI reviewers, hidden prompts, or other manipulative content in the main paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces an end-to-end multi-view diabetic retinopathy (DR) grading framework that avoids reliance on external lesion or vessel annotations. The Grade-Activated Lesion Proposal (GALP) module uses stage-wise auxiliary classifiers and CAMs to generate grade-conditioned evidence maps and select top‑K lesion proposal regions, which are then used by the Cross-View Lesion Expert Guided Regional Fusion (LGRF) module to perform expert-routed, top‑K cross-view attention-based fusion. Experiments on MFIDDR (4 views) and DRTiD (2 views) show that the proposed method outperforms existing end-to-end baselines and is competitive with, or better than, several approaches that use external annotations, with ablations highlighting the contributions of GALP and LGRF.

## Strengths

1. **Clear architectural idea and good integration of components.**  
   The overall pipeline in **Figure 2** is well thought out: GALP builds lesion-centric proposal tokens at multiple stages, and LGRF then uses these proposals for cross-view fusion via expert routing and attention. The design is conceptually coherent, and the stage-wise integration with a Swin backbone is technically sound and well specified in **Section 3** and **Algorithm 1**.

2. **Eliminating reliance on external lesion/vessel annotations while remaining competitive.**  
   On MFIDDR (Table 1), the “Ours (w/o lesion)” variant achieves Acc = 83.9, Kappa = 70.9, F1 = 83.5, surpassing all listed end-to-end methods (e.g., ETMC at 81.5 / 64.8 / 79.7) and matching or outperforming some externally informed methods (e.g., CVSA and LFMVDR). On DRTiD (**Table 3**), the end-to-end variant reaches the highest accuracy (76.0) among all compared methods, including CVSA and CrossFiT which use extra annotations. This is a strong empirical case that self-derived proposals can serve as useful surrogates for external cues.

3. **Reasonable performance gains from the proposed modules, supported by ablations.**  
   The ablation in **Table 4** shows that each component contributes nontrivially: removing GALP reduces Acc from 83.9 to 82.7 and Kappa from 70.9 to 68.5; removing experts (“w/o Experts”) or completely removing LGRF (“w/o LGRF”) gives further performance drops. This suggests that the CAM-based proposal selection and the expert-routed cross-view fusion both add value beyond a simple multi-view backbone.

4. **Stage-wise CAM-based proposal mechanism is practically motivated and well instantiated.**  
   The GALP module is carefully specified. **Equations (3)–(7)** detail how grade-conditioned evidence maps are formed using class weights, normalized, and then aggregated into region scores \(s^{i,r}_{s_n}\), with top‑K selection over non-overlapping \(q \times q\) patches. This is a clean way to derive lesion-like tokens that emphasize grade-discriminative areas without extra labels.

5. **Mildly improved interpretability and controllability over black-box multi-view fusion.**  
   While not deeply explored, the structure of GALP and LGRF naturally lends itself to more interpretable behavior: lesion proposals are spatially localized patches derived from evidence maps, and LGRF explicitly conditions expert routing on the current view’s global tokens. The conceptual story of “lesion proposals corroborated across views by experts” is clinically appealing and is clearly communicated in **Figure 1** and the textual motivation in **Section 1**.

6. **Hyperparameter studies give some insight into design choices.**  
   **Figure 3** systematically varies the token retention ratio \(\alpha\), number of routed experts \(K_2\), and total experts \(M\). The plots (despite being small) show that \(\alpha = 0.5\), \(K_2 = 2\), and \(M = 6\) are near-optimal settings, which somewhat de-risks the concern that the gains are purely due to heavy tuning.

7. **Strong and diverse baselines, including both end-to-end and externally guided methods.**  
   The paper compares against a reasonably comprehensive set of multi-view DR baselines and single-view backbones (the latter in **Table 5**), plus annotation-based methods such as CVSA, WGLIN, SMVDR, LFMVDR, and CrossFiT. This breadth helps position the contribution relative to the field’s current practice and shows that gains are not from trivial baseline choices.

## Weaknesses

1. **Conceptual novelty is moderate; core ideas are combinations of known components.**  
   The two central pieces, CAM-based lesion proposals and MoE-style expert routing with cross-attention, are both well-established techniques. GALP essentially applies stage-wise CAM / LayerCAM to Swin feature maps plus top‑K patch selection, and LGRF is a fairly straightforward MoE-style cross-view attention design. The paper’s main contribution is in integrating them for multi-view DR without external annotations, which is interesting but not a major conceptual shift compared to prior lesion-guided multi-view works (e.g., LFMVDR, SMVDR, WGLIN) that already emphasize lesion-centric fusion. This limits the methodological originality, especially for an ICLR main-track audience.

2. **Limited qualitative analysis of lesion proposals and fusion behavior.**  
   A key claim is that GALP proposals correspond to “lesion-like” regions and that LGRF focuses fusion on corroborated lesions. However, there is essentially no qualitative visualization of GEMs, lesion proposal locations, or attention maps. **Figure 2(b)** only shows a schematic example of evidence maps and proposals but no real data, and the paper does not provide examples of how proposals evolve across stages or align with annotated lesions on MFIDDR (for which lesion masks exist). Without such visual evidence, it is hard to verify that the proposals are indeed capturing microaneurysms or other small lesions rather than coarse vessel/background structure, undermining the interpretability narrative.

3. **Mathematical and notational issues around CAMs and MoE loss.**  
   - In **Equation (3)**, the notation \(\mathbf{w}^{(\hat{\mathbf{y}}^{i}_{s_n})}_{s_n,\tilde{c}}\) is confusing: the index \(\tilde{c}\) appears only in the subscript of \(\mathbf{w}\) but the sum runs over \(c=1,\dots,C_{s_n}\). It is unclear whether \(\tilde{c}\) is a typo for \(c\), or indicates some channel remapping. This ambiguity matters because \(\mathbf{A}^{i}_{s_n}(u,v)\) is the core quantity used for lesion proposals.  
   - In **Equation (11)**, the load-balancing loss \(\mathcal{L}_{\mathrm{load}}^i\) is defined as  
     \[
     \mathcal{L}_{\mathrm{load}}^i = M \cdot \sum_{m=1}^M \left(\frac{1}{B}\sum_{b=1}^B \mathcal{R}s^{i}_{s_n,b,m}\right)\cdot \hat{u}_m
     \]
     and then summed across stages and views. As written, this expression is *larger* when routing probabilities and utilization \(\hat{u}_m\) are more concentrated on a few experts, which seems counter to the stated goal of “encourage equitable utilization of experts”. Typical load-balancing losses either minimize a divergence between routing distribution and uniform or maximize entropy; here there is no negative sign or normalization, so minimizing this product would actually favor small activations (possibly underutilizing experts) unless additional constraints or normalization are in place. This point needs clarification or correction.

4. **Insufficient analysis of where the gains come from and how robust they are.**  
   While **Table 4** shows that removing GALP or LGRF hurts performance, more detailed analysis is missing:
   - GALP includes two intertwined effects: (i) auxiliary supervision on intermediate features and (ii) top‑K region selection based on CAMs. The ablation only removes the whole GALP; there is no variant with auxiliary heads but without top‑K proposal selection, or vice versa. Hence, it is unclear whether the improvements are primarily due to deeper supervision (a standard trick) rather than the specific lesion proposal mechanism.
   - Similarly, the LGRF ablations are coarse (“w/o Experts”, “w/o LGRF”). There is no comparison against a simpler cross-view fusion using all tokens (no proposal filtering) or a single-head, non-MoE attention over proposals. Given that the gains over strong methods like LFMVDR and WGLIN in **Table 1** are modest (e.g., Kappa 72.3 vs. 71.4), it is important to know whether the extra complexity is justified by a meaningful accuracy/efficiency trade-off.

5. **Missing or underspecified training and evaluation details.**  
   - The paper mentions in **Section 4.1** that images are resized to 224×224 (MFIDDR) and 512×512 (DRTiD) and that Swin-B backbones are used, but several implementation details that affect reproducibility are missing or vague: optimizer type, learning rate schedule, batch size, number of epochs, data augmentation, and how early stopping or model selection is performed (validation split vs. test set).  
   - For MFIDDR, they use the “official split 70/30 train/test”, but never mention a separate validation set. It is unclear how hyperparameters such as \(\lambda_{\mathrm{load}}\), patch size \(q\), or MoE configuration were chosen without risk of test-time overfitting. **Figure 3** presents hyperparameter sweeps, but it is not stated whether the plotted metrics are on a validation set or directly on the test set.

6. **External-lesion variant design is underspecified and blurs the “annotation-free” story.**  
   The “Ours (with lesion)” variant fuses lesion segments using SPADE, but **Section 4.1** only states “lesion segments are fused with the original images via SPADE” without clarifying:
   - At which layer(s) SPADE is inserted into Swin-B and how it interacts with GALP and LGRF.  
   - Whether lesions are used only during training or also required at inference.  
   - Whether the lesion masks are ground-truth, model-generated, or post-processed.  
   This makes it hard to judge how fair the comparison in **Table 1** is and to what extent the architecture is truly generalizable to settings without lesion annotations.

7. **Limited discussion of failure modes and per-grade trade-offs.**  
   **Table 2** gives per-grade F1/Precision/Specificity. For Grade 4, even “Ours (w/o lesion)” achieves F1 = 36.0 and “Ours (with lesion)” 51.6, which are still relatively low compared to Grades 0–3, and in several competitors simply pushing precision to 99.9 yields comparable or better F1. The paper briefly notes that performance on severe grades improves, but does not analyze why Grade‑4 remains a weak point, whether data imbalance is addressed beyond focal loss, or how sensitive the model is to thresholds in highly imbalanced categories. This is particularly important clinically.

8. **Computational complexity and scalability are not addressed.**  
   The LGRF module introduces a multi-stage tokenization, proposal selection, multiple Transformer experts per stage and view, and a load-balancing term. There is no discussion of training/inference time, memory overhead, or scaling to more views or larger resolutions. A simple comparison of FLOPs or runtime against a strong baseline like MVCINN or LFMVDR would help justify the extra machinery. Without this, it is possible that similar gains could be obtained with simpler models that are more deployment-friendly.

9. **Some related work omissions in an actively developing niche.**  
   Given the very specialized multi-view DR grading literature, the Related Work section is relatively short and omits several directly relevant recent works that also focus on multi-view/multi-modal fundus fusion and lesion-guided attention (see below). This weakens the positioning and makes it harder to see how different the proposed system is from the latest attention-based fusion or graph-based multi-view models.

## Potentially Missing Related Work

1. **Xu et al., “Prior-Guided Attention Fusion Transformer for Multi-Lesion Segmentation of Diabetic Retinopathy”, 2024.**  
   While focused on lesion segmentation rather than grading, this work introduces a prior-guided attention fusion module that integrates self- and cross-attention for DR lesions. It is highly relevant to the GALP design and the idea of prior/lesion-guided attention. It should be discussed in **Section 2** in the context of lesion-centric attention mechanisms, and the authors should clarify how GALP differs from or generalizes such prior-guided attention to grading without explicit priors.

2. **Li et al., “Learning to Fuse and Reconstruct Multi-View Graphs for Diabetic Retinopathy Grading”, 2026.**  
   This paper proposes a multi-view graph fusion framework for DR grading, also aiming to leverage complementary information from multiple views. It is directly comparable to LGRF as an alternative multi-view fusion mechanism. It should be added to **Section 2** and ideally compared against as an additional SOTA baseline on at least one dataset (or discussed if code/data are unavailable), with a discussion of how graph-based vs. token/attention-based fusion differs.

3. **Huang et al., “Multi-Modal and Multi-View Fundus Image Fusion for Retinopathy Diagnosis via Multi-Scale Cross-Attention and Shifted Window Self-Attention”, 2025.**  
   This work introduces a multi-modal and multi-view fusion model that uses multi-scale cross-attention and windowed self-attention, very much in the same design space as LGRF. It should be cited in **Section 2**, with a comparison of how LGRF’s expert routing and top‑K proposal restriction compare to multi-scale cross-attention, especially since both target similar goals of localizing salient regions and aggregating multi-view evidence.

4. **Luo et al., “Provenance-Enabled Multi-View Diabetic Retinopathy Diagnosis Through Interpretable Process Mining”, 2025.**  
   This work proposes an interpretable, provenance-enabled framework for multi-view DR diagnosis, focusing on interpretability and explicit reasoning about multi-view evidence. Given that this paper also emphasizes interpretability via lesion proposals and expert guidance, it should be discussed in **Section 2**, with an explicit comparison of interpretability mechanisms and clinical usability.

(If some of these methods cannot be directly compared experimentally due to unavailable code or different data, that should be clearly stated, but they should still be discussed to better position the proposed approach.)

## Questions

1. **Clarification of CAM computation and notation in Equation (3).**  
   Please clarify the role of the index \(\tilde{c}\) in \(\mathbf{w}^{(\hat{\mathbf{y}}^{i}_{s_n})}_{s_n,\tilde{c}}\). Is this simply a typo for \(c\), i.e., using the class weight vector over channels, or is there an additional mapping or normalization step? A more explicit expression (e.g., \(\sum_c w_c f_c(u,v)\)) would help.

2. **Sign and behavior of the load-balancing loss \(\mathcal{L}_{\mathrm{load}}\).**  
   As written in **Equation (11)**, minimizing \(\mathcal{L}_{\mathrm{load}}\) seems to either push routing scores or utilization toward zero or produce unintended behavior. Could you derive the intended objective more explicitly and show that it indeed encourages a more uniform expert usage (e.g., by relating it to a variance or KL term)? If there is a missing negative sign or normalization, please correct it.

3. **Disentangling the impact of auxiliary supervision vs. proposal selection.**  
   Could you include (or at least discuss) an ablation variant with auxiliary classifiers but *without* top‑K proposal selection (i.e., all tokens used in LGRF) and another with proposal selection but no auxiliary loss? This would clarify how much of the gain in **Table 4** is attributable to deeper supervision versus the specific GALP proposal mechanism.

4. **Qualitative examples of lesion proposals and cross-view fusion.**  
   Can you provide qualitative visualizations (e.g., as additional figures) overlaying GEMs and selected proposal regions on fundus images, especially on MFIDDR where lesion masks exist? Also, examples of cross-view attention maps from LGRF would be helpful to validate that the model actually focuses on clinically meaningful lesions rather than spurious patterns.

5. **Training protocol and hyperparameter selection.**  
   How were hyperparameters such as \(\lambda_{\mathrm{load}}\), patch size \(q\), and the MoE configurations chosen? Was there a separate validation set for MFIDDR and DRTiD, or was cross-validation used? Please clarify whether **Figure 3** uses validation metrics or test metrics to select the final configuration.

6. **Complexity and scalability.**  
   Could you report approximate training and inference times (or FLOPs / parameter counts) for your model compared to a strong baseline like MVCINN or LFMVDR? Also, how does complexity scale with the number of views \(N\), given that LGRF currently fuses only adjacent views? This would help assess practical deployability.

7. **Details of the lesion-assisted variant (Ours with lesion).**  
   Please detail how SPADE is integrated (which layers, what conditioning inputs), and whether lesion maps are required at inference. Is the “with lesion” variant still using GALP + LGRF, or does SPADE replace some of these components? Clarifying this would help interpret the SOTA numbers in **Table 1** and **Table 2**.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The architecture and training objective are mostly sound and clearly specified, and the empirical results support the main claims. However, some mathematical details (especially load-balancing loss) and the disentanglement of what drives performance gains need clarification.

## Presentation Rating

3: good.  
The paper is generally well written, with clear figures (**Figure 1**, **Figure 2**) and tables (**Tables 1–4**), but a few key implementation and notation details are missing or ambiguous, and the qualitative/interpretability story is underdeveloped.

## Contribution Rating

2: fair.  
The contribution is valuable within the niche of multi-view DR grading and shows strong empirical performance without external annotations, but the conceptual novelty is moderate, primarily combining known elements (CAM-based proposals, MoE routing, cross-view attention) rather than introducing a fundamentally new learning principle.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper presents a well-executed, empirically convincing system that advances annotation-free multi-view DR grading and is relevant to representation learning for medical imaging. The main limitations are modest conceptual novelty, limited qualitative and analytical depth, and some unclear mathematical/implementation details. With clarifications and additional analysis in rebuttal, it would be a reasonable accept; as it stands, I lean slightly positive but see room for strengthening.

## Reviewer Confidence

4: confident.  
I am familiar with multi-view representation learning and lesion-guided DR grading, and I have carefully checked the method and experimental setup, though I did not reproduce results.