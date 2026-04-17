---
job_id: b4154fe1-806a-4b97-99f5-32625c631f7b
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 7L7kmHHfgf.pdf
paper: PIRN: Prototypical-Based Intra-Modal Reconstruction with Normality Communication for Multimodal Anomaly Detection
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new multimodal anomaly detection framework based on prototype-driven reconstruction, vector quantization, and optimal transport within ViT architectures, which is squarely within ICLR’s scope of representation learning, multimodal learning, and computer vision.

## Minimum Quality
Pass ✅.  
All major sections (Abstract, Introduction, Related Work, Method, Experiments, Results/Analysis, Conclusion) are present and reasonably complete. The method is technically coherent, experiments are extensive with strong baselines and ablations, and the exposition is clear enough to evaluate. I do not see fundamental methodological or statistical flaws that alone would warrant desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
Within the main paper content there are no instructions to reviewers, no hidden prompts, and no signs of manipulative text targeting automated reviewing systems.

---

# Expected Review Outcome:

## Summary

The paper introduces PIRN, a prototype-based framework for few-shot multimodal anomaly detection using RGB images and surface-normal maps. PIRN uses a vector-quantized prototype codebook per modality, a Balanced Prototype Assignment (BPA) based on entropic optimal transport to avoid prototype collapse, an Adaptive Prototype Refinement (APR) module that updates prototypes via OT-weighted GRU at test time, and a Multimodal Normality Communication (MNC) module that exchanges prototype-level normal cues across modalities. Experiments on MVTec 3D-AD, Eyecandies, and Real-IAD D3 show that PIRN improves over existing multimodal and prototype-based baselines, especially in few-shot regimes, while being computationally efficient.

## Strengths

1. **Clear, coherent architecture with nontrivial design choices.**  
   The overall framework in **Figure 2** is well-motivated and logically structured: frozen DINOv2 ViT encoders, a stacked prototype-aware decoder, per-layer APR → BPA → MNC, and final reconstruction-based anomaly maps. The design explicitly targets few-shot multimodal AD, rather than just throwing prototypes on top of existing fusion architectures.

2. **Balanced OT for prototype assignment is technically sound and empirically important.**  
   BPA formulates token-to-prototype assignment as a balanced OT problem (Eq. (1)), with equality constraints \(T \mathbf{1}_K = \mathbf{1}_N\) and \(T^\top \mathbf{1}_N = \frac{N}{K}\mathbf{1}_K\), and reconstructs tokens from prototypes via Eq. (2). This is a clean and principled way to enforce uniform prototype utilization and avoid the softmax-collapse issue.  
   Empirically, **Figure 1 (Right)** shows t-SNE plots where softmax assignment leads to underused prototypes, whereas BPA yields a much more spread-out codebook anchored across the normal feature manifold. **Table 9** further supports this: “Balanced Optimal Transport” outperforms softmax, linear, and sigmoid attention for assignment by a large margin in AUROC\(_I\) and AUPRO, indicating that the OT constraints are actually contributing meaningful performance gains, not just adding complexity for aesthetics.

3. **Adaptive test-time refinement of prototypes (APR) is well-specified and ablated.**  
   APR uses a second OT plan \(\Gamma^*\) and column-normalized weights \(\bar{\Gamma}^*\) to construct per-prototype contexts \(c_k = \sum_n \bar{\Gamma}^{*}_{nk} z_n\), and updates prototypes through a standard GRU (Eqs. (5)–(8) in Appendix B.1). The mechanism is mathematically straightforward, gating new context into prototypes while allowing them to stay near their pre-trained normal representation when the context is unreliable.  
   **Table 7** shows that APR with balanced OT aggregation slightly but consistently improves over no APR, global averaging, and top-\(k\) averaging for prototype updates. This gives concrete evidence that the added GRU + OT complexity translates into measurable robustness improvements, especially in few-shot settings with distribution shift between train and test.

4. **Cross-modal prototype communication is handled at a high level, not via brittle dense alignment.**  
   MNC’s two-stage process (Section 3.4) is conceptually appealing: first, build a graph over RGB and SN prototypes, refine them with a GAT, then use them as keys/values in cross-attention where queries are intra-modally “purified” tokens \(Z' = Z \cdot \sigma(Z^{\text{bpa}})\). This respects the paper’s claim that dense patch-to-patch alignment is unreliable in few-shot regimes and uses prototypical “normal” anchors instead.  
   **Table 2** convincingly isolates MNC: removing MNC from the full model (BPA+APR+MNC → BPA+APR only) degrades AUROC\(_I\) from 0.922 to 0.867 on MVTec 3D-AD (10-shot). This is a sizeable drop, suggesting that cross-modal normality injection is actually helping rather than just adding ornamentation.

5. **Strong empirical results with comprehensive baselines and ablations.**  
   - **Table 1** (MVTec 3D-AD and Eyecandies) shows consistent improvements across 5-, 10-, 30-shot and all-shot regimes. For example, at 10 shots PIRN attains AUROC\(_I\)=0.922 vs. the next-best INP-Former’s 0.885 on MVTec 3D-AD, and 0.912 vs. 0.872 on Eyecandies. The gains are not huge but systematic across datasets and metrics (image AUROC, pixel AUROC, AUPRO).  
   - **Table 8** (Real-IAD D3) shows that PIRN achieves the best average localization performance (AUROC\(_P\)=0.864) among listed methods and very competitive AUROC\(_I\), despite using only RGB+SN versus D\(^3\)M’s tri-modal inputs.  
   - Ablations on codebook size (Table 5), decoder depth (Table 6), modality usage (Table 3), assignment strategies (Table 9), and backbone choices (Table 10) provide a reasonably thorough picture of where performance comes from and of the sensitivity to hyperparameters.

6. **Qualitative analyses and visualizations directly support claims.**  
   - **Figure 3 (Left)** compares anomaly maps across methods and visually shows that PIRN tends to be sharper with fewer background false positives; **Figure 3 (Right)** demonstrates better separation between normal and abnormal score distributions across several categories.  
   - **Figure 4**’s displacement visualizations are an insightful diagnostic: normal tokens move only slightly in prototype space, whereas anomalous tokens require larger displacements. This provides an intuitive geometric justification for why prototype-based reconstruction produces a strong anomaly signal.  
   - **Figures 5, 7, and 8** highlight the complementarity between RGB and surface-normal branches: texture-only defects are mainly captured by RGB, geometric defects by SN, and the fused maps benefit from both, which directly justifies the MNC design.

7. **Nontrivial efficiency advantage.**  
   Despite using OT, GAT, and cross-attention, PIRN is computationally efficient relative to recent multimodal baselines. **Table 4** shows that at 10-shot MVTec 3D-AD, PIRN slightly edges FIND in AUROC\(_I\) (0.922 vs 0.921) but uses ~85% fewer FLOPs (103G vs 728G) and is 4.3× faster in latency. This makes the method more appealing for real-world deployment in industrial inspection systems.

8. **Positioning within prototype-based and multimodal AD is reasonably aware.**  
   The related work section correctly discusses 2D prototype-based AD (HVQ-Trans, RLR, DPDL, INP-Former, MemAE, template-guided restoration) and existing multimodal AD methods (CFM, LSFA, M3DM, SG-DM, 3D-ADNAS). The paper also points out that existing prototype-based methods are single-modal or rely on external memory, and cross-modal MAD typically uses dense alignment or memory, which motivates a prototype-centric MAD framework.

## Weaknesses

1. **Conceptual overlap with existing prototype-based and multimodal prototype works is underplayed.**  
   While BPA and APR are solid instantiations, the high-level idea of using prototypes plus OT for robust representation is not that far from prior prototype-based or OT-based schemes, and the paper’s novelty claims could be calibrated more carefully. In particular:
   - Section 2 mentions Mao et al. (2025) and INP-Former (Luo et al., 2025), but the connection to PIRN is treated somewhat superficially. There is no explicit comparison against prototype-centric *multimodal* AD works like UIP-AD (see Missing Related Work) that also learn unified prototypes across modalities.  
   - The idea of cross-modal prototype alignment via graph-based message passing in MNC Stage 1 resembles multimodal prototype transfer/aggregation approaches in other domains (e.g., prototype-level cross-modal transfer for segmentation or few-shot multimodal learning). The paper references some of these (Pahde et al., 2021; Tang et al., 2023; Huang et al., 2025) but does not clearly articulate what is conceptually new beyond applying them to MAD.

2. **Some ambiguity and potential redundancy between APR and BPA OT formulations.**  
   - Both APR and BPA solve seemingly similar balanced OT problems between tokens \(Z\) and prototypes \(P\), but the paper only explicitly writes Eq. (1) in Section 3.2 and then says “Similar to Eq. (1), we derive the OT plan \(\Gamma^*\)” in Section 3.3. It is not completely clear whether \(\Gamma^*\) and \(T^*\) share the same cost \(C\), regularization, and marginals, or whether they differ in a principled way (e.g., different entropic strengths, different marginals, or top-\(k\) cutoffs).  
   - As a result, APR and BPA risk looking like two near-duplicate OT solvers running back-to-back each layer. It would be helpful to mathematically clarify if they are indeed solving the *same* OT problem with different usages (APR uses column-normalized \(\Gamma^{*}\) for aggregation; BPA uses row-normalized \(T^{*}\) for reconstruction) or whether there are meaningful differences in constraints or regularization. Some justification of why two separate solves are needed rather than reusing a single OT plan would also strengthen the technical clarity.  

3. **Test-time prototype adaptation relies on fairly heuristic robustness arguments.**  
   APR is supposed to update prototypes at test time without corrupting normality when anomalies are present. The paper argues that:
   - Balanced OT will assign anomalous patches diffusely with low affinity to any prototype, hence their contribution to \(c_k\) is small.  
   - The GRU gates will close when \(c_k\) is unreliable, preventing prototype drift.  
   However, these are qualitative claims; there is no explicit anomaly-aware masking or theoretical guarantee. For instance, if anomalies occupy a large fraction of the object and share superficial similarity with certain normal modes, OT could still assign them with non-negligible mass.  
   **Figure 6** shows qualitative examples where APR works in the presence of large anomalies, but a more quantitative analysis (e.g., performance with increasing anomalous area percentages, or ablations where APR is disabled specifically under large anomalies) would make this robustness story more convincing. Right now, APR’s safety at test time is plausible but not rigorously substantiated.

4. **Some mathematical and notational details are sloppy or inconsistent.**  
   A few examples that matter for reproducibility and clarity:
   - In Section 3.2, the notation toggles between \(s_n^{\text{bpa}}\) and \(z_n^{\text{bpa}}\) / \(\mathbf{Z}^{\text{bpa}}\). Eq. (2) defines \(s_n^{\text{bpa}}\), but later the text refers to \(z_n^{\text{bpa}}\). It should be consistent that the reconstructed token is \(z_n^{\text{bpa}}\).  
   - The symbol “\(\mathrm{AUROC}_1\)” appears in some tables (e.g., Tables 4, 5, 6, 7, 9, 10) while the text largely uses \(\mathrm{AUROC}_I\). This looks like a typo but is repeated across the appendix, which could be confusing.  
   - Algorithm 1 (Appendix A) has multiple typos and mismatched variable names: e.g., lines 13 and 20 use “Zbpba” and “Zbpa”, line 17 uses “ZsN” and reuses Zbpba in both branches for purification, which does not match the main text where \(Z_{rgb}^{\text{bpa}}\) and \(Z_{sn}^{\text{bpa}}\) are distinct. This suggests small but nontrivial implementation details are left implicit and could hinder faithful re-implementation.  
   While these are not fatal errors, they are numerous enough that a careful reader must reverse-engineer the intended behavior.

5. **A few experimental comparisons are missing or limited given the claims.**  
   - In multimodal anomaly detection, the paper compares with BTF, AST, M3DM, CFM, 3D-ADNAS, and an adapted INP-Former, which is good. But methods that use multimodal prototypes/unified intrinsic prototypes for MAD (e.g., UIP-AD) are absent from the baselines. Given that PIRN’s main novelty is prototype-based multimodal MAD, a direct comparison (or at least a discussion) is needed to judge the incremental contribution.  
   - In Real-IAD D3 (Table 8), PIRN has the second-best average AUROC\(_I\) (0.873) compared to D\(^3\)M’s 0.890, although it achieves better average AUROC\(_P\). The narrative highlights the localization advantage but downplays the detection disadvantage. It would be useful to see a more balanced discussion, plus perhaps a few qualitative examples from Real-IAD to understand the failure modes.

6. **Limited analysis of failure cases and per-category discrepancies.**  
   **Table 11** shows that PIRN’s AUROC\(_I\) per category is not uniformly superior: for instance, it lags behind AST or CFM for some categories (e.g., Bagel: PIRN 0.971 vs. CFM 0.994 or Shape-Guided 0.986). The paper reports the mean but does not analyze which types of anomalies or object categories are challenging for PIRN and why.  
   A more nuanced error analysis would be valuable: for example, are very subtle texture anomalies with minimal geometric deformation harder, perhaps due to prototype granularity \(K=10\)? Or are large homogeneous regions a weakness of balanced OT constraints? Without such analysis, it is hard to know where PIRN is likely to fail in practice.

7. **Few-shot setting and data usage could be more precisely defined.**  
   The paper mentions “5-shot, 10-shot, 30-shot” few-shot regimes, but the exact meaning is only implicitly described as “number of normal examples per class.” It would be helpful to clarify:
   - Are shots sampled once and fixed across methods, or averaged across multiple random seeds?  
   - Are hyperparameters tuned on a separate validation split, or on the few-shot set itself?  
   - Do baselines use the same DINOv2 backbone and feature extraction settings?  
   Although the paper claims to ensure fair comparison, some important fairness details (e.g., whether AST and BTF are reimplemented with DINOv2 ViTs or kept as originally specified) are not spelled out. This is a common but still relevant concern when interpreting the reported margins.

8. **The information bottleneck story is more intuitive than quantified.**  
   The introduction repeatedly frames prototype-based reconstruction + small codebook size as realizing an “information bottleneck” effect (citing Alemi et al., 2017; Seo et al., 2023; Zhang et al., 2024b), but there is no quantitative or theoretical analysis of mutual information or compression.  
   **Table 5** partially supports the bottleneck intuition by showing that increasing \(K\) from 10 to 50/100 degrades performance, but this remains empirical and heuristic. Since the bottleneck framing is part of the conceptual pitch, either a more formal connection or a toned-down claim would be preferable.

9. **Some experimental details around OT are missing.**  
   Given that the OT plans are central, the implementation details matter a lot for reproducibility:
   - The entropic regularization strength, number of Sinkhorn iterations, and whether OT is computed in double precision or with any stabilization tricks are not reported.  
   - Complexity: per-layer, per-modality OT is \(O(NK)\) per Sinkhorn iteration, which in theory could be heavy. **Table 4** shows overall FLOPs/latency vs baselines, which is reassuring, but it would still be useful to know token counts \(N\), \(K\), and Sinkhorn settings to be confident that the efficiency comparison is apples-to-apples.

Overall, the weaknesses are mostly about positioning, clarity, and deeper analysis rather than fundamental method flaws. Still, they matter for a top-tier conference.

## Potentially Missing Related Work

1. **Peng, B., Xu, K., Pan, Y. (2025). “UIP-AD: Learning Unified Intrinsic Prototypes for Multimodal Anomaly Detection.”**  
   - *Why directly related*: UIP-AD explicitly proposes learning unified intrinsic prototypes for *multimodal* anomaly detection, which is extremely close in spirit to PIRN’s prototype-based multimodal MAD with cross-modal communication.  
   - *How to integrate*: This work should be discussed in Section 2 (Multimodal Anomaly Detection) as a prototype-based multimodal baseline, and ideally included in experimental comparisons or at least qualitatively contrasted in terms of prototype learning (unified vs per-modality + alignment) and reconstruction vs decision-based use of prototypes.

2. **Li, C., Zhou, S., Kong, J. (2025). “KAnoCLIP: Zero-Shot Anomaly Detection through Knowledge-Driven Prompt Learning and Enhanced Cross-Modal Integration.”**  
   - *Why directly related*: Although KAnoCLIP is zero-shot and uses vision-language models, its core contribution is improved cross-modal integration for anomaly detection. This is closely related to PIRN’s Multimodal Normality Communication, which also focuses on cross-modal knowledge transfer.  
   - *How to integrate*: KAnoCLIP should be briefly discussed in Section 2 as a complementary line of work that leverages language as another modality for anomaly detection, with a comment on how prototype-level cross-modal fusion in PIRN differs from prompt-based CLIP integration.

3. **Wang, W., Guo, J., Cai, Y. (2026). “Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection.”**  
   - *Why directly related*: This paper learns multi-modal prototypes for cross-domain few-shot detection. While the task differs (object detection vs anomaly detection), the multi-modal prototype design and few-shot considerations are closely aligned with PIRN’s goals.  
   - *How to integrate*: It would be appropriate to mention this work in Section 2 in the broader context of multimodal prototype learning under data scarcity, highlighting similarities (multi-modal prototypes, few-shot) and differences (detection vs AD, cross-domain vs normal-only training).

## Questions

1. **Clarification of APR vs BPA OT problems.**  
   - Are \(\Gamma^*\) in APR and \(T^*\) in BPA computed with *exactly* the same cost matrix \(C\), regularization coefficient, and marginal constraints \((\mathbf{a}, \mathbf{b})\)?  
   - If yes, why not reuse a single OT plan per layer and split it into row/column-normalized versions for reconstruction and context aggregation? If not, please specify the differences and motivate why two distinct OT problems are needed.

2. **Handling heavy anomalies in APR.**  
   - In Figure 6 you show qualitative robustness when anomalies occupy a large fraction of the object. Could you provide a quantitative experiment where the fraction of anomalous pixels is systematically varied (e.g., by synthetic masks or category selection) and compare performance with/without APR?  
   - Is there any evidence (e.g., distributions of update gate \(u_k\) in Eq. (8)) that GRU gates indeed close when context is anomalous?

3. **Prototype and OT hyperparameters.**  
   - What are the exact settings for the OT solver (entropic regularization strength, maximum iterations, convergence tolerance)?  
   - How sensitive is PIRN to these OT hyperparameters? Have you tried smaller/larger entropic coefficients, and does that change prototype utilization (e.g., measured as entropy of row/column distributions of \(T^*\)) or performance?

4. **Fairness of baselines’ backbone and feature usage.**  
   - For BTF, AST, M3DM, CFM, and 3D-ADNAS, do you re-implement them with the same frozen DINOv2 ViT-B/14 encoders and surface-normal generation pipeline as PIRN, or use their original backbones?  
   - If they use different encoders, can you add an experiment where at least one reconstruction-based baseline (e.g., M3DM or CFM) is re-run with DINOv2 to isolate the contribution of your decoder vs better features?

5. **Real-IAD D3 performance trade-offs.**  
   - In Table 8, PIRN has lower average AUROC\(_I\) than D\(^3\)M but higher AUROC\(_P\). Could you comment on how you view this trade-off in practical industrial settings? Are there categories where PIRN’s detection fails but localization is good (or vice versa), and what patterns do you see there?

6. **Stability of few-shot results.**  
   - Are the few-shot metrics in Table 1 averaged over multiple random seeds/shots, or from a single fixed split? Given the small number of training samples per class, variance could be high; some indication of standard deviations or confidence intervals would increase confidence in the reported gains.

7. **Computation budget breakdown.**  
   - Table 4 provides total FLOPs and latency. Can you roughly decompose where the compute goes (OT vs GAT vs cross-attention vs encoder), and confirm whether the two OT solves per layer per modality are the dominant cost? This would help others evaluate whether replacing OT with a cheaper approximate assignment is worthwhile.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A. The work is on industrial anomaly detection from RGB and 3D surface-normal data, with no human subjects, sensitive attributes, or obvious downstream misuse beyond standard caveats for automated inspection systems.

## Soundness Rating

3: good.  
The technical components (OT-based prototype assignment and refinement, GAT-based prototype alignment, cross-attention with gating, reconstruction losses) are standard and correctly used; the experimental methodology is solid with reasonable baselines and ablations. Some robustness arguments (APR under heavy anomalies) are more heuristic than rigorously proven, but not fatal.

## Presentation Rating

3: good.  
The paper is generally well-written, with informative figures and tables and a clear high-level narrative. However, there are several notational inconsistencies, small typos (e.g., AUROC\(_1\) vs AUROC\(_I\)), and some mismatches between Algorithm 1 and the main text that slightly hinder precise understanding.

## Contribution Rating

3: good.  
The paper offers a meaningful step forward in few-shot multimodal anomaly detection through a well-constructed combination of prototype-based reconstruction, balanced OT, test-time refinement, and cross-modal prototype communication, and demonstrates consistent empirical gains with efficiency advantages. The conceptual novelty is moderate rather than dramatic, and some closely related multimodal prototype works are missing, but overall the contribution is valuable to the community.

## Overall Rating

8: Accept, good paper (poster).  
The method is technically sound, empirically strong across three benchmarks (with clear improvements and good ablations), reasonably efficient, and addresses an important and challenging setting (few-shot multimodal AD) with a principled prototype-based approach. The main drawbacks are around positioning relative to closely related multimodal prototype works, some clarity/notation gaps, and mostly qualitative robustness arguments for APR. These are fixable issues that should not block publication.

## Reviewer Confidence

4: confident.  
I am familiar with anomaly detection, multimodal representation learning, and prototype/OT-based methods, and I carefully checked the main mathematical constructions and experimental tables. Some implementation and hyperparameter details are underspecified, but the overall assessment is unlikely to change drastically.