=== CALIBRATION EXAMPLE 46 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title "MAP AS A PROMPT: LEARNING MULTI-MODAL SPATIAL-SIGNAL FOUNDATION MODELS FOR CROSS-SCENARIO WIRELESS LOCALIZATION" is descriptive and accurately reflects the core contributions (map prompts, foundation model, cross-scenario). The abstract succinctly states the problem, the two key innovations (cycle-adaptive masking, map-as-prompt), and the main results (SOTA, strong zero-shot generalization). Claims are specific and appear supported by the results presented later.

**Introduction & Motivation:** The introduction effectively surveys the evolution from model-based to data-driven and foundation model approaches, clearly identifying limitations of prior work (inadequate handling of signal periodicity, superficial geographic integration). The research gaps (Sec 1.1) are well-articulated and logically lead to the stated contributions (Sec 1.2). The contributions are clear and substantive, aligning with ICLR's emphasis on novel methodological advancements.

**Preliminaries (Section 2):** This section provides necessary background on channel modeling and problem formulation. The inclusion of the ray-tracing abstraction (Eq. 2) and the formal problem statement (Eq. 3) is excellent for clarity and reproducibility. The connection between the physical model and the proposed map integration is well-motivated. No major issues.

**Methodology (Section 3):** This is the core technical section.
*   **Overall Framework & Signal Representation:** The two-stage paradigm (pre-train + prompt-tune) is standard but appropriate. The signal preprocessing (Eq. 5) is sensible.
*   **Cycle-Adaptive Masked Modeling (Sec 3.3):** This is a key innovation. The motivation—preventing models from exploiting periodic shortcuts in CSI—is convincing. However, the technical description is slightly insufficient for full reproducibility. Equation 6 and the surrounding text describe the mask shape but do not fully detail how the critical parameters \(d_{\text{final}}\), \(j_0\), and \(w\) are *dynamically* computed from the "detected periodicity." While Figure 3 and Appendix B.4 (cross-correlation analysis) provide intuition, a more precise algorithmic description of the dynamic adaptation process in the main text would strengthen the section. The claim that this forces learning of "globally meaningful" representations needs more explicit validation in the experiments (e.g., via representation analysis).
*   **Geographic Prompt Tuning (Sec 3.4):** The "map-as-prompt" concept is novel and well-explained. The graph construction (Delaunay triangulation of building vertices and BS positions) and the GCN formulation are clear. The integration of the prompt token into the frozen transformer is correctly described. A significant concern is the **scalability and practicality** of the prompt generation. Constructing a graph from a detailed 3D building mesh (vertices) for every new scene during inference could be computationally heavy. The paper does not discuss the complexity or potential approximations for real-time deployment. Furthermore, the assumption of having a precise 3D mesh for any environment is a strong one, which is partially addressed by the 2D ablation but remains a key limitation.
*   **Task-Specific Adaptation (Sec 3.5):** The design for single and multi-BS scenarios is straightforward and appropriate.

**Experiments & Results (Section 4):** The experimental design is comprehensive, addressing the key questions.
*   **Datasets & Baselines:** Using DeepMIMO and WAIR-D is standard and good. The chosen baselines (OMP, CNN, SWiT, LWLM) represent relevant paradigms (model-based, supervised DL, SSL). This is fair.
*   **Main Results (Tables 1, 2):** Results are impressive, showing clear gains, especially in the challenging single-BS NLoS case. The inclusion of "SIGMAP (w/o map)" as an ablation within the main comparison is excellent. However, there is a **major inconsistency**: Equation 11, which describes an "NLoS-aware attention mechanism," appears abruptly in Section 4.2 and was **not mentioned or derived in the Methodology**. This is a significant omission. If this mechanism is a core part of the model, it must be detailed in Section 3. If it is merely a post-hoc analysis tool, it should be clearly stated as such and not presented as part of the result explanation.
*   **Ablation Studies (Sec 4.3, 4.4):** The ablation on masking strategies (Table 3) effectively demonstrates the benefit of the adaptive strategy. The map ablation (Table 4) is crucial and shows the value of 3D information. The observation that 2D maps retain most of the benefit is insightful for practicality.
*   **Generalization (Sec 4.5 & Table 5):** The zero/few-shot cross-dataset results are strong and directly support a key claim. The parameter efficiency (0.4% tuned) is compelling for a foundation model.
*   **Missing Analysis:** While quantitative results are strong, there is a lack of **qualitative analysis or case studies**. For instance, visualizing where the model succeeds/fails in complex NLoS scenarios, or analyzing the learned attention maps (especially related to the unexplained Eq. 11), would greatly enhance interpretability. An analysis of what the cycle-adaptive masking and geographic prompts are actually learning (e.g., via probing tasks or visualization of embeddings) would strengthen the claims about learning "meaningful representations" and "geometric constraints."

**Conclusion, Acknowledgments, Appendices:** The conclusion summarizes the work and points to valuable future directions. The reproducibility statement and detailed appendix (hyperparameters, dataset config, augmentation details) are commendable and meet ICLR standards. Appendix B.4 and C provide useful, domain-specific justification for the method's design.

**Writing & Clarity:** Overall, the paper is well-written and logically structured. The major clarity issue is the **unexplained Equation 11** in the results section, which creates confusion. Some figures are referenced (e.g., Fig 1, 5, 6, 8, 9) but not included in the provided text, which hampers assessment but is assumed to be a parser issue.

**Limitations & Broader Impact:** The paper lacks an explicit "Limitations" section. Key limitations that should be acknowledged include: (1) The reliance on accurate 3D/2D environmental maps, which may not be available in all settings (though future work with visual prompts is mentioned). (2) The computational cost and procedural complexity of generating Delaunay graphs from meshes for prompt creation during inference. (3) The use of simulated ray-tracing data (DeepMIMO, WAIR-D); while standard, performance on real-world, noisy CSI data may differ. A broader impact statement is also absent but could note positive applications (autonomous systems) and potential risks (surveillance).

### Overall Assessment

This paper presents a novel and well-executed foundation model approach for wireless localization, featuring two technically sound innovations: a periodicity-aware masking strategy and a novel geographic prompt-tuning mechanism. The experimental validation is thorough, demonstrating state-of-the-art and strong generalization results with high parameter efficiency, which aligns with ICLR's interest in foundational methods. However, the contribution is currently marred by a **significant methodological omission** (the unexplained NLoS-aware attention mechanism in Eq. 11) and a lack of explicit discussion of limitations, particularly regarding the practicality of map-dependent prompt generation. If the authors can integrate the missing methodological details and provide a more comprehensive analysis/limitations section, this paper represents a strong contribution suitable for ICLR. As it stands, these issues require revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes SigMap, a multimodal foundation model for wireless localization that introduces a cycle-adaptive masking strategy for self-supervised learning on Channel State Information (CSI) and a "map-as-prompt" framework for parameter-efficient fine-tuning using 3D geographic information. The model is designed to achieve robust and generalizable localization across diverse environments, demonstrating state-of-the-art performance on simulated datasets and strong zero-shot generalization to unseen scenarios.

### Strengths
1. **Novel and Well-Motivated Masking Strategy**: The cycle-adaptive masking strategy is a thoughtful adaptation of masked autoencoding to the wireless domain. It explicitly addresses the periodic shortcuts in CSI data by dynamically generating masks based on cross-correlation analysis (Section 3.3, Eq. 6). The ablation study (Table 3) demonstrates its clear advantage over fixed masking patterns.
2. **Effective and Parameter-Efficient Integration of Geographic Context**: The map-conditioned prompt tuning mechanism is a clever and practical innovation. It encodes 3D map information via a lightweight GNN and injects it as a soft prompt, allowing the pre-trained backbone to remain frozen (Section 3.4, Algorithm 1). This leads to significant performance gains (Tables 1, 2) while updating only a tiny fraction of parameters (0.7%), aligning well with the foundation model paradigm.
3. **Comprehensive and Rigorous Evaluation**: The paper provides extensive experiments across single- and multi-base station localization, includes detailed ablation studies on masking and map modalities, and, most importantly, evaluates zero-shot/few-shot generalization on completely unseen environments (Section 4.5, Table 5). The use of established datasets (DeepMIMO, WAIR-D) and standard metrics (MAE, RMSE, CDF) strengthens the empirical claims.

### Weaknesses
1. **Heavy Reliance on Simulated and Perfect Map Data**: All experiments are conducted on ray-traced datasets (DeepMIMO, WAIR-D). While this is common, the performance gap between having a perfect 3D mesh and no map is substantial (Table 1: 1.564m vs. 2.275m MAE). The model's real-world applicability is not tested, and the assumption of having accurate, detailed 3D building meshes available at inference time is a significant practical limitation that is only briefly acknowledged in the conclusion.
2. **Insufficient Analysis of the Learned Representations**: While the model achieves strong results, the paper provides limited analysis of *what* the foundation model actually learns during pre-training. There is no probing of the representations (e.g., visualizing attention maps, analyzing feature clusters) to validate the claim that the model learns "globally meaningful signal representations" or to explain how the geographic prompts modulate the model's reasoning.
3. **Limited Comparison with Truly General-Purpose Foundation Models**: The baselines are mostly specialized wireless localization models (LWLM, SWiT). A stronger baseline would involve adapting a large, general-purpose vision transformer (e.g., ViT) pre-trained on a massive dataset to the CSI data, to better isolate the benefit of the proposed domain-specific innovations (cycle-adaptive masking, wireless-specific architecture) from simply using a larger, more powerful generic backbone.

### Novelty & Significance
The core ideas—cycle-adaptive masking for periodic signals and map-as-prompt for geometric conditioning—are novel contributions to the wireless localization literature. The work successfully translates concepts from NLP/vision foundation models (masked modeling, prompt tuning) to the wireless domain with thoughtful, physics-informed adaptations. The demonstrated parameter efficiency and cross-scenario generalization are significant steps toward practical, scalable localization systems for 6G. The paper meets ICLR's bar for a clear, novel methodological advance supported by solid experimentation.

### Suggestions for Improvement
1. **Conduct a Real-World Pilot Study**: To address the simulation gap, collect and evaluate on a small-scale real-world dataset (even if limited), or perform a sensitivity analysis using the ray-tracing simulator with added noise, occlusions, and map inaccuracies to better understand performance boundaries.
2. **Deepen the Interpretability Analysis**: Include visualizations or quantitative analyses to show how the model uses the geographic prompts. For instance, analyze the attention weights between the map prompt token and CSI tokens in different (LoS vs. NLoS) scenarios, or use perturbation studies to show which map features are most critical.
3. **Strengthen the Baseline Comparison**: Add comparisons with strong, generic foundation models (e.g., MAE-pre-trained ViT) applied to the CSI data, possibly with simple adaptations. This would help the community better appreciate the necessity of the proposed domain-specific components.
4. **Expand the Discussion on Map Availability**: The future work mentions using images or point clouds when 3D maps are unavailable. This critical direction should be expanded in the discussion or limitations section, outlining concrete strategies and the associated challenges (e.g., cross-modal alignment).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Lack of Real-World Validation**: All experiments use synthetic ray-tracing datasets (DeepMIMO, WAIR-D). To support claims of practical applicability, validation on at least one real-world measured CSI dataset is essential. Without it, the performance in real environments is unproven.
2. **Incomplete Baseline Comparison**: The paper does not compare against recent foundational models cited in related work (e.g., LWM, WirelessGPT) in the main localization tasks. To substantiate SOTA claims, direct comparison with these contemporary SSL/foundation models is necessary.
3. **Ablation on Map Robustness**: The map prompt ablation only compares 2D vs. 3D maps. A critical missing experiment tests the model's sensitivity to map inaccuracies (e.g., missing buildings, positional noise) to assess real-world viability where perfect maps are unavailable.
4. **Cross-Domain Generalization with Varying BS Configurations**: The zero-shot test uses unseen *environments* but with similar BS setups. An experiment generalizing to unseen and diverse BS configurations (e.g., different antenna counts, array geometries) is needed to validate the core claim of configuration-agnostic generalization.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of Learned Representations**: The paper claims cycle-adaptive masking learns "globally meaningful signal representations." This requires proof via analysis (e.g., visualizing attention maps, probing task performance on intermediate features, or clustering analyses) to show the model captures propagation physics rather than shortcuts.
2. **Interpretability of Map Prompts**: The "interpretable fusion" claim for geographic prompts is unsupported. An analysis is needed to show *how* map information influences predictions (e.g., via attention visualization between prompt tokens and CSI tokens, or saliency maps linking predictions to specific map structures).
3. **Failure Mode Analysis**: No analysis of where and why the model fails (e.g., error correlation with specific environmental features like dense NLoS, distance from BS). This is crucial for understanding limitations and trusting the robustness claims.

### Visualizations & Case Studies
1. **Qualitative Error Visualization on Maps**: Overlaying localization predictions and errors on the corresponding 2D/3D map for specific challenging cases (e.g., deep NLoS, urban canyons). This would visually demonstrate the value of map integration and expose systematic failure patterns.
2. **Case Study on Prompt Influence**: A side-by-side comparison, for a few complex scenes, of the model's attention patterns and predicted paths (LoS/NLoS) with and without the map prompt. This would concretely illustrate how the prompt resolves ambiguity.

### Obvious Next Steps
1. **Incorporate Real-World Data**: Given the synthetic-only evaluation, the most immediate next step is to fine-tune and evaluate on a public real-world CSI localization dataset (e.g., from previous data-driven localization papers) to bridge the sim-to-real gap.
2. **Proper Ablation of Prompt Design**: The paper should have included an ablation of the GNN-based prompt encoder itself (e.g., comparing to simpler map encodings like rasterized images or point clouds) to justify its architectural complexity.
3. **Quantify Parameter Efficiency Against Full Fine-Tuning**: The parameter efficiency claim (0.7% updated) is presented but not compared to the baseline of full fine-tuning the backbone. A direct comparison of performance vs. tuned parameters is needed to validate the efficiency advantage.

# Final Consolidated Review
## Summary
This paper proposes SigMap, a multimodal foundation model for wireless localization that introduces a cycle-adaptive masking strategy for self-supervised pre-training on Channel State Information (CSI) to prevent exploitation of periodic shortcuts, and a novel "map-as-prompt" framework that integrates 3D geographic information via lightweight GNN-generated prompts for parameter-efficient cross-scenario adaptation. The model demonstrates state-of-the-art accuracy on simulated datasets and strong zero-shot generalization to unseen environments.

## Strengths
- **Novel, domain-adaptive masking strategy:** The cycle-adaptive masked modeling is a principled innovation tailored to wireless signals. It dynamically generates masks based on detected CSI periodicity (via row-wise cross-correlation, detailed in Appendix B.4) to disrupt simple interpolation shortcuts. The ablation study (Table 3) confirms its superiority over fixed grid or strip masking.
- **Effective and parameter-efficient geographic conditioning:** The "map-as-prompt" mechanism is a novel and practical method for injecting environmental context. It encodes a 3D scene graph (building vertices and base stations) via a lightweight GNN to produce a soft prompt token, allowing the pre-trained backbone to remain frozen. This yields significant performance gains (e.g., 34.4% MAE improvement over the best baseline in single-BS NLoS) while updating only 0.7% of parameters (Table 5), aligning with the foundation model paradigm.
- **Comprehensive and convincing empirical validation:** The paper provides extensive experiments on standard ray-traced datasets (DeepMIMO, WAIR-D), covering single/multi-base station tasks, thorough ablations (masking strategies, 2D vs. 3D maps), and, most importantly, demonstrates strong few-shot/zero-shot generalization to entirely unseen environments (Table 4.5), substantiating the core claim of cross-scenario adaptation.

## Weaknesses
- **Heavy reliance on accurate simulated maps and data:** All experiments use synthetic, ray-traced CSI data and assume access to precise 3D (or 2D) building meshes at inference time. The performance gap between having a perfect map and no map is large (Table 1), and the model's robustness to real-world imperfections (noisy measurements, incomplete or inaccurate maps) is untested. This is a significant practical limitation for deployment, only briefly noted in the conclusion.
- **Insufficient analysis of learned representations and prompt mechanism:** While quantitative results are strong, the paper provides limited evidence for *how* the model achieves its gains. Claims that cycle-adaptive masking learns "globally meaningful signal representations" and that geographic prompts enable "interpretable fusion" are not substantiated with probing tasks, attention visualizations, or feature analysis. This lack of interpretability makes it difficult to validate the core learning mechanisms.
- **Missing explicit limitations and broader impact discussion:** The paper lacks a dedicated limitations section, omitting critical discussion of the computational overhead for constructing Delaunay graphs from dense meshes during inference, the strong assumption of map availability, and the sim-to-real gap. A broader impact statement regarding applications (e.g., autonomous systems) and potential risks (e.g., surveillance) is also absent.

## Nice-to-Haves
- Including a qualitative analysis, such as visualizing localization errors and attention patterns on example maps, would improve interpretability and help identify failure modes.
- A sensitivity analysis testing the prompt mechanism's robustness to map inaccuracies (e.g., missing buildings, positional noise) would better characterize its practical boundaries.
- A comparison against a strong, generic vision foundation model (e.g., a MAE-pre-trained ViT) adapted to CSI data could help isolate the benefit of the proposed domain-specific components from simply using a powerful backbone.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strength: "Well-written" or "topic is important"** - These are generic and apply to many papers.
- **Weakness: "Unexplained Equation 11 is a major methodological omission"** - This equation appears in Section 4.2 (Results) as part of explaining the model's attention behavior in NLoS scenarios; it is not presented as a core, previously undisclosed component of the methodology. The description of the model's architecture in Section 3 is complete.
- **Weakness: "Demand for real-world validation in the main revision"** - While a critical limitation, mandating new real-world experiments for acceptance is beyond typical review expectations. The weakness is acknowledged, but the suggestion is moved to "Nice-to-Haves."
- **Weakness: "Requirement for comparison against LWM or WirelessGPT"** - The paper does compare against LWLM (Large Wireless Localization Model), a directly relevant foundation model. The other cited works (LWM, WirelessGPT) are not primarily designed for localization, as noted in the introduction, making a direct comparison less central.
- **Weakness: "Need for ablation on GNN prompt encoder vs. simpler encodings"** - This is a useful architectural investigation but is not required to validate the core contribution of using map information as a prompt.

## Novel Insights
The paper's core novel insight is the translation of the "prompt tuning" paradigm from NLP to the geometric domain of wireless perception. It demonstrates that environmental constraints, encoded as a graph of spatial relationships, can be effectively distilled into a compact, learnable token that conditions a frozen signal foundation model. This enables parameter-efficient adaptation and provides a principled pathway for integrating heterogeneous side information (maps) into an end-to-end localization system. The complementary insight is identifying and mitigating periodicity-based shortcuts in CSI through adaptive masking, a domain-specific refinement of masked autoencoding.

## Suggestions
- Add a "Limitations" subsection to the conclusion, explicitly discussing the reliance on simulated data and accurate maps, the computational cost of prompt generation, and the sim-to-real gap.
- Strengthen the analysis in a revised version by including at least one form of representation probing (e.g., t-SNE plots of features with/without map prompts, or attention visualizations for a few key examples) to substantiate claims about what the model learns.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 6.0]
Average score: 5.3
Binary outcome: Accept
