=== CALIBRATION EXAMPLE 31 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title "Parameter Release and Knowledge Reuse for Class-Incremental Semantic Segmentation" accurately reflects the core contributions. The abstract clearly states the problems (parameter competition, underutilized knowledge) and the proposed solution (DKD with minimization-maximization). Claims of "excellent balance" and performance "approaches that of joint learning" are supported by the results in Tables 1 and 2. The abstract is well-written and sets appropriate expectations.

### Introduction & Motivation
The introduction effectively frames the CISS problem, the limitations of replay and dynamic architecture methods, and the specific shortcomings of static-architecture KD methods (parameter competition, wasted knowledge). The motivational analysis in Fig. 1 is compelling. The core research question ("Can we release parameters and simultaneously reuse the acquired knowledge...") is clearly stated and directly addressed by the method. The contributions are listed concretely.

**Concerns:**
1.  The claim that KD "inherently preserves the old-knowledge distribution with minimal modification" and that "the acquired old knowledge is often ignored" seems to slightly mischaracterize the purpose of KD, which is precisely to *retain* old knowledge by constraining changes. The authors' point is that this rigid retention creates competition, but the phrasing could be more precise to avoid appearing to dismiss the intent of prior work.

### Method / Approach (Section 3)
This is the most critical section. The overall minimization-maximization framework is novel and well-motivated. However, several aspects require clarification and justification.

**3.2 (a) Parameter-driven minimization (`L_Min`):**
*   The parameter release via norm-based pruning (`τ=0.1`) is straightforward but **lacks strong justification**. Why is the L2 norm of a filter/unit a good measure of its "sensitivity" to old classes? The choice of threshold `τ=0.1` is empirical (Appendix C.5). More concerningly, pruning is performed on the *old model's parameters* (`θ_{t-1}`), and these pruned weights are then used to generate a target `y*_c` (Eq. 2). The claim that this "releases parameters" is valid in the sense that pruned weights are zeroed, but it's unclear how this directly alleviates competition in the *new model* (`θ_t`), which is trained from scratch/continued. The mechanism is indirect: the loss `L_Min` encourages the new model's output on old classes to match the (potentially degraded) output of the pruned old model, thereby theoretically freeing capacity. This needs clearer explanation.
*   Eq. 3: The notation `y_t(h,w)` is confusing here, as it previously denoted a predicted category label (Sec 3.1), but in this loss it must represent a vector of probabilities/logits for all classes at pixel (h,w). This should be clarified.

**3.2 (b) Laplacian-based projection estimation (`L_Esti`):**
*   This is the most **confusing and underspecified part** of the method.
    *   Eq. 4: The definition of the position map `P_t(h,w)` is problematic. The expression `∂²/∂w² ( y_c* (h,w) )` seems to take the second derivative of a scalar label `y_c*` with respect to spatial coordinate `w`. This is non-standard. The text says it computes the "second-order gradient to identify low-curvature regions," but the mathematical operation is unclear. Is `y_c*` a probability map? The subsequent term `|| f_t(h,w) - f_{t-1}(h,w) ||_2` suggests an intent to weight by feature difference. The description needs a rigorous reformulation.
    *   Eq. 5 & `L_pro`: The confidence map `C_t(h,w)` is defined as the cosine similarity between the old-class label vector `y_c*` and the current feature `f_t`. It's unclear why a one-hot (or pseudo-label) vector `y_c*` has the same dimensionality as the feature `f_t` for a dot product. This requires explicit dimensionality definitions.
    *   The overall purpose—finding regions where old and new knowledge can coexist—is intuitive, but the mathematical implementation is hard to follow and assess for correctness.

**3.2 (c) Entropy-induced optimization (`L_Max`):**
*   The use of marginal and conditional entropy to maximize shared knowledge is interesting and theoretically grounded. However, the practical implementation raises questions:
    *   `y_{t,c}^{(b)}` is defined as the "predicted probability" for a batch. Is this an average over pixels? For a segmentation model, outputs are per-pixel. How are batch-level probabilities `y_{t,c}` and `y_{t,c}^{(b)}` computed? Aggregation over spatial dimensions and batch must be explicitly defined.
    *   The claim that minimizing `L_Max` maximizes mutual information `I(Y_t; Y_{t-1})` (Appendix A.3) is technically true from the definition, but `Y_{t-1}` here is not a random variable from data; it's the old model's output distribution. The intuition of encouraging dependence on old knowledge is clear, but the information-theoretic interpretation should be carefully stated.

**Reproducibility:** The core algorithms are described, but the ambiguities in Sec 3.2(b) would hinder exact replication without the code.

### Experiments & Results (Section 4)
The experimental setup is comprehensive, following community standards on datasets (PASCAL VOC, ADE20K), protocols (overlapped setting), and metrics (mIoU). The comparison with state-of-the-art methods across **nine different incremental settings** is a major strength and demonstrates broad applicability.

**Strengths:**
1.  Tables 1 & 2 show consistent and significant improvements across almost all settings, especially on challenging many-step scenarios (10-1, 2-2). The claim of nearing joint-training performance is largely supported on ADE20K.
2.  Ablation studies (Table 3, Fig 6) convincingly show the contribution of each component (`L_Min`, `L_Esti`, `L_Max`).
3.  Qualitative results (Fig 4) and t-SNE visualizations effectively illustrate reduced confusion and better feature separation.
4.  The analysis of training convergence (Fig 5) and error analysis (Table 5/13) adds robustness.

**Concerns / Missing Analyses:**
1.  **Baseline Comparisons:** While comparisons are extensive, some very recent strong baselines (e.g., from CVPR/ICCV 2024) beyond MBS, Nest, and Adapter-T might be missing. The authors should justify the chosen comparison set.
2.  **Ablation on Pruning Strategy:** The parameter release component is central. An ablation studying *different pruning criteria* (e.g., magnitude, gradient-based importance) and its *impact on old-class performance immediately after pruning* (beyond Tab 11) would strengthen the method's foundation. Table 11 shows minor drops, but how does this affect final incremental performance?
3.  **Component Interdependence:** The ablation (Table 3) shows individual gains, but does not deeply analyze the *interaction* between `L_Min` (release) and `L_Esti`/`L_Max` (reuse). Does reuse work better precisely because parameters were released?
4.  **Computational Cost:** The claim of "no additional inference burden" is true and important. However, the 7-second/epoch training overhead (vs. CKD) is noted. A more detailed breakdown of training time per component would be informative.

### Writing & Clarity
The paper is generally well-written. The high-level narrative is clear. The major clarity issues are confined to the technical description of the Laplacian-based projection (`L_Esti`), as detailed above. Figures 2 and 3 are helpful for overview.

### Limitations & Broader Impact
The "Limitations" section is absent. A dedicated discussion is needed. Potential limitations include:
1.  The sensitivity of the pruning threshold `τ` and the hyperparameter `γ` (though explored in Tabs 4, 10).
2.  The performance on truly long sequences (e.g., 50+ steps).
3.  The method's reliance on a reasonably accurate old-model output (`y_c*`) for pseudo-labeling; performance under severe forgetting in earlier steps could degrade.
4.  The complexity and potential instability introduced by the `L_Esti` component.

The **Broader Impact** statement is generic but adequate. The **Ethics Statement** and **Reproducibility Statement** are thorough and meet expectations.

### Appendix
The theoretical analysis (Appendix A) attempts to provide gradients and justification. However, it often feels like a *post-hoc* derivation rather than a foundation guiding the design. For instance, the connection between the Laplacian derivation (A.2) and the actual implementation (Eq. 4) is tenuous. The additional experiments (C.1-C.7, D) are valuable and address some concerns (disjoint setting, per-class results, backbone generality).

### Overall Assessment
This paper presents a thoughtful and empirically powerful approach to CISS. The core idea of actively managing the parameter distribution via a minimize-maximize strategy is novel and well-motivated. The experimental results are extensive and strongly support the claims, demonstrating state-of-the-art performance across a wide range of settings. The main weaknesses lie in the **opaque formulation of the Laplacian-based projection component (`L_Esti`)** and the **somewhat heuristic justification for the parameter release mechanism**. While the results suggest the overall package works, the reviewers will likely demand clearer, more rigorous explanations for these parts. If the authors can successfully revise the method description to resolve these ambiguities and provide additional ablation studies on the pruning mechanism, this could be a strong contender for ICLR. As it stands, the solid empirical contribution is slightly hampered by presentation gaps in the methodological details.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes Distribution-based Knowledge Distillation (DKD), a novel method for class-incremental semantic segmentation (CISS). DKD addresses two identified limitations of standard knowledge distillation: parameter competition between old and new classes, and underutilization of previously learned knowledge. It employs a minimization-maximization strategy: minimizing the old knowledge distribution after releasing low-sensitivity parameters to free up capacity, and maximizing shared knowledge distribution via Laplacian-based projection estimation and entropy-induced optimization to reuse old knowledge for learning new classes. The method achieves state-of-the-art performance across multiple incremental settings on Pascal VOC and ADE20K, approaching the joint-training upper bound.

### Strengths
1.  **Clear Problem Formulation and Motivation:** The paper compellingly identifies and analyzes two core issues in CISS: parameter competition in static architectures and the waste of acquired knowledge. This is well-illustrated with conceptual diagrams (Fig. 1) and empirical analysis (Fig. 2), providing strong motivation for the proposed work.
2.  **Comprehensive and Rigorous Evaluation:** The experimental setup is extensive, covering nine diverse incremental settings (including challenging ones like 10-1 and 2-2) across two standard datasets (Pascal VOC, ADE20K). Comparisons are made against a wide array of recent SOTA methods using both ResNet and ViT backbones. The results are convincing, showing consistent and often significant improvements, with average performance nearing the joint-training upper bound (Tables 1, 2).
3.  **Strong Ablation and Analysis:** The paper includes thorough ablation studies (Table 3, 12) to validate each component of DKD (\(L_{Min}, L_{Esti}, L_{Max}\)). Additional analyses on hyperparameters (Table 4), error/robustness (Table 5, 13), backbone compatibility (Table 8), and component effectiveness provide solid evidence for the method's design choices and generalizability.
4.  **Theoretical Grounding:** The appendix provides a detailed theoretical analysis connecting the proposed losses to information-theoretic principles and optimization behaviors (e.g., gradient derivations for \(L_{Min}\), analysis of \(L_{Esti}\) via reverse triangle inequality, and interpretation of \(L_{Max}\) as maximizing mutual information). This strengthens the methodological foundation.

### Weaknesses
1.  **Limited Insight into "Low-Sensitivity" Parameter Release:** While the parameter release mechanism via a fixed norm threshold (\(\tau=0.1\)) is simple, the justification for what constitutes "low-sensitivity" and why this specific pruning strategy is optimal is somewhat superficial. The paper does not deeply explore alternatives (e.g., magnitude-based vs. gradient-based importance) or analyze the semantic impact of the pruned filters/units on old class representations. The claim that it does not cause significant forgetting (Table 11) is based on a single-step evaluation; its effect over many incremental steps warrants more discussion.
2.  **Complexity of Laplacian-based Projection Estimation:** The construction of the position map \(P_t(h,w)\) (Eq. 4) and confidence map \(C_t(h,w)\) (Eq. 5) is technically sound but adds conceptual and implementation complexity. The link between the second-order gradient of a distillation loss and identifying "low-curvature coexistence regions" could be more intuitively explained. The choice of hyperparameter \(\gamma\) is shown to be sensitive to the number of incremental classes (Table 4), suggesting the method requires careful tuning for different scenarios.
3.  **Overstated Claim on Inference Cost:** The abstract and introduction claim the method introduces "no additional inference burden." While this is true for the final deployed model, the training process involves computing Laplacian projections and additional loss terms (\(L_{Esti}, L_{Max}\)), which the paper notes incurs a ~13% training time overhead per epoch compared to baseline CKD. This trade-off between training cost and final performance should be more transparently acknowledged in the main claims.

### Novelty & Significance
**Novelty:** The core idea of actively "releasing" parameters from the old model's representation and systematically "reusing" old knowledge to guide new class learning is novel within the CISS literature. While knowledge distillation and parameter pruning are established techniques, their combination in this specific minimization-maximization framework, guided by distributional constraints and Laplacian-based estimation, represents a fresh and thoughtful contribution.

**Significance:** The work addresses fundamental challenges in continual learning (stability-plasticity dilemma, catastrophic forgetting) for the practically important task of semantic segmentation. The demonstrated performance, particularly its proximity to the joint-training upper bound across varied and challenging settings, indicates a meaningful advance. The method is also shown to be compatible with different backbones (ViT, ResNet). If reproducible, this approach could influence future research in CISS and potentially other incremental learning domains.

### Suggestions for Improvement
1.  **Deepen the Analysis of Parameter Release:** Conduct a more detailed study on the released parameters. For example, visualize the feature maps or channels that are pruned at different incremental steps to better understand what "low-sensitivity" knowledge is being shed. Compare the proposed fixed-threshold pruning with adaptive or learning-based masking strategies.
2.  **Simplify or Better Motivate the Laplacian Component:** Consider providing a more accessible, intuitive explanation for how the Laplacian-based position map facilitates knowledge reuse. Alternatively, explore if similarly effective guidance could be derived from a simpler heuristic (e.g., based on prediction entropy or feature consistency) to reduce complexity.
3.  **Clarify Computational Trade-offs:** Revise statements about "no additional inference burden" to accurately reflect the added training-time computation. A brief discussion on the training cost vs. performance benefit would provide a more complete picture for practitioners.
4.  **Expand Qualitative Error Analysis:** While qualitative results (Figs. 4, 7-10) show successes, include a dedicated analysis of failure cases or remaining confusion patterns. This would provide clearer boundaries for the method's effectiveness and highlight areas for future work.
5.  **Strengthen the Reproducibility Statement:** The reproducibility statement is good. To further enhance it, the authors could explicitly state the expected variance in results (beyond the 0.1 mIoU std. dev. mentioned) and provide estimated total training time (GPU-hours) for key experiments like 10-1 on VOC.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on parameter release mechanism:** No systematic study compares the chosen L2-norm pruning against other criteria (e.g., gradient-based importance, first-order Taylor). Without this, the core claim of effective "parameter release" is not substantiated. The fixed threshold τ=0.1 is justified by a small table but lacks analysis of its impact across layers or architectures.
2. **Comparison with efficient dynamic architecture methods:** The paper dismisses dynamic architectures due to inference burden but does not compare against recent parameter-efficient dynamic methods (e.g., prompt-based or adapter-based). This omission weakens the claim that a static architecture with DKD is inherently superior.
3. **Evaluation on more challenging CISS benchmarks:** Experiments are limited to Pascal VOC and ADE20K. To prove robustness, tests on more complex datasets (e.g., Cityscapes, COCO-Stuff) or under domain-incremental settings are needed. The generalizability claim is not fully supported.
4. **Detailed computation analysis of DKD components:** The paper notes a 7-second/epoch overhead but does not break down the cost of Laplacian projection, entropy calculations, or pruning. A FLOPs/memory footprint comparison is necessary to assess practical efficiency.

### Deeper Analysis Needed (top 3-5 only)
1. **Sensitivity analysis of parameter release:** It is unclear whether the released parameters are truly low-sensitivity for old classes. Analyzing the correlation between pruning and forgetting rate, and visualizing which filters/neurons are pruned per step, would validate the mechanism.
2. **Interpretability of knowledge reuse:** The position and confidence maps are derived mathematically but not analyzed visually or semantically. Showing example maps on images would clarify what "reusable knowledge" looks like and whether it aligns with intuitive spatial/feature reuse.
3. **Granular class confusion analysis:** The class similarity matrix and t-SNE are high-level. A per-step confusion matrix (especially for challenging class pairs) is needed to pinpoint where confusion is reduced and whether the method truly addresses the claimed problem.

### Visualizations & Case Studies
1. **Visualization of pruning effects across layers:** Plotting the distribution of pruned parameters per layer over incremental steps would show how release progresses and whether certain layers are more critical for retention.
2. **Failure case visualization:** The paper shows success cases. Displaying failures (e.g., where new/old classes are confused, or boundaries are poorly segmented) would provide a balanced assessment and highlight limitations.
3. **Overlay of position/confidence maps on input images:** Superimposing the computed position and confidence maps on sample images would demonstrate whether the estimated reusable regions correspond to semantically meaningful areas.

### Obvious Next Steps
1. **Integration with rehearsal strategies:** The method is purely regularization-based. Combining DKD with a small replay buffer (even a few samples per class) should be explored, as this is a common and powerful hybrid approach in continual learning.
2. **Extension to other dense prediction tasks:** The approach should be tested on class-incremental object detection or panoptic segmentation to demonstrate broader applicability beyond semantic segmentation.
3. **Ablation on backbone architecture:** The paper shows one experiment with ResNet101 but does not systematically compare ViT vs. CNN backbones. The effectiveness of DKD across different architectures (e.g., hierarchical vs. transformer) is not established.

# Final Consolidated Review
## Summary
This paper proposes Distribution-based Knowledge Distillation (DKD) for class-incremental semantic segmentation. DKD introduces a minimization-maximization strategy: it minimizes the old knowledge distribution after releasing low-sensitivity parameters to alleviate parameter competition, and maximizes shared knowledge distribution via Laplacian-based projection and entropy optimization to reuse old knowledge when learning new classes. The method achieves state-of-the-art performance across nine diverse incremental settings on Pascal VOC and ADE20K, with average results approaching the joint-training upper bound.

## Strengths
- **Comprehensive and compelling empirical validation.** The method is evaluated across nine challenging incremental learning settings (including many-step scenarios like 10-1 and 2-2) on two standard datasets. It consistently outperforms a wide range of recent competitors, and its average performance is shown to be close to the joint-training upper bound, particularly on ADE20K (Tables 1, 2).
- **Thorough ablation and analysis.** The paper provides extensive ablation studies (Table 3, 12) to validate each component of DKD. Additional analyses on hyperparameters, error robustness, backbone compatibility (showing effectiveness on both ViT and ResNet101), and training stability convincingly support the design choices and the method's generalizability.
- **Novel and well-motivated core idea.** The explicit identification of parameter competition and underutilized knowledge in standard knowledge distillation, addressed via a dedicated minimization-maximization framework for parameter release and knowledge reuse, constitutes a clear and novel conceptual contribution to the CISS literature.

## Weaknesses
- **Clarity and justification of the Laplacian-based projection component.** The construction of the position map \(P_t(h,w)\) (Eq. 4) and its connection to identifying "low-curvature coexistence regions" is technically presented but lacks an intuitive, accessible explanation. While the mathematics in the appendix provides a derivation, the operational logic of this component within the overall framework remains the most complex and least clearly motivated part of the method.
- **Somewhat heuristic foundation for the parameter release mechanism.** The parameter release via a fixed L2-norm threshold (\(\tau=0.1\)) is simple and shown to be effective, but the paper provides limited analysis justifying why this criterion best identifies "low-sensitivity" parameters for old classes. A comparison with alternative pruning criteria (e.g., gradient-based importance) or a deeper investigation into what knowledge is being shed would strengthen the methodological foundation.

## Nice-to-Haves
- A more detailed visualization or semantic analysis of the position and confidence maps could provide intuitive insight into what "reusable knowledge" regions correspond to.
- Exploring the integration of DKD with a very small replay buffer could be an interesting direction to push performance even closer to the joint-training upper bound.
- Extending the evaluation to other dense prediction tasks (e.g., incremental object detection) could demonstrate the broader applicability of the core ideas.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Claim that `y_c*` in Eq. 4 is a scalar label.** The paper defines `y_c*` as a vector (the output of the old model after parameter release), making the second-derivative operation on it valid in context.
- **Claim that the mutual information interpretation in Appendix A.3 is invalid because `Y_{t-1}` is not a data variable.** The paper's use of the old model's output distribution as a reference for the new model's distribution is a valid and clear formulation for the objective, even if the information-theoretic analogy is slightly informal.
- **Criticism that very recent baselines are missing.** The paper compares against multiple 2024 methods (e.g., MBS, Nest, Adapter-T), which is sufficient for establishing state-of-the-art comparison.
- **Suggestion that the training overhead is inadequately disclosed.** The paper explicitly mentions the 7-second/epoch training time increase in Section 4.2, which is a reasonable level of disclosure.

## Novel Insights
The paper's core insight is that standard knowledge distillation in a fixed-capacity network creates a suboptimal equilibrium: rigidly preserving old knowledge crowds the parameter space for new classes, while also wasting the potential for old knowledge to actively guide the learning of new concepts. The proposed DKD reframes this as a distributional management problem, introducing the novel dual strategy of dynamically *releasing* low-sensitivity parameters from the old knowledge distribution and proactively *reusing* old knowledge to shape the new distribution. This shifts the role of old knowledge from a static anchor to a dynamically adjustable resource, offering a fresh perspective on balancing stability and plasticity.

## Suggestions
- Revise Section 3.2(b) to provide a more intuitive, step-by-step explanation of how the Laplacian-based projection estimation works and why it is a sensible mechanism for identifying reusable knowledge. A brief, concrete example could be helpful.
- In the discussion or appendix, consider adding a short analysis comparing the chosen L2-norm pruning to at least one alternative importance metric (e.g., a first-order Taylor estimate) to better justify the design choice, even if the results are similar.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0]
Average score: 3.3
Binary outcome: Reject
