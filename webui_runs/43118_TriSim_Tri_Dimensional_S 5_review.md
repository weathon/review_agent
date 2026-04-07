
========================================================================
INDIVIDUAL REVIEWS
========================================================================

────────────────────────────────────────
HARSH CRITIC (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Section-by-Section Critical Review

### Title & Abstract
The title is descriptive and accurately reflects the paper's core contributions (tri-dimensional similarity, EVT, false-negative mitigation). The abstract clearly states the problem, the proposed solution (TriSim, 3D space, EVT, two tail strategies, intra-modal refinement), and claims of superiority. The abstract's claims are supported by the reported results. A minor point: the claim that TriSim is the *first* framework to exploit a 3D similarity space with EVT is a strong novelty claim that the rest of the paper must solidly defend.

### Introduction & Motivation
The introduction effectively establishes the importance of RSITR and the specific challenge of False Negative Samples (FNS) due to high intra-class similarity. The critique of single cross-modal similarity thresholds (due to semantic overlap and gaps) is well-motivated and illustrated in Figure 1. The transition to proposing a 3D similarity space as a more robust foundation is logical. The contributions are clearly listed. However, the connection between "cross-modal semantic gaps" (which should lead to low similarity) and their role in causing FNS (typically associated with high similarity) could be explained more clearly.

### Method / Approach (Section 3)
This is the core of the paper and requires careful scrutiny.

**3.1 & 3.2 (Problem Definition & EVT-guided Modeling):**
*   The construction of the tri-dimensional similarity space (img-img, img-txt, txt-txt) is novel and well-founded for the RS domain.
*   The application of Extreme Value Theory (EVT) to model the tail of this distribution for FNS identification is a creative and statistically principled approach.
*   **Major Concern 1: Justification of EVT Assumptions.** The paper states "the similarity distribution within this 3D space exhibits a pronounced long-tail pattern" and applies the Pickands–Balkema–de Haan theorem (GPD for exceedances). However, no empirical evidence (e.g., a Q-Q plot or mean excess plot) is provided to validate that the tail of the `min(τ_ij)` distribution (or the Mahalanobis distances) actually follows a GPD. This is a critical step for the validity of the entire EVT-based weighting scheme. The parameter estimation (Eq. 9) is standard, but the foundational assumption needs verification.
*   **Major Concern 2: Probabilistic Weight Derivation.** The derivation of the discard probability `p^d_ij = (p_g + F_GPD(y_ij) - 1)/p_g` (for selected samples) and its use in the Bernoulli sampling `r_ij` is not intuitively explained. Why is this specific formula appropriate? How does it translate the "tail probability" into a meaningful training weight? The link between the statistical model and the training dynamic needs more elaboration to ensure reproducibility and understanding.
*   The two complementary tail-selection strategies (Mahalanobis distance and extreme quantile) are sensible and well-described.

**3.3 (Intra-modal Guided Discrimination Optimization - IGDO):**
*   This module is innovative, aiming to refine features by highlighting regions discriminative for a specific FNS pair.
*   **Clarity Issue:** The description becomes hard to follow. The process of computing saliency vectors **b** and **b'** from attention matrices **a** and **a'** is clear. However, the subsequent steps for constructing the gain matrix are less so. The trainable MLP generates **m^g**, which is supervised by **m_DSR** (Eq. 18). Then, **a^g** is mentioned ("guide the learning of the gain matrix **a^g**") but its definition and how it is learned from **m^g** are absent. Eq. 17 shows MLP producing **m^g**, not **a^g**. The statement "**a^g** is subsequently integrated" implies **a^g** exists, but its learning process is a black box. This is a significant reproducibility gap.
*   The intuition—amplifying features unique to `v_i` but not salient in the true match `v_j`—is good, but the implementation details are opaque.

**Overall Methodological Assessment:** The high-level ideas (3D space, EVT, intra-modal saliency) are strong and novel. However, the paper falters in providing complete, rigorous justification and description for key steps (EVT assumption checks, probability weight derivation, gain matrix learning). For ICLR, where methodological novelty and rigor are paramount, these are substantial concerns.

### Experiments & Results (Section 4)
*   **Baselines & Comparison (Table 1):** The experimental setup is extensive, comparing against 25 methods across two benchmarks. The results show consistent and meaningful improvements, particularly on the more challenging RSITMD dataset for text retrieval. The claim of superiority is supported.
*   **Ablation Studies (Tables 2 & 3):** These are essential and well-executed. They clearly show the contribution of each main module (ETSM, IGDO) and the superiority of the combined tail-selection strategy (A+B) over baselines (CT, AT).
*   **Parameter Sensitivity (Figure 6):** A good analysis showing performance is not overly sensitive to the chosen hyperparameters (`q_u`, `β`, `p_g`) within reasonable ranges.
*   **Qualitative Analysis (Figures 3, 4, 5, 7):** The visualizations are effective. Figure 3 nicely illustrates the complementary selection strategies. Figure 4 provides a compelling visual argument for the 3D space over a 1D threshold. Figure 5 offers some interpretability for the IGDO module. Figure 7 shows reasonable retrieval results.
*   **Missing Analysis:**
    1.  **Computational Cost:** No discussion of the training or inference overhead introduced by the 3D similarity computation, EVT fitting, and IGDO module. This is important for practical adoption.
    2.  **Statistical Significance:** While improvements are shown, are they statistically significant? Reporting standard deviations over multiple runs would strengthen the claims.
    3.  **Failure Case Analysis:** The demo in Figure 7 shows an error. A brief discussion of typical failure modes would strengthen the paper.

### Writing & Clarity
The paper is generally well-structured and readable. The mathematical notation is mostly clear. The primary clarity issues are concentrated in Section 3.3 (IGDO), as noted above, where the process for obtaining and using the gain matrix **a^g** is ambiguous. Some equations have minor formatting artifacts (e.g., Eq. 5, 6), but these do not impede understanding.

### Limitations & Broader Impact
The conclusion summarizes the work but does not contain a dedicated "Limitations" section. This is a significant omission for an ICLR submission. The paper should explicitly discuss:
1.  The computational complexity of the method.
2.  The reliance on EVT assumptions and the sensitivity if these do not hold perfectly.
3.  Potential failure modes (e.g., when intra-modal similarities are also misleading).
4.  Any broader societal impacts (likely minimal for RS retrieval, but could be stated).
The absence of this discussion reduces the paper's completeness and scholarly rigor.

### Overall Assessment
This paper presents a novel and conceptually appealing framework (TriSim) for a well-identified problem (FNS in RS retrieval). The core ideas of using a 3D similarity space and EVT for tail modeling are innovative and have clear potential. The experimental results are comprehensive and demonstrate state-of-the-art performance. However, for acceptance at a top-tier venue like ICLR, the **methodological exposition must be strengthened**. Key concerns are the lack of validation for EVT assumptions, the insufficient justification/derivation of the probabilistic weights, and the unclear learning process for the gain matrix in the IGDO module. Furthermore, the omission of a limitations section is a notable weakness. The contribution is promising but currently stands on somewhat shaky methodological ground. **Major revisions** addressing these issues are required for the paper to meet ICLR's high bar for technical rigor and completeness.

────────────────────────────────────────
NEUTRAL REVIEWER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## Balanced Review

### Summary
This paper introduces **TriSim**, a novel framework for remote sensing image-text retrieval (RSITR) that aims to mitigate the influence of false negative samples (FNS). The core idea is to construct a tri-dimensional similarity space (image-image, image-text, text-text) and use Extreme Value Theory (EVT) to model the tail distribution of this space, identifying FNS as statistical anomalies. Two complementary tail-selection strategies (Mahalanobis distance and extreme quantile filtering) are proposed. Selected FNS are then softly weighted in a probabilistic triplet loss. A second module (IGDO) further refines features by using intra-modal saliency to guide a learnable gain matrix, amplifying discriminative regions. The method shows state-of-the-art performance on two RS benchmarks (RSICD and RSITMD).

### Strengths
1.  **Novel and Well-Motivated Technical Approach:** The formulation of a 3D similarity space to capture intra- and cross-modal relationships for FNS identification is a creative and well-justified solution to a recognized problem in RSITR (high intra-class similarity). The application of EVT for tail modeling is a principled statistical approach, moving beyond heuristic thresholds.
2.  **Strong Empirical Performance:** The experimental results are comprehensive and convincing. *TriSim* achieves new SOTA results on both datasets, outperforming a wide array of 25 baselines, including strong recent CLIP-based models like AIR and GLISA. The ablation studies (Tables 2 & 3) are thorough and clearly demonstrate the contribution of each component (ETSM and IGDO).
3.  **Clear and Insightful Analysis:** The paper provides excellent qualitative analysis. Figure 3 effectively visualizes the complementary nature of the two tail-selection strategies. Figure 4 offers a compelling visual argument for why the 3D space is superior to a simple cross-modal threshold. Figure 5 convincingly shows the learned discriminative masks.

### Weaknesses
1.  **Overstated Novelty Claim:** The claim of being the "first framework to exploit a tri-dimensional similarity space, combined with EVT" (Sec. 1) is somewhat overstated. While the specific combination for FNS in RSITR is novel, the core ideas of using multiple similarity views and statistical outlier detection for hard/negative sample mining have precedents in broader machine learning and metric learning literature. The introduction could better position the work relative to these foundational concepts.
2.  **Limited Discussion of Computational Overhead:** The framework introduces significant additional computation: constructing and processing the 3D similarity space for all pairs in a batch, fitting GPD distributions, and the IGDO module's MLP and mask supervision. While performance gains are clear, the paper does not analyze training/inference time or parameter count increases compared to strong baselines like AIR or GLISA, which is relevant for practical deployment.
3.  **Superficial Analysis of Text Retrievability Discrepancy:** The paper notes a performance discrepancy (stronger text retrieval vs. image retrieval on RSITMD) and attributes it vaguely to "limited diversity of textual descriptions" (Sec. 4.3). This observation is important but under-analyzed. A deeper discussion or experiment probing *why* the method is more effective for one retrieval direction would strengthen the analysis.
4.  **Reproducibility Gaps:** While hyperparameters are listed, key details for exact reproduction are missing: the specific architecture/dimensions of the "lightweight trainable MLP" for the gain matrix, the values of thresholds `ε` and `ε'` in Eq. 15, and the balancing parameter `λ` (Sec. 3.3). The lack of publicly released code (standard for ICLR) further impacts reproducibility.

### Novelty & Significance
**Novelty:** The work presents a **moderately novel** integration of established concepts (multi-view similarity, EVT for outlier detection, saliency guidance) into a coherent framework tailored for the RSITR problem. The primary novelty lies in the specific design and application to mitigate FNS in RS data, which has high intra-class similarity.
**Significance:** The significance is **moderately high**. The paper addresses a concrete, known challenge in RSITR and demonstrates a clear, measurable improvement over existing SOTA methods. The proposed techniques (3D EVT modeling) could inspire new approaches for handling ambiguous negatives in other contrastive learning tasks within remote sensing and potentially other domains with similar data characteristics.

### Suggestions for Improvement
1.  **Reframe Novelty Claims:** Temper the "first" claims in the abstract and introduction. Instead, emphasize the novel *application and integration* of EVT with a 3D similarity space to solve the specific FNS problem in RSITR, acknowledging related ideas in broader literature.
2.  **Add Computational Analysis:** Include a subsection or table comparing training time, inference speed, and parameter count against one or two key baselines (e.g., AIR, GLISA). Discuss the trade-off between performance gains and computational cost.
3.  **Deepen the Text-vs-Image Retrieval Analysis:** Conduct a simple analysis to test the given hypothesis. For example, compute the semantic similarity (e.g., using sentence embeddings) within the text corpus of RSITMD versus RSICD to quantify "diversity." Alternatively, discuss if the 3D space or IGDO module inherently favors one modality.
4.  **Enhance Reproducibility:** Provide full architectural details for the MLP and all hyperparameters (`ε`, `ε'`, `λ`, MLP dims) either in the main text or a clear appendix. A strong commitment to releasing code upon acceptance would be essential for an ICLR submission.
5.  **Improve Clarity on EVT Application:** While technically sound, the explanation of how EVT is applied (Sec. 3.2) is dense. Adding a brief, intuitive explanation in the main text about why the tail of the min(similarities) is modeled, and how the GPD parameters are estimated (e.g., MLE), would improve accessibility for a broader audience.

────────────────────────────────────────
SPARK FINDER (deepseek/deepseek-v3.2 via OpenRouter)
────────────────────────────────────────
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation comparing EVT to a simple fixed threshold in the same 3D space.** The paper claims EVT is essential for adaptive tail modeling, but without comparing to a simple percentile threshold (e.g., 90th percentile) on the same tri-dimensional features, it’s unclear if the statistical machinery is necessary or if a heuristic would work just as well.
2. **Direct comparison with state-of-the-art false-negative mitigation methods from the general vision-language domain.** The baselines are mostly RS-specific or general retrieval models. Methods like FNE [26] or others that explicitly model false negatives (e.g., via soft labels or noise correction) should be compared to isolate the benefit of the 3D space versus other mitigation strategies.
3. **Evaluation on a larger-scale RS dataset (e.g., RS5M).** The method is evaluated on two relatively small datasets (RSICD has ~11k images). To claim robustness and scalability—critical for ICLR—testing on a larger dataset like RS5M is necessary to show the method generalizes beyond curated benchmarks.
4. **Ablation on the intra-modal saliency module (IGDO) with and without the gain matrix.** The paper ablates IGDO as a whole, but does not disentangle whether the improvement comes from the saliency mask or the learned gain matrix. Removing the gain matrix and using the mask directly would clarify its contribution.

### Deeper Analysis Needed (top 3-5 only)
1. **Statistical validation that the tail of the 3D similarity distribution actually follows a GPD.** The paper assumes the excesses are GPD-distributed but provides no diagnostic plots (e.g., quantile-quantile plots, mean excess plots) to justify this assumption. Without this, the EVT modeling is not scientifically sound.
2. **Analysis of what types of false negatives are being caught by each selection strategy (Mahalanobis vs. quantile).** Are they capturing semantically related but mismatched pairs, or just visual/textual duplicates? A breakdown by semantic category or similarity pattern is needed to understand the mechanism.
3. **Computational overhead analysis.** The method requires computing triplet similarities for all pairs in a batch and fitting GPDs. The added time/memory cost relative to baseline contrastive learning should be reported, as efficiency is a practical concern for retrieval.
4. **Error analysis on retrieval failures.** Where does the method still fail? Analyzing cases where TriSim retrieves incorrect results despite FNS mitigation would reveal remaining limitations (e.g., due to extreme cross-modal gaps).

### Visualizations & Case Studies
1. **3D scatter plots of the full similarity space with tail samples highlighted.** Current visualizations are 2D projections (e.g., image-text vs. image-image). Showing the actual 3D distribution with the selected tail would make the selection process concrete and allow assessment of coverage.
2. **Side-by-side retrieval examples comparing TriSim with a strong baseline (e.g., AIR or GLISA).** The paper shows retrieval demos for TriSim only. Showing where TriSim succeeds and the baseline fails (or vice versa) on the same queries would directly evidence the benefit of FNS mitigation.
3. **Visualization of the gain matrix’s effect on attention maps.** How does the gain matrix actually modify the self-attention in the Transformer? Overlaying the gain values on image patches would show whether it truly amplifies discriminative regions.

### Obvious Next Steps
1. **Apply the method during pre-training of a vision-language model.** The false-negative issue is acute in contrastive pre-training. Fine-tuning a pre-trained backbone is less compelling; applying TriSim during pre-training of RemoteCLIP would be a stronger demonstration of impact.
2. **Extend to other cross-modal RS tasks (e.g., VQA, captioning).** The tri-dimensional similarity modeling could, in principle, help in any task suffering from semantic overlap/gaps. A pilot experiment on RSVQA would show generality.
3. **Release code and pre-trained models.** For reproducibility and to facilitate adoption, the authors should commit to releasing their implementation. Many ICLR papers are rejected for lack of reproducibility plans.

────────────────────────────────────────
POTENTIALLY MISSED RELATED WORK (deepseek/deepseek-v3.2:online via OpenRouter)
────────────────────────────────────────
Related work search was skipped.

========================================================================
FINAL CONSOLIDATED REVIEW (deepseek/deepseek-v3.2 via OpenRouter)
========================================================================

## Summary
This paper introduces TriSim, a framework for remote sensing image-text retrieval that mitigates false negative samples (FNS) by constructing a tri-dimensional similarity space (image-image, image-text, text-text) and applying Extreme Value Theory to model the tail of this distribution for FNS identification. It includes two complementary tail-selection strategies and an intra-modal saliency-guided feature refinement module. The method demonstrates state-of-the-art performance on two benchmarks, RSICD and RSITMD.

## Strengths
- **Novel integration of a tri-dimensional similarity space with EVT for FNS mitigation.** The construction of a 3D similarity space to capture intra- and cross-modal relationships directly addresses the core problem of semantic overlap and gaps in RS data, and the application of EVT provides a principled statistical approach to tail modeling, moving beyond heuristic thresholds. This is evidenced by the clear performance gains over threshold-based baselines in ablation studies (Table 3).
- **Strong and comprehensive empirical validation.** The method outperforms 25 diverse baselines, including strong recent CLIP-based models, on two standard RS benchmarks. The ablation studies (Tables 2 & 3) are thorough and convincingly demonstrate the contribution of each proposed component (ETSM and IGDO). Qualitative visualizations (Figures 3, 4, 5) effectively illustrate the complementary selection strategies and the model's interpretability.

## Weaknesses
- **Lack of validation for core EVT assumptions.** The paper applies the Generalized Pareto Distribution (GPD) to model the tail of the similarity distribution but provides no diagnostic plots (e.g., Q-Q plots, mean excess plots) to empirically justify that the data's tail actually follows a GPD. This is a significant methodological gap, as the validity of the entire EVT-based weighting scheme depends on this assumption.
- **Unclear learning process for the gain matrix in the IGDO module.** The description of how the gain matrix **a^g** is learned is ambiguous. While the paper describes generating a mask **m^g** via an MLP supervised by **m_DSR**, it does not explicitly define how **a^g** is derived from **m^g** or how it is integrated into the similarity matrix. The statement "guide the learning of the gain matrix **a^g**" and the equation **ã = a + λa^g** present **a^g** as a pre-existing entity, creating a reproducibility gap.
- **Missing analysis of computational overhead and scalability.** The framework introduces non-trivial additional computation: constructing the 3D similarity matrix for all batch pairs, fitting GPD distributions iteratively, and running the IGDO module's MLP. No discussion of training/inference time, memory cost, or parameter count relative to strong baselines is provided, which is important for assessing practical utility and adoption.

## Nice-to-Haves
- A direct comparison showing that the EVT-based tail modeling is superior to applying a simple fixed percentile threshold within the same 3D similarity space would strengthen the justification for the statistical machinery.
- A deeper analysis of the observed performance discrepancy (stronger text retrieval vs. image retrieval on RSITMD), perhaps by quantifying text corpus diversity or analyzing modality-specific benefits of the 3D space.
- Providing full architectural details (e.g., dimensions of the MLP in IGDO, values for thresholds ε and ε', λ) in an appendix to enhance reproducibility.
- A brief discussion of potential failure modes or limitations, such as performance when intra-modal similarities are also noisy.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength/Weakness about being the "first" framework:** The claim of being the "first" to combine a 3D space with EVT is a subjective novelty claim. The review should focus on the objective contribution (the novel application and integration for RSITR) rather than debating "firstness."
- **Weakness about missing statistical significance tests:** While reporting standard deviations over multiple runs is good practice, single-run evaluation on these established benchmarks is common in the field. This is a generic rigor request not specific to the paper's core claims.
- **Weakness about lack of comparison to general-domain FNS methods (e.g., FNE):** The paper already includes a wide range of RS-specific and CLIP-based baselines. Demanding comparison to every possible general-domain method is scope creep; the paper's contribution is evaluated effectively within its domain.
- **Weakness about not testing on larger datasets (e.g., RS5M):** The paper demonstrates effectiveness on two standard benchmarks. Requesting evaluation on additional, larger datasets is a "nice-to-have" for generalizability but not a core flaw.
- **Criticism about the connection between "cross-modal semantic gaps" and FNS:** The paper's Figure 1b and accompanying text explain that gaps cause low cross-modal similarity, leading to mismatched pairs being incorrectly retained as true negatives. This is a valid part of their problem motivation, not a confusion.

## Novel Insights
The paper's core novel insight is the framing of false negative identification as an extreme value detection problem within a constructed tri-dimensional similarity space. This allows the model to leverage intra-modal relationships (image-image and text-text similarities) as contextual signals to disambiguate whether a high cross-modal similarity indicates a true semantic match or a false negative due to partial overlap. This is a principled shift from relying on a single, fragile cross-modal threshold and directly addresses the unique challenges of semantic overlap and modality gap in remote sensing data.

## Suggestions
- **Add empirical validation for EVT assumptions.** Include diagnostic plots (e.g., a mean excess plot or a Q-Q plot against the fitted GPD) in the supplementary material to justify the use of the GPD for the tail distribution of `min(τ_ij)`.
- **Clarify the gain matrix learning process in Section 3.3.** Explicitly define how **a^g** is parameterized and learned (e.g., is it a direct output of the MLP, or a function of **m^g**?). A short paragraph or an additional equation would resolve the ambiguity and ensure reproducibility.
- **Include a subsection or table analyzing computational cost.** Report the extra training time per epoch, inference latency, and the number of added parameters compared to a strong baseline like AIR or the base RemoteCLIP backbone. This is essential for a complete systems contribution.

========================================================================
PREDICTED SCORE
========================================================================

Score: 5.1
Decision: N/A
Total Cost: $0.0270
