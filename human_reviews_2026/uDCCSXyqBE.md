# Dissecting Representation Misalignment in Contrastive Learning via Influence Function

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 2, 6

## Abstract
Contrastive learning, commonly applied in large-scale multimodal models, often relies on data from diverse and often unreliable sources, which can include misaligned or mislabeled text-image pairs. This frequently leads to robustness issues and hallucinations, ultimately causing performance degradation. Data valuation is an efficient way to detect and trace these misalignments. Nevertheless, existing methods are computationally expensive for large-scale models. Although computationally efficient, classical influence functions are inadequate for contrastive learning models, as they were initially designed for pointwise loss. Furthermore, contrastive learning involves minimizing the distance between positive sample modalities while maximizing the distance between negative sample modalities. This necessitates evaluating the influence of samples from both perspectives. To tackle these challenges, we introduce the Extended Influence Function for Contrastive Loss (ECIF), an influence function crafted for contrastive loss. ECIF considers both positive and negative samples and provides a closed-form approximation of contrastive learning models, eliminating the need for retraining. Building upon ECIF, we develop a series of algorithms for data evaluation, misalignment detection, and misprediction trace-back tasks. Experimental results demonstrate that our ECIF advances the transparency and interpretability of CLIP-style embedding models by offering a more accurate assessment of data impact and model alignment compared to traditional baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes ECIF, which adapts influence functions to contrastive learning by modeling both positive and negative sample roles. ECIF enables efficient data valuation, misprediction trace-back, and misalignment detection without retraining. Experiments demonstrate several applications of the method, and show it achieves similar accuracy to retraining with lower cost, outperforming existing data attribution methods.

### Strengths
- The paper proposes a clear and well-motivated extension of influence functions to contrastive learning. The idea is novel and fills a real gap in the literature.

- The paper demonstrates several useful applications (data valuation, misalignment detection, and misprediction trace-back). I see potential for this method to have real practical implications.

- Experiments are reasonably thorough and show that ECIF can approximate retraining closely while being faster.

- The empirical results also indicate that ECIF performs substantially better than previous baselines, which supports the main claim that existing methods are not directly suitable for contrastive objectives.

### Weaknesses
- I didn’t find an analysis for Figure 1(b). Looking at the plot, I think the expected pattern should be that ECIF results in lower accuracy than random in Figure 1(b), yet this pattern is not consistently observed across the ratios.

- The paper would benefit from including quantitative results (e.g., precision, recall, AUROC, etc.) to show how well the method identifies mislabeled training samples across different noise ratios and random seeds.

### Questions
Most experiments focus on identifying harmful or low-quality data. I wonder how well ECIF performs in finding the most valuable samples, when compared against other baselines.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes ECIF (Extended Influence Function for Contrastive Loss), an influence-function framework tailored to contrastive learning that decomposes the contribution of a training pair into its roles as both positive and negative samples. Formally, the paper defines the batch contrastive loss  and derives closed-form, Hessian-vector–based approximations for the “positive-IF” and “negative-IF” , then combines them to estimate parameter changes from data removal without retraining. ECIF is reported to approximate retraining accuracy closely while cutting runtime.

### Strengths
The method is novel and derives an explicit negative-sample term via softmax reweighting, improving attribution fidelity for contrastive losses. Provides closed-form positive/negative IF and ECIF decomposition.


Analyzes ECIF approximation under convexity, and discusses computation efficiency.

### Weaknesses
(1) The linear superposition of positive/negative influences is assumed but its validity for non-convex deep models is not justified.

(2) Experiments are restricted to models with LoRA, impacts for larger backbones or full parameters finetuning remain unknown.

(3) ECIF combines positive-IF and negative-IF, but experiments do not isolate their separate contributions to performance.

(4) Lack large-scale benchmarks(e.g. ImageNet), constraining claims of scalability to large multimodal datasets.

### Questions
Please see weaknesses. If some of the issues can be resolved, I will raise my score. Currently, I'm leaning towards 5.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ECIF (Extended Influence Function for Contrastive Loss), a method for quantifying data attribution in contrastive learning models like CLIP. The authors address the challenge that classical influence functions, designed for pointwise loss, cannot directly apply to contrastive loss which involves both positive and negative sample pairs. ECIF separately quantifies the influence of data points as both positive samples (matched pairs) and negative samples (non-matched pairs), providing closed-form approximations without requiring model retraining. The method is applied to three tasks: identifying influential data for fine-tuning, misprediction trace-back, and misalignment detection. Experiments on several vision datasets (FGVC-Aircraft, Food101, Flowers102, CIFAR-10/100) demonstrate that ECIF can approximate retraining results with 80-90% computational savings while effectively identifying harmful and valuable training samples.

### Strengths
Novel technical formulation: The dual-perspective approach (positive-IF and negative-IF in Definition 4.3) is a genuine technical contribution. The mathematical derivation for handling negative samples through the Taylor expansion approach (Section 4.2, Equations 3-4) is creative and appears technically sound.

Computational efficiency demonstrated: Table 2 shows concrete evidence of 2-3x speedup over retraining (e.g., 456s vs 1174s on FGVC-Aircraft) while maintaining comparable accuracy (22.77% vs 23.07%), which validates the practical utility of the approximation.

Comprehensive theoretical analysis: Theorem E.7 provides an error bound for the approximation, showing the error scales with O(|D*|²), which gives theoretical backing to when the method should be reliable.

### Weaknesses
Missing critical baseline comparisons: The paper completely ignores the substantial literature on data selection for contrastive learning and CLIP models. Works like DataComp (Gadre et al., 2023), LAION filtering methods, and quality-based data selection approaches (e.g., CLIP-score based filtering as briefly mentioned in G.2) have implicitly solved data valuation for contrastive learning at scale. The comparison to CLIP-score in Figure 4 is relegated to the appendix and shows only marginal improvements, raising questions about practical utility. The paper needs head-to-head comparison with:
- Quality-based filtering methods (aesthetic scores, CLIP-score thresholds)
- Diversity-based sampling methods
- Curriculum learning approaches for contrastive learning
- Recent data attribution methods like DataInf (Kwon et al., 2023) which the authors cite but don't compare against

Weak motivation for the problem: The introduction (lines 33-43) claims that "robust evaluation mechanisms for data quality remain lacking" but doesn't acknowledge that the CLIP training community has extensively tackled data quality through heuristic methods with proven success at billion-sample scale. Why is influence function-based attribution necessary when simpler methods work? The paper doesn't establish that existing approaches fail or are insufficient.

Limited evaluation framework: The experimental evaluation primarily tests whether ECIF can identify "random," "harmful," or "valuable" samples (Section 6.2-6.3). However:
- These categories are themselves defined using ECIF's task-related influence scores, creating circularity
- Any reasonable data selection method can be evaluated in this same framework by ranking samples and binning them into these categories. The paper doesn't show ECIF provides qualitatively different insights than simpler metrics

Scalability concerns not addressed: 
- All experiments use relatively small datasets (Food101: 101 classes x 1000 images; FGVC-Aircraft: 10K images). Modern contrastive learning operates at the scale of millions/billions of samples (LAION-5B, DataComp-1B).
-  The computational requirements of computing Hessian-related quantities (Equation 2, Algorithm 1 line 8) scale poorly. The paper acknowledges using LOGRA for efficiency (Appendix C) but provides no wall-clock time comparisons or scalability analysis.

Experimental methodology issues:
- Baseline methods (IF-EKFAC, TARK, TracIN) in Table 1 show suspiciously poor performance (e.g., 18.27% vs 23.50% for TARK on FGVC-Aircraft), suggesting potential implementation issues or unfair comparison
- The "harmful" data removal experiments (Figure 1a, Table 1) artificially create noise by mislabeling. Real-world data quality issues are more subtle
- No comparison of the types of samples identified as valuable/harmful between ECIF and simpler methods to show qualitative differences


Missing ablation studies:
- No ablation on the relative importance of positive-IF vs negative-IF components
- No sensitivity analysis on hyperparameters (regularization δ, projection rank in LOGRA)
- No analysis of when the method fails or performs poorly

### Questions
Mentioned in weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the performance degradation problem in contrastive learning caused by misaligned data. As classical influence functions cannot handle contrastive loss, this paper proposes ECIF, the first no-retraining data valuation tool designed specifically for such models. ECIF's core contribution lies in its dual-perspective analysis: it decouples a single data point's influence into its contribution as a positive sample and its contribution as a negative sample. This dual-evaluation mechanism enables ECIF to efficiently and accurately identify harmful, misaligned data and trace model errors, which is something traditional IF methods cannot do.

### Strengths
1. The paper addresses an extremely important and urgent problem in multimodal and contrastive learning how to efficiently detect and evaluate misaligned training data. This is the first no-retraining influence function designed specifically for the contrastive loss function.

2. It cleverly decouples the influence of a single data point into positive influence Positive-IF and negative influence Negative-IF, which is technically sound and directly confronts the fundamental challenge of why standard IF cannot handle contrastive loss.

3. The theory and experiments are well combined. The empirical results decisively validate the paper's core theoretical claim by showing that ECIF succeeds where all standard influence function baselines fail.

### Weaknesses
The paper's motivation is to develop a computationally efficient method applicable to large-scale models, but it lacks a clear analysis of the actual computational cost of ECIF. The paper omits runtime comparisons with baseline methods that it outperforms in accuracy.

### Questions
1. The theoretical error bound (Theorem E.7) relies on the assumption of convexity (Assumption E.3), but the contrastive loss is non-convex. Could you clarify why ECIF remains an accurate approximation in this practical, non-convex setting?

2. Proposition 5.3 appears to contain several typographical errors, making it difficult to understand. Could you please provide the correct mathematical formulation for this proposition?

### Soundness
3

### Presentation
2

### Contribution
3
