# Learning from Adversity: Semantic-Aware Mask Refinement through Adversarial Perturbation

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 2, 6

## Abstract
Despite significant advances in image segmentation, even state-of-the-art models produce masks with imperfect boundaries, semantic inconsistencies, and structural errors. Mask refinement addresses these limitations, yet current approaches rely on simplistic synthetic noise that fails to capture the complex error patterns of real segmentation models. We introduce Phoenix, a novel framework that leverages adversarial learning to generate semantically meaningful noise patterns and contrastive learning to model refinement relationships. Our approach consists of two key innovations: (1) Adversarial Mask Perturbation, which employs embedding attacks to create semantic-aware noise that mimics real segmentation errors, and (2) Contrastive Mask Refinement Learning, which establishes a tri-directional framework that ensures feature consistency within semantic regions while maintaining separation between classes. Experiments demonstrate that Phoenix significantly outperforms existing methods across diverse tasks, while consistently enhancing state-of-the-art segmentation models with substantial improvements.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Although recent advances have greatly improved image segmentation, even state-of-the-art models still produce masks with inaccurate boundaries, semantic inconsistencies, and structural artifacts. While mask refinement techniques aim to address these limitations, existing approaches typically rely on simplistic synthetic noise that fails to represent the complex error patterns found in real segmentation outputs. 
The authors present Phoenix, a novel framework that combines adversarial and contrastive learning for realistic mask refinement. Phoenix introduces two core components: (1) Adversarial Mask Perturbation, which employs embedding-level attacks to generate semantically meaningful noise resembling real segmentation errors, and (2) Contrastive Mask Refinement Learning, a tri-directional contrastive formulation that enforces feature consistency within semantic regions while maintaining clear separation across classes.

### Strengths
- The authors address an interesting and important topic.
- I appreciate that the authors go into more detail about the limitations of other papers in section 3.
- The method is well described and motivated - everything is easy to follow. 
- The approach of adaptive, threshold-controlled noise generation is useful.
- The datasets, evaluation metrics, and models were appropriately selected.
- The results (Tables 2 and 3) are very good compared to SOTA.
- The authors have conducted a large number of ablation studies and made good comparisons. The ablations address each component of the presented method.
- In the appendix, more complex datasets and zero-shot approaches were tested - so great additional results.
- Implementation details are presented, and the authors intend to make code available at a later date.
- The limitations are reflected in failure cases, and based on these and their findings, the authors outline the next possible steps.

### Weaknesses
- The related work section seems superficial to me, as if the authors are only focusing on the works they use as baselines.


Remarks: 
- The theoretical analysis of semantic distribution is a good idea, but I find the section insufficiently explained and would therefore perhaps only explain it in the appendix. 
- I think details for the point annotations are worth adding.
- A brief explanation of the challenging DIS task would be helpful.
- The order of the figures in comparison to when they are referenced in the text is somewhat confusing.

### Questions
- Regarding the novelty, are you the first to use adversarial perturbation for mask refinement?
- Why were the NB and PointWSSIS methods chosen as baseline models for (weakly) semi-supervised?

### Soundness
4

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
3

### Summary
The paper introduces Phoenix, a semantic-aware mask refinement framework that leverages adversarial perturbations and contrastive learning. The method consists of two main modules: (1) AMP, which generates semantically meaningful noise to simulate realistic segmentation errors; and (2) CMRL, a tri-directional contrastive framework aligning ground-truth, noisy, and refined masks.

### Strengths
- The idea of using adversarial embedding perturbations for generating realistic segmentation noise is interesting and technically sound.
- The framework is systematically evaluated across multiple segmentation settings, with comprehensive ablations and efficiency analyses showing thoughtful experimentation.

### Weaknesses
1. Is the proposed adversarial mask perturbation conceptually similar to augmentation strategies that add positive (or forward) perturbations to enhance robustness? If so, how does Phoenix differ in motivation or mechanism—does it essentially act as a form of task-specific adversarial regularization?
2. Are there cases where adversarial perturbations introduce unrealistic distortions that negatively affect training stability? If so, how are such cases identified or mitigated during training?

### Questions
1. Is the proposed **adversarial mask perturbation** similar to these augmentation strategies that add positive perturbations to improve robustness? Is Phoenix effectively a form of task-specific adversarial regularization?  
2. Are there cases where adversarial perturbations may introduce unrealistic distortions that harm training stability? If so, how are such cases handled or filtered?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new framework for segmentation mask refinement. The method proposes two primary contributions to generate semantic-aware noisy masks for training and optimize the refiner. The paper's narrative is framed around the AMP component, hypothesizing that this "adversarial noise" better mimics real segmentation errors than the morphological noise used by prior work.

### Strengths
1. The core premise of using adversarial attacks not as a test-time failure mode but as a training data generation mechanism is an interesting methodological direction for this problem.
2. The method is benchmarked against SAMRefiner, which also uses the SAM backbone.

### Weaknesses
1.  The paper attributes the performance gains to "Learning from Adversity" (AMP), as implied by the title, abstract, and introduction. However, it simultaneously introduces a second, complex, and independent contribution: the CMRL loss. It fails to experimentally disentangle the effects of these two new components, making it impossible to verify the central hypothesis.
2.  The ablation studies are inadequate and confusing. They fail to provide a clear analysis that isolates the individual contributions of AMP and CMRL.
3.  The large performance gap between all SAM-based methods and non-SAM methods suggests the ViT-H backbone itself is a dominant factor. The paper's narrative under-emphasizes this, attributing the improvement primarily to the noise generation strategy.

### Questions
1.  To evaluate the paper's central hypothesis, could the authors provide an ablation study that isolates the individual contributions of AMP and CMRL? 
2.  Could the authors please clarify the exact settings for the baselines in Tables 4c and 4d? Does the "Morp" setting (Table 4d) use the full CMRL loss?
3.  Could the authors comment on why the paper's narrative (title, abstract) is framed exclusively around the AMP ("Adversity") component?

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
4

### Summary
The paper introduces Phoenix, a framework for mask refinement in image segmentation, aiming to address persistent issues in mask boundaries, semantic consistency, and structural integrity. Phoenix augments the current state-of-the-art by leveraging adversarial perturbation in the embedding space (Adversarial Mask Perturbation, AMP) to generate semantically-plausible noise, and a tri-directional Contrastive Mask Refinement Learning (CMRL) loss to refine noisy masks. The approach is built atop the Segment Anything Model (SAM) and is extensively evaluated—outperforming strong baselines in semi-supervised, weakly-supervised, fine-grained, and zero-shot transfer settings across multiple datasets.

### Strengths
**Innovation in Noise Modeling:** The introduction of AMP, which generates mask perturbations via adversarial embedding-level attacks rather than naive morphological operations, provides a more challenging and semantically aligned training regime. As shown in Figure 2, adversarial noise patterns demonstrate a higher semantic correlation with segmentation errors compared to morphological noise.

**Powerful Tri-Directional Contrastive Loss:** The CMRL loss explicitly models relationships between ground-truth, noisy, and refined masks, encouraging foreground–background separation, intra-class consistency, and self-improvement. The design is mathematically well-motivated—Section 3.4 details feature-space losses that move beyond basic pixel-level objectives.

**Comprehensive Empirical Evidence:** The paper presents broad quantitative comparisons (see Tables 1, 2, and 3), consistently demonstrating strong improvements over state-of-the-art refinement strategies, especially SAMRefiner and SegRefiner. The results generalize across full-supervision, semi-/weak-supervision, and domain transfer tasks.

### Weaknesses
**Theoretical Clarity of AMP Mechanism:** While the AMP methodology is generally well-explained (see Algorithm 1), the theoretical analysis connecting adversarial perturbation in embedding space to task-specific semantic error diversity is somewhat hand-wavy. The justification for why embedding-level attacks yield realistic error patterns is primarily based on empirical distributions (see Figure 2c), and the claimed proportionality between gradient norm and local uncertainty relies heavily on assumed model properties. No ablation examines weaknesses of AMP, e.g., failure modes if embedding space is not predictive of real error patterns.

**Contrastive Loss Formulation Details:** The mathematical construction of the tri-directional loss in Section 3.4 (multiple feature regions derived from mask overlaps) is intricate but under-explained for readers unfamiliar with the domain. The paper would benefit from explicit algorithmic pseudo-code or stepwise explanation of how region masks and projection features are computed and batched in a practical setting. Furthermore, the mapping from features to projected space (projector $g$) is presented as an afterthought, with no architectural or optimization ablation provided.

**Ambiguities and Reproducibility Gaps:** Key implementation choices for both guiding mask selection and the adversarial update process are reported as “randomly chosen” or based on heuristics (Section 3.3 and Appendix B). For instance, Table 4d (mask perturbation method) and Table 8d (guidance mask ablation) demonstrate sensitivity to mask type, but practical selection protocols are not systematically explored. Minor but notable: several equations (especially for the loss in Section 3.4) lack definitive bounds for summations, and some index notations could be clarified for reproducibility.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
