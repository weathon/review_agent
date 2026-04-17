# BioTamperNet: Affinity-Guided State-Space Model Detecting Tampered Biomedical Images

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
We propose BioTamperNet, a novel framework for detecting duplicated regions in tampered biomedical images, leveraging affinity-guided attention inspired by State Space Model (SSM) approximations. Existing forensic models, primarily trained on natural images, often underperform on biomedical data where subtle manipulations can compromise experimental validity. To address this, BioTamperNet introduces an affinity-guided self-attention module to capture intra-image similarities and an affinity-guided cross-attention module to model cross-image correspondences. Our design integrates lightweight SSM-inspired linear attention mechanisms to enable efficient, fine-grained localization. Trained end-to-end, BioTamperNet simultaneously identifies tampered regions and their source counterparts. Extensive experiments on the benchmark bio-forensic datasets demonstrate significant improvements over competitive baselines in accurately detecting duplicated regions. All source code and dataset will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces BioTamperNet, a new deep learning model for detecting tampering in biomedical images. The authors target the poor performance of existing forensic models on specific biomedical data (microscopy, blots, etc.). The model uses an "affinity-guided" attention mechanism, reportedly inspired by State Space Models, to capture similarities. This includes both a self-attention module for intra-image similarities and a cross-attention module for inter-image correspondences . The main contribution is a unified architecture designed to handle three distinct tasks: External Duplication Detection, Internal Duplication Detection, and Cut Sharp Transition Detection. The paper claims state-of-the-art (SOTA) results on the BioFors benchmark.

### Strengths
The model's strongest point is its unified design. The strategy to convert single-image tasks into "pseudo-pairs" (as shown in Fig. 4) is a clever idea, allowing a single Siamese architecture to handle all three major forgery types

The affinity-guided attention modules are well-thought-out. The design explicitly models the source-target relationship, which is fundamental to copy-move detection. Details like the spatial suppression kernel (Eq. 10) to handle high self-correlation and the bidirectional softmax (Eq. 11) show a careful approach to the problem 

On paper, the model shows impressive quantitative results, apparently outperforming all baselines by a large margin across all categories in Table 1

The ablation studies (Table 3) are thorough and effectively demonstrate the utility of each proposed component (the affinity block, the SSM, etc.) . The analysis of catastrophic forgetting during cross-modality fine-tuning (Table 4) is also a valuable addition

### Weaknesses
The model's greatest weakness is its complete reliance on synthetic training data. This raises serious doubts about its real-world performance. The model might just be overfitting to the specific artifacts of the synthetic generation pipeline (e.g., GAN artifacts, specific blending boundaries) rather than learning the semantic inconsistencies of real human-made forgeries.

The paper's claims of being "lightweight" are unsubstantiated and appear contradictory to the methodology. The architecture in Figure 5 is extremely complex. More importantly, the Affinity Block (Section 3.3) computes a full N*N affinity matrix (where N=H*W), which is a computationally prohibitive O(N^2) operation that SSMs are typically designed to avoid

The "pseudo-pair" generation strategy for IDD/CSTD is critically under-described. The paper only states images are "split into two parts" (Fig. 4). It is unclear if this is a naive centerline cut, a random cut, or a mask-guided split. This is a crucial detail, as a naive split could introduce new edge artifacts that the CSTD detector might be learning as a spurious cue.

A closer look at the results in Table 1 reveals a significant performance gap. For the EDD Microscopy task, the model achieves a high Image-level MCC (0.739) but a much lower Pixel-level MCC (0.487). This disparity suggests that while the model can successfully flag an image as "tampered," it struggles with the precise pixel-level localization of the forgery, which is arguably the core goal.

The architecture includes several arbitrary design choices that are not justified by ablation. For instance, the "Affinity-Guided Self-Attention" module uses three parallel AGSSM blocks whose outputs are averaged (Eq. 13). There is no explanation for why three blocks are used, or why this averaging approach is superior to a simpler, single block.

### Questions
1.Regarding the N*N affinity matrix: Can the authors clarify the computational complexity? How can the model be "efficient" if it computes this quadratic matrix? Please provide concrete metrics (FLOPs, parameters, inference time) comparing BioTamperNet to the ViT and CNN variants from Table 3.

2. Please provide a precise description of the "pseudo-pair" generation for IDD/CSTD. How are the images "split"? Did you investigate if this splitting process itself introduces artifacts that the CSTD detector might be learning?

3. Can the authors comment on the large discrepancy between Image-level (0.739) and Pixel-level (0.487) MCC for EDD-Microscopy in Table 1? Does this indicate a failure in precise segmentation, and if so, why?

4. What was the design rationale for using three parallel AGSSM blocks (Eq. 13)? Have the authors ablated this choice (e.g., using one or two blocks)?

5. What was the process for selecting the sigma^2 hyperparameter in the spatial suppression kernel (Eq. 10) ? How sensitive is the model's performance to this value?

### Soundness
2

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
4

### Summary
This article provides an original and empirically validated method for biomedical image tampering detection. But its clarity, interpretability, and wider comparability need to be further improved to enhance its readability and credibility.

### Strengths
This work presents a novel and unified framework for tampering detection in biomedical images. The key innovation lies in its integration of affinity-guided attention with state-space models, yielding a computationally efficient and high-performing architecture. Extensive experiments across diverse biomedical modalities and benchmarks demonstrate clear superiority over prior works

### Weaknesses
1、The affinity-guided attention mechanism is conceptually appealing, but lacks qualitative visualizations (e.g., attention heatmaps, affinity evolution) that illustrate how the model distinguishes genuine vs tampered regions.

2、The model’s training relies heavily on synthetic duplications and GAN-generated patches. This may cause domain shift when applied to real, unannotated biomedical images.

3、Some equations and architectural diagrams are dense, reducing accessibility for readers. Section 3 could be reorganized with clearer sub-figures and step-by-step pseudocode.

4、The paper primarily compares to traditional forensic or transformer methods, but omits recent vision SSMs (e.g., Vision Mamba, S4ND) that could provide meaningful context.

5、 The anonymous code link provided by the author returns a 404 error.

### Questions
1、How robust is BioTamperNet to real-world artifacts such as illumination changes, JPEG compression, or partial occlusion—especially when no duplicated texture is visible?

2、Could the authors provide interpretability analysis (e.g., affinity heatmaps, attention rollouts) to demonstrate what visual cues the model relies on when detecting duplications?

3、How does BioTamperNet generalize when trained purely on synthetic duplications but evaluated on unseen real manipulations, such as retouched microscopy or gel bands?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the critical and increasingly relevant problem of forgery detection in biomedical images. The authors propose a unified framework, BIOTAMPERNET, designed to detect various manipulation types, including copy-move, splicing, and inpainting.

### Strengths
The motivation is good, as the authors correctly identify the potentially severe long-term consequences of tampering in the biomedical domain, which could be as, if not more, damaging than in the natural image domain. The paper presents a commendable application and includes extensive comparisons against a range of existing methods.

### Weaknesses
Despite its promising direction, the manuscript suffers from several major flaws in its current form that significantly weaken its conclusions. These concerns relate to the validity of the experimental evaluation, the rigor of the theoretical claims, the completeness of the literature review, and the clarity of the methodology and figures. 

1. The evaluation protocol: The authors state that the BioFors dataset provides annotations in the form of bounding boxes, not precise pixel-level masks. However, the paper insists on using pixel-level segmentation metrics to evaluate the model's performance.

2. The literature review overlooks recent and highly relevant work in the field of scientific image forgery detection. For example, the paper fails to compare against [r1] URN (Uncertainty-guided Refinement Network), a method explicitly designed to expose image splicing in scientific publications. Furthermore, the associated SciSp dataset introduced in [r1] is not considered.

[r1] Exposing Image Splicing Traces in Scientific Publications via Uncertainty-guided Refinement. Patterns 2024.

3. The authors present Proposition 1 and a corresponding Proof to justify their Cross-View Duplication Detection mechanism. However, I must point out that the provided text is not a formal proof but rather an intuitive, high-level description of the module's intended behavior. It suffers from several critical flaws:

    (1) The entire argument relies on qualitative language ("emphasizes," "mainly attends to," "close to") and the use of an approximation symbol in the proposition itself. A formal proof would require precise definitions, equalities, or explicitly bounded inequalities, none of which are present. 

    (2) The proof's cornerstone is the assumption that "position i attends mainly to j." This is precisely what the module is supposed to achieve, and thus cannot be used as a premise. This argument borders on circular reasoning. It also fails to consider practical scenarios, such as the presence of multiple similar patches (distractors) that would dilute the attention weights, invalidating the claim that the output is "close to Linear(V_2(j))".

    (3) Key components like AGSSM_Self_Attn, Linear, and the "projected cross-view affinity" are treated as black boxes. Without formal definitions, the proposition is unverifiable.


4. The BioFors dataset is known to have a severe class imbalance between tampered and authentic samples. A naive training approach would likely result in a model that defaults to predicting "authentic," as this would easily minimize the training loss. The authors do not specify how they address this critical issue. Furthermore, other implementation details appear to be missing.

5. In Section 4.4.1, the authors claim that modality inconsistencies can cause a ViT to overfit or forget the learned patterns. This is a strong assertion made without sufficient evidence or rigorous analysis. For instance, CLIP, which is also based on a ViT architecture, successfully handles a wide range of modalities (e.g., RGB, depth, infrared) without the "forgetting" phenomenon described.

6. The organization of Figures 3 and 4 is confusing. At first glance, it appears that the same input is fed into BIOTAMPERNET and then produced as an output. The transition labeled "IDD TO EDD" in Figure 4 is difficult to understand without multiple readings. These figures should be redesigned to clearly illustrate the model's workflow, distinguishing between inputs, intermediate representations, and final outputs.

7. Several details in Figure 1 hinder comprehension. For example: (1) In the AFFINITY BLOCK, it is unclear if both V1 and V2 are inputs to the ELU function; the arrows do not clearly indicate this. (2) The diagram does not explicitly show how the data flow differs for IDD versus CSTD. The authors should revise Figure 1 with clearer arrows and layout to precisely trace the data paths for different forgery types.

8. The quality of the samples generated by the proposed Synthetic Training seems to be relatively low and may not always be semantically consistent with real biomedical images. This raises a question: could this online data augmentation technique also improve the performance of other baseline methods? An ablation study applying this augmentation to one or two baseline methods would be very insightful. It would help determine if the performance gain is a general benefit of the augmentation strategy or if it is uniquely synergistic with the proposed architecture.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes BioTamperNet, a unified framework for detecting duplicated regions in tampered biomedical images. The authors generate synthetic forgeries to train their model, as real forged training data is unavailable, and claim state-of-the-art performance on the BioFors test set of real manipulated images.

### Strengths
The main strength is that the paper proposes a unified framework that simultaneously handles EDD, IDD, and CSTD in biomedical images by converting single-image tasks into pseudo image pairs, enabling one model to support all three forgery detection tasks without architectural changes.

### Weaknesses
1) The paper has poor writing and organization. The training dataset description should move from Section 2 to the experimental setup.  
2) State Space Models are basic model to the proposed method. Section 2 should be renamed “Preliminaries” and include SSMs.  
3) The paper claims BioTamperNet uses affinity-guided SSM attention modules but gives no clear explanation of how they integrate with or adapt SSMs.  
4) Tables list baseline methods without citations which hurts readability and reproducibility.

### Questions
1) The paper claims to use State Space Models (SSMs), but the method only uses linear similarity (Equations 6–8) without key SSM components like selective scan or input-dependent parameters. Where exactly are SSMs used? If it’s just linear attention, why call it SSM-based?

2) All training data is synthetic (geometric transforms + GANs), but real biomedical forgeries may have complex patterns not covered by synthesis. Is there evidence that the high performance isn’t just due to accidental alignment between synthetic training and real test forgeries?

3) The ablation study is weak. Table 3 only removes modules but doesn’t test: (a) the benefit of the “pseudo-pair” strategy vs training separate models; (b) whether the proposed affinity-guided attention is better than standard linear attention.

### Soundness
2

### Presentation
1

### Contribution
2
