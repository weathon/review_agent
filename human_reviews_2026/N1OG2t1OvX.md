# Semi-3DETR: Semi-Supervised Detection Transformer for 3D Object Detection

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
DETR-based 3D detectors have recently emerged as a popular alternative to voting- and voxel-based methods, which offer end-to-end set prediction without handcrafted priors or voxelization. However, they remain unexplored under semi-supervision, where the scarcity of annotated 3D data impedes their widespread adoption. In this work, we present Semi-3DETR, the first framework to systematically adapt DETR to semi-supervised 3D object detection by addressing challenges unique to 3D. Compared to 2D semi-DETR, semi-supervised 3D DETR faces amplified issues of fragile volumetric pseudo-labels, unstable query alignment, and noisy bipartite matching. Our Semi-3DETR mitigates these issues by introducing three core components: Robust Pseudo-Label Denoising (RPLD) to filter and refine volumetric pseudo-labels against orientation and depth errors, Query Alignment Consistency (QAC) to stabilize teacher–student query correspondence under 3D transformations, and a Hybrid Matching Strategy (HMS) to balance one-to-one and one-to-many assignments under noisy supervision. We further adopt a softmax classifier to enforce class exclusivity and improve pseudo-label reliability in semantically ambiguous 3D categories. Extensive experiments on ScanNet and SUN RGB-D demonstrate the feasibility of our Semi-3DETR with promising results compared to fully supervised and semi-supervised baselines. The source code will be released upon paper acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Semi-3DETR, the first framework to adapt DETR-based methods to semi-supervised 3D object detection. The work addresses three key challenges unique to 3D semi-supervised DETR: fragile volumetric pseudo-labels, unstable query alignment, and noisy bipartite matching. The proposed solution introduces three core components: (1) Robust Pseudo-Label Denoising (RPLD) with a query feature-driven IoU head, (2) Query Alignment Consistency (QAC) for stable teacher-student correspondence, and (3) Hybrid Matching Strategy (HMS) with matching degeneration and softmax classification. Experiments on ScanNet and SUN RGB-D demonstrate the feasibility of the approach.

### Strengths
1) The paper clearly articulates the unique challenges of semi-supervised 3D DETR compared to 2D counterparts: volumetric pseudo-label fragility, 3D query grounding instability, and amplified matching noise. These challenges are well-motivated and specific to the 3D domain.
2) The three proposed components (RPLD, QAC, HMS) directly address the identified challenges with principled solutions. The query feature-driven IoU head for geometric confidence is particularly novel and well-motivated for 3D scenarios.

### Weaknesses
1) The evaluation is restricted to only two indoor datasets (ScanNet and SUN RGB-D). The lack of outdoor datasets (KITTI, nuScenes) significantly limits the generalizability claims. Additionally, the performance improvements are modest and inconsistent across different data ratios.
2) Several design decisions lack proper justification. Why is the matching degeneration from one-to-many to one-to-one optimal? The choice of softmax over sigmoid is presented as essential for 3D but lacks thorough empirical validation. The specific architecture choices for the IoU head and query reconstruction are not well-justified.
3) The paper lacks comprehensive ablation studies on key components. How does each component contribute individually? What is the computational overhead of the additional modules? The sensitivity analysis for hyperparameters (λ, thresholds) is missing.

### Questions
1) How does the method perform on outdoor datasets like KITTI or nuScenes?
2) What is the computational overhead of the additional components compared to the baseline V-DETR?
3) How sensitive is the method to the choice of hyperparameters (λ, confidence thresholds)?
4) How does the IoU head performance correlate with actual detection quality?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Semi-3DETR, the first semi-supervised framework for DETR-based 3D object detection, addressing challenges like fragile volumetric pseudo-labels via three 3D-specific components (RPLD, QAC, HMS) and a softmax classifier. It outperforms supervised/semi-supervised baselines (e.g., +8.9% mAP@0.25 on ScanNet with 5% labels vs. V-DETR) on ScanNet and SUN RGB-D.

### Strengths
1.The paper is well-substantiated overall, with attractive figures and tables.

2.Numerous experiments were conducted, and the proposed method demonstrated superior performance in the figures and tables.

3.The ablation experiments appear to be well-substantiated.

### Weaknesses
1. It is suggested to conduct tests on more datasets, such as Kitti and nuScenes.

2. The experiments are mostly carried out in indoor scenes. Is the proposed method more suitable for indoor scenes? If not, how does it perform in outdoor scenes?

3. Tables 1-2 show that the proposed method can outperform fully supervised methods, which confuses me. Do the fully supervised methods and semi-supervised methods use the same fully supervised data?

4. Why does DQS3D in Table 2 outperform the proposed method by such a large margin? The authors need to provide a reasonable explanation to demonstrate the necessity of the proposed method.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Semi-3DETR, the first framework that systematically extends the DETR paradigm to semi-supervised 3D detection. It improves pseudo-label reliability and teacher–student consistency through three designs: query-driven 3D IoU confidence estimation (RPLD), query alignment consistency (QAC), and an one-to-many schedule during supervised pretraining that switches to one-to-one with a softmax classifier in the semi-supervised stage.

### Strengths
1. Clearly identifies the problem and presents the first semi-supervised framework for 3D DETR.
2. Designs are tailored to 3D characteristics and the three modules work in a complementary manner.
3. Outperforms DQS3D on ScanNet while using limited labels.

### Weaknesses
1. The introduction does not sufficiently explain why prior methods cannot be directly applied to semi-supervised DETR for 3D detection, which weakens the motivation’s specificity.
2. Dataset and scenario coverage is limited, since experiments are confined to indoor ScanNet and SUN RGB-D, with no outdoor evaluation.
3. The method is sensitive to threshold and consistency details. A low τ introduces noise and a high τ discards supervision, and removing the QAC attention mask leads to clear drops, which implies nontrivial tuning for practical deployment.
4. Absolute performance is not universally superior. On SUN RGB-D, the voxel-based DQS3D remains stronger.

### Questions
Please address my concerns in Weaknesses. I hope the authors can address the questions raised above to clarify these concerns. If these issues are satisfactorily resolved, I would be willing to raise my score.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a pipeline for semi-supervised 3D detection from point clouds, with a particular focus on improving the performance on DETR-like detectors. The authors discuss varies challenges for this task, focusing on improving the metric for identifying good pseudo-labels, sharing some queries between teacher and student for straightforward alignment, and separating matching process for supervised vs semi-supervised training. The paper reports good performance on the ScanNet and SUN RGB-D dataets, and ablates components of their framework.

### Strengths
- The diagram is drawn clearly, outlining the data and feature flow of the entire model.
- Prior work is extensively covered, and I appreciate the subtlety in explanation that Diff3DETR is not a traditional DETR-esque method.
- Proposed components are clearly outlined in text, with accurate equations.

### Weaknesses
- To the best of my understanding, the proposed "Proposed Pseudo-Label Denoising" (RPLD) process seems like an 3D IoU prediction head, whose prediction is used to get pseudo-labels on unlabeled data. This appears to me the same as the method proposed in 3DIoUMatch, which similarly has a 3D IoU head, and filters labels based on class, IoU, and objectness confidence. In this aspect, I am hesitant to cite this section as a contribution of this work. I am, however, confused by the ablation in Table 5, where the authors compare with the IoU module of 3DIoUMatch and report better performance. What, precisely, is the difference between these two methods?
- Regarding query alignment consistency, this paper writes "Unlike Semi-DETR Zhang et al. (2023), which enforces cross-view consistency only on 2D image queries, our method reconstructs 3D object queries that capture both semantic features and geometric attributes of center, size, and orientation." The author emphasizes the difference between 2D and 3D, but functional, the methods appear very similar. Semi-DETR takes pseudo-label boxes on unlabeled images, extracts features from the input (image), and puts them into the teacher and student to enforce consistency between their predictions. The proposed work seems to similarly take pseudo-labels from the teacher, get the corresponding content query & position query (since V-DETR has two types of features for each query), and similarly puts them through the teacher & student to enforce consistency. While I recognize the benefits of leveraging methods developed in 2D for the 3D task, this does seem to weaken this paper's contribution.
- The Hybrid Matching Strategy is also very similar to Semi-DETR's Stage-wise Hybrid Matching. While there is a difference - the proposed work explicitly does one-to-many matching during pre-training and one-to-one during semi-supervised, while Semi-DETR trains semi-supervised initially with one-to-many before switching to one-to-one, Semi-DETR does not have an explicit labeled-only pre-training phase, making the two pipelines functionally similar. 
- The authors mention the necessity of comparing improvement, not absolute performance, due to the difference in base architectures. However, the proposed model has smaller improvement than baselines in Table 1, albeit it has a higher starting point, and it also does not achieve as good performance in SUN RGB-D.

### Questions
While this paper demonstrates strong performance in SSL 3D detection, it seems like most of the module contributions are brought from prior work Semi-DETR and 3DIoUMatch. While I believe that deriving best practices from previous methods is important and I also acknowledge the difficulties in transfering some 2D techniques to 3D, it also does seem to me that this paper's unique contributions may not be sufficient. At this stage, the differences between the proposed modules and prior work is not sufficiently discussed - if the authors can discuss this, I will consider raising my score. At this stage, I recommend a 2.

### Soundness
3

### Presentation
3

### Contribution
2
