# T-3DGS: Removing Transient Objects for 3D Scene Reconstruction

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Transient objects in video sequences can significantly degrade the quality of 3D scene reconstructions. To address this challenge, we propose T-3DGS, a novel framework that robustly filters out transient distractors during 3D reconstruction using Gaussian Splatting. Our framework consists of two steps. First, we employ an unsupervised classification network that distinguishes transient objects from static scene elements by leveraging their distinct training dynamics within the reconstruction process. Second, we refine these initial detections by integrating an off-the-shelf segmentation method with a bidirectional tracking module, which together enhance boundary accuracy and temporal coherence. Evaluations on both sparsely and densely captured video datasets demonstrate that T-3DGS significantly outperforms state-of-the-art approaches, enabling high-fidelity 3D reconstructions in challenging, real-world scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces T-3DGS, which reconstructs static 3D scenes from monocular videos that contain transient objects. The pipeline begins with a Reconstruction Uncertainty Predictor (RUP) that uses DINOv2 features to produce a binary transient mask, which is then refined spatially with SAM and temporally with SAM2. During 3D Gaussian splatting, masked regions are excluded so optimization focuses on static content. The paper also introduces the T-3DGS dataset, a more challenging benchmark than prior sets. Experiments on three datasets show that T-3DGS achieves state-of-the-art performance.

### Strengths
1. This paper is easy to follow, the idea of detecting transient objects with feature extraction and refining them with SAM and SAM2 is reasonable and demonstrates good performance.
2. The proposed new dataset, T-3DGS, is more challenging than previous benchmark for evaluating models' performance on this task, and it is useful for the community.
3. Each contributions are properly evaluated through ablation studies.

### Weaknesses
1. My main concern is that this method is a combination of existing foundation models DINO, SAM and SAM2. It provides little novelty or inspirations, given these models are already well-explored in this community.
2. The comparison with Easi3R seems unfair, because as a feed-forward based method,  Easi3R is good for its inference speed and generality, and as optimization based method, the proposed method is expected to have better per-scene performance.
3. The TMR module comprises two components—SAM for spatial refinement and SAM2 for temporal refinement. What are the respective contributions of each to the refinement process, qualitatively or quantitatively?
4. Table 2 suggests most of the gains come from TMR. Since TMR leverages off-the-shelf SAM/SAM2, the novelty and impact of RUP are not convincingly demonstrated.

### Questions
1. Is the model structure of RUP introduced in supplementary material B?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes T-3DGS, a framework for robust 3D scene reconstruction that removes transient and semi-transient objects from video sequences during 3DGS optimization. The key contribution is an unsupervised Reconstruction Uncertainty Predictor (RUP) that identifies transient distractors using multivariate uncertainty modeling with KL divergence and a Transient Mask Refiner (TMR) that enhances mask spatial and temporal consistency. Extensive experiments show that the proposed method outperforms baselines.

### Strengths
1. The paper is well motivated, addressing a practical problem in 3DGS-based scene editing. 
2. The proposed T-3DGS dataset fills a gap for semi-transient object evaluation.

### Weaknesses
1. Limited novelty: The main contribution seems to be a combination of existing techniques (uncertainty estimation, semantic guidance, and SAM-based propagation), rather than addressing the deeper underlying issue of poor extrapolation and generalization in 3DGS.
2. Insufficient analysis: The paper lacks detailed sensitivity studies for key thresholds and hyperparameters, and the divergence-based uncertainty formulation remains largely heuristic without strong theoretical or empirical justification.
3. Dependence on SAM: Further evaluation is needed regarding the method’s reliance on SAM; segmentation noise or boundary inaccuracies may significantly influence transient detection and overall reconstruction performance.

### Questions
Does the SAM-based refinement introduce temporal artifacts or error propagation in long sequences?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents T-3DGS, a framework for reconstructing static 3D scenes from monocular video containing transient objects. T-3DGS is based on 3DGS, and introduces two key components: 1) reconstruction uncertainty predictor (RUP) that detects transient regions using semantic features and KL-divergence–based uncertainty modeling; 2) transient mask refiner (TMR) leveraging SAM/SAM2 for spatial refinement and temporal propagation. The combination allows 3DGS to focus on static content, which improves the reconstruction quality in real-world conditions. The authors conduct experiments on several datasets, and show better performance over existing methods.

### Strengths
1. The authors study an interesting problem of transient object, which is important for real-world scene reconstruction.
2. The method is well-designed and theoretically grounded.
3. The pipeline utilize DINOv2 features to provide robustness against color similarity and high-frequency textures.
4. The experiments compare against several baseline approaches and ablate key modules. The authors also introduce a new dataset with transient objects.
5. The authors provide qualitative results to clearly demonstrate better reconstructions.

### Weaknesses
1. The training pipeline is too heavy, which uses DINOv2 to extract features, use SAM for spatial refinement and SAM2 for temporal refinement.
2. Each submodule is adapted from existing techniques.
2.1 RUP uses DINOv2 features for semantic understanding, follows WildGaussians to build per-pixel residual with FeatUP and DSSIM, follows NeRF-W use uncertainty in 3DGS and separate static and transient objects.
2.2 The first part of the TMR uses SAM to clean up the noisy binary masks predicted by RUP.
2.3 The second part of TMR uses SAM2 to propagate masks.
3. The ablations do not fully separate the effects of KL, semantic feature, and TMR propagation.

### Questions
1. What happens if SAM or SAM2 produces incorrect boundaries?
2. What is the training interaction between RUP and 3DGS?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper tackles the task of removing transient objects from a video. The transient objects break the assumption of static scenes that most 3D reconstruction approaches rely on. To mitigate this issue, the authors propose an uncertainty modeling-based approach to detect whether a mask is transient or not. Further, to ensure temporal consistency, the authors propose to use temporal refinement to enhance the mask quality. Experiments on various datasets demonstrate the effectiveness of the proposed approach.

### Strengths
- originality-wise: the idea of utilizing uncertainty modeling and mask propagation to handle dynamic objects is interesting.
- quality-wise: qualitative and quantitative results demonstrate the effectiveness of the proposed approach.
- clarity-wise: the paper is well-written in general.
- significance-wise: the problem of removing transient objects is important for downstream tasks of 3D reconstruction from in-the-wild videos.

### Weaknesses
1. For temporal refinement (L315): can we just use forward or backward propagation? How bad will the performance be qualitatively and quantitatively?

2. Can authors provide some runtime analysis?

3. How to determine the extent of dilation (L290) as it seems important from Tab. 3?

4. From the Fig. 2, it seems like RUP is not updated, which contradicts L170. Can authors clarify?

5. For Fig. 9, the T-3DGS's results do not seem to be from the same camera as the other methods or GT. Is this a bug or using the wrong one?

5. Please add a colorbar to Fig. 3. I am not sure which color means high uncertainty.

### Questions
See "Weakness"

### Soundness
3

### Presentation
3

### Contribution
3
