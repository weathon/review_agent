# Unsupervised Domain Adaptation for 6-DoF Pose Estimation with Contrastive Alignment and Pseudo-Label Refinement

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Unsupervised domain adaptation (UDA) enables robust transfer of knowledge from simulated to real environments while exploiting a subset of unlabeled target data to improve real-world performance. Existing UDA methods for 6-DoF object pose estimation often rely on global feature matching, multi-stage larger frameworks, or image translation pipelines, which tend to overlook the pose-specific information embedded in feature representations. To bridge this limitation, we introduce CAPLR that targets the adaptation of pose-sensitive features in localized regions, ensuring that domain alignment preserves the geometric cues essential for accurate pose estimation. CAPLR achieves UDA with three key components: (1) Efficient Cross-Domain Pairing strategy leveraging intermediate features to identify pose similar image pairs across domains without supervision; (2) Contrastive Alignment to perform feature alignment at localised regions in both intermediate and task-specific representations; and (3) Consistency-Based Pseudo-Label Refinement to improve reliability by encouraging stable target predictions. Extensive experiments demonstrate that CAPLR achieves state-of-the-art performance across multiple well-known 6-DoF object pose estimation benchmarks featuring diverse and challenging scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes CAPLR, a keypoint-based unsupervised domain adaptation (UDA) framework for 6-DoF pose estimation. CAPLR addresses synthetic-to-real domain gaps through three key components: Cross-Domain Pairing (CDP) for identifying pose-consistent source-target image pairs, Local Patch Contrastive Alignment (LPCA) for dual-level (backbone and head) feature alignment, and Consistency-Based Pseudo-Label Refinement (CBPR) for stabilizing predictions via augmented views. Experiments on LineMOD, Occluded-LineMOD, HomeBrewedDB, and SPEED+ datasets show state-of-the-art performance.

### Strengths
+ An interesting aspect of the method is its two-step, cross-domain pairing strategy, where initial anchors derived from heatmap activations are subsequently refined using pose errors. The ablation study in Table 5 also verifies the effectiveness of this strategy.

+ The pseudo-label refinement strategy leveraging spatial distance and confidence effectively identifies more reliable keypoint pseudo-labels.

### Weaknesses
-- The comparative experiment settings on SPEED+ may be unfair. Did the other methods use all the test data for domain adaptation? This should be explicitly labeled in the table, and it should be ensured that the other methods also use the identical training and testing setup.

-- The evaluation is limited by the absence of two key comparisons for the proposed method: one trained only on synthetic data and one trained on annotated synthetic and real data. This omission makes it difficult to assess lower and upper bounds of the method's performance.

-- The method heavily relies on keypoint predictions and their corresponding heatmaps. For symmetric objects where reliable keypoints cannot be obtained, or in cluttered environments where background objects with similar appearances lead to false positive keypoints, the method may fail. To understand whether the method exhibits this issue, additional experiments on datasets like T-LESS could be beneficial.

-- It would be better to move key ablations to the main paper rather than in the appendix.

Minor

-- Why does the actual reference entry for Self6D++ correspond to GDR-Net?

-- The reference entry for PoseCNN is duplicated.

-- L320-323: Missing commas in these sentences.

### Questions
-- Does this keypoint-based method include a 2D object detection module? If not, how does it handle multiple instances of the same object within a single image?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes an unsupervised domain adaptation algorithm for the 6-DoF object pose estimation task. Given (source) synthetic datasets having annotations and (target) real-world data with small scale data, the task needs to reduce the domain gap and predict the accurate objects' pose information.

This paper proposes (1) Cross-domain pairing, (2) contrastive alignment, and (3) consistency based pseudo label refinement. The proposed schemes are to reduce the domain gap in embedding spaces as well as objects' pose space.

Overall, the performance looks good compared to previous studies, but there is no ablation study in the manuscript.

### Strengths
Overall flow of this paper is readable and understandable. The proposed strategies are aligned with the unsupervised domain adaptation using image/pose pairing in source-target data (Sec. 3.2.1) and embedding alignment (Sec. 3.2.2).  Nonetheless it is not that convinced that the proposed methods are novel and unique.

### Weaknesses
__W1. No ablation study in the manuscript?__  
While the authors propose two dominant techniques in the manuscript, but I cannot find the ablation study within the main paper. Surprisingly, the ablation studies were found in the supplementary, but I am a bit worried about such a paper writing. While it is good for me to read the manuscript, but it needs to be more compact to make more margins and locate the ablation studies in the manuscript. I personally guess that the authors should have spent lots of efforts in paper writing. The current writing is not good enough for the submission. It needs reformulation.

__W2. Novelty__
Even though the authors present the background knowledge for the concept of the embedding alignments in Lines 216-247, it is not something new. More technical speaking, it is simply align the two embedding spaces, one for the embedding vectors from backbone network and the other for the objects' poses from the header network, from the target domain and the source domain. Even, the authors simply leverage the InfoNCE loss. Can the authors provide more clues or claims why the proposed schemes are novel?

Moreover, it is not also something new to introduce pseudo labeling in the unsupervised domain adaptation problem. There are lots of related works who also use the same idea, but based on the submission, I cannot clearly tell the differences.

### Questions
I do not have the specific questions. Instead, I hope the authors to answer to my questions writing in the weakness section.

Overall, it is easy to read the paper, but it needs to be more concise and compact. Ablation studies are not written in the manuscript, but the supplementary material. I am not sure why the Section 3.1 is necessary. The paper needs lots of revision.

Beside from the writing, I do not think that the authors present the novel ideas or contributions. There are lots of existing strategy to solve the unsupervised domain adaptation problem for the 6-DoF pose estimation problem. However, I cannot clearly tell the differences between the previous works and the proposed schemes.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to solve unsupervised domain adaptation (UDA) for 6-DoF object pose estimation in a sim-to-real setting. The proposed framework, called CAPLR, introduces three main components: 1) Cross-Domain Pairing (CDP) finds source–target pairs that are likely to depict similar object poses, using a two-stage strategy (first a feature/heatmap-based similarity, then a pose-distance filtering) to reduce noisy matches, 2) Local Patch Contrastive Alignment performs contrastive learning both at the backbone level and at the head level so that features useful for pose prediction are actually aligned, not just globally domain-invariant ones, and 3) Consistency-Based Pseudo-Label Refinement (CBPR) generates pseudo labels for target images and refine them via augmentation consistency so that only stable keypoints are kept. The method is evaluated on standard synthetic-to-real 6D benchmarks (e.g., LineMOD, Occluded-LineMOD, HomeBrewedDB, SPEED-like settings) and claims to outperform or match prior UDA approaches under comparable settings.

### Strengths
1. Dual-level contrastive learning seems effective.
*  Unlike the previous contrastive learning methods that applied at the backbone level feature distribution, the paper exploits both the backbone and the head level feature distribution for 6D pose estimation.

2. Broad experiment and outperformed performance. 
* The proposed method is evaluated on standard synthetic-to-real 6D benchmarks (e.g., LineMOD, Occluded-LineMOD, HomeBrewedDB, SPEED-like settings) and claims to outperform or match prior UDA approaches under comparable settings.

### Weaknesses
1. Failure-case analysis of pairing.
* In the ablation where you compare GT pairing vs. your 2-stage pairing, there is a non-trivial gap. Could you show some qualitative or per-scenario analysis, such as under what conditions (occlusion, illumination, near-symmetric poses, scale mismatch) does your pairing fail most often?

2. Extension to category-level 6D.
* The method is demonstrated mainly on instance-level objects (LineMOD family). However, related work cited includes category-level UDA for 6D. If the method is meant to generalize to slightly different CADs/scales (which is a key challenge in category-level 6D), there should be experiments or at least analysis on how pairing behaves in that setting. If not, can you discuss how the framework can be extended to category-level 6D pose estimation?

### Questions
Please answer the weakness parts.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses unsupervised domain adaptation for 6D object pose estimation and introduces CAPLR, which consists of three key components, including cross-domain pairing, contrastive alignment, and consistency-based pseudo-label refinement. Experiments on LineMOD, LineMOD-Occlusion, and SPEED+ demonstrate the effectiveness of the proposed approach.

### Strengths
- Instead of relying solely on self-supervised learning, CAPLR employs contrastive alignment and pseudo-label refinement for UDA pose estimation, with ablation studies validating the effectiveness of each component.
- CAPLR realizes SOTA on UDA setting on LineMOD, LineMOD-Occlusion, and SPEED+ .

### Weaknesses
- For existing methods, the paper does not sufficiently analyze their limitations in the abstract and introduction. In particular, the discussion of self-supervised learning is too general, using broad statements such as “Many approaches rely solely on self-supervision in the target domain, which has inherent limitations” and “However, these methods are limited when the domain gap is large,” without providing concrete analysis. Consequently, the advantages of the proposed method over these limitations are not clearly highlighted in these sections.
- For cross-domain pairing, I have the following concerns:
    - I am concerned about the robustness of the confidence-weighted embeddings. In keypoint regression, the features of different keypoints are expected to remain distinguishable; however, this design relies on the consistency of both keypoint features and confidence scores across domains. The results in Table 5 do not provide ablation studies to assess this, for example, comparisons with simple average pooling over keypoint features or using masked foreground features. Considering the widespread use of DINOv2 features for template matching in pose estimation, would it be more robust to instead use the CLS tokens from DINOv2?
    - In Table 5, the results based solely on pose prediction outperform those using feature-based selection. Why not reverse the order of the two steps, i.e., perform pose-prediction-based selection first, followed by feature-based selection?
    - Each target image is paired with an optimal source image. Is there a filtering strategy to avoid highly mismatched pairs?
- For pseudo label refinement,  I have the following concerns:
    - What are the augmentations used?
    - Since $L_{task}$ in Eq. (10) is not used at this stage, i.e., no ground-truth supervision is applied, how is it ensured that training progresses in the correct direction rather than falling into shortcuts? Could you provide the performance on the test or validation sets across training epochs? 
- For the results in Tables 1 and 2, it is recommended to provide the lower and upper bound performance of CAPLR, i.e., using only S and using S+R, to better illustrate the gains attributable to the UDA method rather than the baseline model.

### Questions
- For the source domain, why are the predicted poses and keypoints used in cross-domain pairing and contrastive alignment instead of the ground-truth annotations?

### Soundness
3

### Presentation
3

### Contribution
2
