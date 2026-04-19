# CapeX: Category-Agnostic Pose Estimation from Textual Point Explanation

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 5

## Abstract
Conventional 2D pose estimation models are constrained by their design to specific object categories. This limits their applicability to predefined objects. To overcome these limitations, category-agnostic pose estimation (CAPE) emerged as a solution. CAPE aims to facilitate keypoint localization for diverse object categories using a unified model, which can generalize from minimal annotated support images.
Recent CAPE works have produced object poses based on arbitrary keypoint definitions annotated on a user-provided support image. Our work departs from conventional CAPE methods, which require a support image, by adopting a text-based approach instead of the support image. 
Specifically, we use a pose-graph, where nodes represent keypoints that are described with text. This representation takes advantage of the abstraction of text descriptions and the structure imposed by the graph.
Our approach effectively breaks symmetry, preserves structure, and improves occlusion handling.
We validate our novel approach using the MP-100 benchmark, a comprehensive dataset covering over 100 categories and 18,000 images. MP-100 is structured so that the evaluation categories are unseen during training, making it especially suited for CAPE.  Under a 1-shot setting, our solution achieves a notable performance boost of 1.26\%, establishing a new state-of-the-art for CAPE. Additionally, we enhance the dataset by providing text description annotations for both training and testing. We also include alternative text annotations specifically for testing the model's ability to generalize across different textual descriptions, further increasing its value for future research. Our code and dataset are publicly available at https://github.com/matanr/capex.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper focuses on the category-agnostic pose estimation task, which learns to estimate the keypoints of novel classes based on a few support images and text descriptions.
Specifically, the paper proposes a pose-graph and use nodes to represent keypoints that are described with text, which could take advantage of the abstraction of text descriptions and the structure imposed by the graph.
On the MP-100 benchmark dataset, the paper achieves a notable performance boost of 1.27% against SOTA methods of category-agnostic pose estimation.

### Strengths
- The paper proposes to make full use of text to boost CAPE, which seems novel and interesting.
- The proposed method seems resonable and works well for CAPE.
- The achieved performances are significant against existing works of CAPE.
- The paper is well-written.

### Weaknesses
- The description about the related work MetaPoint could be improved, which actually first assigns and then refines.
- The paper composing could be improved. E.g., Fig. 3 and Fig. 4 waste too much space and provide limited information.
- The limitations on using text could be discussed more deeply, because some keypoints are hard to describe using text.
- The experiments could be enhanced with cross super-class transfer as in MetaPoint, CapeFormer and POMNet.

### Questions
See Weaknesses*. Overall, the idea of paper seems interesting, and I would like to see more future works. My rating could be increased if my concerns are solved.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Recent CAPE works have produced object poses based on arbitrary keypoint definitions annotated on a user-provided support image. Our work departs from conventional CAPE methods, which require a support image, by
adopting a text-based approach instead of the support image. Specifically, we use
a pose-graph, where nodes represent keypoints that are described with text. This
representation takes advantage of the abstraction of text descriptions and the structure imposed by the graph. 

Our approach effectively breaks symmetry, preserves
structure, and improves occlusion handling. We validate our novel approach using the MP-100 benchmark, a comprehensive dataset covering over 100 categories
and 18,000 images. MP-100 is structured so that the evaluation categories are unseen during training, making it especially suited for CAPE. Under a 1-shot setting,
our solution achieves a notable performance boost of 1.27%, establishing a new
state-of-the-art for CAPE. Additionally, we enhance the dataset by providing text
description annotations for both training and testing. We also include alternative
text annotations specifically for testing the model’s ability to generalize across
different textual descriptions, further increasing its value for future research.1
.

### Strengths
1. The motivation is clear  and the paper reads smooth. 

2. The design of the presented method is reasonable.

3. Experiments are conducted on the standard MP-100 dataset, establishing a new state-of-the-art for CAPE. The presented method shows superiority to previous CAPE models. Especially, the experimental analysis (e.g. Figure 6 and 7) is very interesting.

### Weaknesses
1. The novelty of the presented paper is a little bit concerning. (1) Open-vocabulary (textual prompt) keypoint estimation is not new, there are works such as CLAMP (Zhang et al. 2023) and KDSM (Zhang et al. 2024). (2) Graph representation for few-shot/zero-shot keypoint estimation is not new, as there are works like GraphCape (Hirschorn & Avidan, 2023). Could you highlight the difference between these methods? 

2.  Pose-Graph seems to be the most important contribution. However, experimental analysis about the pose-graph is insufficient. For example, there are different graph structures, what if using a fully-connected graph AND different graph structural settings (e.g. Tree-structure)?  

3. The reviewer suggests adding "Cross Super-Category Pose Estimation" on MP-100 evaluation, by training the model on all but one super-categories, and evaluate the performance on the remaining one super-category. This is to further evaluate the generalization ability. 

4. Since there already exists an open-vocabulary keypoint detection work like CLAMP (Zhang et al. 2023), what is the performance when comparing your method to CLAMP on Animal pose dataset in the setting of five-leave-one-out problem? Namely training on four species while testing on the left one species. The results of PCK @ 0.1 should be presented in paper for comparisons.

[Zhang et al. 2023] CLAMP+Prompt-based Contrastive Learning for Connecting Language and Animal Pose @ CVPR'23

### Questions
1. It seems that both visual prompt (support images) and text prompt (keypoint name) have their own limitations. For example, human face may have 68 keypoints, and it seems hard for textual prompts to precisely describe the keypoints along the edge of the face. How to solve this? Could you please give some examples about how to describe these keypoints in language?

2. Figure 6 seems interesting. If I understand correctly,  the experiments about "Modified text descriptions" are evaluated based on the same model without retraining. Is it correct?

3. In "Modified text descriptions", could you add experiments with more detailed description (longer description) about the keypoint. The reviewer is curious about the generalization ability of the system.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a novel approach that diverges from traditional CAPE methods, which typically rely on a support image, by utilizing a text-based framework instead. By implementing a pose-graph in which nodes signify keypoints described through textual annotations, this method leverages the abstraction inherent in text and the structural organization provided by the graph. This strategy effectively mitigates symmetry issues, maintains structural integrity, and enhances the handling of occlusions. The authors enrich the dataset by incorporating text description annotations for both the training and testing phases. They also provide alternative text annotations to specifically evaluate the model's ability to generalize across varying textual descriptions.

### Strengths
1. This paper addresses a very important issue: relying on visual information from a support image to locate keypoints in a test image is not reliable.

2. The use of a graph and text representation is intriguing, and I believe it provides a simple yet effective solution to the challenges associated with relying solely on visual information for localization.

3. The paper expands the MP-100 dataset, enabling graph-based CAPE.

4. The experiments in the paper are thorough, and the comparisons are comprehensive.

### Weaknesses
I don’t have many questions regarding this paper, just a few minor issues. 

1. Were all the keypoint texts in the dataset generated by off-the-shelf foundation models? When many keypoints are very close to each other and have highly similar semantics, how are these very similar keypoints distinguished in the text? Could you provide some examples?

2. The authors could provide more analysis regarding the limitations of the paper, explaining why certain failure cases occur.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes to employ textual information connected in a graph structure to tackle the problem of category-agnostic pose estimation (CAPE). The textural keypoint descriptions are annotated on the MP-100 dataset. The experimental results on the MP-100 dataset demonstrate the effectiveness of the proposed method.

### Strengths
This work focuses on the interesting and important task of category-agnostic pose estimation (CAPE). The proposed CapeX utilizes the abstract textual description for keypoint detection to improve the human-computer interaction. Graph structure is applied to capture the relationship between keypoints. The proposed CapeX achieves the state-of-the-art performance on the MP-100 dataset.

### Weaknesses
Pose Anything has designed the graph structure to capture the keypoint correlations, and the textual prompts have been explored for pose estimation in recent works such as KDSM. X-Pose has also provided textual annotations on the MP-100 dataset. Therefore, the contribution of this paper is somewhat incremental.

The graph construction is critical for graph-based approach. Also, manual design of node connections may introduce extra empirical knowledge, leading to unfair comparison. To avoid the above misgiving, the details of graph construction are expected to be clarified.

The cross super-category evaluation on the MP-100 dataset is absent. The cross super-category results would further validate the robustness of the proposed method.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
