# Source-Free and Image-Only Unsupervised Domain Adaptation for Category Level Object Pose Estimation

- Decision: Accept (poster)
- Scores: 6, 6, 6, 6

## Abstract
We consider the problem of source-free unsupervised category-level 3D pose estimation from only RGB images to an non-annotated and unlabelled target domain without any access to source domain data or annotations during adaptation. Collecting and annotating real world 3D data and corresponding images is laborious, expensive yet unavoidable process since even 3D pose domain adaptation methods require 3D data in the target domain. We introduce a method which is capable of adapting to a nuisance ridden target domain without any 3D data or annotations. We represent object categories as simple cuboid meshes, and harness a generative model of neural feature activations modeled as a von Mises Fisher distribution at each mesh vertex learnt using differential rendering. We focus on individual mesh vertex features and iteratively update them based on their proximity to corresponding features in the target domain. Our key insight stems from the observation that specific object subparts remain stable across out-of-domain (OOD) scenarios, enabling strategic utilization of these invariant subcomponents for effective model updates. Our model is then trained in an EM fashion alternating between updating the vertex features and feature extractor. We show that our method simulates fine-tuning on a global-pseudo labelled dataset under mild assumptions which converges to the target domain asymptotically. Through extensive empirical validation, we demonstrate the potency of our simple approach in addressing the domain shift challenge and significantly enhancing pose estimation accuracy. By accentuating robust and less changed object subcomponents, our framework contributes to the evolution of UDA techniques in the context of 3D pose estimation using only images from the target domain.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a new domain adaptation method for category-level object pose estimation. The method uses only RGB images, without relying on source domain data or 3D annotations in the target domain. The authors represent an object that belongs to a known category as a cuboid mesh and utilize an existing method to learn the vertex features. The features are iteratively updated in the target domain based on their proximity to corresponding image features. The experiments show much better performance in the target domain compared with the competitors without domain adaptation.

### Strengths
•	The authors handle the problem that neither depth information nor 3D annotations are available in the target domain, which is challenging yet important in real applications.

•	The intuition behind the presented domain adaptation method is that the local features, which represent specific parts of the object, are more robust than the global features in the target domain, which makes sense to me.

•	The presented method achieves impressive object pose estimation results in multiple datasets with different kinds of nuisance.

### Weaknesses
The majority of the figures in this paper such as Fig.2, Fig.2, and Fig.10, exhibit low quality. It would be better if the authors could consider revising them to enhance the clarity and resolution.

I am not familiar with the topic of domain adaptation, so I would not judge the novelty of this paper. Please refer to “Questions” for my concerns.

### Questions
•	To my understanding, in the experiments, the baseline models are evaluated in the target domain without domain adaptation. In this context, it is reasonable that those methods cannot generalize well in the target domain. I was wondering if there are some existing domain adaption approaches that can be applied to the baseline models. The evaluation would be more convincing, comparing NeMo + 3DUDA with a competitor such as NeMo + another domain adaption method.

•	The pre-render feature maps are generated from the source mesh. As the vertex features are not updated here, how to make sure those feature maps are reliable in the target domain? Are there situations in which the method might be stuck in local optima or even diverge? Conducting an ablation study on the pre-rendered feature maps would be valuable.

•	In Sec.3.2.1, the parameter $\delta$ seems crucial for effectively updating vertex features. The authors mentioned they chose a $\delta$ such that the majority of source domain features lie within the likelihood score. How to set $\delta$ in practice? Is it constant? Intuitively, as the vertex features are updated, one would expect the similarity between vertex features and image features to increase. Does it make more sense to update $\delta$ accordingly?

•	It seems the method is time-consuming due to the iterations and pre-rendering. However, the actual time consumption during testing is unclear.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the task of category-level 3D pose estimation within the setting of source-free and image-only unsupervised domain adaptation. The authors introduce a novel method called 3DUDA, which is developed based on the observation of the invariance of object local parts and is supported by theoretical insights. 3DUDA utilizes a learnable cuboid feature matrix to represent an object category, and assesses the accuracy of a pose by comparing the feature map of the test image with the one rendered from the categorical cuboid feature matrix using the pose. This render and compare approach enables domain adaptation by iteratively updating the categorical feature matrix and fine-tuning the source model. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
- The proposed method, 3DUDA, is developed based on the common observation that certain object parts exhibit invariance across out-of-distribution (OOD) scenarios, and utilizes the categorical learnable cuboid meshes to effectively capture and store the part features at each vertex.
- To achieve domain adaptation, 3DUDA employs an iterative process that involves updating the features of categorical meshes and fine-tuning the source model through feature-level render and compare optimization.
- The paper is well-written and presents its ideas in a clear and understandable manner. It is accompanied by comprehensive supplementary materials and theoretical results, which greatly enhance the persuasiveness of the paper.

### Weaknesses
- It is recommended to evaluate the proposed method on commonly used datasets for category-level pose estimation, such as REAL275 [1] or Wild6D [2].

- It would be beneficial to include relevant works [3,4,5,6] in the paper to provide a comprehensive overview of the existing literature in the field.

[1] Wang et al., Normalized object coordinate space for category-level 6d object pose and size estimation. CVPR2019.

[2] Fu et al., Category-Level 6D Object Pose Estimation in the Wild: A Semi-Supervised Learning Approach and A New Dataset.

[3] Lin et al., Category-level 6D object pose and size estimation using self-supervised deep prior deformation networks. ECCV2022.

[4] He et al., Towards Self-Supervised Category-Level Object Pose and Size Estimation.

[5] Zhang et al., Self-Supervised Geometric Correspondence for Category-Level 6D Object Pose Estimation in the Wild. ICLR2023.

[6] Goodwin et al., Zero-Shot Category-Level Object Pose Estimation. ECCV2022.

### Questions
- How does 3DUDA address the issue of minimizing the impact of translation and object size on neural feature rendering?
- What values are set for the number $R$ of vertices of the mesh and the hyperparameter of $N$ of the clutter model? How do they  impact the performance of the method?
- The format of citations in Table 1 does not align with the citation format used in the rest of the paper.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the challenge of unsupervised category-level pose estimation in a target domain using only RGB images, without access to source domain data or 3D annotations.
The authors introduce a method that adapts to a target domain, even when it is complicated by nuisances, without requiring 3D or depth data.
They represent object categories with simple cuboid meshes and use a generative model of neural feature activations at each mesh vertex.
They focus on updating local mesh vertex features based on their proximity to corresponding features in the target domain, even when the global pose is incorrect.
The key insight is the stability of specific object sub-parts across different scenarios, which allows for effective model updates.

### Strengths
1. This paper suggests effective render-and-compare adaptation pipeline for unsupervised domain adaptation for category-level object pose estimation task.

2. Its proposed method shows the state-of-the-art performance in various corruption scenarios compared to some of the previous methods.

3. The methodology part is intuitive and easy to follow text-wise.

### Weaknesses
1. Poor presentation.
I think this paper holds good insights and corresponding technological contributions.
Yet, the presentation of the whole paper is relatively poor, making it hard to follow the overall message.
Figure placement is not well-aligned with the text context.
Figure 3 seems to be a very important description of explaining one of the main contributions of this paper to match sub-vertices, while there is no mention in the main manuscript referring this.
Moreover, the main manuscript refers to figures in appendix very often, which is very inconvenient to read, while some of these figures seem important enough to be contained in the main paper for effective description. (ex, Figure 5)
I believe that ablation studies regarding several design choices of the proposed method should be contained in the main manuscript as well, since they can effectively validate the authors claim.

### Questions
1. Baseline methods and other datasets
Except OOD-CV results, only previous baseline is NeMo. Can the authors explain why there can't be other methods in this comparison? Being better than only one baseline is not enough to claim the superiority of the proposed method.
Also, while I acknowledge that the paper mainly focuses on data corruption scenarios, is it possible to compare this type of approach in conventional category-level object pose benchmarks like CAMERA and REAL275 datasets provided by NOCS? I believe it would strengthen the authors' motivation if it can be generally applied to conventional syn-to-real UDA scenarios.

2. GT reliability
Ground truth pose illustrated in Figure 1 and 5 seems to be not perfectly aligned with the image. Can the authors explain how these GTs are obtained, and how they are utilized? Are they only used for evaluation?

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper study the problem of source-free unsupervised category-level pose estimation from only RGB images to a target domain without any access to source domain data or 3D annotations during adaptation. The author propose a new pipline which focus on the  individual local mesh vertex features and utilize their pose ambiguity to iteratively update them based on their prox- imity to corresponding features in the target domain even when the global pose is not correct. The proposed method shows good results.

### Strengths
1. The proposed method outperform previous approaches by a large margin
2. The author evaluate their model on real world nuisances like shape, texture, occlusion, etc. as well as image corruptions and show the robustness of proposed method

### Weaknesses
1. As mentioned in the article, an observation is that global information is noisy, but some local details are robust. I hope there is rigorous explanation and quantitative analysis here to support this hypothesis.
2. Ablation is not enough, (e.g., Ablation on top-n 
3. I think this method is similar to the iterative optimization often used in instance-level post-processing. It is unfair to compare this method with other single forward methods.
4. I'm not sure if it's right to say "normalized real-valued feature activations" ? (
The fourth line from the bottom of the third page)
5. The figures in this article are very rough and difficult to understand.

### Questions
Please refer to the weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
