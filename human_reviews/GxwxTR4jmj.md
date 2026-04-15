# Adapting Cross-View Localization to New Areas without Ground Truth Positions

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3

## Abstract
Given a ground-level query image, cross-view localization aims to estimate the location of the ground camera by matching the query to a geo-referenced aerial image that covers the local surroundings. Recent works have focused on developing powerful frameworks trained with ground truth (GT) locations of ground images within aerial images. However, the trained models always suffer a performance drop when applied to images in a new target area that differs from the training data. In most deployment scenarios, acquiring accurate GT location data for target-area images to re-train the network can be expensive and sometimes infeasible. In contrast, collecting images with coarse GT with errors of tens of meters is relatively easier. Motivated by this, our paper focuses on improving the generalization of a trained model by leveraging only the target area images without accurate GT. We propose a weakly-supervised learning approach based on knowledge self-distillation, namely, using predictions from a teacher model to supervise a student model with the same architecture. Our approach includes a mode-based pseudo GT generation for reducing uncertainty in pseudo GT and an outlier filtering to remove unreliable pseudo GT for student training. We validate our approach is generic by performing experiments on two recent state-of-the-art models with two benchmarks. The results demonstrate that our approach consistently and considerably boosts the localization performance in the target area.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper deals with the cross-view localization problem and considers the domain transfer issue for new areas. It proposes a teacher-student pipeline to improve the generation ability of existing works. This work assumes that the images of the target area are available, but there is no label. The experiments are conducted on VIGOR and KITTI dataset.

### Strengths
+ Generalization is very important for the localization system and the work is well-motivated.
+ The proposed method is evaluated on two widely used datasets with detailed evaluation metrics.
+ The writing is easy to follow.
+ The qualitative results look good. The ablation study and analysis are also well presented.

### Weaknesses
-	The proposed method is more like semi-supervised learning rather than weakly-supervised learning, as it generates pseudo labels for training on the target area. Both knowledge distillation and domain adaptation have studied similar problems.
-	The experimental setting could be improved to better support the motivation, i.e. generalization on new areas. The current setting seems to split the images of the target area into several parts. Although the label is not provided, the ground-truth pairs still exist in the training set of the target area, which is not the case for real-world applications. It is very common that some query images may not be covered by the reference images in the target areas. In other words, some images may not have the correct match in the training set of the target area. 

-	The performance improvement is limited, especially on KITTI dataset.

-	The computation cost of the proposed method is not discussed. Given that previous methods have achieved high accuracy on these datasets and the proposed method introduces additional computational cost, It is important to discuss the trade-off between the performance improvement and the additional computational cost.

### Questions
See the weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors propose a weakly-supervised learning approach using knowledge self-distillation to improve the cross-view localization performance in new target areas without accurate ground truth positions. However, the paper is flawed in terms of writing, innovations, and experiments.

### Strengths
+ This article presents a self-distillation framework to enhance the performance of models across domains.

### Weaknesses
- There is a lack of mathematical analysis as to why self-distillation frameworks are able to improve the fine-grained localization by only using coarse labels from target domain. Intuitive explanations and visualisation diagrams alone are not convincing enough.
- The ablation experiments are insufficient. Lacking of comparison with domain adaption methods, and enhancements over baseline methods do not entirely come from pseudo label supervision by the teacher.
- The test results are insufficient. Lack of indicators of the success rate of matching between ground and aerial images e.g. R@1, R@5, Hit Rate. A direct comparison of metre-level localization accuracy in the absence of a matching success rate is meaningless.

### Questions
1.	The proposed introduces the coarse-grained labels from target domain so it is considered a domain adaptation approach. Is the boost due to having seen the distribution of target domains, or is it due to the weak supervision of the pseudo-labels? The addition of ablation experiments to compare other domain adaptation methods is recommended.
2.	Why the poor model (the teacher) can lead good model (the student) in a good direction? Hopefully the authors will give solid mathematical derivations rather than intuitive descriptions and visualisations. This is because visualised heatmaps may simply come from success cases, which cannot be controlled at the time of review.
3.	For the self-distillation approach, it is necessary to maintain the teacher model and the student model in the memory, which is not too demanding in terms of computational resources? I am concerned about the ease of reproducing the method proposed in this paper and suggest that the computational cost be given.
4.	Lacking of indicators of the success rate of matching between ground and aerial images e.g. R@1, R@5, Hit Rate. A direct comparison of metre-level localization accuracy in the absence of a matching success rate is meaningless.
5.	As the most important general framework diagram, the font size in Figure 2 is too small and the overall flow is not clear and concise. It is recommended that it be redrawn.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
