# 3DPAN-CIL: a prototype assisted network of class-incremental learning for 3D point clouds

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
In response to the continuous influx of 3D point clouds encountered in practical training scenarios, we propose a novel incremental learning and classification approach designated as 3DPAN-CIL, specifically tailored for 3D point cloud data. This method initially establishes the 3D category prototype that encapsulates the feature embedding of point clouds within a latent space. Then, we wisely construct an optimal transport strategy on this prototype space for the migration of 3D category prototypes. This alignment ensures that the distribution of new category prototypes adheres as closely as possible to the relative spatial distribution of old category prototypes, significantly reducing the catastrophic forgetting in the training model. Additionally, to tackle the challenge of imbalanced old and new samples, we introduce a prior-guided knowledge distillation strategy aimed at addressing the model’s preference for new knowledge. We conduct a series of experimental evaluations on both synthetic datasets and real scanning datasets, demonstrating that our method surpasses existing state-of-the-art approaches in terms of average accuracy and average forgetting rate. Notably, in the context of average scene partitioning, our method achieves improvements of 4.5% in average accuracy and 1.47% in average forgetting rate compared to other top-performing methods. The model and code are available at: https://github.com/FlRiver/3DPAN-CIL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the application of optimal transport (OT) in incremental learning for 3D point clouds. It proposes a novel framework and provides detailed experimental validation. The study contains interesting findings and contributes valuable insights. However, there are certain aspects that the authors need to address.

### Strengths
1. This paper adheres to the standard academic introduction paradigm, outlining the broad context, specific problems, and limitations of existing approaches.

2. It clearly identifies the core challenges in incremental learning for 3D point clouds: catastrophic forgetting and the imbalance between new and old categories. This demonstrates that the authors have indeed recognised the critical pain points within this field.

### Weaknesses
1. It is recommended to elaborate more thoroughly on the fundamental differences between incremental learning for 3D point clouds and that for 2D images.
2. Whilst the paper identifies shortcomings in existing approaches, a more thorough comparative analysis is recommended. It is suggested to conduct a detailed examination of one or two representative 3D or 3D-adapted incremental learning methods (such as those based on replay or knowledge distillation), elucidating the specific challenges they encounter when processing 3D point cloud data streams. These challenges should encompass aspects such as feature representation stability, preservation of inter-category geometric relationships, and robustness against noise and occlusion.
3. The paper claims to ‘introduce OT for the first time’, which is a strong assertion requiring more thorough contextual support and differentiated analysis. How does your approach fundamentally differ from other works that may have applied OT, in terms of problem formulation, objective function design, or integration with 3D data characteristics?
4. ‘prototype’ and ‘OT alignment’ are not novel concepts in 2D vision. Therefore, the paper's argumentation regarding innovation should be strengthened.

### Questions
1. The paper's use of multiple datasets for evaluation is commendable. It is recommended to briefly justify the selection of these specific datasets (e.g., ModelNet, ShapeNet, ScanObjectNN) within the main text.
2. To ensure the timeliness and comprehensiveness of experimental comparisons, it is strongly advised to incorporate comparative analyses with recently published, directly relevant 3D incremental learning studies or closely related work within the relevant sections.

### Soundness
2

### Presentation
2

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
To address the challenge of continuously arriving 3D point cloud data, this paper proposes 3DPAN-CIL, a class-incremental learning method. This method leverages prototype-based optimal transport and prior-guided knowledge distillation, which effectively mitigates catastrophic forgetting and the imbalance between old and new knowledge. And it shows great performance improvement in the experiment.

### Strengths
1. This article demonstrates a certain level of innovation. The authors propose a novel optimal transport method called k-3DCPOT. This approach leverages optimal transport to align the old categories prototype generated by the network at stage t with those acquired at stage t-1, thereby preserving discriminative capability for previously learned categories and mitigating catastrophic forgetting during the incremental learning process. And they also propose the Priori guided knowledge distillation which can preserve the retention of old knowledge.

2. The article demonstrates clear and logical writing. It is not difficult to understand the paper.

3. The comparative experiment is well-designed and comprehensive, featuring extensive evaluations against numerous existing works across multiple datasets.

### Weaknesses
1. The figure illustrations in this article still require improvement. For instance, Figure 1(a) may use $L_{k-3DCPOT}$ instead of $L_{3DCPOT}$. Additionally, more variables used in subsequent derivations, such as $Z_{t}$ and $P_{t}$, should be included in the figure to enhance readability. The visual clarity of Figures 2(a) and (b) is also insufficient, and they lack detailed explanations.

2. It lacks citations and comparison of recent work, such as [1][2], which fails to demonstrate its competitiveness.

3. The authors conducted comparative experiments focusing solely on the memory consumption of the proposed method without evaluating its computational resource consumption. Meanwhile, they noted in the appendix that the time complexity of the method scales quadratically with the number of classes, yet claimed this to be acceptable. This raises doubts about the method's performance, particularly as the authors did not test its scalability on large-scale class scenarios.

 [1] Qi, Chao, et al. "Boosting the Class-Incremental Learning in 3D Point Clouds via Zero-Collection-Cost Basic Shape Pre-Training." arXiv preprint arXiv:2504.08412 (2025).

 [2] Xiang, Tuo, et al. "Seeing 3D Through 2D Lenses: 3D Few-Shot Class-Incremental Learning via Cross-Modal Geometric Rectification." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

### Questions
1. Why choose Point-BERT as the feature extractor?
2. Have the authors considered the adaptation of this work to other 3D point cloud-related tasks?
3. Have the authors tested the method under smaller memory constraints? The minimum memory limit set in the paper was only 400 features.

### Soundness
3

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
This paper proposes 3DPAN-CIL, a class-incremental learning method for 3D point clouds that leverages optimal transport for prototype alignment and prior-guided knowledge distillation to mitigate catastrophic forgetting. The technique introduces batch-wise optimal transport with a KL-constrained Sinkhorn distance to migrate new category prototypes while preserving their relative spatial distributions with respect to old prototypes. Additionally, a sample-size prior guides knowledge distillation to address old/new class imbalance. The experimental results on ModelNet40 demonstrate state-of-the-art performance, with an average accuracy of 85.25% and an average forgetting rate of 4.02, outperforming 11 baseline methods.

### Strengths
1. The paper introduces OT-based prototype migration to the 3D point cloud domain with batch-wise formulation, providing theoretical grounding with a proof that Sinkhorn distance satisfies distance axioms in continuous form. This addresses the limitation that previous methods use sample-level distillation, which suffers from off-centre distillation issues.

2. The evaluation covers both synthetic (ModelNet40, ShapeNet) and real-world datasets (ScanObjectNN, ScanNet, CO3Dv2), comparing against 11 baseline methods, including both 3D-specific (LwF-3D, I3DOL) and adapted 2D methods (iCaRL, BiC, DECO). The consistent improvements across diverse datasets demonstrate robustness.

3. The paper identifies three key challenges in 3D point cloud incremental learning: distribution shift causing forgetting, prototype disparity between old and new categories, and data imbalance favoring new classes. Each technical component (OT-based migration, category prototype migration, PGKD) directly addresses one of these challenges.

4. Module ablation demonstrates that both PAN and PGKD contribute meaningfully, with the removal of all modules reducing ModelNet40-avg performance by 4.24%. Memory size ablation shows consistent gains across different memory budgets.

5. On ModelNet40-avg, the method achieves 85.25% average accuracy versus 82.05% for the next-best method (DECO), with an average forgetting rate of 4.02% versus 6.38%. Similar improvements are observed on other datasets

### Weaknesses
1. The paper claims to be the ”first use” of class prototype space construction and the ”first to introduce” optimal transport in this context. Prototype-based methods are standard in class-incremental learning, and optimal transport has been used in related 2D CIL work like Co-Transport[1]. The novelty lies in the specific application and batch-wise formulation for 3D, which should be more precisely stated.

2. Since the main method is based on Optimal Transport, the most relevant prior work, Co-Transport (ACM MM 2021)[1], is not included in the experimental comparisons. Without this, it’s difficult to assess the true incremental contribution over existing OT-based methods.

3. While the Appendix provides a theoretical time complexity analysis, it lacks a practical comparison of memory usage during training, or FLOPs, against the baseline methods


[1] Zhou, Da-Wei, Han-Jia Ye, and De-Chuan Zhan. ”Co-transport for class-incremental learning.” Proceedings of
the 29th ACM International Conference on Multimedia. 2021

### Questions
1. Can the authors provide a comparison against Co-Transport (ACM MM 2021)? If you implemented it for 3D, what were the results? If not, could you add a detailed discussion in the related work section on the key algorithmic differences that make your approach novel?

2. Can the authors provide a practical computational analysis, including training time and peak memory usage compared to key baselines like EASE, DECO, and I3DOL?

3. In the conclusions section, you mention limitations on sparse and incomplete point clouds. Could you quantify this?

4. At what level of sparsity or incompleteness does performance begin to degrade significantly?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The article presents a novel class-incremental learning method for
classification of 3D point clouds. The incremental learning aims to
ensure that prototypes of old classes do not change when new classes
are introduced. Optimal transport is used to this end. Prototypes of
new classes are placed among the prototypes of the old classes, while
maintaining their relative distances of older classes. This framework
is combined with a dual branch architecture and distillation to yield
the final class-incremental learning model. Experiments in 5 publicly
available datasets are presented. A large number of alternative models
are used in comparisons.

### Strengths
+ The approach in the prototype space is interesting.
+ The migration part given in Section 3.3.3 with equation 8 and what
  follows is indeed interesting. This concept is rather new and can
  lead to further developments.
+ Integration of distillation in their framework is potentially
  useful. It is well done.
+ The comparisons are made with a large number of models.
+ The results are very promising.
+ Ablation studies are performed to provide an insight to the
  contribution of different model components.

### Weaknesses
- Method description can improve, specifically
  + what is $k$ and $g$ in line 129?
  + $Z$ is used twice in the same paragraph but it does not seem to
    refer to the same thing, I guess. Please clarify.
  + What does "we employ the optimal transport method to assess
    $P_t^{old}$ and to ascertain the variance between $P_t^{old}$ and
    $P_{t-1}$? Do you use optimal transport to ensure they are the
    same?
  + $k$ is used to represent the prototype dimension on page 3 and on
    page 4 it is used to define marginal distributions. This is rather
    inconsistent.
  + What does "quantities of features associated with class $i$ mean
    in Equation 4? This equation seems like averaging of features
    across the samples belonging to class $i$.
  + Equation 5 is not understandable. The summations are over $k$ and
    $l$ but they do not appear in the $d(\cdot, \cdot)$.
- I do not understand the use of optimal transport in this paper.
  + As far as I can see, the model has one prototype vector per
    class. How does one construct the prototype of the old classes in
    phase $t$?
  + Is there an ambiguity in the correspondences between the
    prototypes between model from phase $t-1$ and the model at $t$? 
  + During training phase $t$, the model does not have any samples
    belonging to the classes in phase $t-1$. So what does Equation 5
    exactly do? What are $x_j^i$?

### Questions
- Does the method use positional encoding in the feature extraction?
  If so, how is that defined?
- Your results are really good and I think the model makes
  sense. However, the model description is really not very good. Can
  you please provide a better description of the model, focusing on
  the points I raised above?

### Soundness
3

### Presentation
2

### Contribution
3
