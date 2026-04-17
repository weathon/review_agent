# Generalizable Active Learning: Boosting Out-of-Distribution Generalization in Active Learning with Simulated Generalization via Augmentation

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Active Learning (AL) aims to select informative samples for annotation within a constrained labeling budget. Existing AL methods typically focus on achieving high performance on Independently and Identically Distributed (IID) source data and often terminate once IID performance converges. However, we identify a crucial yet under-explored issue: despite IID convergence, models trained under AL methods often exhibit a significant performance gap in Out-Of-Distribution (OOD) scenarios compared to models trained on the full labeled dataset, and closing this OOD gap often requires a much larger labeling budget. To address this issue, we introduce the task of Generalizable Active Learning (GAL), which aims to improve the OOD generalization while preserving source-domain performance and minimizing additional labeling costs. We further introduce Simulated Generalization Active Learning (SimGAL), a framework that simulates generalization scenarios through data augmentation without incurring extra annotations. SimGAL comprises: (1) Simulated Generalization Augmentation (SGA), which generates augmented samples simulating OOD characteristics for the pool of labeled samples, and (2) Quality Stabilization Module (QSM), which filters out overly distorted augmented samples to ensure stable training. We design two train-test paradigms specifically designed for the GAL task. Experimental results demonstrate that SimGAL significantly enhances OOD generalization performance of AL methods under matched labeling budget and training sample sizes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper highlights a limitation of existing active learning (AL) methods: even after achieving convergence in in-distribution (ID) performance, the out-of-distribution (OOD) generalization performance remains significantly inferior compared to models trained on the full labeled dataset. To address this issue, the authors introduce a new task called Generalizable Active Learning (GAL) and propose Simulated Generalization Active Learning (SimGAL) as a solution. SimGAL consists of two key components: Simulated Generalization Augmentation (SGA), which applies learnable augmentation operations to labeled samples, and the Quality Stabilization Module (QSM), which filters out overly distorted samples based on pixel-level statistics. Experimental results on the Digits and PACS datasets demonstrate that SimGAL consistently improves OOD generalization compared to conventional AL methods.

### Strengths
* The paper addresses a practically meaningful Generalizable Active Learning (GAL) task, grounded in the observation that models trained via conventional AL methods can suffer from significant degradation in OOD generalization.
* The paper clearly differentiates GAL from related problem settings such as active transfer learning, multi-domain active learning, and open-set active learning, making it easy to understand what makes GAL distinct both in motivation and experimental setup.
* The proposed SimGAL framework is empirically validated across two benchmark datasets, Digits and PACS, demonstrating consistent improvements in OOD generalization across multiple standard AL query strategies. Experimental results highlight both its robustness and flexibility for integration with existing AL methods.

### Weaknesses
* While SimGAL shows empirical gains over standard augmentations such as AutoAugment and RandAugment, the advantages of the learnable augmentation strategy (SGA) are not deeply analyzed. The performance gap may diminish depending on how finely these baselines are tuned. A more principled or controlled analysis comparing SGA to advanced augmentation pipelines would strengthen the empirical justification.
* The optimization details of the augmentation parameters $\omega$ are underexplained. Since many standard augmentation operations are non-differentiable, it remains unclear how $\omega$ is updated via gradient-based methods in practice. Moreover, key hyperparameters in the SGA loss, such as $\lambda$ (which controls semantic similarity) and $\alpha$ (a Boolean parameter), lack motivation or ablation. These design choices play a critical role but are left largely implicit.
* The experimental setup does not fully follow common domain generalization evaluation protocols. For example, in the PACS benchmark, it is standard to rotate the source/target domain splits and report average performance across all combinations. This would better validate the generalizability of SimGAL.
* The paper presents the OOD generalization gap in active learning as a central observation and motivation. However, prior work, such as Deng et al. (2023), already reported this phenomenon, particularly in NLP. A more explicit discussion of how this work differs from and builds upon prior observations would clarify its unique contribution.
  * Deng et al. (2023), Counterfactual Active Learning for Out-of-Distribution Generalization, ACL

### Questions
* The objective in Eq. (2) only maximizes performance on the OOD test distribution, but doesn’t explicitly account for maintaining ID performance. Wouldn’t it be more appropriate to either include the IID objective directly or impose it as a constraint?
* Could the authors clarify why increasing the batch size $b$ would reduce the diversity of the augmented samples (Line 256)? The connection between batch size and diversity is not immediately obvious from the current explanation.
* Are the class-centered embedding precomputed from labeled data and kept fixed, or are they dynamically updated during training? If updated, what strategy is used to maintain and refresh these class centers?

### Soundness
1

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
5

### Summary
The paper introduces a new setting: generalizable active learning (GAL), and proposes a data augmentation-based strategy for improving the OOD performance while maintaining the source domain performance, called Simulated Generalization Active Learning. The proposed method has modules that create simulated OOD data while ensuring stable training and is evaluated for performance under budget criteria.

### Strengths
The overall presentation of the paper is clean. Main ideas are clearly explained. The evaluation shows detailed comparisons for each specified setting and provides some ablation study results and feature visualizations.

### Weaknesses
1. The main concern is with the proposed setting. What is the motivation for the generalization to OOD in this particular active learning setting? Figure 1 and Table 1 explain the setting clearly. However, I can not think of a scenario where we query data for label in A and test in A$\cup$B. Why would one assume any generalization ability in this setting? The only reason that the performance on B can improve with queries from A is the similarity between them. However, if we know that there is a similarity, shouldn't adaptation from A to B be a better option? If B is truly OOD and can not be adapted, then I see no reason that sampling from A can improve generalization on B. 
 
2. Another concern is with the model choice for evaluation. The experiments use ResNet-18 pre-trained on ImageNet, meaning that the feature extractor already embeds general knowledge. I'm not sure if generalization on digits and PACS images is meaningful in this setting. Again, what is a specific application where we need to actively sample on datasets like MNIST and generalize to SVHN, using a pre-trained ResNet model? 

3. There is not much theoretical analysis on the generalization problem in this setting. Although I don't think that a theory is necessary, it can definitely be helpful in understanding the setting, especially since distribution shifts and generalization are the main focus. 

4. The lack of discussion on Bayesian active learning methods in this setting is also a concern, due to similar reasons as the last point.

5. Some parts of writing are repetitive. For example, the two observations are mentioned multiple times in the introduction.

### Questions
Following weaknesses:

1. How do authors justify the setting?
2. What is the impact of pre-trained backbone?
3. Why the choice of AL baselines and why the lack of Bayesian AL methods, which can be reasonable for a distribution shift/generalization type of task?

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
This paper introduces SimGAL, a generalizable active learning framework designed to improve performance under unseen domain shift. The key idea is that traditional AL tends to overfit the source domain, leading to poor generalization on target domains. To address this, SimGAL proposes Simulated Generalization Augmentation (SGA), which learns to generate semantically consistent yet distribution-shifted samples by pushing augmented instances away from class centers while preserving their semantics. A Quality-based Sample Mining (QSM) module filters useful augmented examples to avoid semantic drift. Together, these components enable the model to “practice” handling domain shift during active learning without access to target-domain data.

### Strengths
This paper studies an interesting topic that standard active learning can easily overfit the source domain and then struggle when the model is used in a different setting. The main idea of this paper: simulating domain shift during active learning, is creative and makes the problem more realistic for many real-world scenarios. The paper is generally easy to follow, with intuitive motivation and explanations.

### Weaknesses
I have several concerns:
1. It's hard to say the improvement comes from introducing DG-style augmentation (i.e., simulated domain shift with semantic consistency) or advancing the active learning mechanism itself. Besides, in experiments, the paper includes augmentation-based AL baselines such as LADA, it is worth noting that LADA’s augmentation strategy focuses on generating mixed samples to improve uncertainty and diversity for sample selection, rather than explicitly simulating domain shift.  It is better to design new experiments to fully isolate the effect of cross-domain robustness from the benefits of generic augmentation.
2. Experiments are mainly conducted on relatively small-scale and early DG benchmarks (Digits, PACS).  Evaluation on more challenging modern DG benchmarks like WILDs should be included. Besides, these simple DG benchmarks (Digits, PACS), where even after noticeable improvements, the absolute OOD accuracy remains far from practical usability.

### Questions
1. It is still unclear about the definition of "simulated OOD". The transformations appear closer to strong semantic-preserving augmentations within the source domain rather than real OOD examples that genuinely lie outside the source distribution. 
2. The paper pushes samples away from their own class center to simulate domain shift, but could this potentially move them closer to other class centers? Why not constrain distances to all class centers to better preserve semantics?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work focuses that traditional Active Learning (AL) methods, while effective on IID data, often exhibit poor Out-Of-Distribution (OOD) generalization, and introduces the task of Generalizable Active Learning (GAL) to address this gap. The authors propose Simulated Generalization Active Learning (SimGAL), that improves OOD performance without extra labeling costs by using Simulated Generalization Augmentation (SGA) to create OOD-like samples from the labeled pool, coupled with a Quality Stabilization Module (QSM) to filter harmful augmentations. Experiments demonstrate that SimGAL significantly boosts OOD generalization for AL methods within the same labeling budget.

### Strengths
1. The paper is clearly written and easy to understand.

2. The paper achieves good performance.

### Weaknesses
1. There is a lack of introduction to the application scenarios for AL+OOD. This part indeed makes readers confused about this task setting, wondering if it is an important task worthy of research. For example, there are many typical application scenarios for AL: massive data annotation in autonomous driving; expensive data annotation in medical imaging, etc.

2. The proposed method focuses on using Data Augmentation to improve the model's generalization, and its relationship with AL is relatively independent, which makes the combination of the two seem somewhat forced. If AL is performed first, and then the method proposed in the paper is used, it seems that similar performance improvements as shown in the Table could also be obtained.

3. The proposed method adds a loss-driven component on top of conventional data augmentation methods to learn the parameters for various data augmentations. This is the core contribution of the paper. However, such modeling is a bit simple and lacks sufficient innovation.

4. To solve the problem of OOD generalization, would it be better to use unlabeled data for unsupervised/semi-supervised learning?

### Questions
1. The most critical question is still: if the Data Augmentation and AL are separated, performing AL first and then Data Augmentation. Can this also achieve similar performance improvements? If so, it indicates that the method proposed in this paper is more of a general data augmentation method; and the experiments in Table 2 can only reflect that this paper proposes a better data augmentation method.

2. In Table 2, why are the $N_L$ and $N_T$ of SimGAL inconsistent with those of the baseline?

### Soundness
3

### Presentation
3

### Contribution
2
