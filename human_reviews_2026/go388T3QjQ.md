# Long-tailed Learning with Muon Optimizer

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Long-tailed recognition poses a significant challenge in deep learning, as models tend to be biased towards head classes, leading to poor generalization on underrepresented tail classes. A key factor contributing to this issue is that the optimization process for tail classes often stalls in sharp regions of the loss landscape. In this work, we investigate this problem from an optimization perspective and leverage the recently proposed Muon optimizer. We provide new theoretical insights, demonstrating that Muon's gradient orthogonalization enhances the update's projection along directions of negative curvature, thereby facilitating a more effective escape from sharp minima. To further mitigate the additional computational overhead of Muon, we propose Progressive Muon Optimizer (ProMO), a novel hybrid optimization approach that balances performance with efficiency. Specifically, ProMO employs a sinusoidal probability schedule to dynamically alternate between SGD and Muon. This method predominantly uses computationally efficient SGD in the early stages of training and gradually increases the use of Muon as the model approaches convergence when escaping sharp minima becomes critical for tail-class generalization. Extensive experiments on large-scale long-tailed benchmarks demonstrate that ProMO consistently outperforms existing long-tailed recognition methods. These results validate that ProMO effectively improves generalization on tail classes without incurring significant computational costs, highlighting its potential as a practical and effective solution for long-tailed learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates long-tailed learning from the perspective of the sharpness of the loss landscape. It argues that previous sharpness-aware minimization (SAM) methods and their variants incur high computational costs. Similarly, although the Muon optimizer can help models escape sharp regions, it is also computationally expensive. To address these issues, the paper proposes ProMO, a method that applies the Muon optimizer following a sinusoidal scheduling strategy to balance effectiveness and efficiency. Experimental results on several mainstream long-tailed datasets demonstrate the effectiveness of the proposed approach.

### Strengths
1. The paper is well-organized, and the overall writing flow is clear and logical, making it easy to follow.
2. The proposed solution is simple yet technically sound.
3. The theoretical and computational analyses provide valuable insights that strengthen the motivation and understanding of the proposed approach.

### Weaknesses
1. The technical novelty of the proposed method appears limited, as it primarily combines Muon and SGD in a hybrid scheduling manner. To strengthen the contribution, the authors are encouraged to provide deeper insights or analyses on the underlying training dynamics that motivate this design.
2. The paper does not include comparisons with several strong SAM variants, such as ImbSAM [1] and CC-SAM [2], which are highly relevant baselines in this context.
3. Experimental results on the iNaturalist-2018 dataset are missing, which limits the completeness of the evaluation.
4. According to the reported results, ProMO exhibits inconsistent performance improvements across different datasets.

Reference:

[1] ImbSAM: ACloser Look at Sharpness-Aware Minimization in Class-Imbalanced Recognition. ICCV 2023.

[2] Class-Conditional Sharpness Aware Minimization for Deep Long-Tailed Recognition. CVPR 2023.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical challenge of long-tailed learning from an optimization perspective by leveraging the Muon optimizer.
The main contributions are as follows:

1. It demonstrates that Muon’s gradient orthogonalization enhances updates along negative curvature directions, enabling the optimizer to escape sharp minima more effectively.

2. It introduces the Progressive Muon Optimizer (ProMO), a hybrid approach that dynamically alternates between SGD and Muon using a sinusoidal probability schedule to balance performance and computational cost.

In summary, the proposed Muon and ProMO optimizers show potential as replacements for conventional SGD-based methods, achieving improved overall performance at the cost of higher computational overhead.

### Strengths
1. The paper presents a solid theoretical analysis of the Muon optimizer, with Theorem 1 proving that it amplifies gradient projections along negative curvature directions under the Correlated Negative Curvature (CNC) assumption. This provides a sound theoretical foundation for understanding how Muon effectively escapes sharp minima.

2. The experimental evaluation is comprehensive and convincing, covering a wide range of mainstream loss functions and optimization algorithms.

### Weaknesses
1. The Muon optimizer appears to be a general optimization method applicable to existing models, but this paper lacks specific algorithmic design or adaptation specialized for long-tailed learning.

2. The comparison of fine-tuning strategies is missing. For example, recent methods such as LIFT and LPT are not included. 
In addition, the paper does not consider decoupled training strategies. 
It is also unclear whether using a re-balancing classifier would offset the reported improvements.

3. The Muon optimizer sometimes incurs higher computational overhead than SAM and does not consistently achieve superior performance, which may limit its practical effectiveness. (Nevertheless, the proposed ProMO variant shows better efficiency and performance trade-offs. Why not directly treat ProMO as the main part of the proposed mehtod rather than as an auxiliary extension?)

4. The paper does not provide code or implementation details, which prevents further verification and reproducibility of the proposed method.

### Questions
1. What does the deep blue line in the tables represent?

2. Will the authors release the code?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles the challenge of poor generalization in tail classes within long-tailed learning by examining the optimization process through the lens of loss landscape geometry. It demonstrates that tail classes are often prone to converging to sharper minima in the loss landscape. The authors further show that the recently proposed Muon optimizer enhances gradient projections along directions of negative curvature, enabling faster escape from sharp minima. However, Muon introduces considerable computational overhead. To address this issue, the paper proposes ProMO, a progressive hybrid optimizer that alternates between SGD and Muon according to a sinusoidal probability schedule, effectively balancing computational efficiency and generalization. Extensive experiments validate the effectiveness of the proposed method.

### Strengths
- The paper is generally well-written and easy to follow.
- This paper explores the application of the Muon optimizer in long-tailed learning and shows that it can help escape sharp minima, which is both interesting and meaningful.
- The experiments are fairly comprehensive, covering four mainstream benchmarks and comparing multiple methods and optimizers.

### Weaknesses
- As far as I know, the CNC assumption describes the property that determines an optimization algorithm’s ability to escape saddle points. However, a saddle point is not equivalent to a sharp minimum. In fact, [1] shows that the loss landscape of tail classes has a highly negative minimum eigenvalue, indicating convergence to saddle points, which hinders generalization. Therefore, it would be more informative to further examine whether the minimum eigenvalue under Muon is larger than that under SGD, to demonstrate that Muon facilitates escaping saddle points. In Fig. 2, this trend appears plausible, but it would be better to report the exact eigenvalue values for verification.
- More representative long-tailed learning methods, such as GPaCo [2], CC-SAM [3] and DirMixE [4], should be included in the related work for a more comprehensive review.
- There are also some typos, such as in Figure 1, where "Hessain metric" should be "Hessian metric".

-----

[1] Escaping saddle points for effective generalization on class-imbalanced data, NeurIPS 2022

[2] Generalized Parametric Contrastive Learning, TPAMI 2023

[3] Class-Conditional Sharpness-Aware Minimization for Deep Long-Tailed Recognition, CVPR 2023

[4] Harnessing Hierarchical Label Distribution Variations in Test Agnostic Long-tail Recognition, ICML 2024

### Questions
Please see above.

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
3

### Summary
The paper provides a rigorous theoretical analysis of the Muon optimizer, demonstrating its ability to escape sharp minima by enhancing gradient projection along negative curvature directions. This bridges optimization theory with practical long-tailed learning challenges. The proposed ProMO balances computational efficiency and performance, which has been empirically validated on real datasets. In summary, this paper addresses a significant long-tailed learning problem, offering a promising tool for the whole long-tailed learning community. However, the impact of the proposed optimizer on learned representations is somewhat unclear, as the paper focuses more on the efficiency and accuracy issues.

### Strengths
1. The proposed ProMO addresses a critical gap in long-tailed learning by balancing computational efficiency and performance. The sinusoidal probability schedule is simple yet empirically effective, outperforming alternatives like linear or exponential schedules.

2. The paper follows a clear structure and is easy to follow. The appendices provide pseudo-code and additional proofs, which enhance reproducibility. The figures and tables are well-designed and support key claims.

3. Evaluation is performed on multiple datasets and loss functions, demonstrating the robustness of the proposed method.

### Weaknesses
1. This paper focuses on optimizer design. However, for ICLR, which focuses more on the principles of learning representations, a stronger emphasis on how the method fundamentally improves representation learning beyond just faster convergence is preferable. Although the conference's subject areas include 'large-scale learning and non-convex optimization', my understanding is that it focuses more on how to address the fundamental challenges these issues pose to representation learning, rather than the efficiency limitations imposed by them.

2. The paper does not explore optimizations like gradient approximation or parallelization to further mitigate overhead.

3. The CNC assumption is not validated in a broader context. The claim that Muon’s orthogonalization always enhances negative curvature projection lacks edge-case analysis, e.g., near-convex regions.

4. The source code is not yet publicly available.

### Questions
1. The paper focuses on optimizer design. How does Muon fundamentally change feature learning, beyond just better convergence/efficiency compared to the existing optimizers for long-tailed learning?

2. Have the authors compared adaptive methods, e.g., Adam variants?

3. The experiments focus on vision tasks. Is this proposed optimizer promising on NLP or multimodal long-tailed datasets?

### Soundness
3

### Presentation
3

### Contribution
2
