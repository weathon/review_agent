# Label Smoothing Improves Machine Unlearning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
The objective of machine unlearning (MU) is to eliminate previously learned data from a model. However, it can be challenging to strike a balance between computation cost and performance when using existing MU techniques. Taking inspiration from the influence of label smoothing on model confidence and differential privacy, we propose a simple gradient-based MU approach that uses an inverse process of label smoothing. This work introduces UGradSL, a simple, plug-and-play MU approach that uses smoothed labels. We provide theoretical analyses demonstrating why properly introducing label smoothing improves MU performance. We conducted extensive experiments on several datasets of various sizes and different modalities, demonstrating the effectiveness and robustness of our proposed method. UGradSL also shows close connection to improve the local differential privacy. The consistent improvement in MU performance is only at a marginal cost of additional computations. For instance, UGradSL improves over the gradient ascent MU baseline constantly on different unlearning tasks without sacrificing unlearning efficiency. A self-adaptive UGradSL is also given for simple parameter selection.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a gradient-based machine unlearning (MU) method, the UGradSL, that integrates label smoothing (LS) into the unlearning in the ways of fine-tuning and gradient accent.  
The key idea is to perform gradient ascent with generalized label smoothing (GLS) on the forgetting set $D_f$, and gradient descent on the retaining set $D_r$, forming a mixed-gradient optimization.  
The authors show a theoretical condition under which NLS improves gradient ascent.
Experiments across multiple datasets and model architectures (ResNet18, ViT, BERT) show improved unlearning–retaining trade-offs with small computational overhead.

### Strengths
1. Easily pluggable into existing GA/FT unlearning pipelines without retraining.
2. Broad Empirical Evaluation. Tested on six datasets and multiple unlearning types (class, random, group), consistently outperforming baselines.
3. Provided theoretical intuitions. Explained why NLS can guide ascent toward the equivalent retrained solution.

### Weaknesses
1. The condition of the inner product of $\Delta\theta$'s in Theorem 2 is critical but only verified empirically on one dataset (CelebA, Figure 4 in the Appendix). This lacks theoretical justification and broader empirical validation.

2. Theorem 3's connection to LDP is interesting but the practical meaning is unclear - it provides label-level privacy for the forgetting set, but doesn't directly translate to model-level unlearning guarantees.

3. Hyperparameter sensitivity needs ablation studies. A demonstration about the smoothing rate $\alpha$'s influence on performance is expected. While an adaptive version is proposed, the distance threshold $\beta$ introduces another parameter.

### Questions
Besides the weaknesses:
1. In Algorithm 1, the distance computation $d(z^r_i, z^f_i)$ is not clearly defined. What distance metric? In which space (feature/parameter)?

### Soundness
3

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
4

### Summary
This paper integrates label smoothing into the unlearning loss to reduce time cost and mitigate drops in remaining and test accuracies. It proves that gradient ascent can achieve exact unlearning, and generalized label smoothing tightens errors. Building on this, it introduces UGradSL and UGradSL+, which iterate over retained and forgetting datasets, and show gains on class, random, and sub-class unlearning versus baselines.

### Strengths
1. Label smoothing combined with a gradient-based method is interesting for handling unlearning

2. Both theoretical and experimental evidence support the effectiveness of the proposed method.

### Weaknesses
1. The presentation of the technical sections is not clear, for example, why there exists an $\approx$ symbol in the condition of Theorem 1 and why $\epsilon$ in the conclusion of $\epsilon$-Label-LDP relies on the weights $\gamma_1$ and $\gamma_2$.

2. The proposed method requires calculating the distance with samples in the minibatch, which will lead to a large computation cost.

3. The proposed UGradSL does not work better in performance, while UGradSL+ requires a longer time than other baselines.

4. This paper does not contain essential abolition studies for the GA ratio and the optional smoothing ratio

### Questions
Please refer to the weaknesses.

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
This paper concentrates on the low-computation and high-efficiency machine unlearning in the field of computer vision. The author proposed a novel method UGradSL and UGradSL+ (finetune based) combine the GA (gradient ascent) with the NLS (negative label smoothing) and apply mix-gradient strategy which perform GD (gradient descent) on the retain data and GA with NLS on the forget data. Moreover, the stronger variant self-adaptive UGradSL supports the automatic selection of the smooth rate. The paper provides the theoretical analysis and abundant experiments to prove the effectiveness and robustness of method. In addition, the study also explored the label smoothing and local differential privacy (LDP).

### Strengths
+ This paper is clear, logical and easy to understand. The tables and figure are clear and detailed, with good instructions.
+ The proposed method is a simple and plug-and-play tool, which can directly integrate into the existing gradient-based unlearning methods (such as GA and finetune) and improve their performance.
+ This paper provides the mathematical and theory proof to explain the limitation of existing GA methods and the effectiveness of NLS in specific circumstances. Moreover, the authors innovatively establish the connection between label smoothing and LDP.

### Weaknesses
- Although the baseline methods are representative, the experiments lack comparison with the latest schemes between 2024 and 2025.
- Some symbols and formulas lack precise definitions and explanations or exist clerical, such as “distance d()” in the Algorithm 1 where different distance calculation methods can lead to the difference between computational overhead and performance.
- The performance of the method is sensitive to the hyperparameters settings (such as p and α in Eq.8), which may result in the risk of overfitting to the retain classes.
- Since the time cost is only a part of the computational overhead, the analysis of computational resource overhead is incomplete. And the automatic parameter selection in self-adaptive version also may lead to the additional time cost and computational complexity.

### Questions
* Could the authors do more comparative experiment with new methods, such as [1],[2] and etc.
* The authors should detailly check the symbols in the article and conduct standardized descriptions and definitions.
* The authors should analyze and explain the generalization gap between the RA and TA, where the increase of RA and decrease of TA reflect the overfitting and memory to the retain data rather than learning the generalization features.
* Since the efficiency is the core advantage of the proposed method, the authors should compare the memory cost during the unlearning process (such as peak GPU memory usage and etc.), which may help other researchers make decision with limited resources.
* Could the authors do another ablation experiment to the distance computation and automatic parameter selection in Algorithm 1, which helps the observation of whether the improvement of self-adaptive UGradSL comes at the cost of disproportionate calculation time.

Reference:
[1] Certified Unlearning for Neural Networks
[2] Towards Efficient Machine Unlearning with Data Augmentation: Guided Loss-Increasing (GLI) to Prevent the Catastrophic Model Utility Drop

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes UGradSL, which is a gradient-based unlearning methods that integrate label smoothing. The approach performs gradient ascent on forget data and gradient descent on retain data, using smoothed labels to balance forgetting and retaining.
Theoretical analyses explain when NLS improves gradient ascent and link the method to label-local differential privacy (Label-LDP) guarantees. Experiments on several datasets demonstrate the effectiveness of the proposed.

### Strengths
1. The paper has clear problem framing and the motivation for introducing label smoothing to stablilize gradient ascent is sound.
2. The theoretical analysis part is clear and provides useful insights.
3. The proposed UGradSL is simple yet effective.
4. The experiments cover class-level, random, and group unlearning and sufficient datasets and baselines.

### Weaknesses
1. The forget-set size: The forget-set size appears fixed for each experiment. Varying forget-set sizes is important for assessing the effectiveness of the method. This would reveal whether the proposed method scales well when more data need to be forgotten.
2. Lack of ablation study/sensitivity analysis: The paper lacks a discussion on the contribution of each term in the mixed gradient objective. For example, results for different p should be provided, since it is an important factor used to balance GD and GA.
3. Smoothing rate sensitivity: the effect of the label smoothing rate should be analyzed, and a discussion on how much the observed benefits arise from smoothing is worth adding in.

### Questions
1. About forget-set size: have the authors evaluated the method under different forget-set sizes or compositions? Since forget-set size can significantly affect both unlearning and retention, understanding how the method scales with it would help assess its general applicability.
2. In Eq (8), p controls the balance between gradient ascent (forgetting) and gradient descent (retaining). Could the authors provide experimental results for different p values? This would clarify how sensitive the method is to this balance.
3. How was the smoothing rate chosen? It would be valuable to include sensitivity results or justification for the chosen range.
4. How the density is computed in UGradSL+ and how it interacts with the theoretical framework?
5. The paper suggests that UGradSL can be easily integrated into other unlearning frameworks. Have the authors tested this claim empirically?

### Soundness
3

### Presentation
2

### Contribution
3
