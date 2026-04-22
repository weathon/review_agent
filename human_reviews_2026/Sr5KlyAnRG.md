# Protecting Membership Privacy through Adaptive Logit Scaling

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Machine learning (ML) models are susceptible to membership inference attacks (MIAs), where adversaries attempt to determine whether a specific data point is part of the model's training data. Recent studies suggest that MIAs often exploit the model’s overconfidence in predicting training samples, albeit using various proxy indicators. To mitigate this vulnerability, we introduce Adaptive Logit Scaling (ALS) loss, a simple yet effective modification to the standard Cross-Entropy loss. ALS adaptively constrains the norm of the output logits for each sample during training by decoupling and dynamically scaling overly large logits based on their magnitudes. The proposed approach reduces the models' overconfidence and ensures that they produce less distinguishable output metrics between member and non-member data. Extensive evaluations across four benchmark datasets show that ALS consistently achieves strong membership privacy while maintaining high model accuracy. Further comparisons with eight state-of-the-art defenses demonstrate that ALS effectively optimizes both sides of the privacy-utility trade-off, offering an effective and practical defense against MIAs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces an adaptive logit scaling loss, which scales large logits based on their magnitudes during training. Applying the loss to model training reduces the models’ overconfidence and produces less score distinction between member and non-member data, thereby mitigating the MIAs. Experiments show the performance of the method on various datasets.

### Strengths
1. The writing is clear and easy to follow. The paper clearly presents the proposed method through both mathematical formulations and explanations. The related work provides a comprehensive background that helps readers understand membership inference attacks. In addition, the presentation of the experimental results (including figures and tables) is clear and easy to follow.

2. Extensive experiments show that the performance of the proposed method is evaluated on various datasets across diverse attack and defence methods. The authors also conducted ablation studies on components of the method.

### Weaknesses
1. The paper demonstrates limited novelty. The authors incorporate LogitNorm, which was originally introduced to mitigate overconfidence and enhance out-of-distribution detection, into the training loss to defend against membership inference attacks. Although the authors add an entropy regularization term to LogitNorm, its function is similar to that of LogitNorm in mitigating overconfidence, as described in this paper. 

2. The proposed method lacks a clear justification. At test time, the method requires applying a logit scaling (Is this temperature scaling?) to the model’s outputs. The author claims that the operation is to further reduce the separability between member and non-member data. However, with a sufficiently large temperature, the model can effectively mitigate overconfidence, producing predictions that are nearly uniform. If so, it is unclear why we need to independently add an ALS loss.

3. The paper does not provide a detailed experimental setup. For reproducibility, the paper should also provide the experimental parameters used for the baseline methods. The paper does not provide the AUC score, which is a key metric for evaluating membership inference attacks. In addition, the authors should plot privacy-utility curves to show the privacy-utility trade-offs, rather than reporting only a single point result as in Figure 2. Furthermore, the performance of ALS is comparable to that of baselines on CIFAR-10.

### Questions
1. L1, L2 regularization, and early stopping can also prevent overfitting. How do these methods perform in defending against MIAs compared to ALS?
2. How effective is ALS in defending against gradient-based attack methods?
3. How does ALS perform when non-members are OOD samples?
4. In ablation studies, how effective is using only inference-time scaling as a defence?

### Soundness
2

### Presentation
3

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
This paper studies defense mechanisms against membership inference attacks (MIAs) on classification models. In particular, it proposes to use Adaptive Logit Scaling (ALS), i.e., a novel loss function that dynamically adjusts the scaling of logits based on their magnitude and incorporates an entropy regularization term.
The key idea is that samples with larger logit norms are more likely to be overfitted and thus leak membership information. ALS penalizes these samples by rescaling their logits while encouraging higher output entropy. Experimental results show that this approach substantially reduces attack true positive rates (TPR) at low false positive rates (FPR, e.g., 0.1%) while maintaining competitive model utility.

### Strengths
- The presentation is generally clear, and the paper is overall well-structured.
- The proposed approach is lightweight and easy to implement.
- The experimental results are comprehensive.

### Weaknesses
- The general insight (penalizing overconfidence to improve privacy defense) and the proposed approach, which applies a form of confidence regularization, are not particularly novel. Moreover, the paper does not provide deeper insights that clearly distinguish it from the broader generalization literature.

- The experimental results are not very strong, as the improvements over existing methods  are marginal in many settings.

### Questions
- The role of Proposition 3.1 in the overall argument is unclear. Why is the lower bound of the loss relevant to privacy leakage or privacy defense? Even the standard cross-entropy loss is lower-bounded, yet it can still easily leak membership information.

- It appears that the paper still focuses primarily on standard (or “normal”) data samples, while overlooking literature [1] suggesting that such regularization-based defense approaches may be less effective for real privacy protection, particularly for “hard examples” which are often of greatest interest in the privacy context as they represent the worst-case scenarios.

- After reviewing some of the baseline literature, it seems confusing that the main shortcoming you attribute to RelaxLoss is its utility degradation, which contradicts the main findings of the original paper. Is there any specific reason why the same experimental settings or results could not be replicated in this work?

[1] Aerni, Michael, Jie Zhang, and Florian Tramèr. "Evaluations of machine learning privacy defenses are misleading." Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security. 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Adaptive Logit Scaling, a modification to the standard Cross-Entropy loss to mitigate MIAs in machine learning models. ALS adaptively constrains the norm of output logits during training to reduce model overconfidence, thereby decreasing the distinguishability between member and non-member data. The authors validate their effectiveness through extensive experiments on four benchmark datasets. ALS is compared against eight state-of-the-art MIA defense methods, demonstrating superior performance in balancing privacy and model utility without significant computational overhead.

### Strengths
1. The solution is elegantly simple yet effective, requiring minimal changes to existing training pipelines.
2. The evaluation is thorough, testing across multiple datasets, attack types, and defense baselines.

### Weaknesses
1. It is recommended to include recent attack baselines, such as RMIA.
2. Fig.2 presents only a single privacy-utility point for each defense method, which is insufficient for a comprehensive evaluation of the privacy-utility trade-off. Additionally, one panel in Figure 2 combines different defense methods corresponding to various MIAs, which does not provide a justified assessment. I recommend plotting the privacy-utility curve in one panel for a certain attack, similar to RelaxLoss, to provide a more comprehensive evaluation of the trade-offs between privacy and utility.
3. The ablation studies in Table 3 are insufficient. I suggest including results from privacy-utility curves with varying values of $\alpha$ and $\lambda$ to better demonstrate the impact of these hyperparameters.
4. I question the claim that the model with a training accuracy of 97.06% (Purchase100 in Table 1) exhibits the entropy distribution centered at 4.605. If hyperparameters were tuned solely to achieve this result, it would undermine the significance, as most privacy defense methods can achieve similar outcomes. I recommend including the accuracy in Figure 1 and ensuring that the compared methods exhibit comparable test accuracy for a fair evaluation.

### Questions
Proposition 3.1 provides a lower bound for the loss. Could you elaborate on the theoretical connection between minimizing the ALS loss and the mechanism of reducing the distinguishability between member and non-member distributions?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel defense mechanism called Adaptive Logit Scaling (ALS) loss function. It aims to mitigate the overconfidence of deep learning models by constraining the norm of output logits, thereby reducing privacy leakage in Membership Inference Attacks. Unlike traditional cross-entropy loss functions, ALS prevents the model from making overconfident predictions on training samples by dynamically adjusting the scale of logits for each sample during training and introducing an entropy regularization term to encourage high-entropy outputs. This reduces the distinguishability between member and non-member data. Experimental results show that ALS, evaluated against eight MIA attacks and nine baseline defense methods, demonstrates superior privacy protection performance while maintaining high model accuracy.

### Strengths
Strongness
1、Innovation and Practicality: The ALS loss is a simple yet effective defense mechanism. It can be implemented by merely modifying the loss function, requiring no additional data or complex operations during the inference phase. It significantly enhances privacy protection capabilities without causing substantial degradation in model performance. This lightweight design enables easy integration into existing training pipelines, meeting the needs of practical applications.
2、Comprehensiveness of Experimental Design: The paper evaluates eight MIA attacks (including state-of-the-art LiRA and label-only attacks) and compares ALS with nine other defense methods across four datasets, covering both privacy metrics and model utility. Consistent results show that ALS reduces privacy leakage while the drop in accuracy is negligible, demonstrating its robustness.
3、Theoretical Support and Cross-Domain Value: The paper derives the theoretical lower bound of the ALS loss function, providing mathematical proof for the rationality of the method. Although further deepening is possible, this theoretical work already surpasses most empirical defense studies.

### Weaknesses
Weaknesses
1、Insufficient Depth in Comparison with Existing Work: Although the paper compares ALS with nine defense methods, it does not thoroughly discuss the core differences between ALS and similar approaches (e.g., LogitNorm). For instance, LogitNorm uses fixed norm scaling, while ALS emphasizes adaptiveness. However, the paper only provides qualitative demonstration through Figure 1, without quantitative analysis of the advantages of this adaptiveness on specific samples.
2、Limitations in Experimental Scope: The paper mainly focuses on image and tabular datasets, failing to cover extreme data scenarios such as small-sample datasets, highly imbalanced category data, or high-noise data. It also does not involve more sensitive domains (e.g., medical or text data) nor test large-scale models (e.g., Transformers). Additionally, despite the comprehensive attack evaluation, adaptive attack scenarios (e.g., attackers being aware of the defense mechanism) are not considered, which may overestimate the actual protection capability.
3、Limited Depth in Theoretical Analysis: While the paper provides a proof for the lower bound of the ALS loss, it does not deeply explore its connection with generalization ability or privacy theory. For example, the lower bound only depends on the number of categories k and the hyperparameter λ, with no analysis of its robustness under complex models or distribution shifts. This may limit the theoretical universality of the method.

### Questions
n.a

### Soundness
3

### Presentation
3

### Contribution
3
