# Binary-Feedback Active Test-Time Adaptation

- Decision: Reject
- Scores: 8, 6, 6, 5, 5

## Abstract
Deep learning models perform poorly when domain shifts exist between training and test data. Test-time adaptation (TTA) is a paradigm to mitigate this issue by adapting pre-trained models using only unlabeled test samples. However, existing TTA methods can fail under severe domain shifts, while recent active TTA approaches requiring full-class labels are impractical due to high labeling costs. To
address this issue, we introduce a Binary-feedback Active Test-Time Adaptation (BATTA) setting, which uses a few binary feedbacks from annotators to indicate whether model predictions are correct, thereby significantly reducing the labeling burden of annotators. Under the setting, we propose BATTA-RL, a novel dual-path optimization framework that leverages reinforcement learning to balance binary feedback-guided adaptation on uncertain samples with agreement-based self-adaptation on confident predictions. Experiments show BATTA-RL achieves substantial accuracy improvements over state-of-the-art baselines, demonstrating its effectiveness in handling severe distribution shifts with minimal labeling effort.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Deep learning models often face performance issues when there are domain shifts between training and test data. since Test-Time Adaptation (TTA) methods have limitations, and active TTA approaches with full-class labels are impractical due to high costs, this paper proposes a Binary-feedback Active Test-Time Adaptation (BATTA) setting and a method named BATTA-RL to address the performance degradation of deep learning models due to domain shifts when testing. Specifically, in BATTA, an oracle provides binary feedback (correct/incorrect) on model predictions. This feedback is integrated in real-time to guide continuous model adaptation. A dual-path optimization framework is proposed to leverage binary feedback and unlabeled samples, which is balanced by reinforcement learning. Experimental results demonstrates the effectiveness of the proposed methods.

### Strengths
1. This paper proposes a novel and practical active learning test-time adaptation paradigm. It presents an innovative approach that combines binary feedback with unlabeled samples, effectively addressing the issue of domain shifts in deep learning models when testing. In this setting, an annotator provides binary feedback (indicating whether a model prediction is correct or incorrect) instead of full-class labels. This reduces the labeling burden significantly as binary feedback requires less information compared to full - class labels. For example, the human effort and error rate in providing binary feedback are much lower than in full-class labeling as demonstrated by previous studies.

2. The proposed method, BATTA-RL, shows promising results in improving model performance with minimal labeling effort, which is a significant contribution to the field of test-time adaptation.

3. The experimental setup is comprehensive, and the comparisons with existing methods are thorough, providing strong evidence of the superiority of the proposed paradigm.

### Weaknesses
1. It would be beneficial for the article to further explain why the reward functions for the two paths are set differently. For example, in Binary Feedback - Guided Adaptation (BFA), the reward function value is -1 in the incorrect case, while in Agreemend - Based Self - Adaptation (ABA), the reward function value is 0 in the case of disagreement. What are the considerations behind such designs? This clarification would enhance the understanding of the proposed method and its underlying mechanisms.

2. It would be advisable for the article to further explain in the appendix, using formulas if possible, how the baselines of previous TTA and ATTA are adapted to fit into the proposed setting to ensure fairness. This would provide more transparency and a deeper understanding of the experimental setup and comparisons made in the study.

### Questions
Please refer to Weaknesses.

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
The paper introduces Binary-feedback Active Test-Time Adaptation (BATTA), a novel TTA setting for adapting deep learning models to domain shifts at test time using binary feedback from human annotators. The authors address limitations in prior active TTA methods, which suffer from high annotation costs, especially in multi-class settings. To mitigate this, they propose BATTA-RL, a reinforcement learning-based approach with a dual-path optimization strategy. BATTA-RL combines Binary Feedback-guided Adaptation (BFA) for uncertain samples and Agreement-Based Self-Adaptation (ABA) for confident samples, enhancing model performance on challenging test distributions. Experiments across multiple datasets demonstrate that BATTA-RL outperforms existing TTA methods.

### Strengths
1. **Reduced Labeling Costs**: BATTA minimizes labeling demands by using binary feedback from human annotators instead of requiring full-class labels. This significantly reduces the annotation burden, making it more feasible for real-world scenarios than current ATTA methods.

2. **Dual-Path Optimization with Reinforcement Learning**: BATTA-RL combines Binary Feedback-guided Adaptation (BFA) for uncertain samples and Agreement-Based Self-Adaptation (ABA) for confident samples. Introducing reinforcement learning by binary feedback optimization is interesting and novel in TTA. 

3. **Strong Experimental Results**: BATTA-RL consistently outperforms competing TTA methods and even surpasses the ATTA method with full-class labels.

### Weaknesses
1. **Experimental Setting**: Appendix D.1 mentions multiple epochs of adaptation for all datasets. However, in TTA settings—where real-time, stream-based tasks are required—multiple epochs of adaptation are impractical. This approach hinders real-time inference capabilities.

2. **Scalability of BATTA**: Without extensive experimentation on large-scale datasets, the scalability of BATTA remains uncertain. Testing on a dataset like ImageNet-C with 1000 classes would be insightful. In scenarios with high model error rates, where most feedback might be "incorrect," there is a risk of BATTA collapsing. Additionally, as shown in Table 1(c), BATTA only surpasses SAR by less than 0.3%, which is a marginal improvement.

3. **Frequency of Human Intervention**: The requirement of annotating 3 samples per batch of 64 suggests a substantial annotation budget, even with binary feedback. The need for human intervention in every test batch implies that annotators must be continuously available while the system operates. Reducing the annotation budget and frequency could make the system more practical.

4. **Computational Complexity**: Monte Carlo dropout likely introduces additional computational demands. The manuscript lacks a thorough comparison of computational complexity against competing methods. The statement that “experiments were mainly conducted on NVIDIA RTX 3090 and TITAN GPUs, with BATTA-RL consuming 5 minutes on PACS” is vague. Specific hardware details and wall-clock time comparisons with other methods, per test sample, are necessary for clarity.

### Questions
Please refer to **Weakness**.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Motivated by the high annotation budget of the previous active TTA, where an oracle provides accurate ground-truth labels for selected samples, this paper defines a more realistic active test-time adaptation setting (binary feedback active TTA) with relatively weak assumptions. To achieve this, the authors propose an RL framework, BATTA-RL, consisting of Binary Feedback-guided Adaptation (BFA) and Agreement-based Self-Adaptation (ABA). BFA is proposed to learn from the valuable feedback information and ABA is proposed to improve self-training with MC dropout predictions. The evaluation experiments are conducted on CIFAR10-C, CIFAR100-C and Tiny-ImageNet-C. The results show the effectiveness and excellent improvement of the proposed method.

### Strengths
1. This paper introduces reinforcement learning to test-time adaptation and provides the unsupervised TTA method with low-cost feedback to improve the robustness of the TTA method.
2. The paper is well written and easy to follow, and the experiments on several benchmark datasets validate its effectiveness.

### Weaknesses
1. Using the ensemble predictions of multiple data or feature augmentation (e.g., dropout) to estimate the certainty of the samples or to obtain the robust predictions is already well known in the TTA tasks, such as MEMO[A] and CoTTA[B]. One suggestion for improvement would be for the authors to compare different uncertainty estimation strategies, e.g. those used in MEMO and CoTTA.
2. Using the ensemble predictions to obtain the uncertainty of the samples could be more time consuming. It's better for the authors to calculate the wall clock time for the proposed method and compare it with others.
3. To demonstrate the effectiveness of the proposed sample selection method used in BFA, a comparison with random selection is necessary. What is the performance of the BFA module using feedback from randomly selected test samples rather than those from top-k uncertainty samples?

[A] MEMO: Test Time Robustness via Adaptation and Augmentation
[B] Continual Test-Time Domain Adaptation

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes using binary feedback for active test-time adaptation (TTA) through reinforcement learning. It begins by employing MC-dropout to perform multiple forward passes. Based on the softmax output from MC-dropout, uncertain samples (those with low confidence) are selected and sent to an annotator for binary feedback. This feedback is then utilized in the learning process through the REINFORCE loss. For more certain samples, an Agreement-Based self-Adaptation (ABA) module is implemented, encouraging the model to maintain consistent predictions via a reward signal.

### Strengths
1. The concept of using binary feedback to guide the model is intriguing.
2. Utilizing MC-dropout to estimate sample confidence is also a promising approach.
3. Experimental results demonstrate significant improvements in model performance.

### Weaknesses
1. The signal acquisition process may be impractical; relying on an annotator could be costly or unrealistic in certain settings. However, advancements in foundation models may help address this issue soon.
2. The computational cost of MC-dropout is notable. Since multiple inference steps are required to gather this information, its application to TTA may raise efficiency concerns. Reviewers expect the authors to include experiments that evaluate the running time of the proposed method.
3. The fairness of the experiments is questionable. Given that this approach utilizes multiple MC-dropout steps, it effectively functions as a form of ensemble prediction, which naturally enhances performance. In contrast, other methods typically rely on a single forward pass.

### Questions
Please refer to the Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
BATTA is a novel approach for active test-time adaptation that uses binary feedback from a human to adapt pre-trained models to new domains, allowing reasonable labeling costs compared to methods requiring full-class labels. The BATTA-RL method integrates binary feedback on uncertain samples with self-adaptation on confident samples within a reinforcement learning framework. Extensive experiments demonstrate that BATTA-RL surpasses state-of-the-art test-time adaptation methods.

### Strengths
1. The approach of using reinforcement learning, instead of the commonly used (direct) entropy minimization objective or cross-entropy loss for TTA is intriguing and adds a new perspective.
2. The approach to not solely rely on human feedback but to incorporate self-adaptation on confident samples is realistic and efficient, making it a practical strategy for TTA.

### Weaknesses
1. The experiments conducted in this paper seem quite limited. It is common in current TTA to conduct evaluations on benchmarks such as ImageNet-C (additionally, ImageNet-R, and VisDA-2021). Also, to demonstrate that the use of reinforcement learning algorithms for active test-time adaptation is generally powerful, it would be important to include experiments in more complex scenarios, such as those proposed in SAR [1]: i) dynamic shifts in the ground-truth test label distribution leading to imbalanced distributions at each corruption, ii) single test sample scenarios, and iii) combinations of multiple distribution shifts in more challenging and biased situations.
2. Since the current experiments are conducted on relatively easier datasets like CIFAR-10 and Tiny ImageNet, it is likely that there are a relatively higher number of confident samples. However, in scenarios with fewer confident samples, such as those in ImageNet-C or ImageNet-R (severity level 5), wouldn't the burden of human feedback increase? In such cases, it appears that the performance may work similarly to SimATTA (full labeling). If so, what would be the key distinguishing factor of the proposed method?
3. Furthermore, as noted in DeYO [2], there can be cases where a model makes confident predictions due to spurious correlations. Would BATTA-RL still demonstrate outstanding performance when evaluated on benchmarks with significant spurious correlations? Including related observations or at least discussions on this point would be valuable.



[1] Niu, S., Wu, J., Zhang, Y., Wen, Z., Chen, Y., Zhao, P., & Tan, M. Towards Stable Test-time Adaptation in Dynamic Wild World. In The Eleventh International Conference on Learning Representations.


[2] Lee, J., Jung, D., Lee, S., Park, J., Shin, J., Hwang, U., & Yoon, S. Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors. In The Twelfth International Conference on Learning Representations.

### Questions
Please address the concerns mentioned above in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
