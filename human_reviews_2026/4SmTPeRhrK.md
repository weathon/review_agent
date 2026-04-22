# Positive and unlabeled learning incorporating additional posterior probabilities

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Learning from Positive and Unlabeled (PU) data presents unique challenges in scenarios where negative examples are absent. Many state-of-the-art PU methods are prior-based which assumes that the class probability within the unlabeled data corresponds to the class prior probability. However, this framework often falls short when attempting to accurately represent the complexities of real-world applications, such as industrial anomaly detection, where variations in data distribution within the combined training set are prevalent. In this paper, we introduce a generalized PU framework that models uncertainty via subset-specific posterior probabilities, proposing a posterior-based method (postPU) with theoretically and empirically validated consistency. Further, we establish that sample weighting is fundamental to PU robustness and derive a class-balanced weighting principle to minimize sensitivity to label inaccuracies. Experiments show the effectiveness and robustness of postPU and its capacity to leverage auxiliary uncertain annotations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Although conventional PU learning methods often assume a known class prior, this assumption is unrealistic, and the associated hyperparameter is highly sensitive. To address this, the paper proposes a generalized framework that assigns a posterior probability to each data subset in PU learning. By using estimable per-subset sample weights $\lambda^{+}$ and $\lambda^{-}$, the framework enables PU learning that is both robust to errors in the posterior probabilities and high-performing. The effectiveness of the proposed method is demonstrated theoretically and empirically.

### Strengths
- Once $\lambda^{+}$ and $\lambda^{-}$ are estimated, the procedure closely follows nnPU, making the proposed method simple yet effective.
- The method theoretically and experimentally exhibits strong robustness and its performance surpasses existing approaches.
- The explanation uses figures and tables effectively, making the presentation visually clear and easy to understand.

### Weaknesses
- I have several questions; please see Questions.
- Minor comments: There appear to be many places where \citet and \citep are used incorrectly.

### Questions
- How does the proposed method relate to class-prior-free PU learning approaches such as [1, 2] ? If possible, a quantitative comparison would be valuable.
- This paper primarily reports F1-score as the evaluation metric. Is there a reason for not using AUC? Especially for anomaly detection, AUC might be more appropriate.

[1] Chen, Hui, et al. "A variational approach for learning from positive and unlabeled data." Advances in Neural Information Processing Systems 33 (2020): 14844-14854.

[2] Zhao, Hengwei, et al. "Class prior-free positive-unlabeled learning with Taylor variational loss for hyperspectral remote sensing imagery." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

### Soundness
3

### Presentation
3

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
To address the dual challenges of inaccurate class prior estimation and limited representation in training labels, this paper proposes a posterior-based PU learning framework (postPU) that extends traditional prior-based PU learning by modeling uncertainty through subset-specific posterior probabilities. The authors introduce a class-balanced weighting principle to enhance robustness against class probability estimation errors and provide theoretical guarantees through generalization error bounds. Experiments on CIFAR-10, F-MNIST, and wind turbine anomaly detection demonstrate the method's effectiveness in leveraging auxiliary uncertain annotations.

### Strengths
(1) The extension from prior-based to posterior-based PU learning provides a more flexible formulation that can naturally incorporate uncertain annotations from multiple sources with different confidence levels.
(2) The potential for real-world scenarios is demonstrated through the wind turbine anomaly detection application.
(3) The paper is overall well written with strong theoretical contributions.

### Weaknesses
(1) The paper's motivation centers on addressing inaccurate prior estimation in traditional PU learning. However, the proposed solution pivots to requiring posterior probability estimates for each subset. While the method claims robustness to "coarsely approximated posterior values", the robustness demonstration in Table 4 still requires manual specification of posterior values, which test sensitivity to posterior estimation errors, but the baseline "accurate posterior probabilities" themselves require strong assumptions about the data generating process that may not hold. The experimental setup appears largely self-fulfilling.
(2) In lines 837-843, the paper describes constructing a semi-supervised environment using a heuristic criterion V that identifies samples meeting a specific criterion. But how is criterion V defined or learned? How are the posterior probabilities computed or estimated for each subset? Why should practitioners have access to reliable posterior estimates when prior estimation is already challenging?
(3) Several critical experiments are missing:
(a) The theoretical framework supports arbitrary $m \ge 2$ subsets, but experiments only evaluate $m=2$ and $m=3$ cases. Lack of experiments with $m>3$ leaves the scalability claims empirically unsupported.
(b) The paper lacks the hyperparameter sensitivity analysis on $\beta$.
(c) The comparison lacks several relevant recent PU learning methods, such as HolisticPU, PUL-CPBF, RobustPU and other methods from 2023-2025.

### Questions
Please refer to the Weakness.

### Soundness
2

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
3

### Summary
This paper presents a novel approach in PU learning by introducing postPU, which extends existing approaches to handle uncertainty through subset-specific posterior probabilities. The proposed postPU goes beyond the limitation of prior-based methods that assume the class probability within the unlabeled set is known. The authors provides solid theoretical analysis with convergence guarantees and generalization error bounds, and demonstrate enhanced robustness against existing methods.

### Strengths
1. This work relaxes the strict assumption of traditional PU learning, enables applications to more realistic scenarios with varying data distributions.
2. This work provides a principled approach to incorporate and handle uncertainty in training data.

### Weaknesses
While computational complexity is mentioned, there's insufficient analysis of practical scalability to large datasets or high-dimensional spaces. Runtime and memory comparisons with baseline methods are missing. 
Experiments focus primarily on image datasets. The evaluation should include diverse domains like text classification, time series analysis, or graph data where PU learning is commonly applied to demonstrate broader applicability.

### Questions
See above weakness section.

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
This paper considers a generalized PU learning setting where the class probability within the unlabeled set is unknown. It extends the prior-based PU learning as posterior-based PU learning and represents label uncertainty within unlabeled set by multiple class posterior probabilities, leading to a posterior-based PU risk estimator. Besiders, it also introduces a class-balanced weighting principle to enchance the robustness to estimation errors.

### Strengths
1. The studied problem is important to the literature.

2. Extensive experiments validate the effectiveness of the proposed method.

### Weaknesses
1. This paper attempts to represent label uncertainty by multiple class posterior probabilities for different subsets. However, how to partition data to several subsets is unclear, and the authors introduce an “automatic, albeit imperfect, labeling method that identifies samples meeting a specific criterion” for partitioning data. But what is the specific criterion?

2. In the proposed postPU, the authors estimate PU risks by weighting absolute risks in each subset as shown in Eq. (2). Thus, the weights are important for postPU and also affect its robustness shown in the proposed Theorems. However, how to calculate these weights? Although Eq.(9) presents a primary constraint for them with the posterior probabilities $\pi_i$, it depends on known posterior probabilities $\pi_i$. But the posterior probabilities $\pi_i$ are unknown in practice. 

3. Theorem presentation is difficult to parse; definitions of variables and assumptions should be made explicit before the statement, for example, Theorem 1.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
