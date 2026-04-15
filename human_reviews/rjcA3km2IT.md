# Variation-Bounded Losses for Learning with Noisy Labels

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 6, 1, 5, 8

## Abstract
The presence of noisy labels poses a significant challenge for training accurate deep neural networks.
Previous works have proposed various robust loss functions designed to address this issue, which, however, often suffer from several drawbacks, such as  underfitting or insufficient noise-tolerance. Furthermore, there is currently no reliable metric to guide the design of more effective robust loss functions.
In this paper, we introduce the *Variation Ratio* as a novel metric to measure the robustness of loss functions.  Leveraging this metric, we propose a new family of robust loss functions, termed *Variation-Bounded Losses* (VBL), characterized by a bounded variation ratio.
We investigate theoretical properties of variation-bounded losses and prove that a smaller variation ratio would lead to better robustness. Additionally, we show that the variation ratio provides a more relaxed condition than the commonly used symmetric condition for achieving noise-tolerant learning, making it a valuable tool for designing effective robust loss functions.
We modify several commonly used loss functions to the variation-bounded form.
These variation-bounded losses are characterized by their simplicity, effectiveness, and theoretical guarantees.
Extensive experiments demonstrate the superiority of our method in mitigating various types of label noise.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper first proposed a novel metric called Variation Ratio to measure the robustness of loss functions based on the theoretical investigation that a smaller variation ratio would lead to better robustness. Then the authors proposed a family of robust loss functions called Variation-Bounded Losses for learning against label noise by leveraging this metric. Extensive experiments demonstrated that the proposed method outperforms existing robust loss functions over various benchmark datasets with different types of label noise.

### Strengths
1. This paper is well-written and easy to follow.
2. The proposed Variation-Bounded Losses are built on the property analysis on Variation Ratio, which is theoretically sound and practically relevant.
3. This paper conducted extensive experiments over various benchmark datasets and baseline approaches with different label noise and noise rates.

### Weaknesses
1. This paper introduced a list of hyperparameters (i.e., $\alpha$, $\beta$, and a) which varied across different datasets under different noise rates. It might limit the application scenarios.
2. The superiority of VBL compared with the other asymmetric loss functions is unclear.  As introduced in this paper, the common drawbacks of existing asymmetric loss functions are overly complicated with numerous hyperparameters and often suffer from underfitting. However, the proposed approach also needs a list of hyperparameters and does not have clear clarification about the underfitting issue.

### Questions
1. As the authors only provide hyperparameter details on CIFAR-10/100 in Table 6, how about the hyperparameter choice on datasets in Table 5? Could you please explain how to choose those hyperparameters across different datasets?

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
This paper addresses the challenges of learning with noisy labels. The authors propose a new metric called the *Variation Ratio* to measure the robustness of loss functions. Based on this metric, a new family of robust loss functions, *Variation-Bounded Losses* (VBL), are proposed. The authors analyze the properties of variation-bounded losses theoretically and found that a smaller variation ratio can achieve better robustness. Compared to the symmetric condition, the variation ratio can provide a more relaxed condition to achieve noise tolerance. The experiments on several datasets (CIFAR-10, CIFAR-100, WebVision, ILSVRC12, and Clothing1M) demonstrate the effectiveness of the proposed method. Additionally, sensitivity tests for the hyperparameter $a$ are provided.

### Strengths
1. The authors proposed a new metric called variation ratio to measure the robustness of loss functions. This metric can provide a more relaxed condition to achieve noise tolerance.
2. Detailed properties of variation-bounded losses are analyzed theoretically.
3. The authors generalize commonly used loss functions (Cross Entropy, Exponential Loss and Mean Square Error) to the variation-bounded form.
4. Sufficient experiments are conducted. The authors not only evaluate the test accuracy of the proposed losses on the benchmark datasets but also provide sensitivity tests for the hyperparameter $a$.
5. The authors visualize the learned embeddings using t-SNE on the CIFAR-10 dataset. The visualization results show the embeddings for the proposed losses are well-separated clusters.

### Weaknesses
In the experiments conducted on the CIFAR-10 dataset, the authors report the results for VCE, VEL, and VMSE. However, the results of these methods for the CIFAR-100 and CIFAR-100N datasets are absent. Only the results of NCE-VCE, NCE-VEL, and NCE-VMSE are reported. This inconsistency in reporting across datasets may hinder a comprehensive evaluation of the methods' performance. It would be beneficial for the authors to explain the motivation behind this.

### Questions
In Figure 1, the test accuracy of VCE with $a=10$ is noted to be lower than that with $a=4$. The authors claim that a too small variation ratio may reduce the fitting ability. When the training epoch increases, whether the test accuracy of VCE $a=10$ can surpass that of VCE $a=4$?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
In this article, authors focus on measuring the noise-tolerance of robust loss functions which are used to address noisy labels. The paper points out the limitations of existing robustness discriminative conditions, including symmetric and asymmetric conditions, and proposes a new metric (variation ratio) for loss functions. And authors theoretically analyze the relationship between the proposed metric and symmetric and asymmetric conditions. Additionally, this paper develops several robust losses based on the variation ratio.

### Strengths
This paper establishes a theoretical relationship between variation ratio and both symmetric and asymmetric conditions, demonstrating its effectiveness. Furthermore, variation ratio simplifies the discriminative conditions for asymmetric losses.

### Weaknesses
- The author analyzes the deficiencies of three kinds of robust losses in the introduction. However, the manuscript does not clearly indicate how the proposed loss functions address these issues, such as the slow convergence associated with symmetric loss functions.
- The author is suggested to emphasize the advantages of the proposed variation-based losses in comparison to loss functions satisfying symmetric or asymmetric conditions.
- The proposed variation ratio can not effectively measure the robustness of some typical robust losses, such as GCE mentioned in the manuscript, which exhibits robustness under noise rates exceeding 60%.

### Questions
- It appears that the proposed variation ratio is derived by dividing both the numerator and denominator of the asymmetric ratio [1] by $\Delta u$. However, the calculation of the variation ratio only relies on $L_{active}$, which may lose information about other classes.
- What’s the meaning of “partial noisy labels” in the introduction? It seems that the article lacks experiments about this noise type.

[1] 2021 AAAI Asymmetric Loss Functions for Learning with Noisy Labels

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper investigates new robust loss functions for label noise based on the concept of variation ratio. A smaller variation ratio can potentially enhance robustness, as evidenced by excess risk bounds. The best bounds are achieved when the symmetry condition is satisfied (corresponding to the smallest variation ratio). On the other side, larger ratios may lead to loss functions that are easier to train. Additionally, when the variation ratio is sufficiently small, it it shown that the asymmetry condition is met.

Experiments are conducted with three main loss functions—Variation Cross Entropy (VCE), Variation Exponential Loss (VEL), and Variation Mean Square Error (VMSE)—on standard benchmarks with symmetric, asymmetric, instance-dependent, and natural label noise.

### Strengths
1) The variation ratio is a novel idea and the paper introduces novel loss functions based on this idea.
2) The problem of fighting label noise is of great practical importance and developing robust loss functions that are easier to train is an interesting line of research.
3) Promising Empirical Performance: The empirical results are promising, with the proposed loss functions consistently outperforming previously introduced robust loss functions across multiple benchmarks.

### Weaknesses
1) Definition of the Variation Ratio: The variation ratio is defined in terms of a decomposition of a loss function as the sum of active and passive terms. However, this decomposition is not unique in general. For example, consider the Mean Absolute Error (MAE) loss, $1−u_y$​. It can be decomposed with an active term $l_{active}(u_y)=1−u_y$​ and passive terms $l_{passive}(u_k)=0$ for $k\neq y$. Alternatively, we could represent it as an active term $l_{active}(u_y)=0$ and passive terms $l_{passive}(u_k)=u_k$ for $k\neq y$, since $1−u_y=\sum_{k\neq y}u_k$​. In the first decomposition, the variation ratio is 1, while in the second decomposition, it becomes undefined (0/0). The variation ratio as currently defined in Definition 2 is not an intrinsic property of the loss function but rather depends on the chosen active-passive decomposition. This ambiguity suggests an issue in the definition of the variation ratio. This is very important to solve in my opinion since the variation ratio is the central concept of the paper.

2) Hyperparameter Tuning: In the previous work by [Ye,2023], hyperparameters for the loss functions were selected based on performance at 80% symmetric noise, and the same hyperparameters were applied across different noise rates and types (asymmetric, CIFAR-N). In contrast, the current paper re-tunes hyperparameters for each noise type, which could introduce a favorable bias in the comparisons with prior work. Additionally, the paper should specify the range of hyperparameters considered during tuning to get a sense of the cost for tuning the method. This is particularly relevant since the loss functions in this paper require three hyperparameters, compared to only two in [Ye,2023] and [Ma,2020] (excluding the L1-regularization parameter).

[Ye,2023]: Xichen Ye, Xiaoqiang Li, Songmin Dai, Tong Liu, Yan Sun, and Weiqin Tong. Active negative loss
functions for learning with noisy labels. In Thirty-seventh Conference on Neural Information
Processing Systems, 2023.

[Ma,2020]: Xingjun Ma, Hanxun Huang, Yisen Wang, Simone Romano, Sarah Erfani, and James Bailey. Normalized loss functions for deep learning with noisy labels. In International conference on machine
learning, pp. 6543–6553. PMLR, 2020.

### Questions
1) How can you solve the issue in the definition of the Variation Ratio described in the point 1) of the weaknesses?
2) Can you provide more clarification about the tuning of the hyperparameters for your loss functions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The manuscript introduces the variation ratio, a new metric designed to control label noise in supervised learning. Label noise – which results from human error or incomplete labeling – often degrades the performance of deep neural networks. By using the variation ratio to evaluate the robustness of loss functions, the authors develop a new family of robust loss functions called variation-bounded losses (VBL). These functions have bounded variation ratios, and the work provides theoretical proof that a lower variational ratio results in higher noise tolerance. This approach provides a more flexible alternative to conventional symmetric loss functions, which often have a reduced fitting ability. 

The authors generalize commonly used loss functions to a  variation bounded form, such as Variation Cross Entropy (VCE), Variation Exponential Loss (VEL), and Variation Mean Square Error (VMSE). These adaptations aim to retain the effectiveness and simplicity of the original loss functions while improving their robustness to label noise. Theoretical analyses confirm that VBLs achieve robustness without the added complexity of many hyperparameters typically found in asymmetric loss functions.

Extensive experiments demonstrate the practical advantages of variation-bounded losses in various datasets with synthetic and real-world types of noise. The results show that VBLs in many cases outperform other robust loss functions, achieving higher accuracy and robustness to noise. Furthermore, VBLs excel in scenarios involving real-world noisy datasets, underscoring their applicability beyond synthetic benchmarks. The study concludes that variation bounded losses not only improve the noise tolerance of models, but also provide a structured, less complex way to design effective loss functions in real-world applications with noisy labels.

### Strengths
The paper introduces a the concept of variation bounded losses to asses the noise tolerance of loss functions. 

The paper establishes a connection between variation boundedness and both symmetric and asymmetric conditions. 

New variation bounded variants of well known loss functions are formulated. 

Based on a series of numerical experiments the authors demonstrate that the newly suggested loss functions improve on the state of the art in learning data sets with label noise.

### Weaknesses
I do not understand how the hyperparameters for combining NCE with the different VBLs are determined. 

The paper would benefit from an explanation why the combination of VBLs with NCE is beneficial - this is important since this combination is most successful with CIFAR100. Is the combination still variation bounded?

### Questions
Can the authors  provide details on their hyperparameter selection process for combining NCE with VBLs? For example, did they use a specific optimization method like grid search or random search, or was it based on empirical testing? This information would help readers reproduce the results and understand the practical implementation

Can the authors provide a theoretical analysis or intuitive explanation for why combining VBLs with NCE is particularly effective, especially for CIFAR100? Additionally,  clarification on whether this combination preserves the variation-bounded property would be valuable for understanding the theoretical underpinnings of their approach.

### Soundness
3

### Presentation
3

### Contribution
3
