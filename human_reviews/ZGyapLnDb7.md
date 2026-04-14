# Balancing Gradient Frequencies Facilitates Inductive Inference in Algorithmic Reasoning

- Decision: Reject
- Scores: 3, 8, 6, 6

## Abstract
Inductive inference, or extrapolation of general rules from finite instances, is understood to be the foundation of human intelligence. Unfortunately, Deep Neural Networks (DNNs) struggle with inductive inference and thus fail to learn even the simplest algorithms in Algorithmic Reasoning (AR). Existing research efforts on AR with DNNs are limited to those on the architectural design for DNNs. In this study, we investigate the influence of optimization techniques on AR performance. Through toy experiments designed to understand an optimizer's susceptibility to shortcuts in AR, we reveal that Adam, the naive choice of optimization, is easily fooled by spurious correlations. To overcome this shortcoming of Adam, we propose a novel optimizer that avoids spurious correlations by balancing gradients of low- and high-frequencies (BGF). We present extensive experiments and analyses to demonstrate the broad and multifaceted advantages of BGF across various architectures and AR tasks. In particular, BGF expands the AR capability of all explored DNN models and even shows the potential to enable learning of tasks that they previously failed at. The observed success of BGF in climbing the Chomsky hierarchy underscores the importance of optimization for developing advanced artificial intelligence with DNNs.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces a new optimizer for Algorithmic reasoning tasks to avoid/mitigate shortcut learning for better OOD generalizability. Especially the work looks into the frequency domain of the gradients. The work provides toy experiments and synthesized spurious correlations in Algorithmic reasoning tasks.

### Strengths
•	If the claim is correct, the proposed optimizer would be simple and easy to implement, as it would be just a frequency (low/high) pass filter.

### Weaknesses
•	Most of all, the definition of shortcuts is the most serious problem that I have with this paper. I cannot agree with the authors’ definition of “shortcuts” which is stated as “non-generalizing hypotheses are termed “shortcuts.” Shortcut learning happens when a model can minimize the training loss just by looking at simpler features and giving up learning more complex core features. That means if such non-generalizing features are complex enough or more complex than core features, shortcut learning may not happen. That is why the name is ““short”cut” learning. That is why “non-generalizing hypotheses” cannot be just termed shortcuts.

•	Also, the sentence over lines 191-193 “Shortcuts can be interpreted as learning non-generalizing correlations between inputs and outputs that are valid in the training distribution but do not hold in the target distribution;” is not fully true. It depends on the target distribution. If the target distribution is still in distribution, such correlations will still be valid, but such correlations just won’t be visible.

•	This paper only considered two simple forms of synthetic spurious correlations on “toy” experiments. Algorithmic reasoning tasks are a good start and are used to examine the viability of an idea to apply to real-world tasks. It is unclear how this paper’s outcome can be useful for real-world tasks, vision and language. I am not confident to believe the claims stated in this paper as a general phenomenon. Also, nobody knows if something will go different in such Algorithmic reasoning tasks and conventional vision and language tasks?

•	For experiments with spurious correlations in Fig 2, the authors only experimented with spurious correlations of which portion is over 80% (when P >= 0.8). However, when the proportion is just that high the spurious correlation is trivial and straightforward.

### Questions
-	The following paper seems to have a good overlap with this submission:
“Frequency Shortcut Learning in Neural Networks,” Shunxin Wang, Raymond Veldhuis, Christoph Brune, and Nicola Strisciuglio, NeurIPS workshop on Distribution Shifts, 2022.
Hence, the novelty of this submission may be hurt.

-	For other points, please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This study examines optimization’s impact on AR performance, showing through toy experiments that Adam, a commonly used optimizer, is susceptible to spurious correlations that hinder out-of-distribution (OOD) generalization. To address this, the authors introduce a novel optimization method, Balancing Gradient Frequencies (BGF), designed to reduce shortcut learning by balancing low- and high-frequency gradients, thus promoting stronger inductive inference. Extensive experiments across varied AR tasks and DNN architectures reveal that BGF significantly improves accuracy, accelerates convergence, and smoothens the loss landscape, enabling DNNs to tackle tasks previously deemed unmanageable.

### Strengths
1.	Through extensive experiments, the authors demonstrate that balancing gradient frequencies directly impacts the model’s ability to generalize beyond the training distribution. This insight is particularly valuable, as it highlights the frequency characteristics of gradients as a crucial factor in overcoming shortcut learning, a perspective that has been relatively unexplored in AR.
2.	The study documents a phenomenon termed "train-test splitting," in which DNNs achieve training accuracy early on but require additional steps to attain OOD test accuracy. This splitting behavior resembles the grokking phenomenon and is linked to changes in gradient frequency patterns. The discovery emphasizes the importance of low-frequency gradient components for OOD generalization, offering a new understanding of model behavior during training.
3.	BGF’s architecture-agnostic design is evidenced by improved performance across various DNN types, including RNNs, LSTMs, and Transformer models, on multiple AR tasks. This adaptability underscores BGF’s potential for broad application, marking it as a significant innovation in optimization for generalization across neural network architectures.

### Weaknesses
1. Although the manuscript compares BGF with optimizers like Adam, SAM, SWAD, and LPF-SGD, incorporating more recent advancements in optimization techniques that are known to enhance generalization would further strengthen the case for BGF's superiority.

2. Conduct a hyperparameter sensitivity analysis focusing on λ and other gradient-balancing factors (e.g., α and β). This experiment would provide insights into how these parameters influence BGF's ability to generalize, especially under varying data conditions.

3. To deepen insights into how BGF affects gradient dynamics, examining gradient frequencies across different layers of DNN architectures is recommended. This analysis could reveal whether specific layers benefit more from low-frequency gradients, enabling further fine-tuning of BGF for optimal performance in various model architectures.

### Questions
see above

### Soundness
4

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
This paper presents Balancing Gradient Frequencies (BGF), an optimizer that enhances inductive inference and OOD generalization in deep neural networks for algorithmic reasoning tasks. Unlike traditional optimizers like Adam, which can learn spurious correlations, BGF balances low- and high-frequency gradient components to avoid shortcut learning and improve OOD generalization. Experiments across various DNN architectures and AR tasks demonstrate BGF’s superior performance and faster convergence.

### Strengths
1. The method tackles improving OOD AR tasks by proposing a new optimizer, which is novel and interesting to me.

2. The empirical studies at the beginning of Section 3 are interesting and well-designed, effectively illustrating the susceptibility of standard optimizers like Adam to shortcut learning. 

3. Extensive experiments show that BGF consistently outperforms Adam across multiple architectures and tasks.

### Weaknesses
1. The paper lacks formal theoretical analysis explaining why BGF outperforms traditional optimizers. While the empirical results are strong, a theoretical understanding of BGF’s benefits would enhance the contribution.

2. The study primarily compares BGF with Adam, with fewer comparisons to more recent optimizers designed to improve generalization. Including a wider range of baseline optimizers would strengthen the evaluation.

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new optimization method called BGF (Balancing Gradient Frequencies), aimed at improving length generalization in  Algorithmic Reasoning (AR) through the mitigation of shortcut learning. Building on the experimental findings regarding Adam's vulnerability to spurious correlations in AR tasks, the authors propose that BGF mitigates these correlations by balancing high- and low-frequency gradients. This approach encourages the model to learn general rules rather than relying solely on patterns present in the training data. Extensive experiments are conducted, demonstrating the effectiveness of BGF across various DNN architectures and tasks.

### Strengths
Extensive experiments and analyses;

A novel approach to study the effect of optimization on generalization in AR tasks;

The results on multiple DNN architectures and tasks show that BGF effectively improves generalization on test data.

### Weaknesses
The task description in the main text lacks clarity. There are too many task-related descriptions in the appendix—such as the "Modular Arithmetic," "Solve Equation," and "Bucket Sort" tasks—that interrupt the flow of the main text and affect its readability.

### Questions
Since the low-pass filter is implemented using a moving average filter with a window size of \lambda, what is the basis for selecting the window size \lambda? How does \lambda value affect the results?

In Table 3 [Right], why is the result of gradient filtering with an EMA lower than that with a queue in the LSTM model?

### Soundness
3

### Presentation
3

### Contribution
3
