# Initialization Matters: Unraveling the Impact of Pre-Training on Federated Learning

- Decision: Reject
- Scores: 6, 6, 8, 8

## Abstract
Initializing with pre-trained models when learning on downstream tasks is now standard practice in machine learning. Several recent works explore the benefits of pre-trained initialization in a federated learning (FL) setting, where the downstream training is performed at the edge clients with heterogeneous data distribution. These works show that starting from a pre-trained model can substantially reduce the adverse impact of data heterogeneity on the test performance of a model trained in a federated setting, with no changes to the standard FedAvg training algorithm. In this work, we provide a deeper theoretical understanding of this phenomenon. To do so, we study the class of two-layer convolutional neural networks (CNNs) and provide bounds on the training error convergence and test error of such a network trained with FedAvg. We introduce the notion of aligned and misaligned filters at initialization and show that the data heterogeneity only affects learning on misaligned filters. Starting with a pre-trained model typically results in fewer misaligned filters at initialization, thus producing a lower test error even when the model is trained in a federated setting with data heterogeneity. Experiments in synthetic settings and practical FL training on CNNs verify our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This is a theoretical paper. This paper provides a rigorous analysis of the impact of pre-training on FedAvg's convergence. The authors demonstrate that starting from a pre-trained model can result in fewer misaligned filters at initialization, thus producing a lower test error even when the clients are heterogeneous.

### Strengths
- The authors provide a rigorous theoretical analysis of the impact of pre-training on federated learning, based on a two-layer CNN and synthetic datasets.
- In Proposition 1, the authors present a decomposition of the filter weights into a signal vector and a noise vector. Extending theoretical results from a centralized setting to federated learning can be challenging due to multiple local steps and data heterogeneity. Therefore, I believe that this is a solid contribution.
- Experimental results on synthetic datasets and CIFAR-10 substantiate the theoretical findings.
- This work could be valuable in practical scenarios where the number of data points per client is small, heterogeneity across clients is high, and the image modality resembles that used in training existing centralized models.

### Weaknesses
**Theoretical analysis**: 
- Although the authors explained the reason for choosing the 2-layer CNN and I agree that this can simplify the theoretical analysis. However, this can also  limit the broader impact of the findings. 
- The definition of data heterogeneity implies a binary classification problem. This means that for a multi-class problem, we would need to cast it into multiple binary classification problems to apply the data heterogeneity measure, which is less practical.
- The assumption of the noise vector being orthogonal to the signal can hardly hold in more complicated datasets. 

**Practical implications**: While I understand this is a theoretical paper, I would like to know more about its practical implications. For example:
- The paper demonstrates that a pre-trained model can lead to fewer misaligned filters at initialization and lower test error even with heterogeneous clients. However, are there any insights into determining what types of pre-trained weights are best suited for what particular tasks? 
- In what situations might the theory not hold?

### Questions
- The definitions of alignment and misalignment in Definition 1 are very interesting. Can this be quantified, for example, in an experimental setup where the data is MNIST, the model is VGG, and the task is odd and even number classification?
- Could the authors elaborate further on Figure 4, as I do not see a clear trend over r? Additionally, the green color appears to be missing in Figures 4a and 4d, or is it on top of other colors?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the impact of using pre-trained models on the performance of FedAvg in a data-heterogeneous environment. More precisely, the authors analyze the performance of a 2-layer CNN, both theoretically and practically with synthetic data. They introduce the notion of aligned and misaligned filters in a CNN and show that pre-training the model produces more aligned filters, resulting in higher final accuracy compared to training without pre-trained models when applying FedAvg. They extend their practical observations by running more complex experiments on CIFAR-10 using a deeper CNN.

### Strengths
- The paper is clear and easy to read
- The notion of aligned/misaligned filters is interesting
- The paper present the direct harmful effect on the test error of local steps and heterogeneity when the model has misaligned filter

### Weaknesses
- On measuring data heterogeneity. In the paper, the authors use a data heterogeneity measurement that considers only the label distribution across clients, while data heterogeneity can encompass more than this. Indeed, even for two images of the same label, after computing the model’s prediction and the associated loss for each image, their gradients could point in completely different directions or have completely different norms. For example, two images of cats—one being a real photo and the other an animation—could make the gradient vary greatly. This divergence can complicate the task and impair federated training, even when two clients hold data with the same label. The definition of data heterogeneity presented in the paper does not take into account this aspect. In the literature, a common approach is to consider the actual local gradients; see, e.g., [A] [B]. Could the authors explain their choice of measurement of data heterogeneity?

- From Figure 5 (left plot), it appears that even if the number of misaligned filters is the same for two models (for instance, at iteration 200), we can reasonably conclude from the right plot that the test error for the orange curve may never catch up to the blue one, even with 200 more training rounds. From my understanding, this suggests that even if two models have the same number of misaligned filters at initialization (e.g., using the two models at iteration 200 as initialization parameters), they may still perform differently by the end of federated learning. Therefore, it seems to me that the presence of aligned or misaligned filters alone does not fully explain why a model benefits from pre-training in heterogeneous setting. Could the authors comment on this point?


[A] Karimireddy, Sai Praneeth, et al. "Scaffold: Stochastic controlled averaging for federated learning." International conference on machine learning. PMLR, 2020.

[B] Yuan, Xiaotong, and Ping Li. "On convergence of FedProx: Local dissimilarity invariant bounds, non-smoothness and beyond." Advances in Neural Information Processing Systems 35 (2022): 10752-10765.

### Questions
see comments above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper examines the effects of initialization with pre-trained models on federated learning (FL) performance, presenting theoretical bounds for test errors in federated CNNs. The analysis emphasizes how pre-trained models reduce test error by minimizing misaligned filters, which in turn mitigates the adverse effects of data heterogeneity.

### Strengths
- The paper provides a detailed theoretical analysis for understanding how pre-trained initialization benefits FL, focusing on the alignment of filters and the effects of data heterogeneity on signal learning versus noise memorization.

- The conclusions resonate with intuitive understanding, suggesting that pre-trained models can improve generalization by reducing harmful overfitting due to misaligned filters, which otherwise increase error in heterogeneous FL settings.

- Experimental Support: The paper includes (though limited) experiments to support the theoretical findings.

### Weaknesses
- The analysis relies on a simplified two-layer CNN, raising concerns about the transferability of the derived bounds to more complex architectures often used in FL. This limitation could impact the broader relevance of the findings.

- My main concern is on the unclear assumptions and rationale. Some assumptions necessary for the main theoretical results are not thoroughly justified. For instance, Condition C2 requires a “sufficiently large” dimension d, but it is unclear whether this assumption should be intuitively reasonable. Additionally, the relationship between C1 (restricted number of updates) and C6 (learning rate sufficiently small) seems contradictory, as these conditions might imply opposing constraints on the learning process. 

- While the theoretical results are insightful, it lacks practical interpretations. I would like to see more discussions on how these results might apply to real-world datasets and FL setups with varying levels of data heterogeneity (which is unknown in reality).

### Questions
Pls see my earlier comments.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the question: "Why does pre-trained initialization significantly alleviate the challenges posed by non-IID data in federated learning (FL)?" To explore this, the authors first identify that the reduction in test accuracy observed in non-IID FL compared to IID FL is due to filter misalignment at initialization. They argue that when FL training begins with pre-trained models, most filters are already aligned with the signal, which mitigates the impact of data heterogeneity. The paper is primarily theoretical, with less emphasis on experimental validation.

### Strengths
1. The paper is technically well-written and easy to follow, with well-chosen and consistent notation throughout.
2. The results presented in Section 4 are intriguing and, to the best of my knowledge, novel contributions to the field.

### Weaknesses
1. I understand that the results derived for the two-layer CNN can, to some extent, be generalized to deeper CNN architectures. However, I would appreciate if the authors could discuss how the theorems introduced in the paper might change under such a generalization.

2. While the paper is primarily theoretical, additional experiments on larger datasets would significantly strengthen the paper’s contributions.

3. In line 357, the authors mention: "We focus on centralized pre-training, but our discussion here can be extended to federated pre-training as well." It is unclear why this extension holds. The authors should elaborate on this claim elsewhere in the paper. Overall, the primary concern with this manuscript is that it shifts between centralized and FL settings without clear distinction.

4. Minor issues:

   a) The authors inconsistently capitalize the first letters when introducing abbreviations, e.g., "Independent and Identically Distributed (IID)" versus "machine learning (ML)." Please ensure consistency throughout the paper.

   b) On lines 54-56, the authors mention, "One reason suggested by Nguyen et al. (2022) is a lower value of the training loss at initialization when starting from pre-trained models." Could the authors clarify if this statement refers to the centralized or FL setup?

   c) Line 71: "…two-layer ReLU convolutional neural networks (CNNs) (Zou et al., 2023)..." The current citation format implies that the two-layer ReLU architecture was introduced by (Zou et al., 2023). A clearer phrasing might be: "similar to (Zou et al., 2023), we use a two-layer ReLU..."

   d) Line 116: "Also, [a] denotes {1, 2, . . . , n}." Please correct this typo for clarity.

### Questions
Please address the concerns I raised above, and explicitly clarify whether your theorems apply to centralized settings, federated learning (FL) settings, or both. Provide clear reasoning to support each case.

### Soundness
3

### Presentation
3

### Contribution
2
