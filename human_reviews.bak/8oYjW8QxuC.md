# Pi-DUAL: Using privileged information to distinguish clean from noisy labels

- Decision: Reject
- Scores: 5, 6, 6

## Abstract
Label noise is a pervasive problem in deep learning that often compromises the generalization performance of trained models. Recently, leveraging privileged information (PI) -- information available only during training but not at test time -- has emerged as an effective approach to mitigate this issue. Yet, existing PI-based methods have failed to consistently outperform their no-PI counterparts in terms of preventing overfitting to label noise. To address this deficiency, we introduce Pi-DUAL, an architecture designed to harness PI to distinguish clean from wrong labels. Pi-DUAL decomposes the output logits into a prediction term, based on conventional input features, and a noise-fitting term influenced solely by PI. A gating mechanism steered by PI adaptively shifts focus between these terms, allowing the model to implicitly separate the learning paths of clean and wrong labels. Empirically, Pi-DUAL achieves significant performance improvements on key PI benchmarks (e.g., +6.8% on ImageNet-PI), establishing a new state-of-the-art test set accuracy. Additionally, Pi-DUAL is a potent method for identifying noisy samples post-training, outperforming other strong methods at this task.  Overall, Pi-DUAL is a simple, scalable and practical approach for mitigating the effects of label noise in a variety of real-world scenarios with PI.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a dual structure for learning with noisy labels by separating the training process into regular feature learning and privileged information learning. The regular feature learning module is responsible for the final inference. The effectiveness of the algorithm was validated on three datasets: CIFAR=1-H, CIFAR-N, and ImageNet-PI.

### Strengths
1. The structure of this paper is clear, making it easy for readers to follow.
2. It's intriguing to note that the no-PI network of the Dual structure outperforms previous PI-related works.
3. The results presented in the paper attest to the efficacy of the proposed method.

### Weaknesses
1. While PI is a concept introduced in prior works, this article doesn't offer significant innovations to it. The paper points out that PI-based methods underperform compared to no-PI-based methods. However, it fails to delve deep into the underlying principles causing this discrepancy. The conclusions seem to be drawn mainly from some experimental verifications rather than in-depth analysis.

2. The experimental comparisons are not exhaustive. Several state-of-the-art methods mentioned in Table 1, such as dividemix, weren't comparatively analyzed in the experiments. Given that Pi-Dual incorporates additional information to tackle label noise, comparing it with the current best methods is crucial to gauge the algorithm's effectiveness.

3. Certain ablation studies were not conducted, like the choice of the model backbone and parameters of the additional PI-related modules.

### Questions
Please refer to the Weaknesses section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces PI-DUAL, a novel approach designed to address label noise by leveraging privileged information (PI). Privileged information is only accessible during the training phase. In essence, PI-DUAL employs neural networks to separately model the features of $x$ and the privileged information $a$, and it employs an additional network to determine the weights for the logits of $x$ and $a$. 

To demonstrate the advantages of PI-DUAL, the authors conduct a series of comprehensive studies to investigate the training dynamics of PI-DUAL and to provide valuable theoretical insights. The proposed method is characterized by its simplicity, and experimental results exhibit promising performance when compared to other baseline methods."

### Strengths
- The method is both simple and technically sound. Notably, it represents the first PI-based method to explicitly model label noise.

- The studies conducted on PI-DUAL are thorough. The authors perform a range of experiments, including an exploration of training dynamics, an evaluation of detection performance, and an investigation of the impact of PI length. These experiments collectively contribute to a comprehensive assessment of the effectiveness of PI-DUAL.

- While the theoretical insights are based on the linear layer scenario, their analyses appear to be reasonable.

### Weaknesses
- The primary distinction between PI-DUAL and TRAM [R1] lies in PI-DUAL's approach of separately modeling features and privileged information (PI), as opposed to using two heads on top of the feature vector $x$. While this approach provides some technical novelty, it may be considered somewhat limited.

- I find the experimental results to be somewhat perplexing. It is reasonable that PI-DUAL doesn't need to be compared to two-stage methods like DivideMix or other semi-supervised pipelines. However, the test accuracy presented in Table 2 exhibits a notable gap compared to the results reported in the original papers or public leaderboards [R2]. Additionally, the test accuracy for TRAM on CIFAR-10N, as presented in the original paper, was 71.8, but in Table 2, it's only 64.9. While I acknowledge that differences in training settings may account for this, it would be beneficial to conduct specific experiments to compare each method under their optimal configurations.

- Typos:
    - In Section 3.2: Change "Here, $\gamma_{\phi}$ denotes" to "Here, $\gamma_{\psi}$ denotes."

[R1] Transfer and Marginalize: Explaining Away Label Noise with Privileged Information

[R2] You can access the relevant information at http://www.noisylabels.com.

### Questions
See *Weaknesses above*

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces the innovative "Pi-DUAL" architecture, which effectively uses privileged information (PI) to distinguish between clean and erroneous labels, offering a crucial solution to label noise. Pi-DUAL demonstrates substantial performance enhancements in several benchmark tests and attains a new state-of-the-art accuracy.  It also excels in identifying noise samples post-training, surpassing other methods.  The ablation study is complete and conducted on all benchmarks for better presentation.

### Strengths
First, from the originality point of view, this paper presents a novel architecture, i.e., a noise labeling architecture guided by privileged information (PI), which enables the model to distinguish clean labels and mislabels more clearly. Second, they implement a bidirectional gated output logic structure that decomposes the output logic into a predictive term based on regular input features and a noise-adapted term influenced only by PI. Finally, a PI-driven gating mechanism adaptively chooses between the predictive term and the noise-adaptation term to handle clean and mislabeled learning paths, respectively. 
Second, from a significance perspective, the results are impressive, and the improvement of the proposed methods is significant.  Combined with the novelty, I think overall, this is a sound paper introducing an effective method.

### Weaknesses
I list several questions that may be helpful.
1. What hardware requirements are needed for Pi-DUAL training for large datasets? If a training cost analysis is provided, I think it can be more useful for deploying your method in more scenarios.
2. How does Pi-DUAL perform in terms of security and privacy protection? For example, is there a risk that privileged information may be compromised? If there is no PI exists, what the performance will be? 
3. Are there any parameter tuning for Pi-DUAL for different levels of label noise and datasets? What is your hyper-parameter chosen strategy?

### Questions
Please see above weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
