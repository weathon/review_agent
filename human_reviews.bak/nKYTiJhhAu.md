# REDUCR: Robust Data Downsampling Using Class Priority Reweighting

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3

## Abstract
Modern machine learning models are becoming increasingly expensive to train for real-world image and text classification tasks, where massive web-scale data is collected in a streaming fashion. To reduce the training cost, online batch selection techniques have been developed to choose the most informative datapoints. However, these techniques can suffer from poor worst-class generalization performance due to class imbalance and distributional shifts. This work introduces REDUCR, a robust and efficient data downsampling method that uses class priority reweighting. REDUCR reduces the training data while preserving worst-class generalization performance. REDUCR assigns priority weights to datapoints in a class-aware manner using an online learning algorithm. We demonstrate the data efficiency and robust performance of REDUCR on vision and text classification tasks. On web-scraped datasets with imbalanced class distributions, REDUCR achieves significant test accuracy boosts for the worst-performing class (but also on average), surpassing state-of-the-art methods by around 15\%.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a new method to perform online data selection. The proposed method aims to improve the worst-class performance and maintain its overall performance in the meantime. The method can precisely evaluate the performance of classes and put larger weights on the losses of poor-performance class samples. Experiments show some improvement with respect to the worst-class accuracy compared with several baselines.

### Strengths
1. This paper identifies an important and interesting problem in existing methods where the worse-class performance is overlooked.
2. This paper presents a simple solution to solve the complex max-min optimization in Eq. (3).
3. This paper conducts extensive experiments and ablation studies to validate the effectiveness of the proposed method.

### Weaknesses
1. Regarding the class-irreducible loss, it is not well justified that the model $\phi_c$ can be a good approximation of $\theta_t^{(c)}$.
2. Regarding the class-irreducible loss, training a separate model $\phi_c$ for each class can be computationally prohibited on large datasets. Datasets with larger class space such as CIFAR-100 and ImageNet are missing in the experiments.
3. In Eq. (6-7), the meaning of $c$ and $y$, and their relation, need further clarification.
4. Why can the proposed method prevent from selecting datapoints with noisy labels?
5. What if a clean validation set is not accessible?
6. Some related works are missing from discussion and comparison, such as [1,2]

[1] Heteroskedastic and imbalanced deep learning with adaptive regularization
[2] Robust long-tailed learning under label noise

### Questions
see Weaknesses

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes an online batch selection algorithm called REDUCR to preserve the worst-class generalization performance.
Extensive experiments on multiple datasets show the superiority of the proposed method.

### Strengths
- Clear presentation and easy-to-follow writing.
- Extensive evaluation on multiple datasets with two tasks.

### Weaknesses
Unclear motivation
- Clothing1M is not a proper dataset to evaluate the effect of batch selection in worst-case accuracy, since it contains noisy labels as well. The performance drop of other baselines may be due to label noise other than class imbalance.
- Loss-based batch selection baselines (e.g., Loshchilov et al.(2015)) prefer to select high loss examples. Then, they will automatically select the worst-class example first as it exhibits higher loss (i.e., worse generalized). 
- I think why these baselines fail on Clothing1M is due to the label noise, since noisy examples tend to exhibit higher loss so that easy to be selected [a][b].

[a] Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels. NeurIPS , 2018

[b] Meta-Query-Net: Resolving Purity-Informativeness Dilemma in Open-set Active Learning. NeurIPS, 2022


Less practicality
- Although the author provides an efficiency analysis with respect to training steps (in Fig 3), I think this algorithm might be less practical since it takes time to select batch b_t from B_t by solving the minimax problem at every time step t.
- The author should provide GPU time analysis compared to random batch selection to convince the practicability of this algorithm.

### Questions
How to select batch b_t exactly? All the formulations for selection scores are for a single datapoint, as the authors assume the small batch to a single data point in Sec 4.2. Could you elaborate on how the “batch” selection exactly works (line 6 in Alg 1)? With batch selection, I think the selection should consider the relationship between examples to minimize Eq. (3), so the selection algorithm should be different from the single point.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors propose an approach called REDUCR for online batch selection problem. REDUCR improves existing online batch selection approach RHO-Loss by directly optimizing the worst-class generalization performance.

### Strengths
+ The paper is written well and easy to follow. All the figures and tables are of high-quality.
+ A comprehensive discussion with related works has been provided.
+ Empirical studies show the proposed approach can achieve superior worst-class test accuracy, though this result is not surprising since the proposed approach directly optimizes the worst-class generalization performance.

### Weaknesses
- The contribution and novelty of this paper are limited. Compared with an existing work RHO-Loss, the only difference is that the proposed approach directly optimizes the worst-class generalization performance, while RHO-Loss optimizes the average generalization performance. Other aspects (e.g. techniques for inducing selection scores and approximating class-irreducible loss model) are the same.
- It is not clear why the model induced from Eq. (8) can approximate the so-called class-irreducible loss model. They are totally different models from my perspective.
- The proposed approach improves worst-class test accuracy, but sacrifices the overall average test accuracy.

### Questions
- The contribution and novelty of this paper are limited. Compared with an existing work RHO-Loss, the only difference is that the proposed approach directly optimizes the worst-class generalization performance, while RHO-Loss optimizes the average generalization performance. Other aspects (e.g. techniques for inducing selection scores and approximating class-irreducible loss model) are the same.
- It is not clear why the model induced from Eq. (8) can approximate the so-called class-irreducible loss model. They are totally different models from my perspective.
- The proposed approach improves worst-class test accuracy, but sacrifices the overall average test accuracy.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
