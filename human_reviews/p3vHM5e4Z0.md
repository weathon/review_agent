# Hyper-parameter Tuning for Fair Classification without Sensitive Attribute Access

- Avg Score: 4.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5

## Abstract
Fair machine learning methods seek to train models that balance model performance across demographic subgroups defined over sensitive attributes like race and gender. Although sensitive attributes are typically assumed to be known during training, they may not be available in practice due to privacy and other logistical concerns. Recent work has sought to train fair models without sensitive attributes on training data. However, these methods need extensive hyper-parameter tuning to achieve good results, and hence assume that sensitive attributes are known on validation data. However, this assumption too might not be practical. Here, we propose \sys, a framework to train fair classifiers without access to sensitive attributes on either training or validation data. Instead, we generate pseudo sensitive attributes on the validation data by training a ERM model and using the classifier's incorrectly (correctly) classified examples as proxies for disadvantaged (advantaged) groups. Since fairness metrics like demographic parity, equal opportunity and subgroup accuracy can be estimated to within a proportionality constant even with noisy sensitive attribute information, we show theoretically and empirically that these proxy labels can be used to maximize fairness under average accuracy constraints. Key to our results is a principled approach to select the hyper-parameters of the ERM model in a completely unsupervised fashion (meaning without access to ground truth sensitive attributes) that minimizes the gap between fairness estimated using noisy versus ground-truth sensitive labels. We demonstrate that \sys outperforms existing methods on CelebA, Waterbirds, and UCI datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
•	As sensitive attribute information may be unavailable in training and validation sets, this paper proposes the Antigone algorithm. It leverages the classification outcomes of the classifier to generate pseudo-sensitive attributes on the validation set. Subsequently, it utilizes these generated pseudo-sensitive attributes to guide the hyperparameter tuning of the model.

•	To generate high-quality pseudo-labels, Antigone selects the classifier with the maximum EDM as its labeling model.

•	Building upon some existing fairness methods, the experiments show that Antigone can further improve fairness by fine-tuning its hyperparameters.

### Strengths
•	This paper proposes an new method to fine-tune hyperparameters for existing fairness methods, particularly in scenarios where sensitive attributes are not accessible. 

•	The method introduced in this paper, Antigone, efficiently utilizes the ERM model to produce high-quality pseudo-sensitive attribute labels, denoted as PSA.

### Weaknesses
•	To the best of my knowledge, several methods currently exist that either generate pseudo-sensitive attribute or use proxy sensitive attribute in scenarios where sensitive attributes are unavailable [1,2,3]. However, the related work section provides only a concise overview of these methods. It is recommended to engage in a more comprehensive discussion, with particular emphasis on elucidating the distinctions between these approaches.

•	The experimental section demonstrates that Antigone can yield improved results when fine-tuning existing fairness methods. However, I am still curious about how these results compare to those obtained by other methods that generate pseudo-sensitive attribute labels.

•	In the theoretical section, the MC ideal model is used. To enhance reader comprehension, it would be beneficial to provide a more detailed introduction to the MC model, along with its theoretical guarantees.

1.	Zhao, Tianxiang, et al. "Towards fair classifiers without sensitive attributes: Exploring biases in related features." Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 2022.

2.	Zhu, Zhaowei, et al. "Weak Proxies are Sufficient and Preferable for Fairness with Missing Sensitive Attributes." (2023).

3.	Grari, Vincent, Sylvain Lamprier, and Marcin Detyniecki. "Fairness without the sensitive attribute via causal variational autoencoder." arXiv preprint arXiv:2109.04999 (2021).

### Questions
See weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies how to mitigate bias without access to sensitive attributes. It features two things: (1) it does not require sensitive attributes on the validation set and (2) it does not need (extensive) hyper-parameter tuning.

The idea is to use the target label as the proxy of the sensitive attribute, and then use the metric of mean Euclidean distance between features to tune the hyper-parameters.

### Strengths
The paper explores an important problem with appealing features (i.e. no hyper-parameter tuning and no sensitive attribute on the validation set), if true.

### Weaknesses
1. I am puzzled by the design of using the target label as the sensitive attribute. I do not find any justification, either empirical or theoretical, other than claiming, in the introduction, that correctly classified samples would be over-represented and incorrectly classified samples would be under-represented. 

If the target label can be used as a proxy of sensitive attributes, why would people still need sensitive attributes in fairness study at all? I find this is a very broad claim that should be heavily backed up by either strong empirical evidence or solid theoretical insights.

In addition, I can think of plenty of counter-examples to show it is not true. Consider the following tabular data with 5 samples:

|           | $x_1$ | $x_2$ | $x_3$ | $x_4$ | $x_5$ |
|-----------|-------|-------|-------|-------|-------|
| Y         | 0     | 1     | 1     | 0     | 1     |
| $\hat{Y}$ | 1     | 1     | 0     | 0     | 1     |
| A         | 1     | 0     | 1     | 0     | 0     |


In this case, the incorrectly classified samples ($x_1$ and $x_3$) are not over-presented by the disadvantaged group ($A=0$); in fact, they have no disadvantaged samples at all. The correctly classified samples (the remaining) are not over-represented by the advantaged group ($A=1$); in fact, they have no advantaged samples at all.

The authors might argue this is a cherry-picked corner case and not what happens statistically. But then the point is to have more justification for this strong and general claim.


2. The theoretical justification for using EDM seems to me a misunderstanding of Lamy et al. The assumption of Lamy et al. is not the Eq. (7), but rather replace the $X$ in Eq.(7) with $D$ where $D$ is the joint distribution $P(X, Y, A)$. This difference is vital because in this work, the reason why EDM can be justified is the MC assumption is only applied to feature $X$ and therefore you can only look at feature distance to perform hyper-parameter; and since it assumes has nothing to do with either label $Y$ or sensitive attribute $A$, the work can claim it requires no sensitive attribute labeling on the validation set.

However, this is not the assumption in Lamy et al. If I am not mistaken, the theory would not hold if you simply ignore the MC relationship on $A$ and $Y$ in the assumption. I am happy to change my mind if the authors can point out if I am mistaken. But if not, this seems to be a misquote of the results in Lamy et al., and the consequent justification of EDM does not hold.

### Questions
See weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper explores the problem of balancing model performance across different demographic groups (fairness) without having access to the demographic information. The paper argues that most existing fairness methods without demographics require access to a validation dataset with demographic information, which might not be available. To solve this problem, Antigone is proposed. Antigone is a framework to train fair classifiers without access to demographic information on training or the validation dataset.

### Strengths
•	The paper approaches the problem of hyperparameter tuning when sensitive attributes are unavailable in the validation dataset, an exciting and often overlooked problem. 

•	The results obtained using Antigone were close to the results of using the ground truth labels with JTT, as shown in Table 2. This result indicates that the assumption of demographic information in the validation dataset is unnecessary. 

•	Moreover, the comparison with ARL in Table 4 indicates that Antigone may also improve the performance of methods that do not use sensitive information during validation.

•	The last two points show the flexibility of Antigone and demonstrate how it can be used to improve (either by relaxing data assumptions or improving the performance of existing methods) existing approaches.

### Weaknesses
•	The paper only compares its results for WGA when using GEORGE. Why not show the results for DP and EO too?

•	The paper could compare the results acquired by Antigone with other methods that assume access to sensitive attributes during validation. This could make the point that Antigone is flexible and can successfully replace sensitive attributes in the validation dataset clearer.

•	How does Antigone compare with other methods of predicting PSA?

### Questions
How could Antigone be modified to account for various demographic groups, i.e., A = {1, 2, 3, …, m} with m potentially exponentially large? Does ensuring that the fairness metric for the binary PSA is sufficient in this case? I ask this question because of Lemma 2.2.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
