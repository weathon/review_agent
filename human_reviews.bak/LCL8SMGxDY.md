# For Better or For Worse? Learning Minimum Variance Features With Label Augmentation

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 8, 6

## Abstract
Data augmentation has been pivotal in successfully training deep learning models on classification tasks over the past decade. An important subclass of data augmentation techniques - which includes both label smoothing and Mixup - involves modifying not only the input data but also the input label during model training. In this work, we analyze the role played by the label augmentation aspect of such methods. We first prove that linear models on binary classification data trained with label augmentation learn only the minimum variance features in the data, while standard training (which includes weight decay) can learn higher variance features. We then use our techniques to show that even for nonlinear models and general data distributions, the label smoothing and Mixup losses are lower bounded by a function of the model output variance. Lastly, we demonstrate empirically that this aspect of label smoothing and Mixup can be a positive and a negative. On the one hand, we show that the strong performance of label smoothing and Mixup on image classification benchmarks is correlated with learning low variance hidden representations. On the other hand, we show that Mixup and label smoothing can be more susceptible to low variance spurious correlations in the training data.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
In this research, the authors investigate label augmentation methods such as label smoothing and Mixup within deep learning frameworks. These techniques adjust both the input data and labels to enhance training outcomes. The study delves into how these methods impact feature learning and reduce output variance. The findings indicate that label augmentation helps models develop low-variance features, improving classification results on some benchmarks, though it may increase susceptibility to misleading data patterns. With theoretical backing and empirical results, the paper highlights the strengths and drawbacks of label smoothing and Mixup. The authors conclude that while these approaches bolster model robustness, they may inadvertently promote reliance on low-variance features, affecting model generalization.

### Strengths
1. This paper rigorously analyzes label smoothing and Mixup with mathematical proofs, building a solid theoretical foundation.

### Weaknesses
1. Overall, the authors' argument that both Mixup and label smoothing are effective in reducing variance is not particularly surprising and has been a well-known perspective for some time. For this reason, Mixup has been used for calibration due to its generalizability to the underlying distribution of training datasets.

2. The authors should conduct experiments with diverse spurious correlation setups. For example, both label smoothing and Mixup are widely used to mitigate noisy label learning, which is one common spurious correlation setup. I believe these previous works contradict the authors' arguments.

[1] When and How Mixup Improves Calibration, ICML 2022 \
[2] How Does Mixup Help with Robustness and Generalization?, ICLR 2021 \
[3] Does Label Smoothing Mitigate Label Noise?, ICML 2020 \
[4] mixup: BEYOND EMPIRICAL RISK MINIMIZATION, ICLR 2018

### Questions
Is there any theoretical investigation into whether Mixup and label smoothing perform worse in situations involving spurious correlations? I think the limited experimental setups are insufficient to support a counter-intuitive argument compared to the extensive body of previous work on model calibration and noisy label learning.

### Soundness
3

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
The paper studies feature learning in training with label augmentation techniques like Mixup and label smoothing, etc.. Using a linear binary classification model, and proper conditions on the data distribution, the authors have theoretically shown that in ERM training, the learned model tends to assign more correlations on the high-variance features, while in label augmentation training, it's quite the opposite. They then empirically show that such a behavior can lead to both positive and negative consequences. In particular, they show that the model's learning low variance features correlates to better generalization, while the model's learning only the low variance features, if happening, can  harm the generalization.

### Strengths
1. The studying of feature learning problem on label smoothing is novel
2. The paper have uncovered some interesting and new findings about the phenomenons and properties existed in label augmentation techniques like Mixup and label smoothing
3. The theoretical contributions and their proofs are valid
4. The experiments are comprehensive, and they show not only the validity of the theoretical results, but also the correlationship between those theoretical results and the impact of the label augmentation techniques on model performance

### Weaknesses
Please refer to the Questions session for details of the weaknesses listed here

1. While the concept of "high variancce features" and "low variance feature" appear to be the essential components of the theoretical content, the whole paper seems to lack a strict definition of them. Perhaps that's a common knowledge in the field of feature learning, but since this paper is mostly theory-based, it's better to provide more mathematically strict definition of these concepts.
2. The logic connection from section 3.1 to section 3.2 is not so clear, in fact, from my understanding, they are talking about different things.
3. Since the main theoretical  results are built based on a simple model, one would love to see first a experiment done on such a toy model to show the validity of the theoretical results in the most direct way.
4. Some notations and terminologies can be perfected
5. There lacks an intuitive explanation or an insight behind the connection between learning of low variance features and generalization performance of the trained model.

### Questions
1. Line 135 on page 3. The identity matrix shouldd be denoted by mathematical expression "$I_d$" rather than text "Id" right?
2. In the context of binary classification, the label variable $y$ should be $Y\in \{-1, +1\}$ right? But in that case, the cross-entropy loss defined in equation 2.1 cannot be applied on binary classification. Or, is the paper implicitly regarding the term "$k$-class classification" to be distinct from the term "binary classification"?
3. In equation 2.4, I think the second term inside the expectation should be "$\dfrac{\alpha}{k-1}\sum_{i\in [k]/{Y}}\log{g^i(X)}$".
4. In the definition of Mixup loss in equation 2.8, are the points pairs $(X_1,Y_1)$ and $(X_2,Y_2)$ sampled independently from $\pi$? In that context, I would denote it as $(X_{1:2},Y_{1:2})\overset{\mathrm{iid}}{\sim}\pi$ or something like $(X_{1:2},Y_{1:2})\sim\pi\bigotimes\pi$
5. What are the mathematical definitions of low variance features and high variance features? Is it the overall variance? Or in-class variances? Also, how would one justify if a feature is high variance or low? Should there be a predefined threshold? Or maybe even a post-defined threshold, meaning threshold is defined after the groups of high variance feature and low variance features are manually partitioned?
6. Before assumption 3.2, $\mathcal{L}$ and $\mathcal{H}$ are at first defined as subsets of $[d]$, and then regarded as sets of vectors. I understand this is only for the sake of simplicities in notations, but I would suggest that the paper simply uses different notations.
7. In assumption 3.5, how does the first part ensure that we can obtain good solution using low variance dimensions? First, I think if a data is class-balanced (the marginal distribution on the classes is discrete uniform), and given an early assumption that $\mathbb{E}[X]=0$, then $\mathbb{E}[YX_\mathcal{L}]$ should always be 0. Second, if the data is imbalanced, then this first part might hold true, but then does it mean we are only dealing with class-imbalanced data from now on? Also, even if the data is imbalanced and $\mathbb{E}[YX_\mathcal{L}]$ equals 0, it may still be possible to find a good solution.
8. The assumptions on the data in the setting of linear binary classification don't seem to be so unrealistic. Can the authors provide some toy example experiments to verify the theoretical results more directly?
9. I was expecting section 3.2 would be a generalization of the results in section 3.1. But in section 3.1, the conditions are about the variances in the data dimension. But in section 3.2, somehow it's all about the variances in the model output. I am aware of the statement made in remark 3.11, but I still don't understand how does this section help us gain more understanding of the link between the model's behavior of learning low variance features and the change in the model's generalization peformance?
10. Line 368 on page 7. It better to write "$5\times 10^{-4}$" instead of "$*$".
11. Is there and inituitions or insight as of how does the variance of the learned features affect the performance of the model? Without this, it's pretty hard to use the work in this paper as a cornerstone to initiate further studies, such as the mechanism of label augmentation techniques (in other words, why do they work), the limitations and possible flaws of these techniques, how to improve these techniques or design new algorithms, etc.. This is why I feel like the contribution to the research area in this paper is only fair.

### Soundness
3

### Presentation
2

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
In this paper the authors provide theoretical results pertaining to the variance of features in relation to training with label augmentation (Label Smoothing and Mixup). They show that for a binary linear logistic classifier trained on a certain distribution where there is a high and a low variance feature, label augmentation leads optimisation to only consider the low variance feature. For general models and classification they provide a quantative lower bound on how the variance of output probabilities contributes to the loss. The authors then provide some experiments to empirically verify the theoretical results.

### Strengths
- The paper provides some new theoretical results that contribute to better understanding of the behaviour of label augmentation approaches. I appreciate work that tries to advance knowledge in deep learning rather than simply chasing benchmark scores so this is a big plus for me.
- The experiments are well-targeted to verify the theoretical results.

### Weaknesses
I would like to clarify that I did not have time to comb the fine details of the paper or verify the theory, however I believe I have allocated a reasonable amount of time to reading from start to end. I am also not a "theoretical" machine learning researcher personally, and I am more familiar with the literature for label smoothing than Mixup, which should help put my later comments in context. I welcome the authors to correct any mistakes/misunderstandings that I make in this review.

1. The presentation of the paper inhibits easy understanding of the theoretical takeaways for a general machine learning audience (i.e. the average researcher using label augmentation). There isn't a single visual aid to illustrate any of the theoretical results, making it both difficult to intuit the theory and understand the key takeaways. For example, for the linear binary classifier, the paper would be greatly aided with a simple 2D illustration showing how the optimal decision boundaries differ between with and without Label Smoothing. Similarly I would suggest an illustrative sketch over 3 classes (visually in the same vein as those in [1]) of how greater variance contributes to loss. I think the text could also be improved by highlighting the key theoretical takeaways in plain English (e.g. in a box/table/in bold), allowing the less mathematically inclined reader to not have to parse through the comparatively dense and notation heavy theory.
2. Missing related work: there are a few missing references to label smoothing in the context of uncertainty estimation [2,3], neural collapse [4,5] and transfer learning [6].  [4,5,6]  are especially relevant and should be discussed as they directly measure how tight features cluster together after training with label smoothing. [6] also demonstrates that tighter feature clusters lead to **worse** linear probing performance.
3. Limited insight with regards to generalisation. The paper lacks any direct insight into how label augmentation leads to better generalisation peformance, which is ultimately why these techniques are used in the first place. Although the authors do fully acknowledge this limitation, it doesn't change the fact that this limits the impact of the insights of the paper. 
4. The practical takeaways of the spurious correlation experiments are somewhat unclear. It is not necessarily the case that spurious correlations tend to be lower variance. In the case that they are high variance wouldn't the inverse behaviour manifest? 

[1] Muller et al. When Does Label Smoothing Help? 2019

[2] Zhu et al. Rethinking Confidence Calibration for Failure Prediction, 2022

[3] Xia et al. Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It, 2024

[4] Zhou et al. Are All Losses Created Equal: A Neural Collapse Perspective, 2022

[5] Guo et al. Cross Entropy versus Label Smoothing: A Neural Collapse Perspective, 2024

[6] Kornblith et al. Why Do Better Loss Functions Lead to Less Transferable Features? 2021

### Questions
1. For the CIFAR experiment with the spurious feature, are you training the full model or just probing a frozen ResNet with a linear logistic classifier? Additionally can you clarify the difference between line 65 where you specify "if the high variance feature satisfies a greater separability" and the experiment where the low variance spurious feature is more separable.
2. The experiments all use heavily overparameterised networks. 2048 is somewhat overkill for MNIST and ResNet-18 is not designed for CIFAR but ImageNet. ResNet-18 has over 10 million parameters whilst ResNet-56 which is designed for CIFAR has < 1 million and both achieve similar performance on CIFAR. Can you comment on how overparameterisation/overfitting affects the theory/experiments of the paper? This contrasts to the binary linear scenario where underfitting/model being insufficiently expressive may be more likely(?).

I am happy to raise my score if the stated weaknesses (other than 3 which I feel unfortunately can't be changed) and above questions are addressed.

Edit: as the authors have addressed most of my queries and concerns, and also improved the paper in response to the other reviewers, I have raised my score from 5 to 6.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper presents a theoretical analysis of label smoothing and Mixup techniques. The authors investigate the differences between these label augmentation methods and the vanilla approach (i.e., empirical risk minimization with weight decay) regarding their effectiveness in learning features of varying variance. In a binary linear classification problem, they prove that, under certain assumptions, the vanilla method prefers high-variance features, while label augmentation encourages the model to learn low-variance features. This analysis is then extended to more complex settings by showing that the label augmentation loss is lower-bounded by model variance. The theoretical findings are supported by empirical results on both original and modified CIFAR-10, CIFAR-100, and colored MNIST datasets.

### Strengths
The theoretical results are insightful and are effectively supported by a variety of well-designed experiments.

### Weaknesses
- It remains unclear why learning low-variance features leads to better generalization in practical scenarios, as this approach can result in poorer generalization on synthetic datasets.
- The comparison to previous works on feature learning (e.g., Chidambaram et al., 2023; Zou et al., 2023) is insufficient. I encourage the authors to provide a more detailed comparison with these works (e.g. empirical loss vs population loss). Additionally, the notion of "features" appears to differ between this work and prior studies; clarifying what "features" specifically refers to in this context would enhance the reader’s understanding.

### Minors

- Line 795: “approach of Theorem 3.4” → "approach of Theorem 3.3" and I hope the author provides a complete proof of Corollary 3.4 for clarity, even if it largely repeats the proof of Theorem 3.3.

### Questions
- In the proof of Corollary 3.4, what happens if the max-margin solution does not include high-variance features? It appears that the proof technique may not apply in this scenario. Could the authors clarify whether this case is impossible or if there are additional conditions to address it?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript studies the relationship between label augmentation techniques (label smoothing and Mixup) and feature variance. The authors demonstrate that label augmentation in binary classification with a linear model encourages the model to learn low-variance features. In contrast, ERM with weight decay can lead to learning higher-variance features under certain conditions. They also show that the losses of label smoothing and Mixup for nonlinear models are lower-bounded by a function of the model output variance. Additionally, they provide empirical evidence that label augmentation may negatively impact model training.

### Strengths
1. This manuscript is well-written, and the theoretical parts are clearly presented.
2. This manuscript links label smoothing and Mixup to feature variance, offering new insights into the analysis of label augmentation.

### Weaknesses
1. The main claim that “optimizing the empirical risk with weight decay can learn higher variance features” is built on a strong assumption. As the authors note, Assumption 3.2 is indeed strong, suggesting that the $\\mathcal{H}$ dimensions are inherently better suited for classification than the $\\mathcal{L}$ dimensions. However, it’s straightforward to construct a task that relies more heavily on the $\mathcal{L}$ dimensions than on the $\mathcal{H}$ dimensions. For instance, consider a 2-D distribution where the $x$ coordinates for both classes are uniformly ranging from -1 to 1, while the $y$ coordinates are set at -0.1 or 0.1, depending on the class. In fact, by simply reversing the assumption from $y \\langle v^*, x \\rangle > y \langle u^*, x \\rangle$ to $y \\langle u^*, x \\rangle > y \langle v^*, x \rangle$, the conclusion of Theorem 3.3 changes from $\\|w^*_\\mathcal{H}\\|^2 \ge \frac{1}{2} \\|w^*\\|^2$ to $\\|w^*_\\mathcal{L}\\|^2 \ge \frac{1}{2} \\|w^*\\|^2$.
2. There is no discussion on why label smoothing or Mixup learn more on low-variance features compared to ERM or ERM with weight decay. In fact, by setting $\\alpha$ to 0, the label smoothing loss reduces to the negative log-likelihood loss, allowing one to obtain results similar to those in Theorem 3.6 or Proposition 3.10 for ERM, which also demonstrate that the ERM loss is lower-bounded by a function of the model output variance. Thus, $\\alpha$ plays a central role in learning low-variance features, but this important analysis is missing.

### Questions
1. Can these results extend to multi-label classification?

### Soundness
2

### Presentation
3

### Contribution
3
