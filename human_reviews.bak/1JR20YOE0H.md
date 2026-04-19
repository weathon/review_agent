# On Feature Diversity in Energy-based Models

- Decision: Reject
- Scores: 6, 6, 5, 6, 6

## Abstract
Energy-based learning is a powerful learning paradigm that encapsulates various discriminative and generative approaches. An energy-based model (EBM) is typically formed of inner-model(s) that learn a combination of the different features to generate an energy mapping for each input configuration. In this paper, we focus on the diversity of the produced feature set. We extend the probably approximately correct (PAC) theory of EBMs and analyze the effect of redundancy reduction on the performance of EBMs. We derive novel generalization bounds for various learning contexts, i.e., regression, classification, and implicit regression, with different energy functions and we show that indeed reducing redundancy of the feature set can consistently decrease the gap between the true and empirical expectation of the energy and boosts the performance of the model.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces the concept of feature diversity, represented as $(\upsilon-\tau)$-diversity, to enrich the probably approximately correct (PAC) theory for energy-based models (EBMs). It establishes generalization bounds across various energy functions for regression, classification, and implicit regression. A consistent observation emerges: reducing feature redundancy consistently bolsters generalization performance in energy-based approaches, regardless of the chosen loss function or training strategy. Empirical results across regression, continual learning, and generative modelling consistently demonstrate that increasing $(\upsilon-\tau)$-diversity can enhance model performance.

### Strengths
- The paper provides a theoretical insight into the connection between feature diversity and the generalization capacity of energy-based models. It also suggests a promising direction for enhancing diversity within feature sets in future approaches.
- It is remarkably surprising that such an intuitive and straightforward feature diversity regularisation consistently yields performance improvements.

### Weaknesses
For the generative modelling in the appendix, I have reservations about claiming a 10% improvement in FID as significant, primarily because the FID of the original EBM model is already very low. Conversely, the NLL loss does not exhibit a notable enhancement.

### Questions
- The coefficient $\beta$ is exceptionally minute (e.g., on the order of $1 e^{-11}, 1 e^{-12}, 1 e^{-13}$), making it impractical as it requires predefining $\beta$ based on the value of the second term of $L_{aug}$. Is it feasible to employ Monte Carlo estimation to approximate the second term, thereby preventing it from becoming excessively large and dominating the original loss?
- Could you apply the augmentation loss to generative modelling with high-dimensional datasets, such as CIFAR-10? Achieving success with this straightforward regularization approach in high-dimensional datasets would be both theoretically and practically promising.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper delves into the study of the energy-based model (EBM) which has an inner model formulated as a weighted sum of different features (functions). The authors introduce the concept of the $(\vartheta, \tau)$-diversity, claiming that the enhanced diversity can improve the generalization bounds of the EBM model. Moreover, they substantiate this for specific tasks such as regression, binary classification and implicit regression by theoretically giving an upper bound of the generalization gap depending on $\tau$. Additionally, they apply implement redundancy reduction by incorporating a straightforward regularization term into the loss function. The experimental results demonstrate the improvement brought by the redundancy reduction.

### Strengths
- The paper is well-written and easy to follow. The idea that connecting diversity and generalization is interesting and insightful.
- The implementation of feature redundancy reduction, as evidenced in the experiments, improves the model's performance.

### Weaknesses
- My observation is that the theoretical upper bound is heavily contingent on the specific task. It appears to be a refined calculation of previous work in Lemma 1
- The theoretical contribution is based on the upper bound presented in Lemma 1. However, this result seems to originate from an unpeer-reviewed work. By the way, due to the limited review time, i have not been able to thoroughly verify the proof through this research line. Hence there is a concern about the correctness of the result. Personally speaking, it might be more appropriate to submit this work to theoretical conferences like COLT, ALT, etc.
- The experiment performance is not strong. The tasks are limited to 1-d regression and continual learning, which is insufficient. The improvement exists but is relatively marginal.
- There are no discussions and experiments looking deep into the regularizer (see questions below)


minor typos:
- the definition is $(\vartheta, \tau)$-diversity, but in many places including Def 1 itself the notation is $\vartheta$-diversity.

### Questions
- The diversity concept used in this paper is not usual. Does it has something to do with the homonymous concept in datasets/search [1]? Maybe it is better to use the contrary concept redundancy.
- Besides examples already provideded, are there some other problems can be encoded as EBM?
- is there any generic generalization upper bound for the EBM according to $(\vartheta, \tau)$-diversity, without exploring the specific form of the task?
- What is the correlation between diversity regularizer and other popular regularizers? Can they exist together, or do their functions overlap? I think it is necessary to make a comparison to other regularizers and add an ablation experiment to confirm that diversity regularizer plays a different role with other regularizers.


[1] Improved Approximation Algorithms and Lower Bounds for Search-Diversification Problems

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors study how the generalization of energy-based models (EBMs) ca be improved. Specifically, they study how the diversity of the input features affects the generalization.

They derive generalization bounds for regression, classification and implicit regression. Based on these theoretical results, they propose a simple regularization term that can be added to the standard EBM loss during training.

The proposed regularizer is evaluated on two illustrative 1D regression datasets, as well as on continual learning tasks on CIFAR10 and CIFAR100.

### Strengths
The authors provide a number of theoretical results.

The proposed regularizer term is simple.

Adding the proposed regularizer term consistently improves the performance a bit in the experiments.

### Weaknesses
The paper could be more well written. It contains quite a few typos (see "Minor things" in Questions below).

The paper is not ideally structured. The Introduction is long and contains quite a lot of mathematical details (I would split the Introduction into two sections, Introduction and Background perhaps). There is no related work section.

The experimental evaluation is not very extensive. Two illustrative / toy 1D regression datasets and an experiment on CIFAR10/CIFAR100.

The experimental results are not overly convincing/impressive. The performance gains in Table 2 seem relatively insignificant.

### Questions
1. Could you add at least one more experiment? Since you use the 1D regression datasets from (Gustafsson et al.,
2022), could you not apply the proposed method to one of their four image-based regression datasets as well?

2. Could you add a separate Related work section?

3. In Section 2.1, you write that "The two valid energy functions which can be used for regression are $E_2(h, x, y) = 1/2||G_W(x) − y||_2^2$ and $E_1(h, x, y) = ||G_W(x) − y||_1$". However, this is not what is used in (Gustafsson et al, 2022; 2020b;a) (which are cited just above)?


Minor things:
- I think table captions should be above the tables.
- Section 1: "In machine learning context" --> "In a machine learning context"?
- Section 1: "proportional $\vartheta$, This" --> "proportional $\vartheta$, this"?
- Section 2.3, "improves the generalization of EBM..." EBM --> EBMs?
- Section 2.4, "in the form of regularization Laakom et al. (2023)": Laakom et al. (2023) --> (Laakom et al., 2023)? The same also in the Figure 2 caption? I think \citet should be changed to \citep in many places across the paper?
- Section 3, "Given an EBM model with...": EBM model --> EBM?
- Section 3.1, "we validate the proposed regularizer equation 15 on two...": equation 15 --> in equation 15?
- Section 3.1, "several losses has been proposed": has --> have?
- Section 3.1, "are formed of 10 units expect the final output, which has one hidden units corresponding...": expect --> except? units --> unit?
- Table 1 caption: sigma_1 --> $\sigma_1$.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a definition of feature diversity (called "($\vartheta - \tau$)-diversity") that they use to characterize the generalization performance of energy-based models (EBMs). They then propose a simple regularization term in the loss function that encourages feature diversity and show that it consistently improves the performance of EBMs on small datasets.

### Strengths
## Simple idea with rigorous justification that shows some performance gains on small tasks

- (+) The paper formalizes feature diversity in EBMs and characterizes the generalization performance.
- (+) Encouraging feature diversity is as simple as adding an additional term to the loss function to encourage feature representations to diverge.
- (+) The experimental results show that the feature diversity loss term consistently shows a small improvement on the performance of EBMs on small datasets.

**Originality**: To my knowledge this work is novel in the context of EBMs and has not been previously explored or formalized.

**Quality and Clarity**: The paper is well organized and clearly written, though the mathematical conclusions are a bit beyond my capacity to evaluate.

**Significance**: See weaknesses

### Weaknesses
## Evaluations and empirical characterizations are limited

1. (-) The effect of $\beta$ on the performance improvement is not well characterized (i.e., when is $\beta$ too big that we lose the performance gain of the regularization term?). Only 3 values of $\beta$ are ever considered. 
2. (-) The performance improvement is admittedly quite small and within one standard deviation of the original results reported in Table 5 of [Li et al.](https://arxiv.org/pdf/2011.12216.pdf). The paper should consider reporting the standard deviation of their results as well.

**Originality**: See strengths.

**Quality and Clarity**: See strengths.

**Significance**: Medium-Low. The experimental results and the characterization of the method do not provide enough information to anticipate more than a small impact.

As I clarify in my confidence score, there are many mathematical details that I am not qualified to review, but I do not find the empirical results particularly robust or convincing.

### Questions
1. It should be possible to visualize the learned features on the CIFAR10 datasets with and without the feature diversity loss term. Why is this not done?
2. In the CIFAR-10/100 experiments, did the authors consider applying the regularization term to representations other than that outputted by the last of the intermediate layers?
3. The regularization term in Eq (15) has a computational complexity that will grow quadratically with the number of features. Is this interpretation correct? If so, what would the slowdown of this method be when applied to models with a larger number of features?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper investigates the impact of feature diversity on the generalization ability of energy-based models (EBMs). It extends the theoretical analysis of EBMs and introduces the concept of (ϑ − τ)-diversity, emphasizing the importance of reducing feature redundancy to enhance model performance. The authors derive generalization bounds for different learning contexts, such as regression and classification, using various energy functions. They highlight the theoretical guarantees of learning through feature redundancy reduction, independent of the loss function or training strategy. Additionally, the paper emphasizes the need for novel regularization techniques to address the limitations of empirical energy minimization and underscores the critical role of the feature set's quality in the overall model performance. Overall, the paper provides insights into the significance of feature diversity in EBMs and suggests promising research directions for promoting diversity within the feature set.

### Strengths
1. The paper provides a comprehensive review of existing literature on energy-based models, including references to key works in the field.
2. It presents empirical results and quantitative evaluations of different approaches for generating MNIST images, providing insights into the performance of the proposed method.
3. The paper extends the theoretical analysis of energy-based models and focuses on the diversity of the feature set, which can contribute to a deeper understanding of the model's generalization ability.
4. The authors address the limitations of simply minimizing the empirical energy over the training data and emphasize the importance of developing novel regularization techniques, indicating a critical approach to the research problem.

### Weaknesses
1. The paper may lack a detailed explanation of the specific regularization techniques developed to address the limitations of empirical energy minimization, which could limit the reproducibility of the results.
2. 4. The paper may not provide a clear discussion of the limitations or potential challenges associated with the proposed approach, which could impact the overall assessment of the model's robustness and applicability.

### Questions
NA

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
