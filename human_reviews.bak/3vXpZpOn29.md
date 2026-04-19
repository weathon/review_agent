# Machine Unlearning via Simulated Oracle Matching

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Machine unlearning---efficiently removing the effect of a small "forget set" of training data on a pre-trained machine learning model---has recently attracted significant research interest. Despite this interest, however, recent work shows that existing machine unlearning techniques do not hold up to thorough evaluation in non-convex settings. In this work, we introduce a new machine unlearning technique that exhibits strong empirical performance even in such challenging settings. Our starting point is the perspective that the goal of unlearning is to produce a model whose outputs are *statistically indistinguishable* from those of a model re-trained on all but the forget set.  This perspective naturally suggests a reduction from the unlearning problem to that of *data attribution, where the goal is to predict the effect of changing the training set on a model's outputs. Thus motivated, we propose the following meta-algorithm, which we call Datamodel Matching (DMM): given a trained model, we (a) use data attribution to *predict* the output of the model if it were re-trained on all but the forget set points; then (b) *fine-tune* the pre-trained model to match these predicted outputs. In a simple convex setting, we show how this approach provably outperforms a variety of iterative unlearning algorithms. Empirically, we use a combination of existing evaluations and a new metric based on the KL-divergence to show that even in non-convex settings, DMM achieves strong unlearning performance relative to existing algorithms. An added benefit of DMM is that it is a meta-algorithm, in the sense that future advances in data attribution translate directly into better unlearning algorithms, pointing to a clear direction for future progress in unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces the concept of Oracle Matching to significantly reduce the time complexity of unlearning processes, achieving this in a fraction of the time required for traditional retraining or fine-tuning while maintaining model performance as close as possible to that of full retraining. The authors utilize the concept of "DataModels" to efficiently approximate a proxy for the oracle, leveraging this proxy within the Oracle Matching framework.

### Strengths
The proposed approach is both innovative and highly impactful, offering substantial reductions in unlearning time. The authors demonstrate a profound understanding of unlearning methods based on gradient descent and fine-tuning. They delve into the challenges of these methods, particularly examining their impact on the model's performance after unlearning. Although gradient-based approaches are among the most effective unlearning methods, they often degrade the predictive performance of the model considerably—a drawback that the authors have thoroughly investigated and discussed.

The concept of Oracle Matching, combined with the use of DataModels, greatly reduces time complexity. This approach shows considerable promise for improving the efficiency of unlearning processes without compromising model performance.

### Weaknesses
Grammatical errors:
- Abstract - line 2
- The abstract and introduction was written in rush. please 


Early introduction of specific notions:
- such as $S$, $S_R$ in the beginning of the paper without providing the backgrounds and clearly stating them, confuses the reader. (Intro - second paragraph)
- line 74 - trained model $\theta$

Ambiguity:
- line 49 - Simple models // what is considered to be a simple model?
- line 59 - variety of empirical evaluations and benchmarks // what are these benchmarks? either needs to mention at least one of evaluation criteria, or rephrase the sentence.

Incorrect Statement:
- line 63 - , fine-tuning-based methods typically employ.... - incorrect, the simple fine tuning only focus on the remaining datapoints and fine tune the model on $S_R$. If there is a paper that conducts the fine tuning in the way you mentioned in this line, you need to point it out, but otherwise, check this paper ["Model Sparsity Can Simplify Machine Unlearning"](https://openreview.net/pdf?id=0jZH883i34)

Incorrect notation:
- line 152 - the notation for approximate unlearning is the same as exact unlearning.

line 83 - Empirically, we find that .... - I was expecting to see a comparison between your method and unlearning distillation approach, but you didn't made that comparison.

### Questions
line 246 - specify $\text{safe}(S_F)$ - how do you know this safe(S_F)? How did you calculate this? 

page 6 - line 317 - this description is clear but what exactly distinguishes your approach from distillation. The comparison between your method and a distillation unlearning approach such as [Efficient Two-stage Model Retraining for Machine Unlearning](https://openaccess.thecvf.com/content/CVPR2022W/HCIS/papers/Kim_Efficient_Two-Stage_Model_Retraining_for_Machine_Unlearning_CVPRW_2022_paper.pdf)

Figure 3 - For this evaluation, did you retrain the model and then computed the unlearned model using oracle matching?

line 365: in this formulation, to create the datamodels for remaining data, you used the datamodel of whole dataset minus the datamodel of forget set. if my understanding is correct, doesn't it add the additional computation because you need to estimate two datamodels first. 
also can you prove this formulation result in an exact or good estimation of datamodel for remaining data? 

line 374: the true oracle output - I am just worried about the closeness of the proxy to actual retain datamodel to the.

line 404 - , the datamodel generalizes well to new forget sets in practice. very interesting how do you demonstrate this ?

ORACLE MATCHING FOR LINEAR MODELS: my understanding  is that the quality of unlearning for the oracle matching heavily is influenced by the unlearning quality of approximated oracle model . it would be interesting to investigate the influence of that model.

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
This paper introduces a new Unlearning algorithm called Datamodel Matching (DMM) to unlearn a forget set by fine-tuning a model that has been pre-trained on a larger dataset including the forget and retain set. They use predictive data attribution to approximate the oracle model---the one that is retrained from scratch on the retain set and hence has not seen the forget set at all. Predictive data attribution learns datamodels for each input x to simulate how a model trained on the retain set would behave on x. With this approximation, DMM then applies Oracle Matching to align the model's output distribution with that of the oracle. They introduce KLoM for measuring the unlearning quality and show empirically that their algorithm outperforms previous gradient-based algorithms, achieving a lower KLoM and quickly approaching oracle-level accuracy with only a fraction of the retraining time.

### Strengths
1. The paper is well-written and easy to follow.

2. It provides a solid introduction to Unlearning, offering a detailed overview of related work, recent advancements, and existing challenges. The motivation is well-described, and the flow effectively positions this paper within the broader field, helping readers understand its scope and contributions better. The use of data attribution to approximate the Oracle model is particularly compelling.

### Weaknesses
1. The experimental results, while supportive of the algorithm’s effectiveness, are somewhat limited in scope. The analysis focuses primarily on CIFAR-10 and an ImageNet subset (Living-17), with figures 1 and 3 illustrating outcomes only on these datasets. Expanding the experiments to include additional datasets would enhance the generalizability of the findings. On the Retain set, Oracle Matching gets close to the Oracle in CIFAR-10 but this is not the case with Living-17 which means that while the algorithm can reduce KLoM on the forget set, it does not work as well on the retain set. More discussion on this observation can improve the understanding of the algorithm’s limitations, and I'm interested to know how this observation extends to other tasks.

2. The paper could benefit from clearer explanations and interpretations of the figures and baselines. The discussion around interpreting each figure is limited. Additionally, some of the baseline methods are not well-defined; for instance, SCRUB is introduced without a description, and GD (Gradient Descent) is not explained well until page 9 (also see my question on GD in the Questions).

3. Minor typo: The RHS of equation (2) should be dependent on x.

4. Calculating one $\beta$ vector for each x in the dataset appears computationally intensive. Although the authors mention that this is a one-time process that amortizes over unlearning requests, further discussion on its computational cost relative to the unlearning phase would be helpful.

### Questions
1. This question is regarding the GD baseline. Given a model $\theta_{full}=A(S)$ trained on a full dataset S (including the retain set $S_R$), GD minimizes the loss on the retain set using gradient descent, starting from $\theta_{full}$. Since $S_R$ was already in the dataset S that the model was trained on, this essentially involves further training on $S_R$, which could lead to overfitting without effectively forgetting $S_F$. This may explain why, in Figure 3, GD performs similarly to 'Do Nothing' and, on average, even worse. My interpretation is that further training on $S_R$ may not significantly alter a model that was sufficiently trained on S if the loss is strongly convex and could lead to overfitting (and hence degrading performance on the validation) in more complex landscapes. Could the authors clarify if there’s any specific benefit of GD for unlearning that I might be overlooking?

2. The authors mention the possibility of having duplicates across the forget and retain sets on page 6 when discussing the drawbacks of Gradient Ascent. Given that $S_F$ and $S_R$ are sets and $S_R$ is defined as $S \backslash S_F$, I don't see how this duplication is possible. Could the authors clarify this?

3. Could the authors provide more interpretation of the results in Figure 2? How do you interpret the changes and fluctuations in the red and gray lines?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the problem of _machine unlearning_ (removing the effect of a few training data points "forget set" on the model's outputs) with by reducing it to the problem of _data attribution_ (predicting the effect the training set on the model's outputs). With this, the paper proposes a meta-algorithm Datamodel Matching (DMM) that gets predictions from data attribution on all-but-forget set and finetunes the model to match the predictions. A new unlearning metric KL Divergence on Metrics (KLoM) is also introduced. Finally, the paper presents experiments on unlearning in image classification tasks, showing that DMM is better at unlearning and faster than naive-retraining on the all-but-forget-set.

### Strengths
The paper is written extremely well, and has a natural flow to it. When reading I thought of many questions and added annotations, only to find them answered in the next paragraph or section. E.g. the authors introduce Oracle Matching with the strong assumption of oracle access to data attributor $f^{oracle}$ (Section 4.2), and immediately discuss how to simulate such an oracle without such an access (Section 4.3). In Section 5 as well, the paper attends to a natural question readers could have: is oracle matching useful when the problem is easy to solve with gradient descent. This is good writing, and I appreciate the authors' efforts in putting themselves in the readers' shoes.

In sum, the Oracle Matching and Simulation methods are intuitive yet thoroughly tested on image classification tasks. These methods are original as well, and it is surprising to me that linear datamodels work so well empirically. I'd like to see this paper accepted, and the research directions it inspires.

### Weaknesses
It is unclear that linear datamodels extend to other kinds of tasks, e.g. language modeling or regression problems. I believe this to be a major weakness of the paper. While linear datamodels lead to simple algorithms in this paper, the previous work [1] does not have a good argument for why linear datamodels work [1; Section 7.2]---in fact Figure 6 of [1] display imperfect matching using linear datamodels. It'd be useful to mention this limitation in this manuscript as well, and discuss the limitation's impact to machine learning.

# Suggestions:
1. Line 156. It'd be useful to the reader to add a citation on differential privacy, e.g. one of the standard works like [2].
2. Line 176. $\hat{f}$ should have output range in $\mathbb{R}^k$ since the range of $f_x$ is in $\mathbb{R}^k$. 
3. Line 182. "show" -> "empirically show".
4. Definition 3. Write safe, $S_F$, and input $x$ explicitly in KLoM, otherwise KLoM$(\mathcal{U})$ looks like KLoM of the unlearning function across _all_ safe functions and inputs. I'm curious why the authors wrote KLoM$(\mathcal{U})$.
5. Add a Limitations section.

[1] Ilyas, A., Park, S. M., Engstrom, L., Leclerc, G., & Madry, A. (2022). Datamodels: Predicting predictions from training data. arXiv preprint arXiv:2202.00622.
[2] Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science, 9(3–4), 211-407.

### Questions
# High-level questions
1. Definition 3. Why is KLoM called KL-divergence of Margins? I couldn't find the reasoning.
2. Line 321 and Alg A.1. Assuming that dataset $S$ has distinct datapoints, isn't $S_{finetune}$ the same as $S$, implying that Alg A.1 is simply matching the model's outputs to the oracle? Then I don't understand the novelty of Oracle Matching.
3. Line 364. Why replace the first term with $\beta$? Linear $\beta$ is exactly the definition of the estimator $\hat{f}$, which is only an approximation to $f_x$. It'd make sense to explicitly state that DM-DIRECT approximately simulates the oracle outputs. You could then argue that even this approximate simulator works well empirically.

# Low-level questions
1. Line 173. Is this estimator/datamodel only for a single $x$? Does this mean that distinct inputs might require distinct datamodels?
2. Equation 2. What is the approximation in? Is it in some measure of distributions?
3. Figures. Why are there multiple points on the plot for each legend entry?

### Soundness
4

### Presentation
4

### Contribution
4
