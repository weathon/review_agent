# Unlearning Evaluation through Subset Statistical Independence

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 8, 2, 4

## Abstract
Evaluating machine unlearning remains challenging, as existing methods typically require retraining reference models or performing membership inference attacks, both of which rely on prior access to training configuration or supervision labels, making them impractical in realistic scenarios. Motivated by the fact that most unlearning algorithms remove a small, random subset of the training data, we propose a subset-level evaluation framework based on statistical independence. Specifically, we design a tailored use of the Hilbert–Schmidt Independence Criterion to assess whether the model outputs on a given subset exhibit statistical dependence, without requiring model retraining or auxiliary classifiers. Our method provides a simple, standalone evaluation procedure that aligns with unlearning workflows. Extensive experiments demonstrate that our approach reliably distinguishes in-training from out-of-training subsets and clearly differentiates unlearning effectiveness, even when existing evaluations fall short.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes to use Hilbert–Schmidt Independence Criterion to evaluate the effectiveness of unlearning methods. This novel, statistic-based method facilitates evaluation without a retrained reference model or shadow models.

### Strengths
1. The proposed method is well-motivated, novel, clear, and effective. This paper is well-presented and easy to read.
2. The research questions studied in 4.1 are important.
3. The discussion in Section 5 is comprehensive and valuable.

### Weaknesses
1. Previously, unlearning methods usually played with the metrics, including accuracies and MIA results, via cherry picking hyperparameters with the best performance. It seems HSIC won't stop this game, but just provides one more metric to fit.
2. I think HSIC might be extended for unlearning in generative models, but the authors did not discuss it.

### Questions
The authors claim that HSIC evaluation does not require a retrained reference model. But I'm wondering: Is it possible that one day a new unlearning method achieves higher OTR than a retrained model's? If so, how do we analyze this phenomenon?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper examines machine unlearning evaluation at the subset level. Motivated by insufficiencies in existing evaluation methods such as retraining (which is computationally expensive) and membership inference attacks (MIA) (which lack effectiveness), the authors propose a tailored metric based on the Hilbert-Schmidt Independence Criterion (HSIC) to measure statistical dependence between model representations and the unlearned subset with a proxy. Experiments demonstrate that the proposed method serves as a reliable evaluation tool for machine unlearning.

### Strengths
1. I find the paper very well written and pleasant to read. It is sufficiently motivated by an important problem in machine unlearning and proposes an elegant solution. The presentation is clear and easy to follow.

2. The adoption of subset-level statistical dependence as an evaluation metric is both interesting and clever. It provides a well-defined tool for assessing model-data interactions in machine unlearning (and potentially for measuring neural network memorization more broadly). I believe the proposed method has strong potential to become a standard evaluation tool in the unlearning literature.

3. The paper potentially opens many promising future research directions. I would be particularly interested to see its adaptation to other foundation models, such as those based on contrastive learning methods and generative models.

### Weaknesses
1. It appears that the size of the unlearning subset plays an important role in the effectiveness of the proposed method. The experiments demonstrate that SDE works well for unlearning sets comprising 5-20% of the training data, which represents a relatively large proportion. However, consider a scenario where a user requests the unlearning of only a few samples: would SDE remain effective in this case? In other words, is it possible to quantify the minimum subset size at which SDE provides a reliable evaluation?

2. The experiments show that Unroll obtains very low OTR. Is there any explanation for this phenomenon? Moreover, I am curious whether it is possible to attribute unlearning effectiveness at the individual sample level within the subset. While this may seem overly ambitious from a statistical perspective, I wonder if the authors have any insights or preliminary thoughts on this direction.

Note that neither question necessarily requires additional experiments. The authors are welcome to include a discussion of these points and propose reasonable directions for future work.

### Questions
See above.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces *Split-half Dependence Evaluation (SDE)*, a new way to measure how well a model has “forgotten” data during machine unlearning. Instead of retraining models or using membership attacks, it checks the statistical independence of model outputs on data subsets using the Hilbert–Schmidt Independence Criterion (HSIC).

### Strengths
1. Clear presentation and good illustration.
2. Conceptually new evaluation framework.

### Weaknesses
1. Unclear motivation: The benefits of the proposed method should be better explained and carefully compared with other proposed methods. For a proposal of an "evaluation" method, it is especially important to measure/prove/validate its properties, such as robustness, consistency, among others.
2. Unjustified design choice: Some of the design choices are not well-explained, e.g., HSIC with specific RBF kernels, Mann-Whitney U-test, Jensen-Shannon Divergence, among others. To me, it feels artificial to see these components being introduced to the method without a clear justification.

### Questions
1. Missing reference: Recently, there have been many related works focusing on unlearning evaluation, e.g., [1], where the idea of split sets is also explored.
2. Weakness 1: Can you explain or provide additional experiments that demonstrate some desirable properties for the proposed evaluation metric, compared to others? Also, can you explain why your evaluation metric is better compared to the existing literature?
3. Weakness 2: Can you provide some brief justification for each of the mentioned components, besides your main conceptual novel proposal (SDE)?

[1]: Tu, Yiwen, Pingbang Hu, and Jiaqi Ma. Towards Reliable Empirical Machine Unlearning Evaluation: A Cryptographic Game Perspective.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new metric for evaluating unlearning methods, leveraging a clever connection to statistical independence. The main idea behind the paper is to measure the statistical dependence (using a method called the Hilbert-Schmidt Independence Criterion) between different subsets of examples, and use the computed (in)dependence scores as a proxy to evaluate unlearning efficacy. This allows one to evaluate unlearning without needing ground-truth retrained models.

### Strengths
- The paper identifies an interesting and important problem (improving the efficiency of unlearning evaluations)
- The idea to use statistical dependence testing as a proxy for retraining is quite interesting and novel. It also seems quite convenient to implement, and thus practical to run in actual unlearning setups.
- The paper conducts a thorough ablation study to understand the effect of different design choices on the efficacy of their approach

### Weaknesses
- The theoretical analysis seems quite handwavy and makes lots of approximations without really justifying why we might expect them to be true. To me this is the main concern with the paper, and one that permeates throughout the rest of the weaknesses (see also Q1).
- In Table 4, it's seems concerning that the retrained model does not get 100% according to the metric - doesn't this suggest that the metric is either overly sensitive or otherwise misspecified with respect to the unlearning objective?
- The fact that the method hinges on the selection of good reference sets also limits the practical applicability of the algorithm (this limitation is explicitly acknowledged by the authors).

### Questions
- Is there a simple theoretical setting where this evaluation is exactly the right thing to do?
- Did the authors try other kernel functions outside of RBF?
- Can the authors provide a computational cost analysis of their method? What are the main computational steps, and how does the cost scale with the various dataset sizes?

### Soundness
3

### Presentation
3

### Contribution
3
