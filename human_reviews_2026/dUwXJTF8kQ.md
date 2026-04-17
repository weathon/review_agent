# In-Context Learning Is Provably Bayesian Inference: A Generalization Theory for Meta-Learning

- Decision: Reject
- Scores: 6, 4, 8, 4

## Abstract
This paper develops a finite-sample statistical theory for in-context learning (ICL), analyzed within a meta-learning framework that accommodates mixtures of diverse task types. We introduce a principled risk decomposition that separates the total ICL risk into two orthogonal components: Bayes Gap and Posterior Variance. The Bayes Gap quantifies how well the trained model approximates the Bayes-optimal in-context predictor. For a uniform-attention Transformer, we derive a non-asymptotic upper bound on this gap, which explicitly clarifies the dependence on the number of pretraining prompts and their context length. The Posterior Variance is a model-independent risk representing the intrinsic task uncertainty. Our key finding is that this term is determined solely by the difficulty of the true underlying task, while the uncertainty arising from the task mixture vanishes exponentially fast with only a few in-context examples. Together, these results provide a unified view of ICL: the Transformer selects the optimal meta-algorithm during pretraining and rapidly converges to the optimal algorithm for the true task at test time.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper aims to better understand the underlying mechanism of In-Context Learning (ICL) in large language models (LLMs). The authors propose that ICL may not represent a fundamentally new kind of learning, but rather can be interpreted as a form of implicit Prompt Learning (PL).

They develop a unified theoretical framework suggesting that during ICL, the model effectively infers a task-specific prompt representation from the few-shot context and then uses this implicit prompt to make predictions—analogous to how explicit prompt tuning works with optimized prompt embeddings.

The work mainly consists of theoretical analysis, linking attention-based representations with implicit prompt formation. The motivation, as I understand it, is to clarify what ICL is truly doing and to explain why it can behave like prompt learning, even without explicit gradient updates.

Overall, I appreciate the authors’ effort to bring together ICL and prompt learning under one theoretical framework. However, I have to admit that I struggled to fully understand some of the technical details and proofs.

### Strengths
1. The claim that ICL $\approx$ implicit prompt learning offers a compelling and elegant conceptual bridge between two currently disjoint areas (prompt learning and ICL).

2. This perspective has strong explanatory power and could influence how we interpret emergent few-s

3. hot capabilities in LLMs.

### Weaknesses
1. The paper is purely mathematical. Without numerical examples or simulations, it’s hard to gauge how the theory behaves in practice or whether constants in bounds are realistic.
2. The theory is elegant, but what should practitioners or model designers *do* differently because of it? That remains unclear.

### Questions
1. How should we interpret the Bayes Gap intuitively? Is it like train-test generalization error, or something closer to representation mismatch?
2. In practical terms, is it possible to measure the Bayes Gap or Posterior Variance for large pretrained models (e.g., GPT-style LLMs)?
3. The paper focuses on uniform-attention Transformers. Would the same decomposition hold for standard multi-head attention or linear attention architectures?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper considers a new setting of in-context learning including a new modeling architecture named "Uniform-attention Transformer Architecture" define in this paper coupled with data generation process.
Based on the setting, where the model is trained on finite samples generated from the generation process, the paper derives risk of in-context learning, and decompose it to Bayes Gap and Posterior Variance.
Bayes Gap consists of Approximation error which describe the and Pretraining generalization error from the expressiveness of the model, and the Pretraining generalization error from limited pretraining data.
Posterior Variance describe the bayes process of the in-context learning.
Finally, the paper gives the OOD analysis on the ICL risk.

### Strengths
(1) The paper has a complete system including model architecture, data generation, and risk analysis based on the system. I.e., the paper is self-contained.

(2) The paper explains the risk analysis well, which makes it easy to understand intuitively. For example, in Theorems 1 and 2, the paper decomposes the risk into Bayes Gap (Approximation error + Pretraining generalization error) and Posterior Variance, providing intuition behind the mathematical formulation, making it easier for readers.

(3) The paper analyzes the OOD stability of ICL and gives the risk changes under a distributional shift.

### Weaknesses
(1) The paper is not well-aligned with real-world Transformers. The paper is less aligned with real-world Transformers compared with existing work listed in the related work. Specifically, existing work considers architectures more similar to the Transformer, which utilizes an attention mechanism. In this paper, the defined Uniform-attention Transformer Architecture differs from the Transformer in that it sets Q=K=0 constantly.

(2) Consider ICL, a minor difference between the setting of this paper and ICL is that in this paper, the posterior is calculated based on $D^{k}$; in ICL, the posterior is calculated on $P^{k}$.

(3) The paper lacks some numerical or potentially real-world experiments to illustrate the findings, which could help to better show the findings.

(4) As question (1).

### Questions
(1) While the paper gives the definition of Uniform-attention Transformer Architecture in Sec 2.2, could the author explain why in Theorem 2, the approximation error is independent of the architecture's number of layers?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper theoretically investigates the in-context learning in the lens of statistical learning theory. The study is conducted on certain assumptions, such as boundedness, uniform attention transformer architectures. In addition, the study is currently applicable for the mean square error loss. In this setting, the in-context learning risk (or loss) is decomposed following the well-known bias-variance decomposition. Each of the terms is then analyzed and bounded accordingly to connect to the results of previous studies. Although the current study is limited in several setting, it is an encouraging result and effort.

### Strengths
The results presented in the paper, and in particular Theorems 2 and 3, explain and connect to several observations as well as results of previous studies. For example, Theorem 2 shows that the upper-bound of the bias term (or Bayes gap) depends on the complexity (or expressiveness) of the model used, denoted as $m$. It also depends on the samle size and other parameters. Whereas in Theorem 3, the variance term is mainly dependent on the complexity of data (also known as irreducible noise). In general, the main results of the paper do not arrive with a surprise, but well-organized and tailored towards trendy models.

Besides, the paper provides the details of all the notations used, such as the ones in the appendices. This helps to clarify and ease the understanding of the theoretical analysis.

### Weaknesses
The result presented in Theorem 1 is not a surprise since it resembles the well-known bias-variance decomposition for the mean square error. This also limits the scope and impact of the study because it focuses on the mean square error, which has a nice decomposition property.

In addition, the theoretical results are associated with some strong assumptions on the boundedness of data generation process (e.g., task functions and task data). Nevertheless, this should not be considered as a severe weakness for a theoretical study.

### Questions
The analysis still follows the conventional setting, but rebrand certain things. For example, the prompt used is assumed to be independent data points, which may not mimic the reality where prompt should be time-dependent (order matter). I am aware that the paper makes some assumptions and hence, the uniform attention transforms, but could the authors elaborate further about this matter as well as how close it is to practical settings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the generalization properties of ICL in transformers under a multi-task setting, aiming to provide a theoretical explanation for why ICL performs effectively across multiple tasks. The authors derive generalization bounds that depend on the context length, number of tasks, and feature subspace dimension. Their theoretical analysis offers a comprehensive error decomposition of ICL performance, identifying and characterizing several key sources of error: the Bayesian gap (related to feature dimension and in-context sample size), the task identification error, and the input distribution shift error during inference. These results provide insight into how different factors contribute to the overall generalization behavior of ICL.

### Strengths
Given the widespread success of LLM prompting, characterizing ICL performance is both highly challenging and important. This paper provides a comprehensive theoretical characterization of ICL in a multi-task setting, contributing to a deeper understanding of its generalization behavior.

The writing is clear and well-organized, making the theoretical ideas and results accessible to readers.

### Weaknesses
The analysis is limited to Transformers with uniform attention, which restricts the applicability of the results to more realistic architectures used in practice. Given that the uniform-attention setup is relatively simple and computationally tractable, it would be reasonable to expect some numerical experiments to validate the theoretical claims. However, the paper provides no empirical results, weakening its overall contribution and credibility.

### Questions
Prior works that relate ICL performance to Bayesian analysis often rely on strong assumptions (e.g., [1], [2]). In contrast, this paper’s Assumptions 1 and 2 appear to be relatively mild. Could the authors elaborate on why fewer assumptions are required in this work? Specifically, is this simplification primarily due to the uniform-attention Transformer architecture, or are there technical insights or innovations that allow the analysis to proceed under weaker conditions? Clarifying this point would help readers better understand the novelty and robustness of the theoretical framework.

[1] Xie, Sang Michael, et al. "An Explanation of In-context Learning as Implicit Bayesian Inference." International Conference on Learning Representations.

[2] Zhang, Yufeng, et al. "What and How does In-Context Learning Learn? Bayesian Model Averaging, Parameterization, and Generalization." The 28th International Conference on Artificial Intelligence and Statistics.

### Soundness
3

### Presentation
3

### Contribution
3
