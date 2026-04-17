# Learning Interpretable Models Using Uncertainty Oracles

- Decision: Reject
- Scores: 6, 4, 0

## Abstract
A desirable property of interpretable models is small size, so that they are easily understandable by humans. This leads to the following challenges: (a) small sizes typically lead to diminished accuracy, and, (b) different techniques offer bespoke levers, e.g., L1 regularization, for making this size-accuracy trade-off that might be insufficient to reach the desired balance.

We address these challenges here. Earlier work has shown that learning the training distribution creates accurate small models. Our contribution is a new technique that exploits this idea. The training distribution is modeled as a Dirichlet Process for flexibility in representation. Its parameters are learned using Bayesian Optimization; a design choice that makes the technique applicable to non-differentiable loss functions.  To avoid challenges with high data dimensionality,  the data is first projected down to one-dimension using  uncertainty scores of a separate probabilistic model, that we refer to as the uncertainty oracle.

Based on exhaustive experiments we show that this technique possesses multiple merits: (1) it significantly enhances small model accuracies, (2) is versatile: it may be applied to different model families with varying notions of size, e.g., depth of a decision tree, non-zero coefficients in a linear model, simultaneously the  maximum depth of a tree and number of trees in Gradient Boosted Models, (3) is practically convenient because it needs only one hyperparameter to be set and works with non-differentiable losses,    (4) works across different feature spaces between the uncertainty oracle and the interpretable model, e.g., a Gated Recurrent Unit trained using character sequences may be used as an oracle for a Decision Tree that uses character n-grams, and, (5) may augment the accuracies of fairly old techniques to be competitive with recent task-specialized techniques, e.g., CART Decision Tree (1984) vs Iterative Mistake Minimization (2020), on the task of cluster explanation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a framework for constructing small, interpretable models that achieve high accuracy by leveraging the training distribution through a flexible probabilistic modeling approach. The core idea is to use a Dirichlet Process (DP) to model the training data distribution, enabling adaptive and expressive representation learning that supports small model sizes without sacrificing performance. The DP’s parameters are optimized via Bayesian Optimization, a design choice that allows the method to handle non-differentiable loss functions, which is critical for many real-world interpretability tasks. To manage high-dimensional input spaces, the method employs data projection based on uncertainty scores from a separate probabilistic model. This oracle can be any well-performing model, and its predictions guide the dimensionality reduction, enabling the framework to work across diverse feature spaces. The authors demonstrate that their approach improves the accuracy of small interpretable models across multiple model families. It is modular and general: it can be applied to various model types, different feature representations, and even enhance older, classic algorithms to perform competitively with modern, task-specific methods on tasks such as cluster explanation.

### Strengths
1. The proposed method achieves improvements in accuracy for small, interpretable models addressing a core challenge in the field: the severe accuracy penalty incurred when enforcing model simplicity for human interpretability.
2. The technique is not tied to a specific model class or size metric. It can be applied to a wide range of interpretable models and supports diverse notions of "size," including depth, number of non-zero coefficients, and combined constraints. This versatility significantly broadens its applicability.
3. The method requires only one hyperparameter, the optimization budget, which is a major practical advantage over competing approaches that require tuning multiple, complex hyperparameters. This simplicity enhances usability and reproducibility.
4. By leveraging Bayesian Optimization to tune the Dirichlet Process parameters, the method is inherently suitable for non-differentiable loss functions, a critical feature that enables application to discrete, combinatorial model spaces. This is a notable advantage over gradient-based optimization methods, which fail in such settings.
5. The framework permits the use of an "uncertainty oracle" to guide the training of an interpretable model. This decoupling of the oracle and model feature spaces enables powerful transfer of uncertainty information across heterogeneous representations, enhancing generalization and flexibility.
6. Experiments cover both binary/multi‑class classification, and compare against multiple baselines.

### Weaknesses
1. The reliance on Bayesian Optimization leads to prohibitively long training times. While the authors acknowledge this and suggest alternative optimization strategies, the current implementation remains computationally expensive, limiting scalability to large datasets or real-time applications.
2. Although the intuition behind learning the training distribution is plausible, the paper offers no theoretical analysis to explain why this approach yields such large improvements in accuracy. The absence of a formal argument weakens the manuscript and makes it difficult to predict under what conditions the method may fail.
3. The method’s performance hinges on the quality of the “uncertainty oracle.” However, the paper neither examines how different oracle choices influence results nor quantifies the model’s sensitivity to oracle uncertainty estimates. Potential error propagation from a faulty oracle is unaddressed. Moreover, if the oracle is miscalibrated or exhibits systematic bias, the guided model may inherit these issues. The paper also omits an analysis of robustness to oracle noise.
4. The oracle must be trained beforehand on the full dataset, which can be expensive for large‑scale problems. The paper does not quantify the total runtime overhead relative to training a standalone tree.
5. The optimization budget (3000 iterations) is relatively high for a single depth level. The paper does not discuss the sensitivity to this budget.
6. It would also be valuable to assess the technique on more complex real-world applications beyond the benchmark datasets to better demonstrate the practical applicability and performance of the proposed method.

### Questions
Please refer to the section on weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a two-stage framework: first, an "uncertainty oracle" scores samples, and a Dirichlet Process-driven Infinite Beta Mixture Model is used to learn a target sampling distribution on the uncertainty axis, thereby resampling the training set; subsequently, a small interpretable model is trained on the resampled data. The goal is to improve the interpretability and generalization performance of the final model through "explanation-oriented and robust" data selection.

### Strengths
The method is model-agnostic and applicable to various interpretable models.

### Weaknesses
1. While Step 2 primarily uses small models like decision trees and linear models, which is acceptable given the focus on surrogate models for interpretability, the datasets used in the experiments are also very small, raising doubts about the method's practical utility.

2. Furthermore, the runtime issue highlighted by the authors in the limitations section seems critical. With runtimes already approaching an hour even for small datasets, the time sacrifice may not be justified by the performance gains achieved.

### Questions
1. Since the oracle and the interpretable model are decoupled, their decision boundaries may not align. How can it be ensured that the decision boundary of the model in Step 1 is the same as or congruent with that in Step 2?


2. The framework's effectiveness heavily depends on whether the oracle's uncertainty scores are truly calibrated and reflect the "difficulty relative to the decision boundary." The paper lacks experimental validation of the uncertainty estimation.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
In this paper, a training technique is proposed  to improve small interpretable models accuracy by learning training distribution of uncertainty scores from an oracle model. They iteratively train a Dirichlet Process based mixture model to represent the distribution over 1D-projected uncertainty scores through Bayesian Optimization and target model F1 score. The method is designed to be practically versatile, handling non-differentiable loss functions, cross-feature-space and multivariate model sizes. With thorough empirical experiment, the method demonstrates sound and board improvements across 13 standard classification datasets on a variety of interpretable model families, clearly highlighted with Wilcoxon signed-rank tests.

### Strengths
None

### Weaknesses
The contribution is unclear due to lack of theory grounding, computational costs disadvantage, less technical novelty and issues on methodology.
- Lack of theory grounding - The paper covers very little theories, explanation and interpretations of experiment findings. There are no clear discussions or theoretical justifications on why uncertainty-based sampling improves small models. Formal analysis of convergence properties is also missed in the paper. 
- Unfavorable computational cost and unstudied complexity - The high iterations and runtime overhead for small model training defeats the purpose for practical interpretable ML applications. In  addition, runtime comparisons with standard interpretable ML training could be helpful to understand the limitations of this technique.
- Less technical novelty - The core methodology is a specific combination of well-understood and standard techniques, lacking fundamental innovations.
- Methodological Issues - The method has strong oracle dependence, assuming oracle quality is well constructed and calibrated. There is no discussion on robustness to oracle degradation or choices. Secondly, although the paper provides a suggested default reasonable settings, the "only one hyperparameter" is not factual. A variety of hyperparameters are involved in methodology, including uncertainty metric choice, smoothing parameters, box constraints and oracle architecture.

### Questions
Interpretation on negative improvements: Is there any detailed analysis and interpretation of negative improvements in Table 4?

### Soundness
2

### Presentation
2

### Contribution
1
