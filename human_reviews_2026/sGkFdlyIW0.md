# ICFI: A Feature Importance Measure For Multi-Class Classification

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Feature importance is one of the most prominent methods in eXplainable Artificial Intelligence (XAI). It aims to assess the extent to which a machine learning model relies on different features. However, in multi-class classification, current methods fail to explain inter-class relationships, either because they provide explanations for binary classification only, or because they suffer from aggregation bias. To address these shortcomings, we propose Inter-Class Feature Importance (ICFI), which provides feature importance scores for discriminating between an arbitrary pair of classes. ICFI is a post-hoc, model-agnostic method, which provides bounded scores for interpretability. We empirically demonstrate through extensive experiments on real-world datasets that ICFI effectively captures the discriminating features between class pairs, outperforming existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces Inter‑Class Feature Importance (ICFI), a model‑agnostic, post‑hoc method for multi-class classification that quantifies how each feature helps discriminate a specific pair of classes. ICFI follows a removal‑based principle and uses feature permutation as a measure for feature importance: for a given feature, the method compares changes in empirical error between when the feature is permuted and when the feature is permuted and two target classes are merged. A feature is deemed important when its permutation increases the error of separating the selected classes. Empirically, the method identifies discriminative features for chosen class pairs and shows advantages over baselines.

### Strengths
* The problem of pairwise FI in multiclass classification is clearly defined, an perspective missing from many existing FI tools.
* The method is model‑agnostic and post‑hoc: applicable to arbitrary black‑box classifiers without retraining the base model.
* The proposed measure is intuitive, simple and meet the requirements for an effective FI method. 
* Empirical evidence is presented to demonstrate that ICFI‑derived features are more meaningful and accurate than baselines method.

### Weaknesses
1. **Novelty**: The claim that multi-class explanations are not addressed elsewhere is too strong. For example, Vo et al. [1] present a model‑agnostic, post‑hoc multiclass FI method. The authors are encouraged to further reflect the differences between the two works.  

2.  **Computational cost**: Computing importance requires multiple model evaluations per feature and per class pair, which may be burdensome for large deep models.

3. **Extension to other modalities and large-scale classifiers**: It is unclear how the approach extends to high‑dimensional, less interpretable modalities (e.g., images or videos) as well as other classification tasks such as questions answering where the loss function is not simply cross-entropy loss. In this case, the measure in Def. 2 may no longer satisfy the requirements.  

[1] Vo et al. An Additive Instance-Wise Approach to Multi-class Model Interpretation (ICLR’23)

### Questions
1. Why is permutation used to represent feature removal, instead of common approaches like feature dropping, zero/mean substitution? How sensitive are results to different removal strategies?

2. For images or texts, permutation may break semantics of the input content. Could the authors comment on this issue and how it affects the quality of output features from the proposed method?

3. How would ICFI be adapted for images/videos or pairs of modalities e.g., text-image pairs or text-video pairs? In today's AI landscape, I think it is important to reflect on how the proposed framework would be used for these modern black-box models.  

4. How stable are pairwise importance scores across different random seeds, sample sizes, and class imbalance conditions?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a novel feature importance method for multi-class classification. The method tackles the problem of providing not only one set of feature importances, but one for each pair of classes, thus offering more insight into the classifier. The method is based on the idea of observing decrease in empirical risk when two classes are combined, in combination with permutation-based feature importance. Some experiments are provided showing that the method gives sensible results and outperforms GSHAP adapted to the same problem.

### Strengths
Overall, the paper is well-written and most of the ideas are clearly communicated and easy to understand. Feature importance is a very saturated field, but this work tackles a novel subproblem. I like the basic idea of combining classes.

### Weaknesses
The proposed method is relatively simple and does not bring any extremely innovative methodology or theoretical results, which is nothing wrong by itself, but the I would expect a very strong empirical evaluation or (even better) a practical use-case that demonstrates not only that the method works but that the problem of requiring additional insights into (pairwise) relationships between classes is really a problem in need of a solution.

The current experiments do not convince me (see Questions).  As the authors also say, evaluation of XAI is a big challenge and there doesn't seem to be any shortcut to a sound empirical evaluation (https://icml.cc/virtual/2025/poster/40169). The first two experiments establish that there is nothing clearly wrong with the method, which is OK. The retraining experiment and comparison with GSHAP I do not understand. If the goal of the method is to provide insights into how the model classifies, then this is far from a realistic assessment (yes, it is common to do this in XAI/ML papers, but it doesn't make it any less unrealistic). Also, it seems to me that GSHAP was forced into this comparison, not being a method developed for the same purpose. I might be wrong, but the paper doesn't do a good job of describing exactly what GSHAP is or how it was adapted.

And I might have other issues with the paper on things that I currently don't quite understand and/or were not explained clearly enough (also see Questions).

Minor comments:
- Some extra effort seems to have gone into squeezing this to fit the page limit (Figure 6 caption has no space to breathe, etc.).
- ).One
- The proposed method operates on model risk not on model predictions directly. So, technically, it is not explaining what the model does, but what features contribute to the models predictive performance. Often the same, but not always.

### Questions
Q1:  Finally, why not include some global feature importance into the comparison? The problem of masking the feature importance of globally less important features that are important for certain pairs of classes might be exaggerated. I'd imagine that for a low number of classes the global ordering would be decent (definitely better than random). 

Q2. Computational complexity: First, it would really help if the computational complexity is stated more explicitly, instead of "in line with existing permutation methods but cheaper than SHAP". Second, I'm not convinced that the latter is correct. The proposed method requires for each feature a constant number of permutations and each permutation requires a model prediction? Any decent implementation of SHAP should also be linear in the number of features and will contain the model prediction (you don't go through all subsets of coalitions).

Q3. Permutation importance has certain failure cases, compared to SHAP, for example. Why not combine the idea of combining two classes but then use Shapley values instead of permutation importance? 

Q4.  I'd remove the explicit "Definition 1" from definition of the pairwise feature importance problem. It is not necessary and it is not precise. Informally we would probably agree on what "as it pertains to separating the target classes $\sigma$ and $\rho$" means, but what does it really mean? A model never trully 100% focuses on separating only two classes (unless there are only two classes).

Q5. The interval computation in A.3 seems like overkill. The (Bayesian posterior) mean and standard deviation of a process where 100 independent samples are given is estimated using Markov Chain Monte Carlo? Unless I'm missing something, the only possible justification would be that we use uniform priors on the two parameters and therefore can't use the analytical solution. But if we are going to be so precise as to not allow values outside of [0,1] then why use a Gaussian likelihood, which is clearly not appropriate. Burn-in also doesn't make sense (why not just pick a sensible starting value, like the empirical mean and standard deviation). To summarize, average +/- 1.96 * standard deviation of the sample / sqrt(100) should give essentially the same results.

Q6. I'm unsure about the upper/lower bound requirement. First, the requirements, as stated, would allow for a method that assigns arbitrarily low negative feature importances (we require irrelevant features to have 0 and to have an upper bound; there is nothing saying that a relevant feature can't have a negative importance, for example, if it decreases predictive performance). I'll assume that the intention was for them to be bounded between 0 and an upper bound (which might as well be 1). I'm not convinced by the argument that people prefer bounded things therefore bounding is better. That is, it is mathematically easy to bound things, but with it we change the scale of the feature importance. Are these importances even comparable across class pairs for same risk? Are they comparable across different risks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose ICFI, a global feature importance measure centred on multi-class classification. The proposed approach is a model-agnostic, post-hoc approach that computes feature importance by considering the “pairwise feature importance” problem, this is achieved by evaluating how a feature contributes to the separation of classes $\sigma$ and $\rho$. ICFI offers an elegant and computationally efficient means of assessing feature importance.

### Strengths
•	The related works / references are quite thorough.
•	The idea is interesting and conceptually simple.
•	The experiments favour ICFI.
•	It is to my understanding that this can be applied to any method and does not require any separate model or model retraining.

### Weaknesses
•	Whilst interesting conceptually, the main contribution seems to be merging classes – and the FI measure, while the measure is intuitive, it lacks in axiomatic grounding.
•	I think computational runtime is a strength when compared to GSHAP (SHAP in general), a clear table of runtime – which is not presented as a discussion would be ideal.
•	The comparison is quite limited. There is no benchmarking against other methods such as Expected Gradients (EG), Integrated Gradients (IG), Manifold IG, Layer-wise Relevance Propagation (LRP), SmoothGrad, and other methods which have seen wide application since SHAP for neural network-based experimentation. Libraries such as Captum would help provide a broader set of benchmarks. As well as global feature attribution methods such as SAGE [1].
•	Considering the above, it is also notable that GSHAP is an arXiv paper.  

[1] Ian C. Covert, Scott Lundberg, and Su-In Lee. 2020. Understanding global feature contributions with additive importance measures. In Proceedings of the 34th International Conference on Neural Information Processing Systems (NIPS '20). Curran Associates Inc., Red Hook, NY, USA, Article 1444, 17212–17223.

### Questions
•	Is there a benefit to the proposed permutation approach as opposed to sampling from a background/reference dataset? –  this strategy employed in methods such as EG (an extension of IG).
•	The manuscript proves boundedness and (more centrally) symmetry but does not analyse standard attribution axioms (e.g., completeness/efficiency, monotonicity, linearity). Please (a) state explicitly which axioms ICFI satisfies or violates, (b) discuss practical implications of any violations, and (c) motivate why ICFI’s objective (global pairwise discriminative power) warrants these trade-offs. It is worth considering if there exists a global FI analogue. (I refer the authors to [1])
•	If computational or conceptual mismatch prevents some suggested baselines, please can the authors explain?

### Soundness
2

### Presentation
2

### Contribution
2
