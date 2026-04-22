# How Does the Pretraining Distribution Shape In-Context Learning? Task Selection, Robustness and Generalization

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
The emergence of in-context learning (ICL) in large language models (LLMs) remains poorly understood despite its consistent effectiveness, enabling models to adapt to new tasks from only a handful of examples. To clarify and improve these capabilities, we characterize how the statistical properties of the pretraining distribution (e.g., tail behavior, coverage) shape ICL on numerical tasks. We develop a theoretical framework that unifies task selection and generalization, extending and sharpening earlier results, and show how distributional properties govern sample efficiency, task retrieval, and robustness.
To this end, we generalize Bayesian posterior consistency and concentration results to heavy-tailed priors and dependent sequences, better reflecting the structure of LLM pretraining data. We then empirically study how ICL performance varies with the pretraining distribution on challenging tasks such as stochastic differential equations and stochastic processes with memory. Together, these findings suggest that controlling key statistical properties of the pretraining distribution is essential for building ICL-capable and reliable LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors consider a Bayesian analysis of ICL, generalizing this analysis specifically to situations where the pre-training distribution is heavy-tailed. This work is inspired by earlier work showing that the pre-training distribution can have a significant influence on whether a network acquires ICL abilities during pre-training. The results are derived as concentration bounds on the ICL error given a certain pre-training distribution. 

There are two key results: (1) A bound on the Renyi divergence between the true and inferred task parameters as a function of the number of in-context samples (Theorem 1), (2)  A bound on the generalization error as a function of the number of tasks seen during pre-training and the number of in-context samples.

### Strengths
The technical arguments used in the Appendix are out of my domain and I cannot comment on the correctness of the mathematical proofs. With this caveat, the theorems as stated provide quite general bounds on the task selection and generalization errors.

### Weaknesses
As noted above, I cannot comment on the correctness of the proofs, and will instead comment on the significance of the results and the connection to experiments. With respect to these dimensions, the paper suffers from two major drawbacks:

(1) The authors make two implicit assumptions. The first is that models converge to their Bayes' optimal predictors under the pre-training distribution. The second is that the concentration bounds they derive, though quite general, are sufficiently tight that they make relevant predictions in practice. These are both quite strong assumptions, and I fail to see how these are justified. 

As a result, I find it difficult to connect the takeaways to the stated Theorems. For example, Theorem 1 is a general concentration result on how quickly the data distribution given a \theta^* concentrates as a function of the number of in-context examples. Theorem 1 is stated in terms of the Renyi divergence between the data distributions given \theta^* and a \theta drawn from the pre-training distribution. The takeaway that heavier tails are thus beneficial for task identification seems like a logical leap -- there are several assumptions made to make this connection which I fail to see. What if a model in practice does not perform implicitly perform task selection, or is not a Bayes' optimal predictor? Is it appropriate to compare across different prior distributions if the task selection error metric as defined itself depends on the prior? The prior enters as \log \pi(\theta^*) which suggests that the effect of the heavy-tail has only a weak logarithmic dependence. Given that the result is a concentration bound that typically ignores pre-factors of order 1, how seriously should one take this logarithmic factor? The argument that leads to takeaway #2 also makes a similar logical leap.

(2) The authors claim that the empirical results are consistent with their theoretical results. I am not sure how to interpret the empirical results. For example, in Figure 1, the authors compare the ICL error as a function of the shape parameters of the pre-training distribution. But the metric for ICL error itself changes as the ICL tasks are drawn from a shifted distribution relative to their pre-training distribution. That is, the different ICL error curves are measured on different test distributions. If this is indeed the case, I fail to see how one could compare the ICL error across different values of \nu or \beta. 

I also find the comparisons made across different numbers of pre-training tasks (n) rather puzzling. For example, the three plots in Figure 2 seem to show that ICL error does not change significantly as a function of n. The authors claim that, for smaller n, the lighter tail prior performs better is in my opinion unsupported by the empirical results -- it is not possible to compare curves that are this close to each other without averaging over the many stochastic factors that go into training a model, and sampling a particular set of pre-training tasks. These comments also apply to the results presented in the other figures, and Figure 5 in particular.

### Questions
Please see the Weaknesses section above.

### Soundness
3

### Presentation
3

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
The paper develops a framework for ICL, with a focus on pre-training distribution.
Based on the framework, the paper presents a theorem for task selection, demonstrating that heavy-tailed priors are beneficial for task identification.
The paper then provides Theorem 2, which shows that heavy-tailed priors are harmful to generalization error.
These two together reveal a trade-off.
The paper further couples the theoretical analysis with experiments.

### Strengths
(1) The paper develops a framework making it possible to analyze the relationship between heavy-tail prior and task selection and generalization error. The analysis of the relationship is new to me from this perspective.

(2) The theoretical analyses are further supported by experiments.

### Weaknesses
(1) It is not clear to me how the takeaway #1 comes from. How is the heaviness of tail defined? How do I connect the formulation in Theorem 1 to this takeaway? What is the mathematical intuition behind the takeaway #1 (e.g., the mathematical intuition of a heavier tail improving learning speed)?

(2) While I assume that we want to bound the risk in Theorem 1, the LHS of Theorem 1 is not the expectation of error. Any rationale behind this choice? (Theorem 2 in the paper does consider the error)

(3) The mathematical intuition behind Takeaway #2 is missing.

### Questions
(1) I am not familiar with the description on page 4 L198 "have controlled tails, at most poly(T)" and "$\pi$ admits a second moment". Can the author translate it into mathematical formulations?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a theoretical framework to unify task selection vs. generalization perspectives on ICL and then studies, within this framework, how statistics of data-distribution affect in-context convergence rates. The specific generalization here w.r.t. past work is that of exploring heavy-tailed priors.

### Strengths
The paper is well written (though it ignores quite a bit of relevant related work) and digs into better assumptions for modeling the data distribution.

### Weaknesses
- Relation of Bayesian framework to prior work: My main apprehension with the paper relates to how the paper is contextualized with respect to prior work. For example, the Bayesian interpretation, defining a mixture of tasks, and sampling from them with a prior is a common formalism at this point [1, 2, 3, 4], with [1] very actively stating it. 

- Relation of convergence rates to prior work: Another aspect I'm confused by is the proven rates versus the shown / proven rates in prior work. For example, [1] shows the convergence occurs in a power-law manner with context, while [4] provides rigorous results to this effect in the linear regression setting. These rates can be confirmed to be correct for large scale models as well [5], including for the kind of generative processes studied in this paper (like SDEs). 

- The claim on line 352--354 is somewhat cherry-picked in my opinion. While it is correct that context length 16--64 requires order of 1K tasks in linear regression, this is very much not the case for several other tasks for studying ICL (see [1, 3]). I would strongly recommend this claim be toned down. 

[1] https://arxiv.org/abs/2506.17859

[2] https://pmc.ncbi.nlm.nih.gov/articles/PMC11661294/

[3] https://arxiv.org/abs/2412.01003

[4] https://www.pnas.org/doi/abs/10.1073/pnas.2502599122

[5] https://arxiv.org/abs/2402.00795

### Questions
It would be helpful if authors can contextualize their contributions with respect to prior work listed in weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2
