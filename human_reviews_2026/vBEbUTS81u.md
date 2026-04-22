# Explainable Mixture Models through Differentiable Rule Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Mixture models excel at decomposing complex, multi-modal distributions into simpler probabilistic components, but provide no insight into the conditions under which these components arise. We introduce explainable mixture models (XMM), a framework that pairs each mixture component with a human-interpretable rule over descriptive features. This enables mixtures that are not only statistically expressive but also transparently grounded in the underlying data. We formalize the problem and examine conditions under which an XMM exactly captures a target distribution. We then propose a scalable, differentiable learning procedure for discovering sets of rules. Experiments on synthetic and real-world datasets demonstrate that our method discovers interesting sub-populations in both univariate and multivariate settings,  offering interpretable insights into the structure of complex distributions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
They propose a method for explainable mixture models. The explanations created are a conjunctions of different features. The mixture model created can fit the model well and able to generate logical rules to explain the mixtures created.

### Strengths
* Well written paper.
* Good exploration with synthetic dataset to showcase the capability of the models.
* Approach also fairly robust based on the synthetic dataset noise.

### Weaknesses
* The motivation of the problem with application can be improved.
* Many variants of their method is being introduced and it is a bit unclear on what the conclusion is between them. While they say that over NLL on real dataset is best for EMM-GMM, i am not sure how accurate it would be to claim that to be the best, as different datasets yielded different best model.
* Some human evaluation for some real-world dataset would have been good to evaluate the goodness of the explainability.

NOTE: i did not read the appendix section and my comments are purely based on main text.

### Questions
* Are the definitions in section 3 part of contribution or is it background?? It is a bit unclear. if it is background, then it is good to add ref. for them.
* What kind of noise is added in the robustness experiments and how much??
* In synthetic dataset will the performance change if the data is generated from a random distribution rather than a gaussian and uniform?
* Given the rules are logical statements can the features be continuous or do they have to be discrete/categorical for the approach to work?

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
This paper maximizes the conditional likelihood, where the conditional likelihood is parameterized using a mixture model, and adds a regularizer to the objective to encourage each data point to follow a unique path within the mixture. The authors design the mixture model as a rule function, providing explainability, and learnt by gradient descent.

### Strengths
The paper introduces a novel objective function that maximizes conditional density estimation in mixture-models.

### Weaknesses
How is this different from Mixture of Experts (MoE)?
MoEs use a gating function to decide which expert/block to activate, and are typically trained under a maximum likelihood objective for generative modeling. I see EMMs as a special case of MoEs where the gating logic is determined by learned rules. There are many possible gating mechanisms that can yield explainability, for example by restricting the gating function to be simple. Calling these models “EMMs” and not acknowledging them as a special case of MoEs is, in my view, misleading.

Definitions are not complete
How do you define “human-interpretable explanation”? This is subjective. You need to either define it explicitly or give application-specific criteria before introducing Definition 1 (Marginal-EMM). Otherwise, you cannot really call it a “definition.”

Also, from Definition 1, any tree-based method would qualify as an EMM if you take the path to the node as, $e_i$. Does that mean all tree-based models are EMMs?

Comparison to [1]
This paper modifies the objective proposed in [1] for conditional density estimation. Can you provide a direct comparison with [1]? The only clear difference I see is that your objective removes the entropy term. But [1] can still be used to compute a conditional density. How does the current method compare to [1] in terms of (a) explainability and (b) test-set likelihood?

I also disagree with the claim that tree-based approaches are “prone to overfitting.” Any machine learning method, deep learning models, are prone to overfitting.

Finally, in Eq. (6), since $w_i$ is constrained to be positive, why do you need to square it?

[1] Sascha Xu, Nils Philipp Walter, Janis Kalofolias, and Jilles Vreeken. “Learning Exceptional Subgroups by End-to-End Maximizing KL-Divergence.” ICML, 2024.

### Questions
Can you perform an ablation on $\lambda$, and also explain how increasing the number of mixtures affects the conditional likelihood?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes EMM, explainable mixture model, a principled learning procedure that discovers each mixture component with a human-interpretable rule over descriptive features. Experiments on synthetic and real-world datasets demonstrate the effectiveness of the proposed approach.

### Strengths
The proposed approach is based on a principled machine learning procedure.

### Weaknesses
The experiments are restricted to mixture models, and don't reflect the paradigm shift in the era of deep learning. The experiments should include the results that compare with deep learning models such as VAE, and whether the proposed can be extended deep learning models; the data sets are UCI which are too small, don't support the claim that the proposed method is scalable and applicable to more complex, multi-model distributions.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Given a set of data-value pairs $(x_i,y_i)$, where the covariates $x_i \in \mathbb{R}^d$, the paper proposes an explainable mixture model (EMM) approximation of the conditional density $p(y|x)$. The mixture model is conceptually proposed as follows:
- first, the $p(y|x)$ is approximated by a weighted mixture of $k$ components
- each component ideally represented as a box (intersection of intervals) in $\mathbb{R}^d$ or in a lower dimension. The authors show that if the components are supported on a partition of $\mathbb{R}^d$, then there is no error in the approximation
- In order to apply standard optimisation to learn the parameters, each component is approximated by a  smoothed "box" (equation 10)
- the mixture model parameters are learned by minimising a regularised negative log likelihood
- some further pruning is done to reduce the number of rules learned

The authors demonstrate that the proposed EMM model provides good fit of data, while providing fewer learning rules for the explainable model.

The authors show

### Strengths
The overall EMM model uses a combination of simple ideas, but learns an explainable rule-based model that consistently has fewer rules.
The experiments look convincing (with the caveat that I not an expert in this specific problem)

### Weaknesses
The paper's theoretical contribution is limited. A simple result is provided in the case where the supports of the components form a partition. However, there are no guarantees regarding the severity of the approximation error in the general case.
I was tempted to give a low score due to this, but it turns out that this is not a concern in relevant literature (papers like CADET, CDTree).  

I found parts of the papers a bit difficult to follow, which could be because I am not familiar with this line of work (and generally, less familiar with explainability literature).

### Questions
Could the authors explain how to read the explanation plots (for example, right hand side of Fig 7)? What do the bars and colours mean?

### Soundness
3

### Presentation
2

### Contribution
3
