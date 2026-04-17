# Unifying Formal Explanations: A Complexity-Theoretic Perspective

- Decision: Accept (Poster)
- Scores: 4, 8, 6

## Abstract
Previous work has explored the computational complexity of deriving two fundamental types of explanations for ML model predictions: (1) *sufficient reasons*, which are subsets of input features that, when fixed, determine a prediction, and (2) *contrastive reasons*, which are subsets of input features that, when modified, alter a prediction. Prior studies have examined these explanations in different contexts, such as non-probabilistic versus probabilistic frameworks and local versus global settings. In this study, we introduce a unified framework for analyzing these explanations, demonstrating that they can all be characterized through the minimization of a unified probabilistic value function. We then prove that the complexity of these computations is influenced by three key properties of the value function: (1) *monotonicity*, (2) *submodularity*, and (3) *supermodularity* - which are three fundamental properties in *combinatorial optimization*. Our findings uncover some counterintuitive results regarding the nature of these properties within the explanation settings examined. For instance, although the *local* value functions do not exhibit monotonicity or submodularity/supermodularity whatsoever, we demonstrate that the *global* value functions do possess these properties. This distinction enables us to prove a series of novel polynomial-time results for computing various explanations with provable guarantees in the global explainability setting, across a range of ML models that span the interpretability spectrum, such as neural networks, decision trees, and tree ensembles. In contrast, we show that even highly simplified versions of these explanations become NP-hard to compute in the corresponding local explainability setting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents new findings in the field of formal explainable artificial intelligence (XAI), focusing on the complexity of sufficient and contrastive explanations, including their probabilistic variants. The authors introduce a unified framework that covers both local and global explanations, providing insights into the complexity and approximability of subset-minimal and cardinality-minimal explanations.

The key takeaway from this study is that global probabilistic explanation problems differ significantly from local ones in terms of computational complexity and approximability. Specifically, the objective function for both sufficient and contrastive global probabilistic explanations is monotone increasing (Proposition 2). Additionally, when dealing with joint distributions, the objective function exhibits supermodularity for sufficient global probabilistic explanations (Proposition 3) and submodularity for contrastive global probabilistic explanations (Proposition 4). In contrast, local probabilistic explanations lack these properties.

The authors also focus on empirical distributions, where the objective function can be evaluated in polynomial time for various classifiers. They combine the aforementioned properties with standard greedy algorithms to reveal new results. Notably, subset-minimal global explanations can be computed efficiently for empirical distributions (Theorem 2), and cardinality-minimal global explanations are approximable (up to curvature and a logarithmic term) for empirical joint distributions (Theorems 3 and 4).

### Strengths
**S1.** Despite the paper's dense and technical nature, it is well-articulated: the notation is clear, and the results are easy to understand.
 
**S2.** The related work is well-detailed, including most relevant references. 

**S3.** The unified framework that encompasses both local and global explanations, including probabilistic approaches, is intuitive. 

**S4.** To my knowledge, Propositions 2, 3, and 4 appear to be novel contributions. In most cases, the negative results are not straightforward.

### Weaknesses
**W1.** I find Algorithm 2 confusing, as it seems incorrect; the marginal gain should be maximized. 

**W2.** I think that the positive results presented in the main theorems (1-4) are primarily applications of existing findings. 

**W3.** Theorem 4 does not truly deliver a constant-factor approximation, as the curvature could be unbounded. 

**W4.** The upper bounds for Theorems 3 and 4 appear somewhat loose, and these approximation results lack lower bounds necessary to establish tightness. 

**W5.** Finally, the practicality of Theorems 3 and 4 is questionable, given that empirical “joint” distributions are rarely available in real-world scenarios.

### Questions
------------------------------
### Major Comments:
------------------------------

**C1**. As mentioned earlier, I found Algorithm 2 to be confusing. Typically, a greedy algorithm aims to maximize the marginal gain at each step, which is especially true for the greedy set-cover method. Therefore, unless I missed something, I believe that in Line 3, the `argmin` function should be replaced with an `argmax` function. Consequently, the comment in Line 428 should also be revised accordingly.

**C2**. Essentially, the approximation factor for Theorem 3 consists of a logarithmic term (related to the number of data instances), while the approximation factor for Theorem 4 includes both a logarithmic term and a curvature term. Can we bound the curvature term? If not, the fourth contribution mentioned in the introduction (Lines 120-126) should be rephrased to reflect this.

**C3**. In light of the previous comment, could the authors provide tight lower bounds for Theorems 3 and 4? Specifically, for large datasets, a bound of the form $\log |D|$ in Theorem 3 seems quite loose. However, if the authors demonstrate that approximating the problem within a factor of $(1 - \epsilon) \ln |D|$ is NP-hard, this would strengthen the result. A similar observation applies to Theorem 4, given that the curvature can be significant.

**C4**. In Theorems 2-4, the assumption regarding “empirical” distributions is reasonable; otherwise, the problem could be PP-hard. However, combining this assumption with the condition that $\mathcal D$ is also a “joint” distribution appears unrealistic in practice. Do the authors envision practical scenarios where we need to globally explain a model trained on a dataset with samples drawn from a joint distribution? At present, I find this result to be primarily of theoretical interest, as I have not identified concrete applications for it.

-----------------
### Minor Comments:
------------------

**C5**. In Section 2 (Setting), I suggest introducing the main notation, including the input dimension ($n$), the number of classes ($c$), the underlying distribution ($\mathcal D$), and the training set ($\mathbf D$ or $\mathbf Z$). Additionally, I would like to ask why you are using the subscript $p$ in $\mathcal D_p$; I would recommend simply using $\mathcal D$. Throughout the rest of the paper, I suggest maintaining consistent notation (for example, choose either P or PTIME, but not both).

**C6**. Based on the proof provided, I believe that Theorem 1 applies not only to decision trees but also to orthogonal DNF formulas (i.e., "1-satisfy" DNF formulas). Given that there exists a Fully Polynomial Randomized Approximation Scheme for counting models of arbitrary DNF formulas, I am also curious whether we could find an approximation result for identifying minimal subset global explanations under the uniform distribution.

**C7**. Section 8 is acceptable, but it could be more informative. I recommend including some key open questions that arise from this theoretical study. For instance, can we establish better approximation bounds for minimal-size global explanations when the classifier is a simple function, such as a decision tree or a linear threshold function?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a framework for unifying two types of explanations: contrastive and sufficient reason explanations. The framework models both explanations as minimisations of a value function.

### Strengths
- Clear research question and contribution
- A lot of novel (formal) insights are generated, which I consider to be valuable for the XAI community.

### Weaknesses
- Readability and accessibility can be improved. Most importantly, it looks to me that sufficient reasons are the same as semi-factual explanation, and global contrastive reasoning seems to describe goup/multi-instance counterfactuals. Since semi and counterfactuals are popular and widely used terms in the XAI community, I suggest clearly relating them to the concepts introduced and discussed in this paper. By this, the paper and its contribution will become accessible to a wider audience.

Minor:
- Line 232 "smaller" in XAI people often talk about "simpler explanation". I suggest clarifying the meaning of "smaller" and also including "simpler" to make the paper more accessible to other researchers

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a unified framework for explanation complexity. Global value functions are monotone; under feature independence, sufficient variants are supermodular and contrastive variants submodular. This yeilds polynomial-time algorithms for subset-minimal global explanations and approximations for cardinality-minimal ones. Local explanations remain NP-hard.

### Strengths
The global versus local structural distinction appears to be new. Monotonicity holds without independence. Proofs are rigorous with counterexamples showing the necessity of assumptions.

### Weaknesses
Feature independence is required for approximation results. This limits practical applicability where features correlate.

### Questions
Could you clarify the candidate set specification in Algorithm 2?

### Soundness
3

### Presentation
3

### Contribution
3
