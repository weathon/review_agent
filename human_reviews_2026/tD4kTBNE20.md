# Intervention-based Cumulative Causal Fairness Learning

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Causal inference has emerged as a powerful framework for addressing algorithmic discrimination, offering a principled approach to understand and mitigate unfairness in decision-making systems. Various causality-based fairness notions have been proposed to quantify unfair causal effects stemming from sensitive attributes by leveraging interventions or counterfactuals. Among these, intervention-based fairness has gained prominence as a foundational and widely applicable concept, computable from observational data. However, existing intervention-based fairness notions face critical limitations: (i) they fail to uniquely determine causal effects, and (ii) achieving zero interventional fairness does not guarantee causal fairness. To overcome these drawbacks, we introduce a novel intervention-based fairness metric, the post-Intervention Cumulative Ratio Disparity (ICRD), to rigorously assess causal fairness. Building on this metric, we propose the Intervention-based Cumulative Causal Fairness Learning (ICCFL) framework, which trains causally fair decision models by generating interventional samples and computing differentiable approximations of ICRD. Theoretical analysis and empirical evaluations demonstrate that ICRD provides a robust measure for causal fairness. Extensive experiments on four benchmark datasets demonstrate that ICCFL significantly outperforms six state-of-the-art methods, improving fairness (MMD metric) over 40% on average, and effectively balancing fairness and accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose an approach for assessing and achieving causal fairness, based on a new metric labeled post-Intervention Cumulative Ratio Disparity. Key ideas include:

- Introducing a new metric (ICRD), which instead of comparing only expectations, looks at the distributional distance in terms of the L1 distance of cumulative functions,
- Introducing a computationally tractable relaxation of ICRD using sigmoid functions, which allows differentiable optimization,
- Introducing a result for partial identification of the causal fairness metric under partial causal knowledge (MPDAG).

### Strengths
(S1) The paper deals with an important and timely topic of causal fairness,

(S2) Using a distributional distance (rather than comparing only expectations) is an interesting direction to explore.

### Weaknesses
(W1)  Theoretical contributions are unclear: in current form, the paper does not convey important theoretical results. 

Theorem 1:
For part (1) is the notion of causally independent even defined in the text?
For (2), isn’t the range of ICRD [0, 1] an obvious immediate consequence?
For (3), ICRD is stated to be continuous, _but without even specifying with respect to which variable_ continuity holds? One has to check the appendix for understanding the statement;
Furthermore, the ICRD notation seems abused; in Eq. (4), ICRD is an integral over $\tilde y$; in Eq. (5) and appendix, ICRD takes $\tilde y$ as an argument. 

Theorem 2:
The validity of the sigmoid approximation is a well-known result, so this part does not really add much (it is questionable whether this should be framed as a theorem). Additionally, the inconsistency in notation is very alarming: $n_+, n_-$ used in Eq. (S5), $n_1, n^-$ in Eq. (6), $n_1, n_2$ in Eq. (7). 

Proposition 1: 
The proposition statement is almost identical to the cited previous work of Zuo et. al. and Perkovic et. al. Even the proof in the appendix follows this work almost verbatim.

(W2) Identification concerns not discussed: the result of Proposition 1 establishes identification under a specific assumption (no edge O — V). There is almost no discussion on what happens if this assumption is violated? How common is it to have this assumption satisfied, for any choice of admissible context C? This seems to be a very important point that is not emphasized.

(W3) Role of admissible context not discussed: ICRD definition uses a fixed value C=c; clearly, enforcing this constraint needs to happen for each value of C=c (which could be very many). This important practical challenge is not properly discussed.

(W4) Based on the above, the overall level of rigor in mathematical notation needs to be substantially higher. 

(W5) The writing of the paper is poor. There are a number of typos and errors (e.g., adjectives/nouns for causal/causality and fair/fairness are repeatedly swapped; inconsistent notation, etc.). Also, some of the writing is imprecise: e.g., 

Line 39, “cannot distinguish between discriminatory and spurious correlations” -> this statement is conceptually confused and not grounded in existing literature. The distinction between “discriminatory” and “spurious” correlations is not a recognized or meaningful contrast in causal or fairness analysis. “Discriminatory” refers to normative or causal notions of unfair influence, whereas “spurious” refers to statistical confounding or non-causal association

Line 46, “counterfactual notions require full knowledge of causal model” -> this statement is false, check for instance 
Shpitser, Ilya, and Judea Pearl. "What counterfactuals can be tested." arXiv preprint arXiv:1206.5294 (2012).

Eq. (2) -> path-specific fairness notation not even introduced? (e.g., how to define/interpret $\hat y_{s|\pi, s’|\bar \pi}$).

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies the limitations of existing interventional fairness notions (e.g. K-Fair, Path-Specific Fairness), which fail to capture unfair causal effects of sensitive attributes on outcomes since they only compare average causal effects. The authors propose a new intervention-based fairness notion, post-Intervention-based Cumulative Ratio Disparity (ICRD), which measures distributional causal disparities through post-intervention cumulative prediction probabilities under contest $do(C=c)$. They further establish key theoretical properties and introduce a differentiable approximation to enable optimization. Experiments on both synthetic and real datasets demonstrate that the proposed ICCFL framework achieves better fairness-accuracy trade-offs and remains robust under noisy and incomplete causal graphs.

### Strengths
1. Significance: The paper clearly identifies key issues in prior intervention-based fairness notions, namely that expectation-based criteria may lead to inconsistent or insufficient assessments of causal fairness. It addresses these problems by providing a principled solution.
2. Originality: To the best of my knowledge, this is the first work that explicitly tackles the limitations of mean-based interventional fairness and introduces a distribution-level measure (ICRD) that accounts for the entire predictive distribution.
3. Quality: The paper conducts extensive experiments, including comparative studies and sensitivity analyses. The results consistently demonstrate the superiority of the proposed fairness metric (ICRD) and the robustness of the ICCFL framework across various datsets and causal graph settings.

### Weaknesses
1. The paper’s readability can be improved. There are several minor typos and presentation issues that slightly hinder the reading experience. For example:
    1. In line 191~193, the equation $P(\hat{y}=1|do(S=1), do(D=’A’)) - P(\hat{y}=1|do(S=0), do(D=’A’))=0.4$ seems to contain a typo, since $P(\hat{y}=1|do(S=1), do(D=’A’))=0.4$ and $P(\hat{y}=1|do(S=0), do(D=’A’))=0.2$. 
    2. In line 205~206, the hyperlink for “Figure S1(b)” is not included, though I can guess which figure you want to mention.
    3. In Definition 1 (ICRD), the phrase 'a model is considered as causality fairness' should probably be “a model is considered causally fair.”
    4. In Equation (6), the term $\frac{1}{n-}$ seems to be a typo and should likely be $\frac{1}{n_2}$.
    5. In Proposition 1, the “augmented G” should be introduced or defined earlier for clarity.
2. Some parts of the method and analysis build on existing work (e.g. identifiability analysis and certain modeling components heavily rely on Zuo et al., 2024). While this does not undermine the contribution, it would be beneficial for the authors to better clarify which components are novel and which are inherited from previous frameworks.

### Questions
1. The paper refers to ICCFL as a “novel fair learning method,” but the main difference from Zuo et al. 2024 appears to lie only in the fairness regularization term. Could the authors clarify in what sense ICCFL represents a new *learning framework* rather than an existing pipeline with a new regularizer? Moreover, if the novelty mainly lies in the differentiable estimation of ICRD, why is the variant using K-Fair (i.e. ICCFL-KF) still called “ICCFL,” even though it does not employ the ICRD estimator?
2. The results show that both ICCFL-wF and ICCFL-KF outperform $\epsilon$-IFair, even though $\epsilon$-IFair also adopts a well-designed fairness regularizer within a similar training pipeline. Could the authors provide further analysis or intuition behind this phenomenon? How does this result support or contrast with the claimed advantages of ICRD and the ICCFL design?

### Soundness
3

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
This paper identifies the limitations of existing interventional fairness notions and introduces a new causal fairness metric, Intervention-based Cumulative Rate Disparity (ICRD). ICRD measures the cumulative post-intervention causal effects along the prediction probabilities for any intervention on the context. In addition to formalizing this metric, the authors propose an algorithm to optimize for ICRD. Experimental results demonstrate the effectiveness of the proposed approach.

### Strengths
- The paper makes a meaningful attempt to analyze and reveal the weaknesses of existing interventional fairness notions and to propose a new one.

- The experimental evaluation is comprehensive, including practical settings with imperfect causal graphs.

### Weaknesses
1. While ICRD provides an interesting perspective, the contribution appears modest. The learning method proposed to achieve it is not particularly novel, and the paper would benefit from a clearer discussion of the situations where ICRD might be less suitable than alternative fairness notions.

2. In Theorem 1, the authors mention that ICRD is a continuous function. However, the benefit or implication of this property for ICRD is not discussed.

3. The identifiability analysis in Proposition 1 focuses primarily on MPDAG identification, which appears somewhat tangential to the main contribution of introducing ICRD.

4. Minor Issues / Typos: 
- Please check the notation $s$ in Equation (3). 
- Verify the denominator of $n_2$ in Equation (6).

### Questions
Please refer to the above weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a new causal-based ML fairness metric, which belongs to the category of intervention-based. This is motivated by two limitations of existing intervention-based metrics: Inconsistent assessment of bias and the insufficiency to conclude causal fairness. The new metric is called Intervention-based Cumulative Ratio Disparity (ICRD). The main idea behind this metric is that instead of relying on a specific threshold for the prediction (which might lead to equal averages between sensitive groups, despite discrimination), this metric considers the cumulative distribution function of the prediction. Along with the metric, the paper proposes a fair learning method (ICCFL) that mitigates bias as assessed by ICRD. The experimental analysis considers a significant number of existing causally fair learning methods. The comparison considers the Accuracy and fairness (measured using MMD, a distribution-level distance, to capture the divergence between distributions after intervention). In another set of experiments, ICRD is compared with other intervention-based metrics. The last set of experiments focused on the robustness of ICCFL in presence of noisy graphs.

### Strengths
The paper falls into the category of causal-based ML fairness approaches which is timely and very promising to address the problem of ML Bias.
The proposed ICRD fairness metric is well motivated and formally defined. 
The experimental analysis involves a comparison with a representative set of existing causal-based approaches (which is not common in the literature, so a positive aspect of this work, but also raises some concerns as explained in the weaknesses).

### Weaknesses
1) My main concern with the ICRD fairness metric is its applicability in practice. As defined, it is non-differentiable, and I don't see how it can be reliably computed in practice. There is already major concerns about the practicality of Causal-based fairness metrics, with this proposed metric, it becomes more questionable.
2) Very few details were provided about how existing approaches have been implemented for comparison purposes (e.g. A3). I checked in the appendix and I didn't find further details about how this is exactly done. This is important to show in order to justify the claimed "superiority" of the approach.

Some minor issues:
Equation (1), pa is not defined (I assume it means parents, but not sure).
Equation (3) of K-Fair is flawed (the same terms on both sides).
Equation (6) there should be 2 instead of the dash in n^{-}
Propoosition 1: MPDAG is not defined unless it is a typo for CPDAG. For that proposition, I would suggest an intuitive illustration because it is too abstract to understand (at least in the appendix).
Other metrics can be included in the comparison, such as: 
Pfohl, S. R., Duan, T., Ding, D. Y., & Shah, N. H. (2019, October). Counterfactual reasoning for fair clinical risk prediction. In Machine Learning for Healthcare Conference (pp. 325-358). PMLR.
Also, you can refer to this survey for other causal-based fairness metrics: 
Makhlouf, K., Zhioua, S., & Palamidessi, C. (2024). When causality meets fairness: A survey. Journal of Logical and Algebraic Methods in Programming, 141, 101000.

### Questions
For Causal discovery, why still using PC while newer algorithms are much better (DiffAN / DeepDAG, etc.) ?

### Soundness
3

### Presentation
3

### Contribution
2
