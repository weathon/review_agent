# Resolving the Duplicate-Feature Paradox with ReSHAP: A Redundancy-Weighted Generalization of Shapley Attribution

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 4

## Abstract
Feature attribution methods based on Shapley values are widely used to explain machine learning model predictions. However, these methods suffer from a critical flaw often observed when features are duplicated, which we term the duplicate-feature paradox: when a feature is duplicated (or strongly correlated with another), its total contribution to the model prediction is unfairly inflated, diminishing the attribution of other important features. This paradox arises because traditional Shapley-based methods allocate joint contributions equally across all participating features, regardless of redundancy or informational overlap.

In this work, we propose ReSHAP, a redundancy-aware generalization of Shapley attribution that systematically resolves the duplicate-feature paradox. ReSHAP adjusts the allocation of credit within feature coalitions by down-weighting features that contribute redundant information. We begin by proving that no attribution method can simultaneously satisfy equal division and duplication-invariance, even in instances without redundant features. This reveals a fundamental trade-off in designing fair attribution methods. Building on this insight, ReSHAP redefines how Shapley values are computed by redistributing interaction terms across feature subsets using a recursive weighting scheme. This approach preserves core theoretical properties while providing a practical, computationally efficient correction for redundancy bias, without altering the standard value function or requiring distributional assumptions. We support our theoretical findings with illustrative examples and experiments, highlighting the practical effectiveness of ReSHAP.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents ReSHAP as a new method to compute Shapley Values to compute Shapley Values while addressing what the authors coin the duplication paradox. The duplication paradox is presented as a model that receives the same input feature twice and then assigns different attribution scores to the remainder of the features compared to a model that receives the feature without duplication. To solve this, the paper presents then ReSHAP which redistributes the möbius coefficients onto coalitions that contain duplicated features.

### Strengths
- Shapley Values and attributions are very important!
- The topic of attribution methods' dependence on the underlying feature distributions and potentially un-intuitive explanations is an interesting research direction.

### Weaknesses
- **Contrived Problem:** The paper's contribution addresses a rather contrived problem in my opinion. I have never heard of a "duplication paradox" as the authors present it (lines 64-68) and in my opinion this is a non-problem for Shapely. If you have one model $f_1(x_1, ...,x_n)$ with $n$ features and another model  $f_2(x_1,x_1,...,x_n)$ operating on $n+1$ features, then this is seems to be a modeling issue if the model gets access to the same information in two of its input features and b) totally justified for the Shapley Values to be different for both models $f_1$ and $f_2$. While I would not directly know how the Shapley Values behave as this is extremely dependent on the model architecture (chosen hypothesis space), I do not see a Shapley issue here. Yet, this instantiation is related to correlated features and implications of the removal mechanism selected in the value function for computing the Shapley Values. However, this is a quite broad topic explored in many related works. 
- **Very shallow Empirical Evaluation:** The empirical evaluation of this work is basically not existent. The only thing done is a Jupyter notebook containing case studies on synthetic data and really small datasets. Further no results are placed in the main body of the paper but referred to in the supplement. Thereby the empirical evaluation is not convincing at all. The *"problem"* is not properly motivated and the proposed ReSHAP does not show clear improvements compared to established baselines.
- **Writing and Presentation:** The paper's writing and presentation is not good. The work contains multiple technical examples that are not well placed in the text. The manuscript reads quite jarring and lacks a good structure. Proofs, proof sketches, and examples make up the majority of the methodological section which all take away the space for actually fleshing out the proposed problem more and addressing the core methodological contribution.
- **Unclear:**  It is unclear to me if the method needs to evaluate the power set of all coalitions or if it can be used in a estimation procedure akin to traditional Shapley value estimation methods (such as KernelSHAP, SVARM, or Permutation Sampling). The former is quite a big problem limiting the applicability of ReSHAP drastically and would require proposing an estimation method (and properly testing it) the and the latter still needs to do a proper evaluation of the estimation quality.
- **Missing Ablations.** The paper does not discuss or analyze ReSHAP's behavior on common parameters such as feature-size or hyper parameters (the above mentioned estimation budget).

### Questions
- What is the computational complexity of ReSHAP? Can I call it with a fixed budget for estimation purposes or do I need to evaluate the whole powers of coalitions?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper raises a problem that when duplicate features exist, Shapley-value based explanations may be distorted. For example, the contribution of the remaining features can be significantly diminished. To address this problem, the paper proposes two properties equal division and duplication-invariance. The paper then proposes Reshap that satisfies the duplication invariant property.

### Strengths
1. The problem raised about duplicate or similar features distorting Shapley-value based explanations is valid and significant.

### Weaknesses
1. The paper is not well-written and lacking clear definitions, justifications and intuitions. This makes it hard to understand and assess the significance or soundness of the claims.
    * Why does the paper propose Lemma 3 and 2? Why do we need this alternative form of Shapley value?
    * The use and significance of Harsanyi dividends should be explained.
    * What is the formal definition of equal division in Observation 4? Equal division property does not seem to be proposed by the reference van den Brink & Funaki (2025). They propose an axiomatisation for the equal value solution instead.
    * It is hard to interpret Figure 1. What does each number represent? Why is there two 2s for D?
    * Justify why it is reasonable to define a redundant feature based on the _existence_ of a subset where $j$ adds no value. Why should the total attribution value stay the same when the value of other subsets may change (Definition 7)?
    * The proof of theorem 9 is also unclear. Can you give the function value for every subset? Why should the attribution of feature C remain unchanged in both instances?
    * Can Reshap handle more realistic similar features (instead of duplicates)?
2. The experiments section is severely lacking. Only one dataset and model is used. More information should be given about how the P and R (precision and recall?) are computed.
3. There are undiscussed existing work that consider replication robustness and propose solutions. See Han, D., Wooldridge, M., Rogers, A., Ohrimenko, O., & Tschiatschek, S. (2020). Replication Robust Payoff Allocation in Submodular Cooperative Games. IEEE Transactions on Artificial Intelligence, 4, 1114-1128.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

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
I have just reviewed this manuscript for the AAAI.  As far as I can tell, the manuscript has been reformatted for the ICLR and added a page of 'Analysis and Empirical Evaluation' ($\S 7$), but not otherwise to have taken into account my previous review.

That review can be found at https://openreview.net/forum?id=p8Zao3l8HP&noteId=viyMisQReq

> The Shapley value feature attribution technique randomises uniformly over features when seeking to assign 'value-added' to a feature. Thus, duplicating a feature gives that feature twice as many opportunities to add value - even though no explanatory power has been added to the model. (See Kumar et al. 2020 - as cited by the authors - for an example.)

> The paper introduces ReSHAP, which seeks to correct for interaction effects between features.

> It does so by decomposing the Shapley value using Möbius transforms, following a tradition dating back at least to Grabisch & Roubens (1999); see Pymar et al. (2022) for a comparison of some of the 'interaction' literature, and Stoian (2023) for a recent paper on using this decomposition for faster computations.

> The authors, however, compare their approach to 'correlation-based' approaches - giving the impression that they are not familiar with the existing literature.

> Thus, at this point, it is not clear what distinctly novel issues are introduced by this paper:

> 1. the motivating concern is known.
> 1. no comparison with existing Shapley/Möbius/interaction methods is carried out.
> 1. it does not re-analyse the existing literature in a way that clarifies unresolved issues. (Indeed, it does not demonstrate a tight grasp on it: e.g. the claim that "computation is not guaranteed in general" is unclear in Frye et al., which uses a deterministic algorithm with a guaranteed, unique solution, given some encoding of domain knowledge.)

> I would encourage the authors to focus their rebuttal on how ReSHAP compares to existing techniques.

This remains my view.

### Strengths
See above: understanding how features interact in ML models is an important problem.

### Weaknesses
See above: although this problem has already been studied, the authors do not compare their approach to the most similar approaches in the literature.  

When I first reviewed this paper for AAAI, I wondered whether they were just not familiar with the literature.  Having cited papers in the literature, and recommended that the authors compare their approach to existing ones, that interpretation is no longer tenable.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces ReSHAP, a redundancy-weighted generalization of Shapley-based feature attribution. The method addresses the duplicate-feature paradox - the well-known issue that adding redundant or duplicate features changes attribution results in standard SHAP. The authors propose a recursive redistribution algorithm (Algorithm 1) to adjust the Möbius-based Shapley decomposition and theoretically prove that ReSHAP satisfies duplication invariance. Experiments on a small regression dataset illustrate improved stability of feature importance. The conceptual motivation of ReSHAP is strong and theoretically well presented. However, the algorithm currently lacks comprehensive experimental validation and requires deeper discussion of computational feasibility.

### Strengths
S1) The paper is clearly written and well structured, making the theoretical content relatively accessible.
S2) Theorem 9 provides a meaningful theoretical insight into the trade-off between equal division and duplication invariance.
S3) The recursive redistribution algorithm is original and intuitively motivated.

### Weaknesses
W1) The weighting scheme in ReSHAP is defined heuristically. (D1)
W2) The experimental evidence can be improved. (D2, D4)
W3) Key results such as complexity analysis and comparisons with baselines are placed in the appendix, which weakens their visibility. (D2, D3, D4)
W4) The algorithmic complexity is prohibitively high; the paper should discuss the feasibility of practical computation.

### Questions
D1) Although Theorem 11 justifies duplication invariance, the derivation of the weighting rule lacks rigorous theoretical grounding. It would strengthen the paper to provide an analytical motivation for the weighting rule.
D2) The experiment uses only the Ames Housing dataset and a simple MLP model. Additional datasets of varying dimensionality and correlation structure would better demonstrate generality.
D3) Appendix E indicates that the number of features is only about a dozen. A discussion on how ReSHAP scales to higher-dimensional data, or an approximation scheme for large n, would be valuable.
D4) Appendix G contains no experimental comparison with SHAP or related redundancy-aware methods. Including such baselines—or at least a discussion of expected differences—would make the empirical section more convincing.

### Soundness
2

### Presentation
3

### Contribution
2
