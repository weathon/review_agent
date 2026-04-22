# Doubly-Regressing Approach for Subgroup Fairness

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Algorithmic fairness is a socially crucial topic in real-world applications of AI.
Among many notions of fairness, subgroup fairness is widely studied when multiple sensitive attributes (e.g., gender, race, and age) are present.
However, as the number of sensitive attributes grows, the number of subgroups increases accordingly, creating heavy computational burden and data sparsity problem (i.e., subgroups with very small sample sizes).
In this paper, we develop a novel learning algorithm for subgroup fairness that resolves these issues by focusing on sufficiently large subgroups as well as marginal fairness (fairness for each sensitive attribute).
To this end, we formalize a notion of subgroup-subset fairness and introduce a corresponding distributional fairness measure called the supremum Integral Probability Metric (supIPM).
Building on this formulation, we propose the Doubly Regressing Adversarial learning for subgroup Fairness (DRAF) algorithm, which reduces a surrogate fairness gap for supIPM with much less computation than directly reducing supIPM.
Theoretically, we prove that the proposed surrogate fairness gap is an upper bound of supIPM.
Empirically, we show that the DRAF algorithm outperforms baseline methods on benchmark datasets, particularly when the number of sensitive attributes is large so that many subgroups are very small.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of subgroup fairness in machine learning, particularly when multiple sensitive attributes result in a large number of intersectional subgroups. Traditional fairness frameworks become impractical in such cases due to severe data sparsity and computational explosion.

To overcome these issues, the authors propose a new notion called subgroup-subset fairness, which ensures fairness across sufficiently large and socially meaningful subgroups while maintaining fairness for each attribute. They formalize this idea using the supremum Integral Probability Metric (supIPM) to measure distributional disparities and derive a computationally efficient surrogate formulation.

Based on this, they introduce the Doubly Regressing Adversarial learning for Fairness (DRAF) algorithm, which enforces subgroup-subset fairness using only a single discriminator, avoiding the exponential cost of prior methods. Theoretical analysis shows that this surrogate metric upper-bounds the original fairness measure, providing guarantees without sacrificing efficiency.

Experiments on benchmark datasets (ADULT, DUTCH, CIVILCOMMENTS, COMMUNITIES) demonstrate that DRAF achieves strong trade-offs between accuracy and fairness, particularly when data are sparse. The method successfully aligns subgroup and marginal fairness, offering both theoretical soundness and practical robustness.

### Strengths
The paper’s major strength lies in its theoretical innovation coupled with practical feasibility. The authors do not simply modify existing fairness metrics but instead derive a new conceptual layer, which is subgroup-subset fairness. It realistically balances statistical soundness and social relevance. This idea acknowledges that enforcing fairness over all possible intersectional subgroups is statistically impossible, and instead directs fairness efforts toward meaningful, estimable subsets.

Another strength is the mathematical coherence. The connection between fairness and regression-based measures (through the doubly regressing 𝑅^2) transforms a complex adversarial fairness optimization problem into one that is differentiable and computationally manageable.

From an algorithmic standpoint, DRAF’s design is efficient. By requiring only a single discriminator, the method retains adversarial flexibility without the exponential cost typical of subgroup fairness models.

Empirically, this paper includes multiple datasets with varying degrees of sparsity, showing that the method generalizes well.

### Weaknesses
While the theoretical development is solid, the paper’s presentation of empirical limitations could be more transparent. Although DRAF achieves strong results on several datasets, the analysis does not deeply explore why it performs better in certain settings or whether there are trade-offs when subgroup distributions overlap significantly. A more detailed analysis of failure cases would make the argument more comprehensive. For example, when subgroup definitions are noisy or when marginal fairness conflicts with subgroup fairness, how is the performance?

The paper briefly mentions the threshold parameter γ, which decides which subgroups are treated as “active.” However, this choice will strongly affect the fairness results. A clearer explanation or guideline on how to set γ properly would make the method easier to reproduce and evaluate fairly.

Another weakness is the limited discussion of the robustness of the DRAF. Since this surrogate replaces the original supIPM measure, it is important to know whether it consistently preserves the relative fairness ranking between models. In other words, if one model is fairer under supIPM, is it guaranteed to remain fairer under DR? The paper does not clarify this point. Without such an analysis, readers cannot be certain that the surrogate behaves reliably in all cases. The authors may provide some consistency conditions or give some counterexamples.

### Questions
A natural question concerns how subgroup-subset fairness aligns with broader ethical objectives. Because the algorithm deliberately excludes minimal subgroups to ensure statistical stability, what ethical safeguards ensure that such exclusions do not marginalize vulnerable intersections? Could adaptive weighting schemes or uncertainty-aware selection criteria help mitigate this risk?

Another question arises from the interaction between marginal and subgroup fairness. The authors show that DRAF aligns these two fairness notions empirically, but it would be interesting to formalize the connection: under what theoretical assumptions does ensuring fairness on selected subgroup-subsets guarantee approximate fairness across all subgroups?

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
4

### Summary
The paper proposes subgroup-subset fairness (enforce fairness only on sufficiently large subgroups plus all marginals) and an adversarial learner DRAF that optimizes a surrogate of a new fairness measure (supIPM) using a single discriminator; proves a generalization bound and that the surrogate upper-bounds supIPM; evaluates on ADULT, DUTCH, CIVILCOMMENTS, and a sparse COMMUNITIES dataset.

### Strengths
1. Computational convenience: the paper proposes a novel metric for subgroup fairness, which only considers a representative subset of sensitive attributes, significantly bringing down the dimensions.
2. Sound theoretical contribution: the authors have a rather complete theoretical analysis of their methods.

### Weaknesses
1. Experiments compare to REG, GerryFair, and a sequential post-processor, but these methods don't seem to get extensively discussed in the related work section, e.g., how the proposed method is different from [1]? Also, for multiple sensitive attributes especially in binary classification, there've also been some other literature around, e.g., [2] on binary classification with DP/EO constraints and [3] on EO with adversarial training fashion.
2. “Subgroup-subset fairness” depends on the chosen W. Provide coverage guarantees: if all first-/second-order marginals are in W, what bound (if any) follows for full subgroup fairness? It would be beneficial if the authors can justify why some W is chosen while others are not (I'm imagining there could be some implied unfairness for unseen tiny groups).
3. Datasets. I appreciate the comprehensive results on 4 datasets. However, ADULT is overused (e.g., see [4]); please consider newer tabular suites (ACS/Folktables, HMDA, LawSchool) or at least add some discussion on the limitations of these datasets (e.g., the label threshold for COMMUNITIES dataset).In addition, I feel the tasks for these four datasets are to some degree overlapping and limited, is the proposed method only working for binary classification or binary subgroups? If not, it could be more convicing experimenting on diverse learning tasks.
4. Scalability. One novel point of this paper is that it avoid the exponentially large dimensions of subgroups. Is there any empirical evidence in terms of running time to support this claim? Or, how's the scalability regarding $|\mathcal{S}|$? 


[1] Kearns, Michael, et al. "Preventing fairness gerrymandering: Auditing and learning for subgroup fairness." International conference on machine learning. PMLR, 2018.
[2] Agarwal, Alekh, et al. "A reductions approach to fair classification." International conference on machine learning. PMLR, 2018. 
[3] Lai, Yuheng, Leying Guan. "FairICP: Encouraging Equalized Odds via Inverse Conditional Permutation." International conference on machine learning. PMLR, 2025.
[4] Ding, Frances, et al. "Retiring adult: New datasets for fair machine learning." Advances in neural information processing systems 34 (2021): 6478-6490.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies subgroup-subset fairness, whose goal is to ensure that a prediction model is fair on active subgroups that are not too small. The authors propose to use supremum integral probabilitiy metric (subIPM) for measuring subgroup-subset fairness. To make it easier to calculate supIPM, the authors propose a surrogate of supIPM named DR, which is the z transformation of a smoothed version (by another regression) of supIPM.

### Strengths
S1. The motivation is clear, that the subgroup grows exponentially and that there could be data sparsity in subgroups that make the fairness measure not statistically meaningful.

S2. Theoretical contents are in general clear to me (though I have questions listed in weaknesses part).

### Weaknesses
W1. A main argument of this paper is that we should use active subgroups to study subgroup-subset fairness. But from Figure 10, it seems that with varying $\gamma$, the model doesn't seem to have much difference in performance. (There indeed are differences but just not very significant to me.) Could the authors provide more justifications on the impact of inactive subgroups for learning the fair model?

W2. Following W1, the connection between Theorem 3.1 + Section A.4 and $|W| \geq \gamma n$ is a bit vague to me. Could the authors further elaborate on this?

W3. For DR$^2$, the key idea is to regress $g(f_i)$ and $\mathbf{v}^T c_i$. Could the authors provide more justification on why we can have this $\mathbf{v}^T c_i$ as a smoothed $y_{W, i}$?

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
