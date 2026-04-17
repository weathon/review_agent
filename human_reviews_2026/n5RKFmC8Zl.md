# On the Impact of the Utility in Semivalue-based Data Valuation

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Semivalue–based data valuation uses cooperative‐game theory intuitions to assign each data point a value reflecting its contribution to a downstream task. Still, those values depend on the practitioner’s choice of utility, raising the question: *How robust is semivalue-based data valuation to changes in the utility?* This issue is critical when the utility is set as a trade‐off between several criteria and when practitioners must select among multiple equally valid utilities. We address this by introducing the notion of a dataset’s *spatial signature*: given a semivalue, we embed each data point into a lower-dimensional space in which any utility becomes a linear functional, making the data valuation framework amenable to a simpler geometric picture. Building on this, we propose a practical methodology centered on an explicit robustness metric that informs practitioners whether and by how much their data valuation results will shift as the utility changes. We validate this approach across diverse datasets and semivalues, demonstrating strong agreement with rank‐correlation analyses and offering analytical insight into how choosing a semivalue can amplify or diminish robustness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a unified geometric framework based on the spatial signature and also the robustness metric, designed to help data valuation practitioners assess the sensitivity of semivalue-based data valuation results to the choice of utility in both the multiple-valid utility and utility trade-off settings. They evaluate the effectiveness of the proposed metric through experiments on multiple real-world datasets, considering both the multiple-valid-utility and utility trade-off scenarios.

### Strengths
Authors present another perspective on how the choice of the utility affects the resulting data values and rankings during data valuation.  

To mitigate the noise and randomness introduced by Monte Carlo sampling, they propose the use of aligned sampling.

The theory and empirical setups are well-written and easy to follow. 

Authors conduct multiple experiments to assess the efficacy of their metric, and also make their code available.

### Weaknesses
Restricting the utility to a 2D space spanned by two fixed base utilities u1 and u2 (and extension to class-wise utilities ) makes the robustness metric hard to scale.

It's possible that some information is lost when embedding data to a low dimensional space. It's unclear to me how this affects the results.

The data values from Monte Carlo approximations in regression problems tend to be unreliable due to greater variability in utility estimates when sample sizes are small. So, is there a specific reason why authors restricted the use of regression problems to utility trade-off scenarios, rather than including them in multiple-utility analyses?

There appears to be a misalignment between the claims made in the introduction and the empirical setup. In particular, the treatment of utility trade-offs in the experiments does not accurately reflect the notion of trade-off presented in the introduction.

There are very few results presented for the utility trade-off scenario, with none in the main paper and few in the appendix. Also, the results relegated to the appendix are neither discussed nor explained. For instance, why do Tables 9–14 only consider p=500? And why are only regression tasks used?

While the robustness metric could, in principle, address the question “How robust are my data valuation results to the choice of utility?”, it is unclear how distinct the observed patterns are from those obtained using Kendall or Spearman correlations. For instance, based on the results for both Kendall and the robustness metric, the cases where Banzhaf outperforms Shapley appear identical. The authors note that, in the absence of ties, the robustness metric “captures how far, in expectation, one must move from a utility direction before the Kendall rank correlation degrades by at least 2p/N.” While this is insightful, the results presented in the tables and figures reveal only broad trends, which seem largely similar across both metrics.



The authors briefly mention in the appendix that “one can alter the utility either by changing the algorithm A or by changing the performance metric,” an important consideration also discussed in several of the cited works. In my opinion, this is a strong consideration, and seeing how the robustness metric performs across datasets would have been helpful. However, the results presented in Table 15 are reported using Spearman correlation rather than the proposed robustness metric. Also, these results are not discussed, and key details of the experimental setup are not given, such as the number of runs. 



The authors should consider adding a discussion section. Overall, the experimental results are insufficiently discussed, making it difficult to highlight the main contributions of the proposed metric, both theoretical and empirical.



Minor.
The references section needs a second look. Cite the published versions of the papers where they exist and not their arXivs. E.g., Jiachen T. Wang, Tianji Yang, James Zou, Yongchan Kwon, and Ruoxi Jia. Rethinking data shapley
for data selection tasks: Misleads and merits, ICML’24

### Questions
How does embedding the data point to a low dimensional space affects its value and ranking?

Is there a specific reason why authors typically restrict the use of regression problems to utility trade-off scenarios, rather than including them in multiple-utility analyses? 

Why do Tables 9–14 only consider p=500?

How distinct are the observed patterns with the robustness metric from those obtained using Kendall or Spearman correlations? In what ways is the robustness metric a great addition to the data valuation practitioner’s toolbox?

In what ways do dataset characteristics, data preprocessing choices, and algorithmic choices, e.g., chosen classification/regression algorithms, affect the results? 

Why restrict the multiple-valid-utility scenario to binary classification problems?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates how sensitive semivalue-based data valuation scores are to the practitioner’s choice of utility function. It introduces the dataset spatial signature, an embedding in ℝ² where any candidate utility function corresponds to a linear functional. In this space, ranking data points with a semivalue reduces to projecting the dataset signature onto a direction that represents the chosen utility.

### Strengths
The paper is clearly written and easy to follow.

It presents an elegant and intuitive geometric representation that translates variations in utility functions into simple linear projections.

The proposed method is validated across a wide range of datasets and semivalues (Shapley, Beta Shapley, and Banzhaf).

The authors have thoroughly addressed the major limitations of the previous version by adding regression and multiclass tasks, introducing a formal robustness metric, and conducting experiments to assess algorithmic modifications.

### Weaknesses
I have reviewed this paper before, and the authors have resolved most of my confusion, but I still have a remaining question.

The experiments on multiple-valid-utility scenarios remain somewhat limited and would benefit from further expansion.

### Questions
1. Currently, empirical studies of multiple-valid-utility systems are mainly conducted in binary classification. It is also recommended to conduct multiple-valid comparisons of "multiple equally-valid indicators" in multi-class classification, rather than just trade-offs (e.g., macro-F1 vs macro-Recall vs Accuracy). 2. While $R_p$ provides a theoretically elegant proxy for stability under utility shifts, the paper would be significantly strengthened by connecting $R_p$ to downstream selection stability. It is recommended to report the relationship between top-k overlap@k/Jaccard@k and $R_p$.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper provides an in-depth study of stability of semi-value based data valuation methods when there are shifts in utility. The authors  introduce the concept of spatial signature, where a training data point is projected on to a low-dimensional space. The authors assess the robustness of semi-value using a geometric representation of the utility space, specifically a unit circle, such that the metric quantifies how much utility direction must be rotated before the ranking undergoes a significant change. The authors use a geodesic angular distance over the unit circle, and the robustness metric Rp is measured as normalized minimal geodesic distance of rotation needed to achieve p swaps in ranking. The paper offers an evaluation of their metric against data valuation methods Data Shapley, Data Banzhaf and Beta Shapley.

### Strengths
1. The introduction of a novel geometric perspective on and a formal robustness metric for semi-value based data valuation offers a way to benchmark data valuation frameworks and test their stability. In real-world datasets, the utility function can change anytime, reinforcing the importance of this metric. 
2.  The authors provide a rigorous evaluation of their metric on existing data valuation methods and utility functions. 
3. The paper also offers insights into why some valuation methods tend to collinearize the spatial signature and in doing so achieve higher robustness to utility shifts. 
4. The robustness metric proposed is shown to be computationally efficient via the closed-form expression for the expected minimal geodesic distance needed to induce p pairwise swaps in ranking.

### Weaknesses
1. The paper focuses on utility functions specific to binary classification and raises the question of whether the findings will translate to other utility functions / learning tasks. Similarly the evaluations focus on two utility systems, and multi-utility systems are yet to be explored.
2. Benchmarking the stability of data valuation methods is not new. The key contribution of the paper is the geometric perspective - and this paper would be strengthened by additional experiments that show the true value of this geometric nature of the stability problem. While this paper discusses collinearity of spatial signatures leading to better robustness, the experimental validations focus on the robustness metric and not the geometric properties (eg. how the spatial signatures are distributed for a given valuation framework / dataset or the practical impact of this collinearity in geometric structure).

### Questions
1. I’d be interested in the authors' perspective on whether the robustness metric reflects an inherent property of the data valuation framework, or if it is more influenced by the dataset and the choice of utility function. For example, in imbalanced binary classification tasks, the rankings generated by different utility functions (e.g., F1 score vs. accuracy) can vary substantially. How does this variation impact the stability of data rankings and the robustness metric? And if it is affected by dataset and utility functions, then in what setting do the authors advice using this robustness metric in practice ?

2. How do the authors propose choosing p (especially as a factor of the size of the training dataset) for a given dataset?

### Soundness
3

### Presentation
2

### Contribution
3
