# On the (In)Significance of Feature Selection in High-Dimensional Datasets

- Decision: Reject
- Scores: 2, 2, 10

## Abstract
Feature selection (FS) is assumed to improve predictive performance and highlight meaningful features. We systematically evaluate this across $30$ diverse datasets, including RNA-Seq, mass spectrometry, and imaging. Surprisingly, tiny random subsets of features (0.02-1\%) consistently match or outperform full feature sets in $27$ of $30$ datasets and selected features from published studies (wherever available). In short, any arbitrary set of features is as good as any other (with surprisingly low variance in results) - so how can a particular set of selected features be ''important'' if they perform no better than an arbitrary set?  These results indicate the failure of the null hypothesis implicit in claims across many FS papers, challenging the assumption that computationally selected features reliably capture meaningful signals. They also underscore the need for rigorous validation before interpreting selected features as actionable, particularly in computational genomics.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies feature selection for classification tasks. It uses established methods and metrics to perform feature selection on a long list of datasets, focusing on biological data (and microarray data within that).

The paper provides an empirical study on the effectiveness of randomly choosing a small, random subset of features from a given dataset for downstream applications (here, explicitly, classification). It claims that randomly selected features perform at least as well as 'cleverly' selected features. It also claims that a small, random subset of features can achieve the performance obtained by the full feature set and proposes that the minimum random set size that achieves this performance is a useful metric in describing the data.

### Strengths
- Reporting results on a long list of datasets. This is common in feature selection literature.
- Demonstrating with workflows implementing established methods
- Proposing a metric, minimum sufficient random sample size, that interpretably evaluates the collective strength of the features of a dataset

### Weaknesses
- The results do not seem to support a key claim of the paper (randomly selected features performing at least as well as cleverly selected features)
  - In Table 2, all values in column D are lower than the corresponding values in column A.
  - In Table 2, 2 out of 6 values in column E are higher than those in column A, but this is problematic:
    - The difference between columns D and E is that E uses an ensemble of classifiers. Column A could presumably also benefit from such ensembling.
    - The procedure for column E is not clear. The text says "ensemble of LR, RF, and XGB trained on different random subsets of features of same size as the published study" while the table says "same number of randomly selected features". If each ensemble uses a different random subset of the same size, then column E effectively uses up to 3 times more number of features. Alternatively if the total number of features available to column E is the same as that to column A, then it is not clear how the sizes of feature sub-subsets were allocated between LR, RF, XGB.

- With the exception of the Madelon and Gisette subsets, the sample count is less than the feature count for all studied datasets. (In Madelon, the paper's claim does not hold and all features are needed to achieve full performance. In Gisette, sample count and feature count are close to each other; 7000 vs 5000.) However, sample counts being much larger than feature counts is very common in modern datasets. For instance, it is nowadays routine to profile millions of cells with scRNA-Seq. It is possible that the trend the authors observe concerns primarily the #feature < #sample regime.

- The datasets are heavily skewed towards biological datasets. It is possible that the trend identified in this paper does not apply to all domains. I think the claim needs to be narrowed accordingly.

### Questions
- Could you try the proposed experiment on a larger scRNA-Seq dataset? Many such datasets are publicly available together with their cell type annotations, disease status, etc.

- (line 4141-416) This is confusing. Did the authors mean "The only three cases where **the full feature set** does better than chance..."? Table 1 does not show results for feature selection that is not random.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents a large-scale empirical analysis challenging the assumption that feature selection improves predictive performance in high-dimensional datasets. Across 30 datasets, mostly gene expression microarray data focused on cancer the authors show that small random subsets of features (0.02–1%) consistently match or outperform models trained on full feature sets or published feature selections.

### Strengths
The authors address an important question regarding the utility of feature selection in high-dimensional datasets, a topic with significant implications for machine learning and computational biology.

### Weaknesses
The paper has several limitations that reduce the strength of its conclusions:
1) The dataset selection process is not described. The authors provide no inclusion or exclusion criteria, making it unclear how the 30 datasets were chosen.
2) The dataset pool is heavily biased toward cancer-related gene expression studies from the Gene Expression Omnibus (GEO), yet the conclusions are generalized to the entire field of feature selection.
3) Cancer and inflammation datasets are known to display large-scale, disease-specific expression changes, which increases the likelihood that randomly selected genes still carry predictive signal.
4) Training on all available features may introduce substantial noise, artificially lowering performance and making small random subsets appear stronger than they are.
5) The paper overlooks much of the recent literature on feature selection and related theoretical analyses.

### Questions
I would ask the authors to address the weaknesses above:
1) Please specify the inclusion and exclusion criteria for dataset selection. For the microarray datasets in particular, were they drawn from a defined subset of GEO, or selected manually from thousands of available series?
2–3) The analysis should be extended to more diverse domains. Even within transcriptomics, including single-cell RNA-seq datasets would provide a stronger test case. In single-cell data, subtype classification often depends on a few key marker genes, making it unlikely that random subsets would perform comparably.
4) Please include actual feature selection algorithms in the experimental comparisons rather than relying solely on published feature sets.
5) I recommend consulting prior work on the multiplicity of genomic signatures, which offers theoretical grounding for the observed phenomena:
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000790
https://jmlr.org/papers/v14/statnikov13a.html
https://link.springer.com/article/10.1007/s10618-020-00731-7

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This is a refreshingly honest paper that questions the validity of results published in 30 biological studies and this paper shows that randomly choosing a small collection of features and training a simple ensemble model with Linear Regression, Random Forest and XGBoost, gives comparable results to the published results!

### Strengths
ML as a community needs more such papers that challenge the conventional norm.  The results are quite powerful and a bit surprising. Having analyzed many of these datasets, I couldnt but wonder why others havent published such papers earlier. 

A biologist can have one of two reactions to the claims in this paper: "wow, this is surprising/shocking" or "the analysis is flawed because of ..."! For either of these extreme reactions, this paper may be worth accepting to provoke deeper discussions. I actually think ICLR is the wrong venue for this paper for the visibility but the authors should have sent this to Nature/Science sub-journals.

### Weaknesses
The paper may have flaws in the analysis but I think the authors have been honest about their analysis and opened up their code and tried to validate them independently.

### Questions
A biologist can have one of two reactions to the claims in this paper: "wow, this is surprising/shocking" or "the analysis is flawed because of ..."! For either of these extreme reactions, this paper may be worth accepting to provoke deeper discussions. I actually think ICLR is the wrong venue for this paper for the visibility but the authors should have sent this to Nature/Science sub-journals.

If you want to cherry pick specific issues in the analysis, here are some trivial basic questions:
a. When they randomly select a sub-set of features, did the features they picked have strong correlations with the features deemed relevant by the published paper. If even a few features have strong correlations, then the random selection defeats the purpose.

b. Did the authors check how was the original datasets generated? Was the single cell RNA or bulk RNA data imputed using any zero imputation algorithm before the dataset was published? This completely changes the equation. Many imputation algorithms used in the literature give different imputations on the same data with limited agreement in the generated matrix. This could completely change the results. 

c. Did the authors check if the generated data had any notion of experimental validation of gene types or cell types in the sequencing methodology?

d. Did the authors check the number of zeroes in the matrix? Many raw RNA matrices have more than 50-95% of the features to be zero before imputation. So if you randomly choose 0.5-2% of numbers, most of the numbers may be zero. If this is not the case, then there may be serious issues from the imputation pipeline.

e. Why did the fraction of randomly sampled features change from paper to paper? Are the authors optimizing their parameters to get the best outcome result for each dataset? That would constitute cheating in the analysis. They should clarify the same.

f. Many of the published results typically highlight very few edges as "outcome edges" that have significance from their analysis. Now, are the authors are comparing against the right metrics reported from these papers. The 95+% accuracy with a limited set of features published in these papers seems to imply that many of these papers are reporting several features. Most Nature/Science/Cell/Immunity papers focus on "one inference edge" in a paper. Something seems off in the way the results are reported.

Despite all these questions, I do love this paper and I think ICLR should accept such papers even if the results are flawed! Papers that challenge conventional wisdom are much harder to publish and should be encouraged.

### Soundness
3

### Presentation
3

### Contribution
3
