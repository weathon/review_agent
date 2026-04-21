# Differentiable Optimization of Similarity Scores Between Models and Brains

- Avg Score: 6.25
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 5

## Abstract
How do we know if two systems - biological or artificial - process information in a similar way? Similarity measures such as linear regression, Centered Kernel Alignment (CKA), Normalized Bures Similarity (NBS), and angular Procrustes distance, are often used to quantify this similarity. However, it is currently unclear what drives high similarity scores and even what constitutes a "good" score. Here, we introduce a novel tool to investigate these questions by differentiating through similarity measures to directly maximize the score. Surprisingly, we find that high similarity scores do not guarantee encoding task-relevant information in a manner consistent with neural data; and this is particularly acute for CKA and even some variations of cross-validated and regularized linear regression. We find no consistent threshold for a good similarity score - it depends on both the measure and the dataset. In addition, synthetic datasets optimized to maximize similarity scores initially learn the highest variance principal component of the target dataset, but some methods like angular Procrustes capture lower variance dimensions much earlier than methods like CKA. To shed light on this, we mathematically derive the sensitivity of CKA, angular Procrustes, and NBS to the variance of principal component dimensions, and explain the emphasis CKA places on high variance components. Finally, by jointly optimizing multiple similarity measures, we characterize their allowable ranges and reveal that some similarity measures are more constraining than others. While current measures offer a seemingly straightforward way to quantify the similarity between neural systems, our work underscores the need for careful interpretation. We hope the tools we developed will be used by practitioners to better understand current and future similarity measures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the properties of several similarity metrics in various neural activity datasets. The goal of similarity metrics is to quantify how well models of brain align with neural data. However, there are inconsistencies across different metrics, i.e., some metrics score high while others score low. This paper aims to address this inconsistency problem and propose a model-agnostic synthetic dataset optimization to analyze the properties of similarity metrics. The optimization dynamics in numerical experiments reveal that there is no single metric that is universally applicable for all dataset since the concept of a good score is highly dependent on the dataset. Additionally, the authors provide a python package that includes various similarity metrics.

### Strengths
* The problem is well-formulated in the introduction and clearly illustrated in Figure 2.
* Numerical experiments are presented clearly for the reader.
* The published code is well-structured, enhancing reproducibility.
* The observations made in Figure 3 are interesting (that some scores are good for some datasets while they are bad for other datasets).

### Weaknesses
The paper has some clarity and novelty issues in my opinion. Please see the points below and the questions section.

* One premise stated in the abstract is that the paper offers a theoretical analysis to show how similarity metrics are dependent on the principal components of the dataset. However, this premise appears weak to me because: (i)  it does not seem to be a novel analysis but rather a predictable outcome of using Frobenius and nuclear norms in metrics CKA and NBS; (ii) the assumption $\langle u_X^i, u_Y^i\rangle \approx 0$  is introduced without sufficient context and is unclear; and (iii) the assumption is said to hold with large sample sizes, validated in a numerical experiment, yet I did not see a clear mention of dataset sizes in the paper. Including more details on the datasets and clarifying the underlying assumptions would be helpful.

* The introduction and related work section suggest that prior research lacks practical guidance on metric selection given a dataset. However, I am uncertain if this paper proposes such a guidance. Suppose I have a neural dataset $X$ and model representations $Y$ to compare. How should I choose the most suitable metric based on this paper? My understanding is that I can optimize a synthetic dataset $Z$ using various metrics, observe the optimization dynamics, and then choose a similarity metric for $X$ and $Y$. Is that correct? I am asking this since I am struggling to understand how this paper offers a method for selecting an appropriate metric for a given task, if indeed it is promising.

* The claim between lines 267-270 is not detailed enough in the paper. The authors mention testing a hypothesis, but they simply state "we tested this hypothesis ... but this did not change the results" without further context. The results for that is not shared in the paper. Including these results in the appendix would enhance the paper's clarity.

* The joint optimization method in Section 4.4 and Appendix C.3 is unclear, as the details on experiments in this section are sparse. I think the paper can benefit from more details.

* The term "Proof" in Appendix C.2 and Section 4.3 seems a bit strong without an accompanying theorem or lemma, especially since the assumption is only noted in the appendix. Revising the word "proof" might be appropriate.

**Minor Comments** 

* The sentence in line 417 is repeated; it was already mentioned that Williams et al. (2021) advocate taking the arccos of CKA to align with distance metric axioms.

* Appendix B.1 could provide more dataset details rather than referring readers to other works. Including information such as dataset dimensions and data collection methods would be helpful.

### Questions
* What is the reason for using ridge regularization in the $R^2$ definition (in the numerator in line 208)? In that case, the numerator will not be the residual square and I do not see the rationale behind this. 

* Why is line 512 specifically bold-faced, but not the next one? According to Figure 7, the relation that high value of angular Procrustes implies a high score for linear regression appears more established compared to the relation of angular Procrustes and CKA scores.

* Given the findings, the main takeaway seems to be that similarity metrics are highly sensitive to different data aspects and may be mutually independent. How, then, would the authors suggest selecting the best similarity metric for a given dataset?

**Post rebuttal comment:** I appreciate the authors' detailed responses and the revisions made to the paper. These changes have significantly improved its clarity in my opinion. Considering the positive feedback from other reviewers as well, I am pleased to raise my score to 6.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to compare how different similarity measures such as CKA and linear regression behave, based on both prior theoretical work as well as by optimizing synthetic data through Adam to become more similar (under some similarity measure, e.g. CKA) to a reference neural dataset. The paper analyzes what properties a synthetic dataset can be expected to have (e.g. with respect to decodability of task relevant variables) at various levels of similarity to a reference dataset under a range of similarity measures.

### Strengths
- Although optimizing a set of features to become more similar to neural data has been done (e.g. optimizing neural network models of the brain), specifically optimizing a synthetic dataset to in order to gain insight into how similarity measures behave, especially at various intermediate levels of similarity, is novel. 
- Most discussions of similarity measures have focused on the special case where the similarity score is 1 (for example, what happens when response profiles X and Y are equivalent under some similarity measure such as CKA), so discussion of how intermediate values behave for different measures is a good contribution, especially since we are often dealing with intermediate levels of similarity in practice, e.g. when comparing models to brain data.
- CKA and linear regression are widely used methods of measuring similarity, so this paper can potentially be useful to many researchers comparing models to brain data.

### Weaknesses
- In this paper, the regularization level for Ridge Regression is fixed to some chosen level (and the authors do consider results for different fixed levels of lambda). However it seems to me that, because of the probability of overfitting in high dimensional data settings, it is generally preferable to tune the ridge penalty through some cross-validation method (searching over a range of possible alpha values) such as k-fold so as to select the lambda that will maximize generalization performance on the chosen data.
- Linear regression is only done in one direction, from the model to the reference neural dataset. This is good to know about, but it also would be useful to see what happens when linear regression is done in both directions, i.e. if synthetic data is optimized so that it predicts the brain and the brain predicts the data as well. 
- RSA is also widely used as a similarity measure, but is not mentioned at all in the paper. It would be very useful if the paper included an analysis of RSA, especially since RSA is mathematically very closely related to linear CKA, but the formulas for RSA and linear CKA are not identical. It would be good to therefore have an analysis of the relationship between these two methods, as well as empirical simulations showing how the intermediate values compare for RSA and CKA (just as the authors did for other methods, like comparing CKA to NBS).
- While many parts of the paper are clearly written, Figures 5 and 6 were hard for me to understand, and there was not much explanation in either the caption or main text. See questions below.
- While I understand the intended application of these results is to help researchers better understand similarity scores when comparing models to brains, the paper title seems a bit misleading, since it gives the impression that the paper is optimizing the similarity between an actual ANN model and the brain, whereas what is actually done here is optimizing similarity scores between a randomly initialized matrix and the brain features. Perhaps the title doesn't need to be fixed, but initially the title gave me a different idea of what the paper was going to do.

### Questions
In Figure 5, what does PC explained variance mean? What is the PC threshold? (I'm also not sure which PC we are talking about here - is it the first, largest PC?) Why is it that the score to reach PC threshold is *larger* for a smaller PC explained variance? Shouldn't explaining less variance require a smaller score? 

In Figure 6, what does it mean to perturb a single PC? How much do you perturb that PC? (it isn't stated, but I assume that how much you perturb it is very important for what the resulting similarity score should be).

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
4

### Summary
The paper aims to compare various similarity measures by optimizing randomly initialized datasets to various neural recording datasets. The authors find that some measures such as CKA can have high scores without sufficiently encoding task relevant information. The paper then investigates how much of a dataset needs to be captured before a certain score is achieved. It finds that some of the measures that have high scores without encoding task-relevant information also are most sensitive to high variance principal components. The authors complete theoretical and perturbation experiments to validate this hypothesis.

### Strengths
1. Very Clear writing, easy to see what analysis is being done and why.
2. Evaluating in a model agnostic way puts the focus on the measures and leads to a better understanding of the relevant differences for completing model-brain comparisons.
3. This analysis is fundamental to the field. Understanding what aspects lead to a high similarity score is extremely important to guide development of new models and to properly apply the modeling results to the brain.

### Weaknesses
1. Ridge-Regression seems to be the most widely-used measure in the field although most of the comparisons focus on CKA vs Angular Procrustes. Would be nice to see more commentary on this especially in Fig. 7 where it seems independent from angular Procrustes.
2. From the start of the paper it seems like it will answer the question: "What metrics should guide the development of more realistic models of the brain?" The discussion seems to attempt to avoid this question: "Our findings demonstrate that the interpretation of these scores is highly dependent on the specifics of both the metric and the dataset. We do not claim that one metric is superior to another, as indeed, they are sensitive to different aspects of the data and in some cases can be largely independent. Rather, we emphasize that the concept of a "good" score is nuanced and varies with context." I would like the authors to comment more directly on what should be done. Should this style of analysis be done for every new dataset which can provide a score range that encodes certain relevant variables? Is there some other guideline? It makes sense that there isn't one best choice but the question that starts off the paper doesn't seem to be addressed.
3. The datasets are all electrophysiology datasets whereas comparisons are often also done with fMRI datasets, will these results still hold for these datasets? Especially with the difference in sampling between the methods.

### Questions
1. Do the authors have suggestions on how to use these measures? Or do they have a suggestion of what analysis is still needed before picking a measure?
2. Why does ridge regression seem to be independent from Angular Procrustes (Fig 7)?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies several popular methods to quantify the similarity between models and neural data by applying them to five neural data from several studies. The approach is to directly optimize synthetic datasets to maximize their similarity to neural recordings.  The work is of expository nature and there have been several reviews on similarity measures, but this work is model-agnostic and can shed light on how different metrics prioritize various aspects of the data, such as specific principal components or task-relevant information.

### Strengths
Similarity measures have played a pivotal role in guiding the development of more realistic models of the brain. This work provides new insights and challenges of such measures.

### Weaknesses
This work is of expository nature, so by this nature its advancement in methodology and theory is less significant.

### Questions
1. What guidance will you provide to scientists in choosing a suitable similarity measure? 

2. How will this work have impact in the way that similarity scores are applied in practice?

### Soundness
3

### Presentation
3

### Contribution
3
