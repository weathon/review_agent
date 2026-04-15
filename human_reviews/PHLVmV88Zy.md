# Unconstrained Stochastic CCA: Unifying Multiview and Self-Supervised Learning

- Decision: Accept (poster)
- Scores: 8, 5, 8

## Abstract
The Canonical Correlation Analysis (CCA) family of methods is foundational in multiview learning.
Regularised linear CCA methods can be seen to generalise Partial Least Squares (PLS) and be unified with a Generalized Eigenvalue Problem (GEP) framework.
However, classical algorithms for these linear methods are computationally infeasible for large-scale data.
Extensions to Deep CCA show great promise, but current training procedures are slow and complicated.
First we propose a novel unconstrained objective that characterizes the top subspace of GEPs.
Our core contribution is a family of fast algorithms for stochastic PLS, stochastic CCA, and Deep CCA, simply obtained by applying stochastic gradient descent (SGD) to the corresponding CCA objectives.
Our algorithms show far faster convergence and recover higher correlations than the previous state-of-the-art on all standard CCA and Deep CCA benchmarks.
These improvements allow us to perform a first-of-its-kind PLS analysis of an extremely large biomedical dataset from the UK Biobank, with over 33,000 individuals and 500,000 features.
Finally, we apply our algorithms to match the performance of `CCA-family' Self-Supervised Learning (SSL) methods on CIFAR-10 and CIFAR-100 with minimal hyper-parameter tuning, and also present theory to clarify the links between these methods and classical CCA, laying the groundwork for future insights.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
UPDATE:
The authors have done a good job engaging with the reviewers during the discussion period. I have updated my score accordingly to an 8.

Classical algorithms for solving linear problems arising from CCA, PLS, and GEP are computationally infeasible (slow and memory-intensive) for huge datasets. The authors formulate top-subspace GEPs via a new unconstrained objective. The main result is then a family of algorithms that solves stochastic PLS, stochastic CCA and deep CCA by applying SGD to this objective. Empirical results show faster and better convergence than previous methods. For the first time, a very large biomedical dataset is tackled with PLS.

### Strengths
- This paper does require a relatively strong background and interest in linear algebra (GEP), optimisation, classical self-supervised/unsupervised techniques (PCA, CCA, PLS), and I have to admit that I found this paper challenging to read. However, when I did check isolated individual details carefully, I was not able to spot any immediately obvious issues. (Soundness +) 
- While the material is dense, the paper is clearly laid out and well-written. I particularly like section 2.2 and section 3.2/3.3, which first present general frameworks and then discuss special cases, which is a helpful tutorial-like description. (Presentation +)
- The paper is of general interest to the community, particularly those interested in (self/un)supervised methods. Unifying theories under a single framework is an attractive story for a paper.

Overall I find this paper to be good quality, however I have low confidence in my evaluation. I am receptive to increasing my evaluation and confidence if the authors can provide a good rebuttal and help clarify my queries.

### Weaknesses
Clarity:
- I do not see any precise discussion of Barlow Twins or VICReg in the main paper, however Proposition 1 informally states a result related to these formulations. I wonder what is the value in this --- a reader who knows about these formulations might dismiss the informal result (looking at the Appendix instead), and a reader who does not know about these formulations is left completely in the dark. Perhaps section 3.4 could be removed and replaced with a sentence referring to appendix D. (Presentation -)
- I checked the appendix I.3.4 but was not able to properly understand basic properties of dataset UK Biobank. Are the features in this data categorical, integer, real-valued, mixed, ... ? What are the modelling and "convenience" considerations when applying your method? In section 5.4, it is mentioned that "this can reveal novel phenotypes of interest and uncover genetic mechanisms of disease and brain morphometry." What is the precise task that is being performed here, with respect to the PLS formulation (e.g. with reference to the symbols in equations 4, 5, 6, 7)? (Presentation -, and perhaps Soundness -)

Minor and typos:
- It is not clear what is meant by "which should be true in general" in footnote 3. Perhaps if the observations are drawn from a continuous probability distribution, with probability 1 they are linearly independent?
- Check the references. E.g. "stochastic optimization for pca and pls" should be "stochastic optimization for PCA and PLS". Perhaps the authors could also seek advice from the area chair (if the paper is accepted) about the most appropriate way to cite the multiple papers with huge author lists (this might be common in other fields where these papers are published, but in ML it looks really bizarre).

### Questions
- I am having trouble with Lemma 3.2. When we assume that there is a final linear layer in each neural net, does this parameter $\theta$ only relate to this final layer, or to the whole neural net? How then is it intuitively possible that $\theta$ could be a local optimum?
- Ridge-regularized CCA. Does the ridge term also change the aforementioned "notions of uniqueness" to a stronger notion?
- "unlike Gemp et al. (2022) we do not simplify the problem by first performing PCA on the data before applying the CCA methods, which explains the decrease in performance of γ-EigenGame compared to their original work. " What then is the motivation for not first simplifying the problem using PCA preprocessing?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a new unconstrained loss function that introduces solutions to GEPs, paving the way for solving Canonical Correlation Analysis in large scale setting in an efficient manner.

### Strengths
The paper conducts extensive experiments across various benchmarks, and proves the effectiveness of the proposed method by the superior empirical results, compared to baselines such as SGHA and $\gamma$-EigenGame.

### Weaknesses
I have concern for the novelty of the method compared to Gemp et al. (2022). “unlike Gemp et al. (2022) we do not simplify the problem by first performing PCA on the data before applying the CCA methods “  The author needs to show insights as for the innovation of the proposed method.

### Questions
Justification of the novelty of the proposed method.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new formulation for general CCA problems, based on an Eckhart-Young inspired objective for the Generalized Eigenvalues Problem. Such objective is prone to being implemented in a stochastic way (i.e. with minibatches of dataset samples), and can be used for various variant of CCA such as the original linear CCA, PLS and Ridge Regularized CCA, Multi-view CCA, as well as Deep CCA. The theoretical advantages of such formulation are given, and an extensive experimental benchmark is performed, showing the advantage of the introduced method on several tasks, over state of the art methods.

### Strengths
### Originality

I believe the work is original, as it is, up to my knowledge, the first of its kind to consider an Eckhart-Young inspired objective (and the corresponding stochastic optimization algorithm to optimize it) for CCA-type problems.

### Quality

I believe the work is of quality, with theoretical results clearly described as well as their link the the related literature, and experimental results performed extensively.


### Clarity

I believe the work is clear, with a clear description of the methods and of the state of the art. I believe the experiments are clearly described and the code provided eases the reproducibility of the methods.


### Significance


I believe the work is significant, since CCA-like methods and the related Self-Supervised learning methods have been shown to be very important these last years, in particular for modern machine learning applications, including those which use deep neural networks to learn data representations: offering a theoretically grounded view of such problems, as well as an efficient and still theoretically grounded implementation (namely, the stochastic algorithm), I believe this type of methods is very interesting to the community.

### Weaknesses
Although the literature is extensively cited across the paper, and well detailed, and although the experiments show a clear advantage to the presented method, I still find it a little bit hard to compare the theoretical and computational guarantees of the presented method vs. the other methods from the benchmark: I believe it might be useful, (for instance in Appendix), to provide some short structured summary (perhaps as a table for instance), on all the properties of the related methods in the literature vs. the one presented (e.g. biasedness/unbiasedness of the gradients, complexity of computing one iteration as a function of the batch-size, dimension, etc, guarantees of convergence towards global/local optimum/saddle point, etc). However, I am aware that such comparison may not always be doable (in particular because the guarantees of convergence for instance, are not always simple to express), so this is just a suggestion,  andI believe the paper in its current form may already provide the necessary information to the reader to study such aspects.

### Questions
I just noticed one typo below: 
- In section 3.3: “we this flesh out” —> “we flesh this out”

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent
