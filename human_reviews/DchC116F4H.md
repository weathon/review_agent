# Non-negative Probabilistic Factorization

- Avg Score: 4.75
- Decision: Reject
- Scores: 3, 8, 3, 5

## Abstract
Non-negative Matrix Factorization (NMF) is a powerful data-analysis tool to extract non-negative latent components from linearly mixed samples. It is particularly useful when the observed signal aggregates contributions from multiple sources. However, NMF only accounts for two types of variations between samples - disparities in the proportions of sources contribution and observation noise. Here, we present VarNMF, a probabilistic extension of NMF that introduces another type of variation between samples: a variation in the actual value a source contributes to the samples. We show that by modeling sources as distributions and applying an Expectation Maximization procedure, we can learn this type of variation directly from mixed samples without observing any source directly. We apply VarNMF to a dataset of genomic measurements from liquid biopsies and demonstrate its ability to extract cancer-associated source distributions that reflect inter-cancer variability directly from mixed samples and without prior knowledge. The proposed model provides a framework for learning source distributions from additive mixed samples without direct observations.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this submission, the authors claim to propose a probabilistic extension of NMF, namely VarNMF, that introduces a novel type of variation between samples. They learn this variation through a straightforward expectation-maximization procedure. The proposed method is demonstrated on a dataset of genomic measurements from liquid biopsies to show its effectiveness.

### Strengths
The performed experiments on real data seem to be of some interest.

### Weaknesses
- Equations (2), (3), (4) and Figure 1 are fairly standard for people familiar with NMF. However, the authors dedicate considerable space to these concepts and even replicate them later, such as in Equations (8) and (9).

- It appears that the proposed VarNMF is a simplified version of the model/framework designed in  

Tan, V. Y., & Févotte, C. (2012). Automatic relevance determination in nonnegative matrix factorization with the/spl beta/-divergence. IEEE transactions on pattern analysis and machine intelligence, 35(7), 1592-1605.

The authors of the current submission should not be unaware of this closely related work. 

- The mathematical notations in the paper are somewhat messy. For example, the dimension of $V$ is first mentioned on page 3, but $H_k \in \mathbb{R}^M_{\ge 0}$ suddenly appears on page 2. On page 4, the notation $V = (V[1], \ldots, V[N]) \in \mathbb{R}^{N\times M}_{\ge 0}$ is written in a confusing style. Additionally, some simple equations such as (8), (9), and (12) are repeated (although notations are slightly different) and occupy entire lines.

### Questions
No

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Non Negative Matrix Factorization can be understood as a particular case of blind source separation. In blind source separation problems, we are given several measurements (the so-called channels) corresponding to noisy observations of the mixing of different sources. The goal is to recover the sources (and the mixing components of the different sources for each channel). One main limitation of NMF models is that the sources are considered to be the same across the different channels: an assumption that is not always satisfied in practice.

In this work, the authors propose an extension of the standard NMF model where the different source signals are sampled from some probability distribution in the feature domain for each channel. Contrary to previous works, the distribution of each source is not pre-learned assuming that observations of the sources are available but they are inferred from the channel data.

### Strengths
- The paper is well written and provide a clear description of previous works in the field of non negative matrix factorization. 

- The authors conduct experiments on both simulated and real world datasets. They show that their method leads to better results to take into account the variability in the sources.

- The authors properly address several important aspects of their method. In particular, they provide in the Appendix a detailed explanation of the way to mitigate the heavy computational cost of the E-step of their EM procedure when dealing with real data and a large amount of features.

### Weaknesses
- One possible limitation of the proposed approach is the assumption that the probability distribution of source $k$ can be factorized in the feature domain (i.e. independence of the features of source $k$ is considered). This assumption might be limiting the expressivity of the proposed model. In particular, it would have been particularly interesting to go beyond this independent assumption for the application on real data since it is well-known that genes can be up or down regulated by other genes.

- Another limitation of the method is the heavy computational cost of the algorithm, even if the authors propose approaches to cope with this issue.

- The method proposed by the authors is specific to the Poisson distribution for the observational probability (and to the assumptions made on the dependence structure of the sources).

### Questions
I thank the authors for this very nice work. The paper is well written and it was a pleasure to read it. My question are the following.

- How the method can be adapted to consider other observational probability such as a negative
binomial distribution (instead of the Poisson) ? The negative binomial distribution would be particularly valuable for the biological application suggested by the authors as it enables the modeling of overdispersion, a phenomenon known to occur in such biological settings.

- In the same spirit, would it be possible to change the model to have dependence between features of source $k$ ?


Here are some typos (or minor comments):

- It might be beneficial to also include a reference to Section A.7 of the Appendix in the section discussing experiments on real data, which explains how you tested your model on fresh data. Indeed, I was briefly confused because NMFs are known to encounter the cold start problem, and I was curious about the authors' approach to testing the model on new data.

- At Eq.(11), a closing parenthesis should be removed. Same at Eq.(35).

- At page 11, "Differentiating" should be "differentiating".

- In Lemma 5, it might be good to specify which parametrization of the negative binomial distribution you consider.

- I think the logarithm should be removed in Eq.(21).

- In Eq.(30), the symbol "sum" does not display properly.

- After Eq.(37), the same style is not used for the log-likelihood function ($l$ instead of $\ell$).

- At Eq.(40) (second line), I think you should remove one "$p(.)$".

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a probabilistic non-negative matrix factorization (VarNMF) in which components vary per sample, rather than being "static" coefficients.
To achieve this variability, the authors introduce a Poisson Factorization formulation in which per-observation-components are independently distributed.
The result is an intractable likelihood function that the authors simplify in terms of additional auxiliary variables, which enables model-estimation using EM.
In two case-studies, the authors apply VarNMF to bioinformatics problems.

### Strengths
The per-sample variability increases the modeling space.
The illustration (Fig 1) is insightful (but could be significantly smaller).
The Bayesian factorization model and resulting likelihood has been described well.
The bioinformatics experiments are detailed.
The bioinformatics data has been described in detail (potential candidate for the appendix)

### Weaknesses
1. The related work has not been discussed thoroughly.
The paper only shallowly discusses related work on Poisson Factorization and misses out on Poisson Factor Analysis.
Probabilistic Matrix Factorization and Bayesian Matrix Factorization are not discussed in detail.

2. Given that the authors consider the 'mean signals' of the variable components in their interpretation and analysis, it is not clear if one could achieve the same by using a Bayesian Matrix Factorization and an appropriate prior distribution.
Given the lack of contextualization, it is not clear how novel the proposition is.

3. The paper lacks in experimental contextualization.
That is, the paper does not properly compare VarNMF to the state-of-the-art of direct competitors from Poisson Factorization, Poisson Factor Analysis, or Bayesian Matrix Factorization. Comparing with Hierarchical Poisson Factorization (HPA) as a baseline might be interesting. 
That means, benefits and limitations of VarNMF compared to the state-of-the-art remains unclear.

4. The experimental section lack in breadth.
Since the focus lies on bioinformatics case-studies, it is unclear how VarNMF performs in other domains. 
Considering the interests of our ICLR community, readers probably expect a broader evaluation.
Given the related work, it would be highly beneficial to include experiments on recommender systems or computer vision datasets, such as Movielens, Netflix Prize (which all are ideally suitable for the Poisson Factorization), or Olivetti Faces.

5. The empirical evaluation lacks in depth.
The robustness of VarNMF under noise has not been evaluated.
An empirical evaluation on convergence is not included.
The paper does not compare the runtime of VarNMF to competing methods.
The performance characteristics of VarNMF on a wide range of datasets is unclear.

6. The paper does not include a Reproducibility Statement and the submission does not include a Reproducibility Package.

7. The title of this paper is quite general and does not match the specific contribution of this paper.

### Questions
What is the purpose of \hat{K}-NMF?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The manuscript proposes a probabilistic version of nonnegative matrix factorization (NMF) where the latent factor factors (H or the basis matrix) are assumed to be sampled per data point. This modeling leads to the implementation (by expectation and maximization) that each data point can have its own H-matrix as a parameter. The proposed method is applied to a cfChIP-seq dataset to extract inter-cancer variability.

### Strengths
By relaxing the assumption of standard NMF that the constant components (H matrix) are shared between data points, the proposed factorization model can capture the variations in the signals between data points.

### Weaknesses
1. I am concerned about the high computational complexity of the proposed model (the number of parameters increases with the number of data points).

2. I think that the presentation could be further improved. 
1) The symbols are used inconsistently. For example, the symbol R is used to represent a function and a variable; it seems that there are two indexing systems used for matrices. V[i]_j and H_{k,j} in eq. (2), but H[i] also represents a matrix in eq. (3).
2) From the introduction and the model descriptions, it seems that the proposed method aims at solving source separation or deconvolution. For example, "the objective of NMF is to decompose the mixed single into K sources...  " in Section 2. If this is what the authors intended to claim, the authors should provide rigorous analysis or experiments to prove whether the proposed method can recover the sources from the mixed signals. If not, I think the current presentation is confusing and should be modified. 

3. The manuscript does not include extensive or rigorous experiments with real data. In section 4.3, it is stated that the experiment was done once with the randomly split dataset.

### Questions
Regarding the training/testing experiments in sections 4.2 and 4.3. Each data point has its H[i] matrix. How can we define H[*] for a test point *? 

Have you compared the proposed NMF method with other Bayesian NMF models on the dataset used in the experiments?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
