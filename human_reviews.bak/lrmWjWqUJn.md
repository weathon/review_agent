# Covariate-moderated Empirical Bayes Matrix Factorization

- Decision: Reject
- Scores: 5, 5, 8, 6, 5, 5

## Abstract
Matrix factorization is a fundamental method in statistics and machine learning for inferring and summarizing structure in multivariate data. Modern data sets often come with ``side information'' of various forms (images, text, graphs) that can be leveraged to improve estimation of the underlying structure. However, existing methods that leverage side information are limited in the types of data they can incorporate and rely on specific parametric models. Here, we introduce a novel method for this problem, covariate-moderated empirical Bayes matrix factorization (cEBMF). cEBMF is a modular framework that accepts any type of side information that is processable by a neural network. The cEBMF framework can also accommodate different constraints and assumptions about the factors through the use of different priors and takes an empirical Bayes approach to adapt the priors to the data. We demonstrate the benefits of cEBMF in simulations and in an analysis of spatial transcriptomics data.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a novel approach, covariate-moderated Empirical Bayes Matrix Factorization (cEBMF), which is aimed at enhancing the prediction of missing values in matrix factorization problems where side information can be leveraged flexibly. Compared to existing models, cEBMF allows prior knowledge to be incorporated into its design as a modular component, similar to a neural network module. The parameter estimation in this framework is based on an Empirical Bayes approach, and an estimation algorithm is derived accordingly. Experimental results on three datasets demonstrate that the proposed method achieves better performance.

### Strengths
- The cEBMF model introduces a novel methodology.
- The paper is well-organized and written clearly, with a structured overview of related work that positions the proposed method effectively.
- The code is provided, supporting reproducibility.

### Weaknesses
- There is a lack of comparison with neural network-based approaches. While several models capable of handling side information have been proposed in the literature [1-3], no comparison with these methods is provided.
- The primary contribution of this paper lies in its novel approach to incorporating prior knowledge. While this is valuable within the Bayesian community, its broader impact on the ICLR community, which tends to be more focused on neural network approaches, might not be as apparent.



1. Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: A factorization-machine based neural network for CTR prediction. *Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI)*, 1725–1731. https://doi.org/10.24963/ijcai.2017/239

2. Cheng, H.-T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... & Anil, R. (2016). Wide & deep learning for recommender systems. *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems*, 7–10. https://doi.org/10.1145/2988450.2988454

3. Xiao, J., Ye, H., He, X., Zhang, H., Wu, F., & Chua, T.-S. (2017). Attentional factorization machines: Learning the weight of feature interactions via attention networks. *Proceedings of the 26th ACM International Conference on Information and Knowledge Management (CIKM '17)*, 1019–1028. https://doi.org/10.1145/3132847.3132953

### Questions
- Can the insights gained from this paper be applied to neural network modeling or other machine learning frameworks rather than the Bayesian framework? If so, could you elaborate on potential methods for achieving this?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes an extension of empirical Bayes matrix factorization (EBMF), by incorporating "side information" through an adaptive prior, resulting in a method called covariate-moderated EBMF, or cEBMF. One type of side information discussed in this paper is spatial information, motivated by the recent popular technology, spatial transcriptomics.

### Strengths
1. Matrix factorization is important to the community.

2. Spatial transcriptomics is an interesting application.

3. The cEBMF framework is broader than specific models, such as spatial models, or sparse models.

### Weaknesses
1. It seems that Section 3.2 is incomplete. My understanding is 3.2 discusses the general framework of cEBMF model, followed by some specific examples of side information. Then 3.2.1, as expected, discussed the case where the side information is factor sparsity. As a reader, I was expecting 3.2.2 to be another example, say when side information is spatial, as discussed in both the abstract and the introduction. However, there is no such an subsubsection, and actually 3.2.1 is the only subsubsection of 3.2. Did I miss anything here? 

2. Following the above comment, in 4.1, the simulation settings include sparsity-driven covariate, and tiled-clustering model. However, there is no such a discussion on tiled-clustering model in 3.2. This setting seems a spatial example, so I think it makes more sense to have a section 3.2.2 for it. 

3. The ST analysis need to be improved:

3.1 There is no quantitative metrics comparing methods, for example, Adjusted Rand Index, among others. 

3.2 There are more competitors to be included, like those discussed in the introduction. One obvious example is NSF by Townes and Engelhardt. 

3.3 Further results on other slides in Maynard dataset, shown in Figure 10 and 11, don't seem to support the claim that "cEBNMF tended to produce the largest improvements in accuracy". I might be wrong since this is purely visualization, which again requires some more quantitative scores to better compare the performance. 


Minor issues:

Line 112-113, the citation Gopalan et al. (2014) should be in (), say \citep instead of \cite. 

The order of methods in Figure 4, 9, 10, 11 are different. 

In ST, "slide(s)" are used more frequently than "slice(s)" in my understanding. I personally prefer "slides", but I don't have any strong opinion on this. However, at least it should be consistent within this manuscript. A simple search would find 10 "slices", and two "slides".

### Questions
These are somewhat repetitive. 

1. Can the authors extend 3.2, by adding more settings under which side information can be used (and how).

2. Since one major selling point of cEBMF is the flexibility in incorporating side information, is there any other examples in addition to sparsity and spatial information? If that's all, one potential argument why can't I just use sparse methods and/or spatial methods?

3. Can the authors improve the ST analysis in section 4.2? I mean to include more competitors and quantitative scores. Once the scores are included, it is easier for the audience to compare the methods, over all slides in the Maynard dataset.

### Soundness
2

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
The paper proposed a new matrix factorization method with covariates for rows/columns included in the model. It generalizes previous studies on nonnegative matrix factorization and empirical Bayes matrix factorization. The proposed model can utilize any side information that can be included in a probabilistic model with the flexibility of little to no assumption on the factors.

### Strengths
1. Treating side information through empirical Bayes in the matrix factorization problem is novel. 
2. The paper provides clear derivations and algorithms for the proposed methodology. Part of the result is justified in the appendix.

### Weaknesses
My major concern is the parametric prior assumption in Eq. (4). It can easily overfit the model given the flexibility in Eq. (4). Although the authors mentioned the overfitting problem in the limitation section, it cannot be overlooked.  Details are provided in the Questions.

### Questions
1. The parametric assumption in Eq. (4)  is confusing --- it can easily lead to an overfitted model if the index of the distribution family is arbitrarily related to $\mathbf{x}\_i$. More constraints should be given to control the model complexity. 
For example, it could be $\ell_{ij}\sim g_k^{(l)}(q_i)\in \mathcal G_{l,k}$ with $q_i=q_i(\textbf{x}_i)\in\mathcal Q$, where restriction on the function family $\mathcal Q$ helps control the complexity. Actually, the authors are doing this in later illustrations. For example, in the spike-and-slab prior example in Sec. 3.2.1, $q_i$ is a logistic function of $\mathbf{x}_i$. Further discussion on the choices of the parametric families and the constraints on controlling the model complexity should be provided. 
2. Does the proposed model assume a known number of factors $K$? If not, how should $K$ be determined?
3. If $\mathbf{Z}$ is a symmetric matrix (e.g. covariance) such that $\mathbf{L}=\mathbf{F}$, is there any change to the current process?
4. For the example in Sec 4.2, the side information $\mathbf{X}$ is the genre of the movies. If every movie belongs to exactly one genre (correct me if I was wrong), then the empirical prior appears to be a hierarchical prior (19 priors with parameters from a common hyperprior). Then I don't see why the proposed model is needed. 
5. For the example in Sec 4.2, what is the side information for columns, i.e. $\mathbf{Y}$?
6. Other minor typo/writing issues:
    1. In Eq. (3), $\ell_{ik}$ and $f_{jk}$ are not defined.
    2. In Eq. (5), $\mathbf{\omega}$ is not defined.
    3. In Eqs. (14) and (16), should Eq. (14) be $\overline{\mathbf{R}}^k = \overline{\mathbf{R}} + \overline{\mathbf{\ell}}_k\overline{\mathbf{f}}_k^T$ and Eq. (16) be the other way?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a modular framework for empirical Bayes matrix factorization that can leverage a large variety of models, and can use families of priors that are flexible in form to accommodate different assumptions and constraints, and allows automatic selection of the hyperparameters. Various experiments are conducted to illustrate the effectiveness.

### Strengths
The proposed framework is modular, flexible, and general, encompassing many previous studies as special cases. The presentation of the core ideas and algorithms is clear, supported by the appendix. The experimental results sufficiently demonstrate the method's effectiveness. The paper provides a well-balanced discussion of the approach's strengths and limitations.

### Weaknesses
The overall technical contribution appears moderate, as the generalization from MFAI seems straightforward. The key challenges in model formulation and algorithm design could be better articulated.

### Questions
It would be helpful if the authors could elaborate on the specific obstacles encountered in extending MFAI.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a covariate-moderated empirical Bayes matrix factorization, which borrows information from the side information. The side information is modeled as a prior information. Comparisons are made to some competitors through simulations and real data analysis.

### Strengths
Paper is well-written with detailed introduction on empirical bayes and clarifications on details. 

The whole paper is easy to read and technical parts are easy to follow.

### Weaknesses
All the empirical studies are simple. The dimensions of the simulated data are not large. 1000*200 matrix Z is not enough to show the effectiveness. Even the real datasets are relatively small.

### Questions
The side information can be treated as other views for learning, which is multi-view learning. There are tons of papers about this. It can also be treated as source data in transfer learning. Authors need to discuss why the side information is modeled into the prior as a preferred way. 

Authors mentioned on Line 183 that it can leverage many models. Can authors tell more about the details? Do authors mean that the prior information can be found via modeling? 

How is the rank determined in the paper? It is very difficult to identify it. Also, in the empirical studies, K is treated as known in the simulation. In MovieLens, how all methods choose K based on complexity of data (Line 469)? In DLPFC, I see most people use K=15. If doing clustering, authors need to talk about choosing the number of clusters.....

In Figure 2, if I am right, the uninformative covariate should have worse results than sparsity-driven covariate, but they seem to have better results. Shifted tiled-clustering should have worse results, but MFAI got better. Can authors clarify these?

I used the DLPFC dataset multiple times and it may not be a good way to present it here. Literature about spatial transcriptomics focuses a lot on incorporating the spatial information, e.g., the empirical Bayes model for spatial transcriptomics. How is the spatial information used in this method? People typically use it for clustering, so authors may show the clustering results, ARI etc. Many methods haven been demonstrate on this dataset as well.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The work deals with a Bayesian matrix factorization (BMF) framework that can accommodate the side information using a parametrized representation, e.g., neural network. Such side information allows the framework to handle any prior of the latent in flexible manner. The work extends the EBMF framework proposed in [Wang and Stephens, 2021; Willwersheid, 2021)] along with the classic normal means model. Experiments with both simulated data and real data are presented to support the claims.

### Strengths
Strengths:

1.	The wok is an interesting combination of connecting classical models with more expressive models in deep learning.

2.	Real data experiments especially the genomic data is insightful and the performance of the proposed approach looks reasonable.

### Weaknesses
Weakness:

1.	The discussions and sections could have been organized better. I think, many important technical details and discussions are moved supplementary, which makes hard to verify the technical soundness and the reasoning.

2.	Some relevant baselines are missing from discussion and empirical study.

### Questions
1.	In the introduction, a figure or a clear description could have helped what are the side information in the context of the described genomic data.

2.	It is commented that “Further, there are sometimes benefits to not making strong assumptions about the spatial organization of the data even when we know the data are spatial.”. This is a bit unclear statement as spatial PCA seems perform well in the real data experiments in Figure 4. Could you clarify this statement?

3.	Figure 1 is hard to understand the clustering performance of the different methods. Instead of PCs, it would have been more natural to compare the clustering of the original points in x-y domain. Also, how does different prior assumptions.

4.	What are the connections with deep matrix factorization frameworks? There exists some works by directly modeling the latent using the deep architectures with prior information, e.g.,
a.	Wang, Jianyu, and Xiao-Lei Zhang. "Deep NMF topic modeling." Neurocomputing 515 (2023): 157-173.
b.	Xue, Hong-Jian, et al. "Deep matrix factorization models for recommender systems." IJCAI. Vol. 17. 2017.
And the related references.
These types of frameworks are non-Bayesian, without making any distributional assumptions on the prior. I think, they are very related to the proposed method and could not find any discussion or empirical experiments in the paper.

5.	It is unclear how does the constraints are handled in this case, for e.g., nonnegative constraints in Eq. (15) in the algorithm design. 

6.	Also, if there are missing side information (i.e., matrix entry of Z is present, but side information from X is missing), will the method be able to handle it?

7.	“In practice, the full posterior q is not needed; the first and second posterior moments are sufficient”. However, looking at Eq. (12), it seems that the full posterior is needed (notations are also confusing here, $p$ vs $q$). It is unclear how does this translate to the moments of  $\ell_{ik}$ and $f_{ik}$. Are there for a specific family of distributions?

8.	In the real data experiments with collaborative filtering, there exists side information for only columns. In that case, how does the algorithm handle it? In spatial transcriptomics data, no side information is specified. Spatial PCA seems performing well in the real data. What about the runtime performance of the proposed algorithm and how does it compared to the competing baselines?

9.	I feel the series of equations 31-36 has some issues. I doubt if (31) and (32) are equal, which then questions the correctness of the remaining equations. Also, the distribution $q_{\beta}$ is not defined properly. As the details are missing in the main paper, the soundness of algorithm is hard to verify. Please add more clarity to the algorithm design in the main text. 

10.	Any insights about identifiability of this approach which is a key consideration in matrix factorization-based models?

11.	Minor comments:

a.	Some typos: Notation in (7). Notation confusion: side information is notated using $\bm x_i$ or $\bm y_i$, but then $\bm d_i$ is used in (9)

b.	“D is an invertible diagonal matrix” in Page 4. D can be any invertible matrix

### Soundness
2

### Presentation
2

### Contribution
2
