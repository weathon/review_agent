# Conditional Diffusion Models are Minimax-Optimal and Manifold-Adaptive for Conditional Distribution Estimation

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
We consider a class of conditional forward-backward diffusion models for conditional generative modeling, that is, generating new data given a covariate (or control variable). To formally study the theoretical properties of these conditional generative models, we adopt a statistical framework of distribution regression to characterize the large sample properties of the conditional distribution estimators induced by these conditional forward-backward diffusion models. Here, the conditional distribution of data is assumed to smoothly change over the covariate. In particular, our derived convergence rate is minimax-optimal under the total variation metric within the regimes covered by the existing literature. Additionally, we extend our theory by allowing both the data and the covariate variable to potentially admit a low-dimensional manifold structure. In this scenario, we demonstrate that the conditional forward-backward diffusion model can adapt to both manifold structures, meaning that the derived estimation error bound (under the Wasserstein metric) depends only on the intrinsic dimensionalities of the data and the covariate.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper uses distribution regression to estimate the conditional distribution induced by conditional forward-backward diffusion models. Authors utilize neural network to estimate the key points and also allow for high-dimensional X and Y with low-dimensional structures. Solid theoretical results are provided.

### Strengths
1. Authors introduce SDE in a simple way and made it easy to connect SDE to distribution regression.

2. The paper extends the unconditional distribution estimator to the conditional one and provides theoretical results. Theories are given in an elegant way that they clearly explain the connections to the theories of unconditional distribution estimator, mean regression problems, and nonparametric regression under l2 error. Theoretical results are interesting and inspiring.

### Weaknesses
The paper mentions that it provides practical guidance to design the neural network, including the network size, smoothness level and so on. Though it is a theoretical work, some empirical results would significantly demonstrate the usefulness of the paper. I would recommend authors run some simulations to show how the theoretical results guide the design of neural networks. For example, under what scenarios, the errors are optimally controlled?

The theoretical results are interesting.  Can the authors highlight the difficulties and summarize the new ways of proof if there are? This could be a huge contribution to the statistical learning community. I didn't see the challenges from my end and all seem to be standard.

### Questions
What is p_data on Line 122?

Authors need to check all the parenthetical and narrative citations throughout the whole paper. Many were mis-used. For example, Line 316 should be parenthetical citation.

Definition of neural network class is already given in Tang and Yang, 2024 (AISTATS). Authors need to reference it or directly call it from the citation.

In remark 1, can authors discuss the case when alpha_X>1? What does this mean to the minimax rate?

In Remark 2, can the authors clarify why D_Y need to satisfy this condition to have the first term dominate the second one? 




Rong Tang and Yun Yang. Adaptivity of diffusion models to manifold structures. 27th International Conference on Artificial Intelligence and Statistics, 2024.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- The paper presents a derivation of the estimation error bound for the diffusion models with sampling guided by the covariate regressor. The authors closely build upon the methodology from Oko et al 2023 https://proceedings.mlr.press/v202/oko23a/oko23a.pdf who explored the statistical learning theory for diffusion models via score matching estimators and proved nearly minimax optimal estimation rates in total variation and Wasserstein metric of order 1 for fully connected neural-networks (including suggestions on generalisation to U-nets and ConvNets). 

- The presented paper aims to introduce the statistical learning theory for conditional diffusion models using the distribution regression of the covariate structure guiding the sampling. Authors investigate the properties of the estimators using the finite sample analysis of the statistical error bound under various metrics, the relations of latent lower dimensional data structures and the smoothness properties of the data. 

- The main contribution is: 
   1) Given the assumptions on the size of the architecture,  the estimation error bound of the conditional diffusion estimator is minimax optimal under total variation and Wasserstein distance of order 1, authors include the discussion on the generalization to the unconditional case as a special case, 
   2) if there is a low-dimensional latent space structure within the high-dimensional data manifold (say within the feature space or feature and label space), the resulting conditional distribution estimators adapts to the lower-dimensional latent space.

### Strengths
- The paper is mathematically well written. The notation is consistent and precise. Definitions and proofs are provided in the extensive Appendix. These can be useful for deriving and investigating other novel properties and algorithms.
- The problem is relevant and well chosen and the previous work is well mapped. 
- There is an interesting potential applications for measuring the quality of the conditional diffusion estimators and guided  sampling. The paper also  provides a background for the reasoning about the selection of the model architecture.
- The interesting contribution is in analyzing the latent dimension representations of the data and the adaptability of the higher dimensional estimator.
- The results provide insight into the impact of the smoothness of the underlying data manifold on the performance of the estimator.

### Weaknesses
1) Paper is missing motivational examples and practical experiments to demonstrate how the presented theory can be used on real-life problems. The results are theoretically interesting and there is no issue with the soundness of the arguments. Nevertheless, without a practical demonstration on how to use the derived results, the main contribution of the paper may be harder to access to the wider machine learning community.

2) While it is great to see the methodology related to the proofs as a part of the main paper, it would be great to see particular direct applications in the main body of the paper.

3) The usual approach to estimate the gradients of log p_t(y_t|y) (see line 200) is to use the gradient of unconditional model and add gradient of log-likelihood of the ‘reverse predictor' which detects ‘the covariate values’ x based on the features constructed from whom the unconditional values ‘y_t’. Authors decided to use marginal forward diffusion which results in empirical estimation of the joint density between the covariate X (the guiding covariates) and Y (the unconditional diffusion). The discussion on incorporating computational efficiency of this approach in comparison to the 'reverse predictor' approach is missing.
4) The proven Theorems assume the initial constraints of the sizes of the architecture such as depth, width, norm of the parameters, etc. These constraints are set as functionals of the order of magnitude of the data sample $n$. It is not well discussed what happens if these constraints are not satisfied.

### Questions
1) Are the covariates X also located in the Besov space support?
2) The marginals M_X and M_Y are assumed to be  [-1,1]  D-X, resp. D_Y dimensional cube. How does the scaling of the data to the [-1, 1] cube impact some other intrinsic features, e.g. latent translations? What type of scaling transformations would you recommend?
3) Can you provide guidelines on how you estimate $\alpha_X$ and $\alpha_Y$ or do you assume these values?
4) How does presented results in Theorem 2 relate to latent diffusion models?

5) p.7 Remark 1: Does this remark imply that the kernel estimator can well approximate the performance of the conditional diffusion based estimator?

6) It would be great if authors can  include experiments for different embedded lower dimensional structures, and demonstrate how the presented theory supports and navigates the experimentally obtained results. Using even simple synthetic geometric objects would be great.

7) Can you provide more discussion on the implications of the different regimes discussed in Remark 3? In particular how would you select the smoothness parameters $\alpha_x$ to investigate the transition phases?

8 )   How does Lemma 2 helps to assess the generalization error?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Previous work has studied the min-max optimality of score-based diffusion models in estimating the density of an unknown distribution, and in estimating the intrinsic dimension of data lying on a submanifold of a sample space. 

This work extends these results to the conditional modelling setting, where a conditioning variate changes the distribution being modeled. The distribution is assumed to vary smoothly with the conditioning variate.

These results show that very similar rates of convergence of the total variation and Wasserstein distances between the true and  estimated data distributions hold to the unconditional modeling setting, with additional terms depending on the dimensionality of the conditioning variate. In the limit of the conditioning variate having dimension zero, the same rates as previous works on the unconditional setting are obtained.

### Strengths
- The contributions are, while possibly not surprising, are meaningful and interesting results extending the theoretical analysis of diffusion models.
- The paper is generally very well written and easy to follow, given the technical nature of the content. In particular the remarks on the theorems make understanding the implications of them more accessible.

### Weaknesses
- The practical implications of the theoretical work presented are not completely clear to me. The authors mention that the results suggest how to design networks, but from my understanding on theorem 2, this suggests simply that larger networks produce better bounds?
- The implications of assumption D are unclear to me. The variable $r_0$ is undefined. Is this condition meant to hold for any $r_0$? If so then this implies a need for the decoding to be smooth globally, with I believe would limit topological structure of the low-dimension data manifold. If this is local condition, the is this not a consequence of the smooth manifold structure of $\mathcal{M}$?

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2
