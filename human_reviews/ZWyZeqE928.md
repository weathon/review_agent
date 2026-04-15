# Functional Bayesian Tucker Decomposition for Continuous-indexed Tensor Data

- Decision: Accept (poster)
- Scores: 5, 6, 8, 5

## Abstract
Tucker decomposition is a powerful tensor model to handle multi-aspect data. It demonstrates the low-rank property by decomposing the grid-structured data as interactions between a core tensor and a set of object representations (factors).  A fundamental assumption of such decomposition is that there are finite objects in each aspect or mode, corresponding to discrete indexes of data entries. However,  real-world data is often not naturally posed in this setting. For example, geographic data is represented as continuous indexes of latitude and longitude coordinates, and cannot fit tensor models directly. To generalize Tucker decomposition to such scenarios, we propose Functional Bayesian Tucker Decomposition (FunBaT). We treat the continuous-indexed data as the interaction between the Tucker core and a group of latent functions. We use Gaussian processes (GP)  as functional priors to model the latent functions. Then, we convert each GP into a state-space prior by constructing an equivalent stochastic differential equation (SDE)  to reduce computational cost. An efficient inference algorithm is developed for scalable posterior approximation based on advanced message-passing techniques. The advantage of our method is shown in both synthetic data and several real-world applications. We release the code of FunBaT at {https://github.com/xuangu-fang/Functional-Bayesian-Tucker-Decomposition}.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces the Functional Bayesian Tucker Decomposition (FunBaT)
algorithm. Instead of treating the factors of the Tucker decomposition as
matrices with discrete row indexing, the authors instead think of the factors
as latent Gaussian process function priors. Earlier work in this area of
[Schmidt, ICML 2009] considers a simpler "continuous CP decomposition" for
tensors.
The authors then make connections
to stochastic differential equations (SDEs) and conditional expectation
propagation (CEP) for the learning task, which is somewhat analogous to a
continuous version of alternating least squares. The authors provide experiments
on synthetic and real-world datasets, of order $K=2$ and $K=3$ respectively.

### Strengths
- Extends work from Tucker to CP, since CP is a generalization of Tucker if you
  restrict the core tensor to be diagonal
- A "continuous" version of Tucker decomposition on the factors is an excellent
  idea to explore
- Uses interesting real-world datasets with continuous features (indices), and
  the results are comprehensive and compelling. That said, the US-TEMP
  experiment could be improved since a core shape of $(1,1,1)$ is used, which
  means a product of the factor functions is being learned.

### Weaknesses
- There is not enough emphasis on related work, especially with [Fang et al.,
  ICML 2022], which appears to be a *very similar paper*. This discussion should happen
  much earlier in the paper, and there should be more than one mention of this
  Fang et al. (2022), as it it almost gets buried in its current form. Missing
  seminal work:
  * "Bayesian Tensor Regression" by [Guhaniyogi et al., JMLR 2017]
- While the synthetic experiments have the potential to be very strong and
  compelling, it would be useful to see how the tensor reconstruction looks as a
  function of the number of samples. It may not be surprising in its current
  form that we can recover ground truth with $650$ samples.
- In the Beijing Air experiments, please explain if non-uniform core shapes were
  explored (as they might better help fit the data). A recent ICML paper
  investigates core shape selection and could be of interest:
  * "Approximately Optimal Core Shapes for Tensor Decompositions" by [Ghadiri
    et al., ICML 2023]

### Questions
### Questions
- [page 01] Is Tucker decomposition really a more compact low-rank
  representation? Maybe the factor matrices can be more compact, but doesn't
  this come at the cost of a larger "core structure" then, e.g., CP
  decomposition?
- [page 03] By "successive derivatives of $m$-th order", do you instead mean to
  write $\frac{d^m f(x)}{dx^{m}}$?
- [page 03] "the trained model cannot handle new objects with never-seen
  indices" -- this is not necessarily true, and is the core idea behind tensor
  completion.
- [page 04] Why do you use $\tau^{-1}$ as Gaussian noise (i.e., variance)
  instead of $\sigma^2$?
- [page 06] Is the preset (scalar) mode rank $R$ for a given Tucker core of
  shape $(r_1, \dots, r_K)$ the product? I.e., $R = \prod_{i=1}^K r_i$? If so,
  this should be stated explicitly and you need to remove the claim that "the
  linear costs of both time and space ..." since $R$ is exponential in the order
  $K$.

### Typos and suggestions
- [page 01] "there were finite" --> "there are finite"
- [page 01] "tensor train(TT)" --> "tensor train (TT)"
- [page 01] "Tucker decomposition is famous for ..." --> "Tucker decomposition
  is widely-used for ..."
- [page 01] "and time.." --> "and time."
- [page 01] "or simpler CP format" --> "or the simpler CP format"
- [page 01] "which is with more compact ..." --> "which is a more compact ..."
- [page 02] Inconsistent subheading capitalization (see ICLR style guide and
  example paper)
- [page 02] "under the following settings" --> "under the following setting"
- [page 02] typo at end of sentence with "groups of latent factors" (should be
  a comma instead of a period)
- [page 02] suggestion: consider superscripting factor matrices/vectors as
  ${U}^{(k)}$ instead of ${U}^k$ to remove any ambiguity in meaning
- [page 02] "The classic CANDECOMP/PARAFAC (CP)" --> "The classic CP" (you
  already introduced the acronym)
- [page 02] missing space: factors: $y_{i}$ (same for many other sentences on
  this page)
- [page 03] "denote as $f \sim$" --> "denoted as $f \sim$"
- [page 03] consider using $\ell$ instead of $l$ as a hyperparameter in the
  Matern kernel
- [page 03] missing space: "efficient $O(n)$ inference"
- [page 03] No need to give the meaning of "FunBaT" again
- [page 04] "preset latent rank:$\{r_1,\dots,r_K\}$ --> "preset latent rank
  $(r_1, \dots, r_K)$. Missing space between subsequent sentences.
- [page 04] suggestion: when you shift to "continuous-indexed tensors", it may
  be more clear to say that $(i_1^{n},\dots, i_K^{n}) \in \mathbb{R}^{K}$
- [page 04] missing space: "Namely, "
- [page 05] typo: "massage merging" --> "message merging"
- [page 06] punctuation typos in Algorithm 1 inputs
- [page 06] suggestion: Substantially more discussion / math can be given to
  FunBaT-CP. The focus of the paper is on Tucker decomposition, but this is a
  very nice special case for which it would be nice to know more of its
  properties.
- [page 06] Missing space: "function factorization is(Schmidt, 2009)." along
  with other citations in this paragraph.
- [page 07] typo: "alternating least square" --> "alternating least squares".
  Many more typos / punctuation errors in this paragraph.
- [page 09] typo: "lantide mode"

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed a function Tucker decomposition for tensors with continuous-valued indices. The authors firstly use a Gaussian process to map indices to Tucker factors and then contract these factors to obtain the entry value. To efficiently learn the GP prior, an algorithm based on state-space GP and expectation propagation is derived. For experiments, the authors test the model on synthetic data and several spatiotemporal data imputation tasks.

### Strengths
The authors study the functional tensor decomposition for continuously-indexed tensors. This seems to be an interesting and novel topic in the field tensor decomposition and may have some new applications.

### Weaknesses
1. The setting of continuously-indexed tensors is new in the community of tensor decomposition. However, my main concern is how this task is related and different from traditional regression tasks. Why do we need such construction of tensors and what is the significance of the proposed function Tucker decomposition. Considering the experiments, the authors show applications in spatiotemporal data imputation. However, there are many existing methods, including interpolation like GP, VAE [1], GAN [2], LSTM [3], diffusion models [4] and many of their variants. Besides, the problem setting is very similar to CPNN [5]. For a better understanding of the paper, I think it might be better to have a discussion of these lines of works and empirical comparisons
- [1]. Mattei, & Frellsen. (2019). MIWAE: Deep generative modelling and imputation of incomplete data sets. ICML.
- [2]. Yoon, et al. (2018). Gain: Missing data imputation using generative adversarial nets. ICML.
- [3]. Cao, et al. (2018). Brits: Bidirectional recurrent imputation for time series. NeurIPS
- [4]. Tashiro, et al. (2021). Csdi: Conditional score-based diffusion models for probabilistic time series imputation. NeurIPS.
- [5]. Tancik, et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. NeurIPS.

2. Since the proposed model employs nonlinear GPs, it might be better to compare with some nonlinear or GP-based tensor decompositions. Also, baselines for continuous-time tensor decompositions are also good choices, as the authors also mentioned in the related work.

### Questions
1. Compared with GP regression, the main difference of the proposed model is preserving the Tucker structure in the final layer. I am wondering why this construction is so helpful as shown in empirical results. 
2. There are matrix inversion and multiplications in the update rules. Why is the time complexity linear with the rank $R$? 
3. The authors adopted three resolution settings for BeijingAir datasets. What is the setting for US-TEMP?

typo: In paragraph above Figure 1, whore tensor -> whole tensor?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed an Bayesian method for tensor Tucker decomposition, where the data are assumed to have continues index. Each column of the factor matrices is modeled using Gaussian process (GP), and efficient method was proposed to infer the parameters of the GP. The idea of this paper is quite interesting,
and the experimental results are convincing.

### Strengths
1, the idea that model factor matrices using GP is new.
2, efficient inference methods are proposed to determine the unknown parameters.

### Weaknesses
1, the methods need to set the rank of the tensor manually, which is usually unknown in practice.
3, In table 1, the MSE of the proposed method much lower than the baselines. But no explanation is provided to show why.

### Questions
1, Does the equation 4 implies the smoothness of z(x)?
2, Compared to Tucker decomposition, CP decomposition offers several advantageous properties, particularly its uniqueness, making it more favored in many applications. However, it is worth noting that CP decomposition typically requires larger factor matrices compared to Tucker decomposition. Therefore, it is interesting to discuss the computational complexity of the proposed method in relation to the rank of the tensor and evaluate its applicability in scenarios that require setting a large rank.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a method called Functional Bayesian Tucker Decomposition (FunBaT) to handle continuous-indexed tensor data. Traditional tensor decomposition methods are designed for discrete and finite-dimensional indexes, but real-world data often contains continuous indexes such as geographic coordinates. FunBaT solves this problem by treating the continuous-indexed data as the interaction between the Tucker core and a group of latent functions. Gaussian processes are used as functional priors to model the latent functions, and an equivalent stochastic differential equation is used to reduce computational cost. The paper introduces an efficient inference algorithm based on advanced message-passing techniques. FunBaT outperforms existing methods in synthetic and real-world applications while being able to identify interpretable patterns. The paper also provides explanations on tensor decomposition, function factorization, the use of Gaussian processes as state-space models, and the FunBaT model and algorithm. Overall, the paper offers a solution for tensor decomposition on continuous-indexed tensor data.

### Strengths
1. Extension to continuous-indexed tensor data: The paper proposes a method called Functional Bayesian Tucker Decomposition (FunBaT) that extends traditional Tucker decomposition to handle continuous-indexed tensor data. 
2. Utilization of Gaussian processes as functional priors: FunBaT models the latent functions in continuous-indexed data using Gaussian processes as functional priors. This allows for flexible and efficient modeling of the latent functions. Additionally, the paper reduces computational cost by constructing an equivalent stochastic differential equation, further enhancing the efficiency of the method.
3. Improved performance and interpretable patterns: FunBaT is demonstrated to outperform existing methods in both synthetic and real-world applications. It not only achieves lower prediction errors but also has the ability to identify interpretable patterns in the data.

### Weaknesses
1. The paper's innovation and significance may not be convincingly conveyed. The idea of using Gaussian processes to model N factor matrices for continuous data appears to be straightforward and intuitive. Furthermore, the paper's modeling approach is very similar to that of reference [3], which employs Gaussian processes to model core tensors. 
2. There are now many function decomposition models based on Tucker decomposition, e.g., [1,2], so the Introduction section regarding related works appears to be inaccurate.
3. Furthermore, the significance of this paper isn't fully articulated. Merely stating that there is currently no function Tucker decomposition doesn't seem to provide a sufficiently compelling reason for its importance.
4. Regarding the optimization algorithm, this paper is also closely related to some previous work, e.g., [3]
[1] M.Imaizumi, K.Hayashi (2017). "Tensor Decomposition with Smoothness". PMLR: International Conference on Machine Learning 2017
[2] Luo Y, Zhao X, Li Z, et al. Low-Rank Tensor Function Representation for Multi-Dimensional Data Recovery[J]. arXiv preprint arXiv:2212.00262, 2022.
[3] Fang S, Narayan A, Kirby R, et al. Bayesian Continuous-Time Tucker Decomposition[C]//International Conference on Machine Learning. PMLR, 2022: 6235-6245.

### Questions
1. What are the advantages of this function tensor decomposition method compared to existing approaches? Utilizing the Tucker decomposition allows for the representation of higher-order tensors and effectively mitigates the problem of the curse of dimensionality. Why does the Tucker decomposition lead to a more compact and flexible low-rank representation compared with TT and CP?
2. What is the primary motivation behind this paper in comparison to other function CP, Tucker, and TT decompositions? If this method was established solely because GP-based functional Tucker decomposition does not currently exist, I believe the motivation appears to be rather weak.
3. What is the primary innovation of this paper? Apart from considering continuous tensor indices in the modeling, is there a deeper level of differences compared to Bayesian Continuous Tucker Decomposition?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
