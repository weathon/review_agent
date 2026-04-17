# An Efficient Variational Method for Fitting Log-Gaussian Cox Processes

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Log-Gaussian Cox Processes (LGCP) have been widely used for modeling spatial point patterns. However, fitting LGCP is computationally challenging due to a nested structure involving Poisson process and latent Gaussian random field. To address these issues, we first approximate the intractable LGCP likelihood  based on the Voronoi tessellation method. Then, using variational Gaussian approximation, we transform the problem of fitting LGCP into maximizing the evidence lower bound which admits an explicit expression. We design a novel coordinate ascent maximization algorithm which updates the parameter blocks by Newton method and fixed-point method, respectively. To further enhance the computational efficiency, we adopt a nearest neighbor Gaussian process as the prior for the latent Gaussian random field, and the cost of inverting large covariance matrices is greatly reduced via the Woodbury formula. Theoretically, we prove the existence and uniqueness of the optimal solution to the strongly concave objective function, and the convergence of the proposed algorithm is established. Numerical results demonstrate the computational and inferential benefits of our method in modeling log-intensity surface over competing methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an efficient variational method for fitting log-Gaussian Cox processes (LGCPs). The intractable integral term in the LGCP likelihood is approximated using a Voronoi tessellation approach, and a variational Bayesian framework with a Gaussian proposal distribution is developed. The resulting evidence lower bound (ELBO) is proven to be strongly concave and to admit a unique maximizer. A coordinate ascent algorithm that combines Newton and fixed-point updates, together with a nearest-neighbor Gaussian process and the Woodbury matrix identity, ensures scalable model fitting with respect to both data size and the Voronoi discretization level. Experiments on simulated and real spatial data demonstrate that the proposed method achieves superior predictive accuracy compared with conventional alternatives.

### Strengths
- Since log-Gaussian Cox processes (LGCPs) are widely used models for analyzing spatial event data, a concave formulation for LGCP has significant practical value.
- The paper carefully develops an efficient algorithm for parameter and hyperparameter learning by effectively leveraging existing ideas.
- The manuscript is clearly written and easy to follow.

### Weaknesses
- In the literature of point processes, approximating the integral in the likelihood function by a summation has a long history. Common approaches include the use of uniform grids and Monte Carlo integration, and the Voronoi tessellation method can be regarded as a variant of these. As noted in the paper, this approach has some desirable convergence properties, but the idea itself is not particularly novel: it appears to be a straightforward application of existing techniques. If applying Voronoi tessellation to the log-Gaussian Cox process offers substantial advantages or unique insights, these should be clearly articulated.
- I appreciate the careful construction of an efficient algorithm for model fitting. However, the techniques employed (e.g., low-rank approximations of large matrices) are based on conventional ideas in GP literature and are not non-trivial in themselves. 
- The positioning of the proposed method relative to prior research is ambiguous. In general, when considering spatial point process models, the problem formulation and objectives differ substantially depending on whether or not spatial covariates are assumed to exist. In the related work section, references to these two settings are discussed without a clear distinction, leaving the reader uncertain about how the authors intend to position their proposed model. 
    - If the authors tackle the problem of estimating the relationship between covariates and the intensity function, then prior studies addressing this problem (e.g., [1,2,3]), as well as INLA, should also be reviewed and compared.
    - Alternatively, if the authors consider that spatial covariates are not mandatory, then the comparative discussion about efficient Gaussian Cox processes with the square [4,5] and sigmoidal (e.g., [6,7]) link functions is necessary. Especially, if the reason for focusing on the somewhat classical log-GCP lies in its high predictive accuracy, and the main contribution of this study is to further improve that accuracy, then it would be valuable to include several recent GCP models [4,5,6,7] (not necessarily exhaustively) as baselines and discuss differences in both predictive performance and computational efficiency.

[1] Baddely et al., “Nonparametric estimation of the dependence of a spatial point process on spatial covariates”, Statistics and Its Interface, 2012.

[2] Yu and Loh, “Bayesian semiparametric intensity estimation for inhomogeneous spatial point processes”, Biometrics, 2011.

[3] Kim et al., “Fast Bayesian estimation of point process intensity as function of covariates”, NeurIPS, 2022.

[4] Lloyd et al., “Variational Inference for Gaussian Process Modulated Poisson Processes”, ICML, 2015.

[5] John and Hensman, “Large-Scale Cox Process Inference using Variational Fourier Features”, ICML, 2018.

[6] Donner and Opper, “Efficient Bayesian Inference of Sigmoidal Gaussian Cox Processes”, JMLR, 2018.

[7] Aglietti et al., “Structured Variational Inference in Continuous Cox Process Models”, NeurIPS, 2019.

### Questions
see strengths/weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes an efficient variational inference method for Log-Gaussian Cox Processes (LGCPs). Instead of relying on expensive MCMC, the authors use a Voronoi-based Poisson likelihood approximation, a Gaussian variational family, and a coordinate-ascent optimization scheme. They provide theoretical guarantees, including concavity, existence/uniqueness of the optimum, and convergence. Further computational speedups are achieved using nearest-neighbor GP approximations and matrix identities. Experiments on synthetic and real datasets show comparable or superior accuracy to existing LGCP methods with significantly improved efficiency and robustness to missing spatial regions.

### Strengths
* The paper contributed to scalable Bayesian spatial modeling, with a novel combination of Voronoi approximation + VI + NNGP for LGCP inference.
* The paper has solid theoretical foundations with clear convergence and optimality guarantees.
* The paper provided a practical algorithm that is computationally efficient and implementable.

### Weaknesses
1. Some theoretical sections are dense, and intuition can be made more straightforward.
2. Limited discussion of performance on very large-scale datasets or high dimensions.
3. Comparisons focus mostly on classical spatial models, but deep generative point-process baselines are absent.

### Questions
* How scalable is the approach to very large spatial domains or high-resolution grids?
* Can the method be extended to spatiotemporal LGCPs or non-stationary kernels?
* How sensitive is performance to choices in Voronoi partitioning and variational family?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors focus on modeling spatial point patterns using a Log-Gaussian Cox Process (LGCP). In this model, the point pattern $Y$ follows an inhomogeneous Poisson process, conditional on an intensity surface $\lambda(s)$. The logarithm of this intensity surface is modelled as the sum of a linear predictor and a latent Gaussian random field: 

$$log \lambda (s) = X(s)\beta + Z(s); Z(s)\sim GP(0, K_{\theta}(s,s'))$$

This doubly stochastic, hierarchical construction provides the LGCP with greater flexibility than standard inhomogeneous Poisson models.
- The primary challenge lies in fitting the LGCP and inferring the log-intensity surface, $\lambda(s)$, which models the expected number of points per unit area.
- A typical approach to this problem is using Markov Chain Monte Carlo (MCMC) methods. However, these methods are often burdened by substantial computational costs and difficulties in assessing convergence, limiting their scalability. Other leading approaches include Integrated Nested Laplace Approximation (INLA) and variational inference methods.The authors of this paper have chosen to develop a variational inference framework to better fit the LGCP model.

The key ideas used in the paper are:  
1. authors pre-select n auxiliary integration points within the observation area on the top of the given N observations; using Delaunay triangulation, the authors construct Voronoi tesselation; authors hypothesise that this leads to more accurate and efficient inference of LGCP.

2. The artificial extension creates computational burden as ELBO evaluation is carried on (n+N) points; authors propose to speed up the computation by approximating the Gaussian field in the intensity model via Nearest-Neighbour Gaussian Process, there the full conditional distribution is replaced by the the distribution conditioned only on the set of the nearest neighbours. 

3. Authors discuss in detail how to apply Newton Optimisation and how to update the variational covariance estimate in detail. For the covariance estimation, authors propose using Woodbury formula. It is not clear to me whether this is innovative in connection with Gaussian processes estimation.

Briefly, the method works as follows: 
 - the logprob to estimate the model $p(Y| \lambda)$ contains the product of the intensity values at each observation point and a multiplier corresponding to Poisson term where the intensity is an integrated over the entire domain $\Omega$. 
$$p(Y|\lambda) = exp\Big( - \int_{\Omega} \lambda(s) ds \Big) \Pi_{i=1}^N \lambda(s_i)$$
The integral term over the domain $\Omega$ is evaluated on the grid of Voronoi tessellation; the product term corresponding to likelihood of the intensity itself is evaluated on the observed points
-  the prior Gaussian Process is approximated by NNGP 
- the linear covariate parameter $\beta$ and the mean /drift  $\mu$ of the Gaussian field is computed uby the Newton method
- the covariance matrix of the variational approximation is computed using the matrix inversion 
- the method also optimises the likelihood noise parameter

### Strengths
1. Paper is reasonably well written with good logical structure and easy to follow.
2. The ideas seems to be novel and in general interesting. 
3. Authors provide optimisation algorithm and attempt to safe-guard the proposed methods by theoretical results.
4. Well described optimisation algorithm for fitting the parameters.

### Weaknesses
1. The main contribution of the paper is, in my opinion, in selecting the auxiliary points to improve the integration within the likelihood. While this idea is interesting, authors do not elaborate on how to select these auxiliary points and does not provide any ablation study on this topic. Given that this the major step, I consider this as a crucial weakness. It would be great to see experiment, where the the authors demonstrate the robustness / sensitivity of the method to these auxilary points. 
2. Authors claim they prove theoretical properties of the estimator. Proof for Theorem 1 is only sketched, while for Theorem 2, the provided proof only shows existence of the solution but does not prove the uniqueness. 
3. Paper does not provide experiments showing how the NNGP selection impacts the inference method (in comparison to using the GP without any approximation on n+N datapoints).
4. The experimental results does not seem to over-perform the existing methods. It is not major weakness as authors provide interesting algorithm but I believe that benefits of the proposed should be somehow better demonstrated than by results in Figure 3., e.g. authors would consider to construct a metric that can demonstrate how to measure depicted phenomena.  

Minor typos: 
l 116: dimension m is not defined
l 151: dimension p is not defined in Z_s  / defined on the l 182, slightly confusing

### Questions
1. How do you select the “pre-selected” deterministic integration points (l 108)? How sensitive are the results to this selection? Can you provide an ablation study to demonstrate this?

2. When fitting a Gaussian process, the combination of the hyperparameters is not unique, only the eigen-values of the covariance matrix. Can you elaborate on how this impacts your fits? 

3. When approximating GP with NNGP (l 201), how does this impact the results from the Theorems 1 and 2? What is the impact of NNGP approximation on the performace on the method?

4. For Newton method, how do you select the initial values (l 242)?

5. How numerically stable are the inversions (18) - l 281

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes and studies an variational approach to fit log-Gaussian processes.

### Strengths
The paper is well-written, and I mostly enjoyed reading it. The idea is clear, and the results are scientifically correct. The numerical experiments also support the proposed approach.

### Weaknesses
The paper is more a statistical paper, rather than a ML/AI work. Though the results seems to be novel, the approaches (and the proofs) are standard.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
