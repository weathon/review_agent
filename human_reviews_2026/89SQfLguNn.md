# Graph Random Features for Scalable Gaussian Processes

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
We study the application of graph random features (GRFs) – a recently-introduced stochastic estimator of graph node kernels – to scalable Gaussian processes on discrete input spaces. We prove that (under mild assumptions) Bayesian inference with GRFs enjoys $\mathcal{O}(N^{3/2})$ time complexity with respect to the number of nodes $N$, with probabilistic accuracy guarantees. In contrast, exact kernels generally incur $\mathcal{O}(N^{3})$. Wall-clock speedups and memory savings unlock Bayesian optimisation with over 1M graph nodes on a single computer chip, whilst preserving competitive performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper applies the random feature approach in kernel-based learning to graph problems (such as node classification/regression). For context, graph random features are a recently introduced stochastic estimator of a family of kernel matrices defined on graphs. The authors use these features for node regression and Bayesian optimization. They show an $O(N^{3/2})$ time complexity for kernel parameter estimation and demonstrate that a learnable kernel with random features outperforms the graph diffusion kernel, either in the full matrix form or in its random feature representation.

### Strengths
A main bottleneck of Gaussian processes is the $O(N^3)$ scalability barrier. The proposed approach provably reduces the cost to $O(N^{3/2})$. Additionally, the authors demonstrate experiments to show that they can handle 1M graph nodes comfortably by using a single GPU.

The writing of the paper is excellent and easy to follow, balancing clarity and conciseness for graph random features (GRFs), which contain many subtle details.

(Half strength, half weakness) GRFs are an interesting extension of random features from the Euclidean domain to the graph domain. It is pleasant to see the use of GRFs in machine learning tasks. On the other hand, this paper mostly concerns applying the concept, weakening the contribution. Moreover, as graph neural networks (GNNs) become so widely used for the tasks concerned in this paper, the advantages of GRFs are unclear.

### Weaknesses
On the theoretical side, given the prior work on GRFs and the machinery to perform GP inference and parameter estimation by using random features, the contribution of this work primarily lies in applying the former to the latter. A majority of the background for this application is known (e.g., the concentration of GRF in Theorem 1, the parameter estimation method in Eqn (8)--(11), and the posterior inference in Eqn (12)). To be fair, the authors do provide new results, such as the $O(N^{3/2})$ complexity for inference due to the $O(N)$ spectral norm of the kernel, but this contribution is less significant than the establishment of the GRF itself.

On the practical side, it is unclear how kernel/GP methods fare with GNNs. There are no such experiments in the paper. GNNs scale better than kernels and they are data-efficient for node-level problems. For example, for the Cora graph and the classification task that the authors discuss in Section 4.4, GNN can achieve over 80% accuracy with 20 training nodes per class (a labeling rate of 5.2% only). Can GRF be as data-efficient?

Another practical question is the Bayesian optimization over graphs, which has a finite search space. Since predicting the graph signal on each node is cheap and relatively accurate by using a GNN given a small training set, a natural heuristic is that one does not use Thompson sampling for the acquisition but only picks the best GNN predictions. The practical value of the paper can be strengthened by experimenting with such a simple baseline.

### Questions
See comments above.

### Soundness
3

### Presentation
4

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
This paper proposes Graph Random Features (GRFs) to construct learnable graph node kernels which estimate the sparse GP covariance function reducing the cubic computational complexity of exact GPs and applying these graph-based approximation technique for sparse GP estimation for Bayesian optimization via Thompson sampling. The contributions of this paper are considerably novel, as they build upon the previous work of Reid et. al. on GRFs specifically into GP domain and then provide an application of this technique by applying it into a BO setting showcasing the effectiveness and practicality.

### Strengths
The paper presents a well-motivated contribution that bridges recent advances in graph kernels and scalable GP inference. The use of GRFs for graph node kernels is an innovative idea that leverages random walks to approximate covariance structures efficiently. The theoretical development is sound and neatly connects GRFs with probabilistic kernel approximation, culminating in a clear argument for the claimed time complexity. The results claimed in Theorem2 and Lemma1 also establish theoretical underpinnings of the proposed work, contributing to the theoretical understanding of GRFs.

Empirically, the paper demonstrates wall-clock speedups and scalability to graphs with >1M nodes, which is impressive for a sparse GP-based method. The reported experiments, on regression and BO tasks, are diverse enough to illustrate both predictive and computational benefits. 

Moreover, the paper is clearly written and well-organized. Overall, the paper is theoretically and empirically solid and easy to follow for community working on GPs and BO.

### Weaknesses
While the core idea of the paper is novel, the experimental validation somewhat lacks depth and breadth. The comparisons are limited mainly to diffusion kernels and do not include strong baselines such as sparse variational GPs, approximation based on inducing-point (support set). Without these benchmarks, it is difficult to judge the real-world competitiveness of GRFs in GP regression and BO tasks. The paper also does not sufficiently explore the sensitivity of the method to hyperparameters such as the number of random walks, halting probability, or walk length.

Moreover, the paper does not provide bounds on the GP posterior approximation error, so while kernel estimation is shown to be accurate, the implications for predictive uncertainty are less clear. Empirical metrics focus on RMSE and NLP, but uncertainty calibration and robustness are not analyzed. The BO results show consistent performance gains, but the lack of statistical testing limits the strength of these claims to some extent.

### Questions
I have few concerns.
-- The assumption that the number of random walkers 𝑛 is independent of 𝑁 is theoretically elegant but practically fragile. In dense or long-range graphs, a small 𝑛 can lead to high-variance kernel estimates and poor posterior calibration.
-- The sparse approximation $\hat{K} = \Phi \Phi^{\top}$ is unbiased only in expectation, but in practice, the variance of the estimator is not analyzed in sufficient depth. The paper cites exponential concentration (Theorem~1) but never empirically validates how variance scales with $n$ or the graph topology. As a result, while computational speed is improved, the fidelity of posterior uncertainty (a key factor in Bayesian inference) may degrade unpredictably.

-- The footnote in Section~2 admits an $\mathcal{O}(1/n)$ bias for diagonal entries and notes  that removing it breaks positive definiteness. Yet, the authors use the biased estimator throughout without quantifying the impact of this bias 
on GP likelihood or predictive uncertainty. This is a non-trivial issue --- GP models are sensitive to kernel conditioning, 
and even small biases can affect the posterior mean and variance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Traditional GPs on graphs suffer from cubic time complexity (O(N^3)), limiting their scalability to small graphs. This paper leverages graph random features, which approximate graph kernels via sparse random walk–based feature maps, enabling sub-quadratic inference with strong theoretical guarantees. GRFs yield unbiased kernel estimates with exponential concentration bounds, allowing the GP covariance matrix to be approximated efficiently while preserving accuracy.

### Strengths
1. The authors provide rigorous complexity analysis and concentration bounds, proving that GRF-based GPs achieve ($O(N^{3/2})$) time complexity with probabilistic accuracy guarantees — a solid theoretical contribution that enhances credibility.
2. The experiments are diverse, covering both synthetic and real-world tasks. Results consistently support the theoretical claims and demonstrate substantial speedups without significant performance loss.
3. Despite the technical depth, the paper is well-structured and clearly written. Figures, pseudocode, and explanations make the ideas accessible.

### Weaknesses
1. The empirical evaluation focuses mainly on regression and Bayesian optimization; classification and non-conjugate inference are mentioned but not fully explored. This somewhat limits the generality of the method’s demonstrated usefulness.
2. The performance of GRFs depends on parameters like the number of random walks and the maximum walk length. Although heuristics are provided, the sensitivity to these settings could be further studied or automated.
3. While the paper compares dense and sparse GRFs, it does not extensively benchmark against other scalable GP frameworks (e.g., inducing point methods, variational GPs, or Kronecker-structured GPs), which would strengthen claims of superiority.

### Questions
1. Clarification on Practical Hyperparameter Sensitivity

The performance of GRFs depends on parameters such as the number of random walks, the halting probability, and the maximum walk length. Appendix C.1 partially answers Question 1, but is limited to empirical and qualitative descriptions. Could the authors provide more quantitative guidance on how these hyperparameters influence the trade-off between accuracy, sparsity, and computational cost? For example, is there an empirical scaling rule or automatic tuning strategy that generalizes across datasets? 

2. Comparison with Other Scalable GP Frameworks

While the paper contrasts dense vs. sparse GRFs, it does not directly compare with other established scalable GP techniques (e.g., SVGP, SKI, or Kronecker GPs). Could the authors comment on how GRFs compare empirically or theoretically to these alternatives, especially in terms of scalability and approximation accuracy? 

3. Clarification on Bias and Positive-Definiteness

Section 2 mentions a small $O(1/n)$ bias in diagonal kernel entries due to shared randomness among random walks. Could the authors clarify whether this bias ever affects posterior stability or positive definiteness in practice? Have they observed numerical issues during GP training? A short empirical note or ablation (e.g., comparing one vs. two independent random walk ensembles) would help readers understand if this bias is purely theoretical or occasionally relevant.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a scalable GP methodology for discrete input spaces using graph random features (GRFs) in combination with conjugate gradient (CG) methods and Hutchinson’s trace estimator. A theoretical analysis is presented showing that the computational complexity of the proposed approach scales as $O(N^{3/2})$, which is a significant improvement over the $O(N^3)$ complexity when using exact kernels. Extensive numerical studies are presented to illustrate the performance of the proposed approach.

### Strengths
The paper is well-written and organized. Relevant background and motivation is discussed clearly and details of the setup used for numerical studies are described in sufficient depth in the appendices. The three-step procedure (initialization, hyperparameter learning, inference) is clearly explained. Algorithm pseudocode aids reproducibility.

The application of GRFs to scalable GP inference is novel, though GRFs themselves were introduced recently (Reid et al. 2023, Choromanski 2023). The key theoretical results are: (1) Theorem 2 proving O(N) condition number for the covariance matrix and (2) Lemma 1 showing $O(N^{3/2})$ complexity for CG solves. The paper presents numerical studies demonstrating 50× speedups on graphs with <10K nodes and  Bayesian optimization on graphs with >1M nodes. 

Overall, the paper presents solid methodological work demonstrating that GRFs enable scalable GP inference on large graphs.

### Weaknesses
The theoretical development is largely sound. Theorem 1 (from prior work) provides useful concentration inequalities that the paper builds upon Theorem 2's proof for $O(N)$ condition number follows from standard operator norm arguments. The constant 'c' must be finite for the concentration bounds to hold. While the authors claim this is a “mild assumption,” it effectively requires the spectrum of W to satisfy a specific decay rate. This could be made more explicit.

The analysis in Lemma 1 that claims $O(N^{3/2})$ complexity for CG is rather crude for a formal theoretical result. In practice, CG is terminated when the residual falls below a specified tolerance ($\epsilon$), requiring $O(\sqrt{\kappa} log(1/\epsilon))$ iterations. The paper does not specify what tolerance was used, nor does it analyze how the choice of tolerance affects the approximation quality of the posterior. The $O(N^{3/2})$ estimate is equivalent to assuming that CG iterations are terminated after $\sqrt{\kappa}$ steps. 

The empirical scaling in Table 1 shows ~1.04 (nearly linear), which the authors attribute to a 'fixed iteration budget’. This trend implicitly suggests that the theoretical O(N^{3/2}) bound may not be achieved in practice without careful tuning of convergence criteria.

I was expecting to see approximation error analysis quantifying how using $\hat{K}$ instead of $K_\alpha$ affects the quality of the posterior. Since Theorem 1 provides concentration for individual entries, this should be feasible and would strengthen the contribution.

### Questions
- Section 3.1 mentions that the condition number bound requires $||\phi(i)||_1 \leq c$. Can you clarify if this bound depends on the choice of modulation function?
- Can the authors derive approximation bounds showing how the error bound on $||\hat{K} - K_\alpha||_F$ impacts the error in the posterior predictions?
- The numerical experiments show empirical scaling exponents of ~1.04 for training/inference (Table 1), which is nearly linear rather than N^{3/2}. What was the convergence tolerance used for CG? Was it an upper bound on the residual and/or maximum number of iterations?
- Was a preconditioner used for the CG solver? 
- For the wind interpolation task where GRFs outperform exact kernels: (a) is this trend consistent across multiple runs? (b) Can you comment on whether this is due to overfitting of exact kernels or beneficial regularization from GRF sparsity? (c) does the error over the training/validation set show similar patterns?
- The Woodbury identity approach (Appendix B) trades sparsity for lower-dimensional inversions. Under what conditions is this preferable to the sparse CG approach? Computational cost and approximation quality comparisons would be valuable.

### Soundness
3

### Presentation
4

### Contribution
3
