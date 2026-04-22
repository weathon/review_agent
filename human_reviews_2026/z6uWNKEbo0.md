# Nash: Neural Adaptive Shrinkage for Structured High-Dimensional Regression

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Sparse linear regression is a fundamental tool in data analysis. However, traditional approaches often fall short when covariates exhibit structure or arise from heterogeneous sources. In biomedical applications, covariates may stem from distinct modalities or be structured according to an underlying graph. We introduce Neural Adaptive Shrinkage (Nash), a unified framework that integrates covariate-specific side information into sparse regression via neural networks. Nash adaptively modulates penalties on a per-covariate basis, learning to tailor regularization without cross-validation. We develop a variational inference algorithm for efficient training and establish connections to empirical Bayes regression. Experiments on real data demonstrate Nash’s improved accuracy and adaptability over existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Neural Adaptive Shrinkage (NASH), a novel framework for high-dimensional regression that leverages neural networks to incorporate covariate-specific side information. The core idea is to learn an adaptive, structured penalty function on a per-covariate basis, guided by the side information. This is framed within an empirical Bayes perspective, and the authors propose an efficient "split VEB" (Variational Empirical Bayes) algorithm for model fitting, which decouples prior learning from posterior computation. The method is shown to generalize several existing penalized regression techniques and demonstrates strong empirical performance on various real-world datasets and an image denoising task.

### Strengths
- Provides a novel and unified method to integrate diverse side information (e.g., groups, graphs, time) into regression using a neural network-based prior.
- The proposed split VEB algorithm is scalable and efficient, making the approach practical by decoupling the prior learning and posterior computation steps.
- The method is shown to be highly competitive and often outperforms established baselines across a comprehensive set of real-world experiments.

### Weaknesses
- Performance can be sensitive to the neural network architecture and its hyperparameters, which adds a layer of complexity to its practical application.
- The use of a neural network to define the penalty structure may reduce the model's interpretability compared to classical regularization techniques.
- The paper is primarily empirical and would be strengthened by theoretical results, such as convergence guarantees for the proposed algorithm.

### Questions
Can the trained neural network provide insights into the underlying structure of the covariates? For example, is it possible to interpret what features of the side information the model found most important for determining the regularization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a new method to adaptively penalize the coefficients in linear regression. The authors provide a model, derive a loss and an algorithm, and validate on MNIST and tabular data.

### Strengths
The topic is interesting and the paper is rather easy to read.

### Weaknesses
Weaknesses:
- Clarity:
    - I am not sure I understood the proposed algorithm, would it be possible to encapsulate it in an environment, as done in [1], maybe it would be a good opportunity to highlight the difference with [1]
    - In Figure 1, bottom left, where dore the GNN-based prior comes from? Is it a pre-trained GNN? On other data? Could authors provide more details on this specific figure?
- Novelty: I am not sure I understood the difference with [1], could authors comment on that? Is it a generalization of [1] to graph-based data?
- Experiments
    - Would it be possible to add a vanilla CNN as a benchmark for Figure 2? (I understand this is a different kind of technique, just want to have an order of magnitude)
    - Table 1, how significant are the performance gain?


[1] Youngseok Kim, Wei Wang, Peter Carbonetto, and Matthew Stephens. A flexible empirical Bayes approach to multiple linear regression and connections with penalized regression. Journal of Machine Learning Research, 25

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a new Bayesian method for linear regression.
The prior distribution of the coefficients has product form. Each coefficient \beta_j has prior that is the convolution of a Gaussian with variance \sigma_0^2, and a distribution that depends on some side information d_j and is parametrized by common parameters \theta.
The paper describes a variational method for estimating the parameters of the prior, and performing inference on the linear model coefficients:
- The posterior is approximated by a product posterior.
- The corresponding ELBO is  maximized via coordinate ascent.
The paper illustrates the application of their approach to various regression problems, and how the new prior introduced in this paper can 
model a variety of complex structures (in particular group- and graph- based penalies) and demonstrates effectiveness on a denoising problem.

### Strengths
1. Regression with structural information about the coefficients is an central problem in high dimensional statistics with countless applications. Any progress on this problem is welcome.
2. The proposed framework is very general.
3. Simulations demonstrate some promising results.

### Weaknesses
1. The prior construction seems a straightforward extension of Wang & Stephens (2021) and Kim et al. (2024). The main innovation is the introduction of the "side information" d_j. While this is helpful, especially for graph-based tasks considered here, it is a very natural idea.
2. The variational inference algorithm is an application of standard methodology.

### Questions
1. I think Section 4, 5 describing the application and empirical results the most important of the paper. I think they should be expanded, spelling out in each case what is the architecture, how the prior was constructed, what are the x's and y's, what are the d's and so on. 

2. In contrast, Fig. 1 is fairly obvious/ not informative (and could be removed for reasons of space). Same consideration for the bottom panel of Fig 2

Minor:

1. Eq (8), you should write what expectation is over

2. Also the subscript Nash to the ELBO appears in random positions.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Nash (Neural Adaptive Shrinkage), a framework for high-dimensional sparse regression that leverages covariate-specific side information through neural networks to learn adaptive penalties. The authors develop a split variational empirical Bayes (VEB) algorithm that decouples prior learning from posterior inference. The paper establishes connections to mr.ash and demonstrates competitive performance on four real datasets.

### Strengths
S1. The split VEB approach addresses a real computational bottleneck of  the mr.ash variational formulatio. By decoupling the updates of problems, Nash requires only one neural network update per coordinate ascent iteration per updates. Theorem A.1 provide the lower-bound relationship to mr.ash.

S2. The authors successfully demonstrates how Nash can encompass various structured penalties (group lasso, fused lasso, IPF-lasso) within a single framework. 

S3. The authors clearly explains the variational formulation, coordinate ascent updates, and connections to exiting  variational Empirical Bayes approach.

### Weaknesses
W1. The authors claim Nash is "the first work to propose the use of a neural network to incorporate covariate side information when learning the penalty function," but fail to demonstrate why neural networks are necessary. They can compare more classical baselines including Kernel-based methods (e.g., RBF kernels on side information).

W2. The author employed only 4 real datasets, no synthetic data demonstrating when/why NNs help. It would be helpful if they can provide scenarios with complex non-linear side information where NN superiority would be clear.

W3. The theoretical contribution is limited to Theorem A.1. Although it shows Nash ≥ lower bound of mr.ash, but I want to understand How tight is this bound in practice.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
