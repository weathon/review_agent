# Federated Learning With $L_{0}$ Constraint Via Probabilistic Gates For Sparsity

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Federated Learning (FL) is a distributed machine learning setting that requires multiple clients to collaborate on training a model while maintaining data privacy. The unaddressed inherent sparsity in data and models often results in overly dense models and poor generalizability under data and client participation heterogeneity. We propose FL with an $L_0$ constraint on the density of non-zero parameters, achieved through a reparameterization using probabilistic gates and their continuous relaxation: originally proposed for sparsity in centralized machine learning. We show that the objective for $L_0$ constrained stochastic minimization naturally arises from an entropy maximization problem of the stochastic gates and propose an algorithm based on federated stochastic gradient descent for distributed learning. We demonstrate that the target density ($\rho$) of parameters can be achieved in FL, under data and client participation heterogeneity, with minimal loss in statistical performance for linear and non-linear models: $\emph{(i)}$ Linear regression (LR). $\emph{(ii)}$ Logistic regression (LG). $\emph{(iii)}$ Softmax multi-class classification (MC). $\emph{(iv)}$ Multi-label classification with logistic units (MLC). $\emph{(v)}$ Convolution Neural Network (CNN) for multi-class classification (MC). We compare the results with a magnitude pruning-based thresholding algorithm for sparsity in FL. Experiments on synthetic data with target density down to $\rho = 0.05$ and publicly available RCV1, MNIST, and EMNIST datasets with target density down to $\rho = 0.005$ demonstrate that our approach is communication-efficient and consistently better in statistical performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to enhance Federated Learning (FL) by introducing an L0 constraint on the density of non-zero parameters to enforce model sparsity. This mechanism, implemented using a reparameterization technique with Probabilistic Gates, aims to counteract the issues of overly dense models and poor generalizability arising from unaddressed data sparsity and various forms of heterogeneity common in FL settings. By leveraging techniques like the Binary Concrete Distribution and the Gumbel max trick, the approach allows for flexible model training from either dense or sparse initialization.

### Strengths
The paper introduces a novel application of the L0 constraint via Probabilistic Gates into the Federated Learning (FL) framework, specifically targeting model density. This represents a creative combination of established sparsity techniques (like the Binary Concrete Distribution and Gumbel max trick) to address a challenging problem in the unique, distributed FL setting.

### Weaknesses
1.While heterogeneity is simulated, the paper needs a more direct and comprehensive comparison against other state-of-the-art FL sparsity or pruning techniques (e.g., FL-adapted L1/L2 regularization, magnitude pruning, or methods based on the Lottery Ticket Hypothesis). This would clearly delineate the unique advantages of the L0 constraint approach.
2.The paper mentions that the parameter p controls the expected number of non-zero parameters. However, a detailed ablation study on the sensitivity of the final performance (accuracy vs. sparsity) to the choice of the sparsity hyperparameter is essential for practical use and is currently missing.

### Questions
1.How are the Probabilistic Gate variables specifically treated during the Federated Aggregation step? How does this choice ensure that the sparsity constraint remains satisfied across global model rounds?
2.In highly heterogeneous settings, how are the sparse structures determined by the Probabilistic Gates aggregated across different clients?

### Soundness
2

### Presentation
2

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
The paper presents an FL framework, that enforces $L_0$ constraint on the model parameters to induce sparsity by utilizing probabilistic gates to make the problem tractable. This is motivated by the poor generalization of the dense models in heterogeneous FL settings. The paper further demonstrates that how the $L_0$ constraint is connected to entropy maximization of stochastic gates. They then utilize this insight to derive the $L_0$ constraint for the FL setting. The resulting algorithm called FLoPS allows simultaneous updates of model parameters, gate parameters and a Lagrange multiplier that controls the level of sparsity.

### Strengths
1. The key insight in the paper connecting the $L_0$ constraint to entropy maximization of stochastic gates is novel.
2. The utilization of the above insight for deriving the $L_0$ constraint for the FL settings enables a new learning setup.

### Weaknesses
1. All experiments in the paper are conducted on linear models, therefore the scalability and soundness of the method on more commonly used non-linear models remain untested.
2. The convergence guarantee or stability conditions of the joint optimization is under-discussed.

### Questions
1. Can you provide intuition on the proposed connection between entropy maximization and $L_0$ constraints? 
2. Do all clients contribute parameters in every iteration? If not, how are the missing gates handled?
3. How sensitive is the algorithm to the decay and pruning schedule?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes FLoPS, a federated learning algorithm for learning sparse models with controlled parameter density using probabilistic gates and Hard Concrete relaxation. The authors adapt the L0-constrained optimization framework from Gallego-Posada et al. (2022) to the FL setting, derive the objective from entropy maximization principles, and propose a distributed algorithm based on FedSGD. Experiments on synthetic and real datasets demonstrate superior sparsity recovery and statistical performance compared to magnitude pruning baseline (FedIter-HT) under data and client participation heterogeneity.

### Strengths
S1. New prespective on gate-based L0 sparsity to FL: Addresses a gap in the literature where probabilistic gates have not been applied to federated sparse learning.

S1. Extensive experiments across multiple datasets, sparsity levels (0.5%-95%), and heterogeneity conditions (data, client participation).
Consistent improvements over baseline: FLoPS demonstrates better True Discovery Rate and statistical performance than FedIter-HT across all settings.

S3. Achieves target density through constrained optimization rather than tuning regularization coefficients.

S4. Entropy maximization derivation (Section 2.2) provides alternative theoretical perspective on L0 regularization.

### Weaknesses
W1. Section 2.2 (entropy derivation) is isolated from the FL application. It derives the centralized formulation but provides no insights for distributed optimization, convergence, or aggregation strategy.

W2.  No theoretical analysis of whether FLoPS converges in heterogeneous FL settings. Does the algorithm converge under non-IID data? How does heterogeneity affect convergence rate? What is the relationship between three learning rates?


W3. The paper applies standard weighted averaging to (\hat{\theta}, \phi) without justification. Why is this optimal? Why not aggregate \theta directly? 

W4. How does a consistent global sparsity pattern emerge from heterogeneous local updates when different clients prefer different features? The constraint is only enforced server-side - why is this sufficient?

W5. The paper provide limitted analysis, it requires to have  (i) communication cost comparison, (ii) computational overhead, (iii) ablations on design choices.

W6. Limited baseline comparison: Only compares with FedIter-HT. Missing comparisons with: (i) Lasso-based FL methods (Frandi et al. 2016, Sehic et al. 2022), (ii) Standard FedAvg with post-hoc pruning, (iii) Centralized training (upper bound)

### Questions
Q1. Can you provide convergence guarantees for FLoPS under data heterogeneity? What is the convergence rate and how does it depend on heterogeneity level?

Q2. How do you ensure global sparsity pattern emerges from conflicting local preferences?

Q3. What is the total communication cost (rounds × message size)? Does FLoPS converge faster enough to offset the 2× parameter overhead per round?

Q4. refer to weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors seek to tackle the challenges brought by inherent sparsity in data for Federated Learning (FL). To mitigate the risk of building overly dense models and poor generalizability, the authors propose FL with an $L_0$ constraint on the density of non-zero parameters by extending the idea of sparsity in centralized machine learning. The authors reveal the connection between the objective for $L_0$ constrained stochastic minimization and the entropy maximization problem and propose an algorithm based on federated stochastic gradient descent for distributed learning.  The authors conduct experiments on synthetic data as well as MNIST datasets to demonstrate that the proposed method outperforms other methods in both sparsity recovery and statistical performance.

### Strengths
I like the deep insights between the objective for $L_0$ constrained stochastic minimization and the entropy maximization problem. Motivated from this connection, the authors are able to propose the federated stochastic gradient descent for distributed learning

### Weaknesses
1. While I like the nice insights and connections the authors reveal theoretically, I think there is a big room for improving the experiment section. For example, the largest dataset considered in this paper possibly be MNIST, which is a very small scale dataset. It would lead to questions from readers on the necessity of distributed learning.

### Questions
Federated Learning is distributed learning in essence which are often important when dealing with large scale datasets or when devices are hardware-restricted. I would like to see the authors go beyond the small scale size datasets such as MNIST to avoid the challenges on the necessity of distributed learning.

### Soundness
3

### Presentation
3

### Contribution
2
