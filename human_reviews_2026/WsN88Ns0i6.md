# Variational Deep Learning via Implicit Regularization

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Modern deep learning models generalize remarkably well in-distribution, despite being overparametrized and trained with little to no explicit regularization. Instead, current theory credits implicit regularization imposed by the choice of architecture, hyperparameters, and optimization procedure. However, deep neural networks can be surprisingly non-robust, resulting in overconfident predictions and poor out-of-distribution generalization. Bayesian deep learning addresses this via model averaging, but typically requires significant computational resources as well as carefully elicited priors to avoid overriding the benefits of implicit regularization. Instead, in this work, we propose to regularize variational neural networks solely by relying on the implicit bias of (stochastic) gradient descent. We theoretically characterize this inductive bias in overparametrized linear models as generalized variational inference and demonstrate the importance of the choice of parametrization. Empirically, our approach demonstrates strong in- and out-of-distribution performance without additional hyperparameter tuning and with minimal computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a method to get the benefits of Bayesian deep learning—like improved robustness and uncertainty estimates—without the usual computational cost.

Instead of adding an explicit regularization term (like KL divergence) to the loss function, they rely on the implicit regularization of SGD. They initialize a variational distribution over the network's weights to match a prior, and then train by minimizing only the data-fitting loss.

The key insight is that SGD naturally finds the solution that is closest to its starting point. This behavior acts as a powerful regularizer, achieving strong performance with minimal overhead. They provide proof that this technique actually yields the divergence minimiser for Wasserstein distance in the linear case for overparametrised models. 

They provide empirical results showcasing that their proposed methodology offers competitive performance on a broad range of datasets.

### Strengths
1. Very sound approach and presentation, very much like the simplicity of just not having an extra divergence term to deal with the numerical issues of.
2. Theorem 1 is very nice, gives an intuition as to why this could work for much larger models.
3. Experiments convincing, comprehensive across diverse tasks and datasets

### Weaknesses
1. This method seems to be possibly limited to mean field variational posteriors, and only considers the Wasserstein distance as the divergence term. 
2. Missing baseline comparison, would be nice if you could compare against a variational method that uses Wasserstein distance explicitly in the "ELBO", possibly https://proceedings.neurips.cc/paper_files/paper/2022/file/18210aa6209b9adfc97b8c17c3741d95-Paper-Conference.pdf or any other paper that proposes Wasserstein distance explicitly as a divergence term.

### Questions
1. Can it possibly be generalised to other family of variational posteriors and not just Wasserstein distance for the divergence term?
2. Did you try any other parameterisations of the mean field covariance term? 
3. Would appreciate an ablation study showing that this offers similar performance compared to using an explicit Wasserstein distance divergence term.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
- this paper exploits the implicit regularization effect of (stochastic) gradient descent (SGD) for Bayesian deep learning.
- In particular, the authors characterize the implicit bias induced by SGD for regression and binary classification task in overparameterized linear models.
- Experiments show competitive performance compared to state-of-the-art Bayesian deep learning baselines.

### Strengths
- The paper is well written and structured.
- The authors propose to training a Bayesian neural network using only the expected loss, removing the divergence term from the variational objective. They then show that among the variational parameters (assumed to follow a Gaussian distribution) that minimize the expected loss, SGD converges to the one closest to the prior in terms of Wasserstein-2 distance -- analogous to how SGD converges to minimum-norm solution in overparameterised linear network.
- The experimental results demonstrate that this implicit bias achieves competitive test error and uncertainty quantification across different datasets, thereby empirically confirming the proposed effect.
- Overall, I enjoyed reading this paper.

### Weaknesses
- The theoretical analysis is limited to overparameterized linear model (one layer). Extending the proof to more than 2 layers might not be straightforward.
- The proof assumes Gaussian variational distribution, and the theorems do not generalize to other distributions which might be more effective for other problem settings/ datasets.

### Questions
See the weaknesses.

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
4

### Summary
he authors propose and analyze a variant of the so-called “generalized variational inference” objective (e.g., Eq. 1 of Knoblauch et al.) that drops that regularization term, arguing that the problem is already well-posed from implicit rather than explicit regularization. The well-posedness derives from the fact that in overparameterized models, the set of global optima is quite large (not just a point mass), and so SGD tends to converge to the member of the set of global optima which is nearest to the initialization. Because of this, the problem is implicitly regularized to the prior already (b/c the prior is used for initialization), so explicit regularization is unnecessary. The authors formalize this result for linear regression and binary classification by showing the implicit regularization takes the form of a Wasserstein distance from the the variational distribution to the prior.

### Strengths
- The ideas of the paper are central, yet deep, and well articulated by the authors
- To my knowledge, the contributions are novel in the literature and interestingly meld together the ideas of Bayesian ensembles of models with variational inference and optimization. I find the takeaway message of the paper to be quite “unifying”.
- Extensibility to two different problem settings (regression and classification) augments the impact of the work.
- The authors provide sufficient background to make the work accessible to a broader audience

### Weaknesses
- The selling point of the paper falls short of the results: deep learning is not included in the main results of S4, which assume overparameterized linear models. I still find the contribution interesting even in this context, but I’m unsure how much the results can be said to explain the success of “deep learning”. I understand the discussion on page 5 and the CIFAR experiment tie it more closely to practical deep learning, but the theoretical results do not explain the success of deep learning in the same way that the results of, for example, the neural tangent kernel (NTK) does. The lack of a clear connection to deep learning undermines the contribution somewhat. 
- The main experimental results (Fig. 4) do not demonstrate clear outperformance of the method relative to the competitors. However, I commend the breadth of the competing approaches considered.

### Questions
- It seems (cf. line 45 and line 2228) that the loss function used by the authors is the negative log likelihood. Considering that the proposed method only minimizes this, and nothing else, it’s surprising that other approaches can outperform on this exact metrics (Test NLL for MNISTC, for example, in Figure 5). Can the authors provide any insight on this phenomenon?
- Taking implicit regularization as well-established, can the authors comment more on the desirability of the solutions found by this fact? Is the solution closest to the prior advantageous with respect to generalizability, etc.?
- The authors may consider citing other works that use the negative log likelihood loss, perhaps to comment on differences. For example, the class of neural posterior estimation (NPE) methods comes to mind. Can you elaborate on similarities or differences to this class of methods?

### Soundness
3

### Presentation
3

### Contribution
3
