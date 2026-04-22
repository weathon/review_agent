# Contextual Similarity Distillation: Ensemble Uncertainties with a Single Model

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 10

## Abstract
Uncertainty quantification is a critical aspect of reinforcement learning and deep learning, with numerous applications ranging from efficient exploration and stable offline reinforcement learning to outlier detection in medical diagnostics. The scale of modern neural networks, however, complicates the use of many theoretically well-motivated approaches such as full Bayesian inference. Approximate methods like deep ensembles can provide reliable uncertainty estimates but still remain computationally expensive. In this work, we propose contextual similarity distillation, a novel approach that explicitly estimates the variance of an ensemble of deep neural networks with a single model, without ever learning or evaluating such an ensemble in the first place. Our method builds on the predictable learning dynamics of wide neural networks, governed by the neural tangent kernel, to derive an efficient approximation of the predictive variance of an infinite ensemble. Specifically, we reinterpret the computation of ensemble variance as a supervised regression problem with kernel similarities as regression targets. The resulting model can estimate predictive variance at inference time with a single forward pass, and can make use of unlabeled target-domain data or data augmentations to refine its uncertainty estimates. We empirically validate our method across a variety of out-of-distribution detection benchmarks and sparse-reward reinforcement learning environments. We find that our single-model method performs competitively and sometimes superior to ensemble-based baselines and serves as a reliable signal for efficient exploration. These results, we believe, position contextual similarity distillation as a principled and scalable alternative for uncertainty quantification in reinforcement learning and general deep learning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose  contextual similarity
distillation, an approach that explicitly estimates the variance of an ensemble
of deep neural networks with a single model, without learning or evaluating
such an ensemble in the first place. It can estimate predictive variance at inference time with a single forward pass,
and can make use of unlabeled target-domain data or data augmentations to refine
its uncertainty estimates.

### Strengths
The paper is clear, well-written, solves a cogent problem, and presents convincing experimental evidence.

### Weaknesses
The paper only has a conceptual weakness, that is widespread in the ML literature.

If we posit that epistemic uncertainty (EU) is encoded in the parameter distribution, then a single posterior predictive is not able to gauge EU correctly, since it is the expectation of the likelihood taken wrt the posterior, $p(\tilde{y} | \tilde{x},D) = \int_\Theta p(\tilde{y} | \tilde{x},\theta) p(\theta | D) \text{d}\theta = \mathbb{E}_{\theta \sim p(\theta | D)}[p(\tilde{y} | \tilde{x},\theta)]$, hence all the uncertainty encoded in the posterior gets ``washed away'' by taking the expectation. 

This is not the case in ensembles, where we have different posterior predictives, but if we then distil all the models into only one, and estimate the variance of the latter thinking that it captures the EU, we incur the same problem as above. In symbols, we would have $p(\tilde{y} | \tilde{x},D) = \sum_{k=1}^K w_k \int_\Theta p_k(\tilde{y} | \tilde{x},\theta) p_k(\theta | D) \text{d}\theta = \mathbb{E}_{k} [p_k(\tilde{y} | \tilde{x},D)]$, where the $w_k$'s are the weights that the ensemble method assigns to each model $k = 1,\ldots,K$ (which of course are nonnegative and sum up to $1$).

The way to avoid this is to keep the posterior predictives separate, e.g. by considering their induced convex hull, and use other metrics existing in the literature (e.g. in https://link.springer.com/article/10.1007/s10994-021-05946-3) to quantify EU (and also aleatoric uncertainty). This was already done in https://openreview.net/forum?id=4NHF9AC5ui, and applied to active learning (thus very close to RL) in https://dl.acm.org/doi/10.1145/3716863.3718040, where EU is used to inform the exploration of the state space. 

While the paper solves a very relevant problem, in a creative and innovative way, I urge the authors to (a) either avoid referring to epistemic uncertainty, and rather just call it uncertainty (in which case my previous point can be ignored), or (b) to add even a small discussion where they acknowledge my point above, and state that they keep the term "epistemic" only to conform to the existing literature. 

Moreover, I suggest the authors, in future work, to look into the techniques developed by the imprecise probabilistic machine learning field, that may well turn out useful for the future endeavor pointed out in the conclusion.

### Questions
In line 86, shouldn't it be $\mu \in \mathscr{P}(\mathcal S)$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes contextual similarity distillation (CSD), a single-model method that tries to estimate the variance of a random initialization ensemble of DeepEnsembles by using the kernel similarities in the covariance in the Neural Tangent Model. Empirical results show CSD performs competitively and sometimes superior to ensemble-based baselines in OOD detection and sparse-reward RL environments.

### Strengths
- The paper is generally well-written to understand the key aspects of the proposed algorithm.
- The direction of this paper is important because DeepEnsembles' superiority in uncertainty and robustness, but struggle with computational efficiency.
- Empirical results are quantitatively shown in the experiments and qualitatively shown in Figures 1 and 2 to understand how the method actually works.

### Weaknesses
- Using NTK to get epistemic uncertainty is not new and has been studied in several works [1,2,3]. This limits the novelty contribution of CDS. Note that I still appreciate your contribution to the similarity distillation.
- There is no theoretical contribution in this paper to formally explain why and how CDS works. 
- The experiments lack several computational efficiency baselines in uncertainty and robustness (e.g., BatchEnsembles, Rank1-BNN, etc. [4]) and OOD detection baselines (ReAct, DICE, etc. [5]).
- The main contribution is improving computational efficiency, but there is no computational efficiency evaluation (training, inference speed, GPUs, etc.) in the experiments.
- Contribution to uncertainty also needs to improve beyond OOD detection evaluation, such as Expected Calibration Error (in IID and OOD data), sharpness, etc.
- In writing, background in Reinforcement Learning, such as MDP, Q-function, policy improvement objective, are not related to the main method in Section 3 (many mathematical notations are not reused at all). Note that I still appreciate your additional RL experiments, but I would suggest moving some RL background in Section 2 to the Appendix.

### Questions
1. How does CDS perform on larger-scale datasets such as CIFAR-100 and ImageNet?

2. In a large-scale dataset, where the feature dimension is higher, how about the training/inference efficiency of CDS get impacted? How is the ensemble variance estimator impacted (can the authors formally show the estimator error in this regard?)?

References:

[1] He et al., Bayesian Deep Ensembles via the Neural Tangent Kernel, NeurIPS, 2020.

[2] Sergio et al., Epistemic Uncertainty and Observation Noise with the Neural Tangent Kernel, NeurIPS, 2022.

[3] Seijin et al., Disentangling the Predictive Variance of Deep Ensembles through the Neural Tangent Kernel, NeurIPS, 2022.

[4] Nado et al., Uncertainty Baselines: Benchmarks for Uncertainty & Robustness in Deep Learning, Bayesian Deep Learning workshop, NeurIPS 2021.

[5] Yang et al., OpenOOD: Benchmarking Generalized Out-of-Distribution Detection, NeurIPS 2022 Datasets and Benchmarks Track.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes contextual similarity distillation (CSD), a method which approximates an ensemble of deep networks using a single model. The key idea behind this method is motivated by neural tangent kernels (NTK), which describes how similar the gradients of two inputs are with respect to the network parameters. For a network layer with infinite width, the NTK becomes deterministic and constant (allowing it to be expressed as a Gaussian process). CSD estimates the NTK by training with regression targets corresponding to kernel similarities between data points. Empirical results support theoretical claims against several baseline methods.

### Strengths
1. Strong theoretical grounding. The proposed method builds off of well-studied prior literature (e.g., NTK and Gaussian processes),
2. Reduced computational cost. The proposed method is able to approximate a deep ensemble using a single model, avoiding large training and inference costs. This is supported by the experimental results on tasks like OOD detection and RL navigation.
3. Novelty. The idea to use kernel regression to analytically approximate an ensemble is novel. Furthermore, the addition of a context variable to allow the method to work on arbitrary query points.

### Weaknesses
1. Empirical results. The experiments are limited to small scale datasets like VizDoom and FashionMNIST with well-studied, but somewhat outdated, baselines. Inclusion of larger / more complex datasets and comparison against more recent methods would considerably strengthen the paper and give more context on the scalability of CSD.

### Questions
1. Can the proposed idea of contextual regression be extended to capture aleatoric uncertainty as well, or is it primarily limited to estimating epistemic uncertainty?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper presents an approach to estimate the uncertainty of a neural network prediction through contextual similarity distillation. The method is built upon the Neural Tangent Kernel theoretical results, which express a probability distribution of a neural network function after training based on the distance between a test sample and the training sample as measured by a gradient-based kernel (NTK). Using a clever choice of labels, they train a neural network to predict the uncertainty that would result of an ensemble.

### Strengths
This paper brings in an interesting and novel approach to uncertainty estimation. It is strongly anchored in theoretical analysis (although with several assumptions). It is also more practical than current approaches, as it requires significantly less compute, and it performs well on different datasets compared to baselines.
The paper does a good job at presenting the concept development in a pedagogical and self-contained manner, which is necessary for anyone who is not familiar with the NTK literature.
While some of the assumptions are strong, the discussion section in the Appendix about approximations is relevant and important.
I really appreciated Figure 2 - it demonstrates clearly and convincingly the method’s objective and performance.
I believe this is an impressive piece of work which could have a lasting impact on uncertainty estimation with neural networks.

### Weaknesses
I did not find important weaknesses in this work. Still, elements could be improved.

While the experimental results on image classification are good and convincing, I believe other types of tasks (regression, for example) could strengthen the article and open the use of this approach for a wider range of tasks. 

Regression uncertainty estimation is what is actually showed in the RL environment, but very indirectly. The RL experiment is one of the least strong aspect of this paper. While it indirectly supports the claim about CSD generating good uncertainty estimations, it is not sufficiently rigorous to demonstrate the usefulness of the approach in RL. For a stronger case, more details about the implementation and other environments would be needed. More discussion too - for example, it is surprising that CSD works better than Bootstrap DQN where CSD is supposed to approximate ensembles with lower costs and Bootstrap DQN actually uses ensembles.

### Questions
Questions
* In section 3.1, it is said that $g$ has the same architecture as $f$, therefore $\Theta_g(x, x’) = \Theta(x, x’)$. In the text earlier, there is no justification for this: the closer argument I saw was that $\Theta(x, x’)$ is equal for every stage of training of weights $\theta_f$ under the assumption of gradient flow. Is the two networks having the same architecture sufficient for this equality to hold? If so, could you introduce this somewhere in section 2.2?

* Figure 1, the legend describes the “Kernel Prior $\Theta(x,x)$”, yet the function changes across the three subfigures, where $x_t$ is the element that changes. Should it be $\Theta(x, x_t)$?
* Section 3.3: is the feature vector $\phi(x, \tilde{\theta}_f)$ related to parameterized function $f$? If so, what exactly from $f$ does it refer to?
* Equation 14, is the RHS calculated on the trained $f$ or on the initial $f$ parameters? (This also applies to equation 13 too if $\phi(x, \tilde{\theta}_f)$ is also based on $f$). 
* If it is on the trained parameters, how does that work when it is estimated during a continuous training process, such as in RL?

Comments:
* Introduction, Section 3.1: It is said often in the paper that, in the field of RL, there are often large models or datasets. Depending on the RL application, this may not be true: most applied RL uses small neural networks, and datasets are created from experience (which can be sparse/expensive). Maybe these claims could be more nuanced?
* Section 2: The objective of finding the optimal policy is not RL specific (this is the case for every MDP solving method, s.a. dynamic programming or model-predictive control). The RL specific challenge is to do so through learning on trajectories acquired in the environment.
* Section 2.1: The fundamental challenge of exploration is specific to the case of online RL (to differentiate from offline RL)  
* Insufficient upper margin for eq. (3)
* End of section 2.1: Due to their initialization, ensemble members are not great at the beginning of the RL process to determine the model’s uncertainty: that’s where randomized prior functions (RPF, Osband et al., 2019) are helpful.

Typos:
* Introduction: “distribution shift detection tasks(Van Amersfoort” and “NOTMNIST datasets(Xiao et al.”: space missing before the “(“
* Section 2: “agents […] subsequently receiveS the immediate reward […] and observeS the next state”: there should not be an S as agents is plural
* Section 3.1: “it is our goal IS to estimate”: correct the sentence

### Soundness
3

### Presentation
3

### Contribution
4
