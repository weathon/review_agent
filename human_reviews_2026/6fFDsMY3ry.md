# Surrogate-Based Quantification of Policy Uncertainty in Generative Flow Networks

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Generative flow networks are able to sample, via sequential construction, highreward, complex objects according to a reward function. However, such reward functions are often estimated approximately from noisy data, leading to epistemic uncertainty in the learnt policy. We present an approach to quantify this uncertainty by constructing a surrogate model composed of a polynomial chaos expansion, fit on a small ensemble of trained flow networks. This model learns the relationship between reward functions, parametrised in a low-dimensional space, and the probability distributions over actions at each step along a trajectory of the flow network. The surrogate model can then be used for inexpensive Monte Carlo sampling to estimate the uncertainty in the policy given uncertain rewards. We illustrate the performance of our approach on a discrete and continuous grid-world, symbolic regression, and a Bayesian structure learning task.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors proposed a way to estimate uncertainty for learned policies in GFlowNets, using a polynomial chaos expansion over the ensemble of already-trained flow networks to train a mapping from a latent reward-function space to a policy space. Using this linear model over the reward latent space, the authors estimated the policy uncertainty on several benchmarks, such as discrete and continuous grid-worlds, symbolic regression, and a Bayesian structure learning task.

### Strengths
- The authors addressed an interesting problem from a general GFlowNet problem;

### Weaknesses
- To my knowledge, GFlowNets typically address learning of the sampler in an **acyclic** environment, and making GFlowNet work in a non-acyclic environment requires many additional training tricks (see Brunswic et al. 2024, and Morovoz et al. 2025). The described grid-world environments are exactly non-acyclic, and thus, it is unclear how the model was trained.
- The paper lacks high-dimensional real-world motivational examples: the only not fully synthetic example is Bayesian structural learning, where the state space consists of about $10^4$ states (and even this example cannot be considered as a real-world example since it uses Erdos-Renyi random graphs and not real data). Thus, it is impossible to understand whether this method actually scales.
- Lack of a general mechanism to extract the reward features: for each environment, the authors use ad hoc approaches to do it. In particular, it also makes me doubt about the scaling abilities of this empirical approach.

## References

Brunswic, L., Li, Y., Xu, Y., Feng, Y., Jui, S., & Ma, L. (2024, March). A theory of non-acyclic generative flow networks. AAAI-2024.

Morozov, N., Maksimov, I., Tiapkin, D., & Samsonov, S. (2025). Revisiting non-acyclic gflownets in discrete environments. ICML-2025.

### Questions
- What is a motivation for using polynomial features instead of just a two-layer neural network over the latent space? The linear model lacks expressivity, even in the space of orthogonal polynomials (of course, if the function is smooth, any function can be estimated by a Fourier series, but I don't see any reason to do it there). 
- What is the use of using the reward features as an additional feature for training the initial policies, as in temperature-conditional GFlowNets?
- How is it possible to use this uncertainty score to get an improved training or exploration for GFlowNets?

Kim, M., Ko, J., Yun, T., Zhang, D., Pan, L., Kim, W., ... & Bengio, Y. (2023). Learning to scale logits for temperature-conditional GFlowNets. ICML-2024.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a modeling framework for uncertainty quantification in Generative Flow Networks given the uncertainty in the reward function. The authors suggest to train a proxy model that maps the rewards (or low-dimensional reward representations) into the probability distributions over GFlowNet generation trajectories. To do this modeling step, he authors use a low-dimensional reward representation based on VAE and then fit a system of orthogonal polynomials (Hermite polynomials in most examples) in order to estimate the logits of GFlowNets forward probabilities. With this model, it is possible to estimate the uncertainly in the trajectory-wise generation probabilities without retraining the whole GFlowNet from scratch for each particular reward design. The authors demostrate their approach on a number of synthethic benchmarks, including hypergrids, continuous hypergrid, symbolic regression, and Bayesian structural learning (DAG generation).

### Strengths
The paper is well-written and addresses clearly the problem of uncertainty quantification of GFlowNet-induced policy. The sources of the problem (e.g., the reward model trained from the data) are clear, and the motivation for the study is clear.

### Weaknesses
1. While the paper is rather well written, I feel that its real applicability is rather limited, despite the claims of broader applicability (such as LLMs mentioned in conclusion). It is a classical statistical approach to aprpoximate the unknown mapping from rewards to policies with a polynomial basis, yet this method has its own drawbacks, as all methods in parametric statistics (the curse of dimensionality). This mapping can be non-smooth and complicated, especially in the claimed real-world applications, and the paper does not provide an argument why should we expect that such a mapping can be well approximated by a small-order polynomial.

2. The comparison between surrogate-predicted and ensemble-based policy distributions is visual and qualitative. No quantitative metrics such (KL divergence, total variation distance) are reported.

3.  Is it not clear that the mapping $\Lambda_t$ is well defined. It might be the case that there are more than 1 forward policies in gflownet, each one inducing the correct terminal objects distribution.

4. The authors did not provide theoretical analysis for their method.

Minor issues:

- The chosen notation with $\widehat{R}$ for the true expected reward is rather confusing for readers with statistical background;

- The polynomial chaos expansion in line 176 is a bit inaccurate, as $Y$ is expressed in this case not as a polynomial, but as a series of polynomials.

### Questions
1. Is it possible to consider the approach with more expressive functional family? It is not clear if the basis of polynomials is representative enough, especially beyond the Gaussian latents used in the examples. Is it possible to adopt RKHS or some ideas from Gaussian processes?

2. The polynomial-based regression is known to degrade quickly given the curse of dimensionality. How to choose the degree of approximating polynomial in more "real" applications? At first glance, reader would expect that this degree should quickly increase, making the proposed algorithm an interesting, yet impractical, approach.

3. Is the performance of the surrogate sensitive to the chosen latent representation (for example, for different VAE retraining or different latent dimension of the VAE)? 

4. Are there any accessible metrics between the policy distribution estimated by the algorithm and the "exact" one, with full retraining of GFlowNet for every reward? One can measure some characteristics, such as total variation distance, Jensen-Shannon divergence, etc.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents an approach for uncertainty quantification in GFlowNets. The method learns a mapping from reward functions to a lower-dimensional latent space, after that learning a surrogate model based on polynomial chaos expansion from a small number of GFlowNets trained on given reward functions. The obtained model can be used estimate the uncertainty in the policy given uncertain rewards. The approach is evaluated on discrete and continuous grids, symbolic regression and bayesian structure learning tasks.

### Strengths
Up to my knowledge, the paper presents the first attempt at uncertainty quantification in GFlowNets, which is an interesting and important research direction. The presented methodology is highly novel in the context of GFlowNets.

### Weaknesses
I am struggling to understand the method presented in Section 3.2, which is, in my opinion, the central part of the paper. Do I understand correctly that a separate PCE must be learnt for each pair (state, action) in the environment? Why is there a sum over actions in the loss in the Equation 6? Are coefficients $c_j$ separate across different states and actions, or must they be the same? Why doesn't the predictive model formally depend on the state? If a separate PCE must indeed be learnt for each pair (state, action), I do not really understand the utility of the proposed approach since it cannot generalize between states.

Section 2.3 states that the object of interest is a collection of random variables corresponding to the forward transition probabilities in each state of the trajectory. However, I find this formulation to be a bit confusing since trajectories can have different lengths, thus the size of this collection would also be a random variable. Wouldn't it be more natural to state that the object of interest is the forward policy itself, i.e. a family of probability distributions over actions in each possible state?

In Section 3.1 it is stated that in the case when the output is a discrete probability distribution, logit transformation is applied to probabilities, a separate PCE is fit on each action, and then the result is transformed back into a distribution using softmax. However, applying logit transformation to each element of a probability distribution, and then applying softmax would generally give a different probability distribution than the one we started with. Because of this, I also find this part confusing.

Although the experimental evaluation is conducted on a number of different environments, all of them are either toy (discrete and continuous grids) or very small scale, as symbolic regression task has trajectories of very small length (5) containing symbols from a small set of terms (9), and bayesian structure learning task is limited to 5 variables. I believe that experiments larger in scale are crucial to demonstrate the practical utility of the proposed method.

In addition, line 426 states that a certain trajectory ([2,—,x,+,sin,x]) is considered in the experimental evaluation in the symbolic regression task. I find such experimental setup where only one trajectory is used to evaluate the approach too limited, as it is generally difficult to draw conclusions from the experiment when the model is tested on such a small set of objects.

Finally, I point out a relevant paper that considers a GFlowNet-like problem formulation with uncertain rewards, which the authors may find interesting [1].

References:\
[1] Jiralerspong et al. Discrete Compositional Generation via General Soft Operators and Robust Reinforcement Learning. 2025

### Questions
0) See Weaknesses.

1) It is stated that hypergrid task allows actions of moving left, right, up, and down (line 247). Wouldn't this result in an environment with cycles? The usual GFlowNet formulation works in acyclic DAG environments. There are works that consider GFlowNets in non-acyclic environments [1, 2], but then this should be additionally discussed as this is not a standard setting.

References:\
[1] Brunswic et al. A Theory of Non-Acyclic Generative Flow Networks. AAAI 2024\
[2] Morozov et al. Revisiting Non-Acyclic GFlowNets in Discrete Environments. ICML 2025

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a surrogate modeling framework for uncertainty quantification in Generative Flow Networks under epistemic uncertainty in the reward function.
The authors train a small ensemble and fit a Polynomial Chaos Expansion surrogate instead of training multiple GFNs on perturbed rewards. The surrogate maps low-dimensional representations of rewards to policy statistics. This surrogate enables cheap Monte Carlo estimation of the marginal distribution over policies induced by uncertainty in the reward. Thus, retraining additional GFNs is not needed.
Experiments cover discrete grid, continuous grid, symbolic regression, and Bayesian network structure learning and show that the surrogate reproduces ensemble uncertainty at a reduced cost.

### Strengths
The paper addresses a clear point that Bayesian or Monte-Carlo ensembles for UQ in GFNs are computationally infeasible. The motivation of a surrogate-based hence is clear.
The separation between epistemic uncertainty in rewards and randomness in training (SGD, initialization) is stated clearly.
Experiments cover both discrete and continuous environments.

### Weaknesses
1.Line 125 is misleading: the distribution is built for policies, integrating out reward functions.
2. Though the idea to approximate the mapping with polynomials is appealing, this mapping can be highly non-smooth, discontinuous, and multimodal. No argument or empirical check supports that a low-order polynomial expansion provides a valid approximation.
3. The analysis assumes (Y=f(X)) (the policy statistics) have finite second moments, but no proposition proves that for general GFlowNet training. The PCE representation is mathematically invalid without bounded variance.
3. Authors choose the degree of the PCE arbitrary, no mathematical explanation provided. “lack of impact” as stated is not a justification. 4.Moreover, the influence of the approximation basis should be explored as an ablation. Convergence analysis is missing.
5. The comparison between surrogate-predicted and ensemble-based policy distributions is visual and qualitative. No metrics such as KL divergence, total variation distance, or correlation are reported.

### Questions
1. Should “marginal distribution over (\mathcal{R})” be “marginal over policies (\pi)” obtained by integrating over (\mathcal{R})? Please clarify and restate formally what distribution your method aims to estimate.
2. Under what conditions on the reward distribution and training process does the policy statistic (Y) have finite variance? Can you provide empirical evidence or theoretical reasoning supporting this assumption?
3. How is the polynomial degree selected? Why was no adaptive truncation, error estimator, or cross-validation procedure applied? What is the relationship between number of polynomial terms and GFN state space dimensionality?
5. Have you computed any quantitative divergence metrics (KL, JS, total variation, or correlation) between the surrogate-induced policy distribution and the ensemble reference? If not, how can readers assess the calibration or bias of the surrogate?
6. Why was Polynomial Chaos chosen over Gaussian Processes or kernel surrogates, which could model non-smooth mappings and yield built-in uncertainty estimates?
7. How sensitive is the surrogate performance to the chosen latent representation (PCA vs VAE)?
8. Do you have any empirical justification that the policy manifold in latent reward space is smooth enough for a low-order polynomial approximation?

### Soundness
2

### Presentation
3

### Contribution
2
