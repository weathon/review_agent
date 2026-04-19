# Learning Multi-Index Models with Neural Networks via Mean-Field Langevin Dynamics

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
We study the problem of learning multi-index models in high-dimensions using a two-layer neural network trained with the mean-field Langevin algorithm. Under mild distributional assumptions on the data, we characterize the effective dimension $d_{\mathrm{eff}}$ that controls both sample and computational complexity by utilizing the adaptivity of neural networks to latent low-dimensional structures. When the data exhibit such a structure, $d_{\mathrm{eff}}$ can be significantly smaller than the ambient dimension. We prove that the sample complexity grows almost linearly with $d_{\mathrm{eff}}$, bypassing the limitations of the information and generative exponents that appeared in recent analyses of gradient-based feature learning. On the other hand, the computational complexity may inevitably grow exponentially with $d_{\mathrm{eff}}$ in the worst-case scenario. Motivated by improving computational complexity, we take the first steps towards polynomial time convergence of the mean-field Langevin algorithm by investigating a setting where the weights are constrained to be on a compact manifold with positive Ricci curvature, such as the hypersphere. There, we study assumptions under which polynomial time convergence is achievable, whereas similar assumptions in the Euclidean setting lead to exponential time complexity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the sample and computational complexities of the mean field Langevin algorithm (MFLA). The analysis is centered aroung a newly introduced ``effective dimension'' which measures the complexity of the learning task (in terms of the data distribution and the teacher model). Under several regularity conditions and conditions on various constants, it is proven that the sample complexity and the computational complexity can both be controlled in terms of the effective dimension, which can be significantly smaller than the ambient dimension of the input space. The authors estimate (theoretically) this effective dimension on several examples. While the sample complexity bound is very satisfying and scales linearly in the effective dimension, the computational complexity may scale exponentially in the effective dimension and linearly in the ambient dimension. While this might be an issue in the general case, the authors show that this issue can be alleviate in a compact Riemannian setting. Improving the exponential dependence in the Euclidean setting remains open.

### Strengths
- Both the statistical and the computational complexity are bounded in terms of a newly introduced effective dimension, which is a measure of the complexity of the learning task. It is interesting that the same quantity can be used for both notions of complexity.

 - The study of Mean-Filed Langevin Dynamics may provide important insights on the sample complexity of modern deep learning models.

 - The limitations of the paper are well-discussed.

 - The paper is well-written. Even though the subject is quite technical, good technical background and references are provided, making it sufficiently clear. The assumptions and theorems are clearly stated.

### Weaknesses
- The computational complexity may scale exponentially with the effective dimension (in the worst case). While the authors show that this might not be an issue in a compact Riemannian setting, the issues persists in the Euclidean setting. Additional discussion on how to go beyond this Riemanian setting could enhance the paper. In particular, what are (maybe intuitively) the conditions that could lead to non-exponential complexity?

 - As discussed in the paper, the analysis is restricted to a particular class of 2 layers neural networks. Moreover, the analysis is restricted to ``simple'' learning tasks, ie, the teacher models uses $k \ll d$ neurons. It could be interesting to further discuss what the results become when $k$ Is large.

### Questions
- Can we derive a LSI like in Proposition 2 in a setting where there is no explicit regularization, but instead we make a dissipativity assumption on the model? Could it be of any interest to extend the analysis beyond regularization?

 - Could you explain further the meaning of the expectation $E_\xi [\rho(\xi)]$ is Equation (3.6)?

 - Could part of the analysis be extended to more complex function classes than 2 layers neural networks?

 - Is it possible to construct explicit examples where the computational complexity becomes exponential in the effective dimension?

### Soundness
4

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
3

### Summary
In this work, the authors study the task of learning a general multi-index model using neural networks and the 
mean-field Langevin Algorithm (MFLA). Specifically, the authors introduce a notion of effective dimension in this context, 
which can potentially be much smaller than the ambient dimension when the input distribution is spiked the spikes 
align with the relevant subspace of the target multi-index model. Then, the author derive bounds for the sample and 
computational complexity of MFLA based the effective dimension in both Euclidean and Riemannian settings.

### Strengths
* Overall, the paper is well-written, and the presentation is clear. 
* Proving polynomial convergence guarantees for MFLA is an important and challenging subject. This paper makes process in 
  this direction by proving a bound that is exponential in the effective dimension. They also show that for certain 
  models, the effective dimension can be as small as $O(\mathrm{poly}\log d)$, which leads to a quasi-polynomial 
  convergence guarantee.

### Weaknesses
* My main concern with this paper is that the effective dimension, as defined in Definition 1, is essentially about 
  how spiked the input distribution is (and whether it aligns with the target function) and does NOT capture the potential 
  low-dimensional structure of the target function. 
  In fact, if the input distribution is isotropic, the effective dimension (as defined in Definition 1) will be
  $d$ even when the target function is a single-index model. 
  This limits the applicability of this paper's results, as it does not cover even the simplest Gaussian single-index case.
* The comparison with the information/generative exponent (Section 3.1) is misleading. The authors suggest that one 
  advantage of the MFLA analysis over the information/generative exponent analysis is that the sample complexity is 
  always $\tilde{O}(d)$ (in the isotropic case), while for SGD, (the bounds on) the sample complexity can be $\tilde{\omega}(d)$
  when the information/generative exponent is larger than $2$. However, this is not true if we are allowed to have 
  exponentially many neurons, as assumed in the setting of Corollary 4. It has been shown in the information exponent paper [1]
  that after the neuron attains a constant correlation with the ground-truth direction, the number of samples needed 
  in the final "descent" phase is $\tilde{O}(d)$ regardless the information exponent. Meanwhile, if we have 
  exponentially many neurons, we can ensure that some neuron will have a constant correlation with the ground-truth 
  direction at initialization with at least constant probability. In addition, the number of steps needed in the 
  information exponent analysis is also $\tilde{O}(d)$, which is better than the exponential bound given in Corollary 4. 
* In Sec. 4 (the Riemannian setting), the authors require the existence of a good distribution $\mu$ with a small loss 
  and $H(\mu | \tau) = o(d)$ where $\tau$ is the uniform distribution on the manifold. They justify the second condition 
  in Prop. 8 by showing that if the ground-truth distribution is given by $\mu^* \propto e^f \mathrm{d} \tau$, then we have 
  $H(\mu^* | \tau) \le \mathrm{osc}(f)$. However, this bound is vacuous if $\mu^*$ is supported only on a few points as $f$ would then take $\infty$ (or a large value if 
  we smooth $\mu^*$). This excludes models such as the single-index model and $k$-multi-index model with $k$ being small.
  To be fair, the authors do mention this limitation in the paper. I just think this assumption is too strong/unnatural. 
  It would be great if the authors could give a natural example where this condition holds. 


[1] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Online stochastic gradient descent on non-convex losses from high-dimensional inference. 2021

### Questions
See the Weakness part of the review.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper examines the sample and computational complexity of learning general multi-index models in high-dimensional settings using two-layer neural networks and the mean-field Langevin algorithm. Under very mild assumptions, the authors successfully demonstrate that the sample complexity for such learning scales linearly with the effective input dimension, $d_{\text{eff}}$, which is information-theoretic optimal. However, the computational complexity typically scales exponentially with $d_{\text{eff}}$, which is also in general unavoidable. To address this, the authors explore specific conditions under which computational complexity might have a polynomial dependence on $d_{\text{eff}}$. They identify certain settings—such as particular target functions and constraints on weights to lie on the hypersphere—that allow for the desired polynomial-time convergence.

### Strengths
1. This paper is well-written, clear, and mathematically rigorous.

2. Theoretical guarantees and insights into feature learning in deep learning are notoriously challenging, so any solid progress in this area is highly valuable. In particular, I find the paper’s second contribution quite interesting. Identifying cases where polynomial-time complexity (and thus efficient learning) can be achieved under the mean-field regime is of great importance, and this work is among the first to address this question, even if the current assumptions are somewhat restrictive.

### Weaknesses
I feel that the result in the first part is somewhat expected (though I recognize that proving it rigorously is still a quite valuable accomplishment. For the second part, I wonder if it might be possible to establish a polynomial-time convergence guarantee for a broader class of target functions (beyond the teacher neuron setup) by furthermore just considering the case where we restrict the weights just to the sphere. In general, I am convinced that a more detailed analysis/discussion would be valuable regarding the setups in which polynomial-time convergence with optimal sample complexity can be achieved, even when the input distribution is inherently high-dimensional, such as standard normal distribution with mean zero and covariance $\Sigma = I_d$.

### Questions
See the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores learning multi-index models in high dimensions using a two-layer neural network trained with the mean-field Langevin algorithm (MFLA). The authors identify an "effective dimension" that can significantly reduce sample complexity by adapting to low-dimensional structures in data. While worst-case computational complexity may still grow exponentially, they propose a solution to achieve polynomial-time convergence by constraining weights to a compact manifold under specific conditions.

### Strengths
* The presentation is good, which is easy to follow.
* Discussion on previous works is extensive and informative, and comparison with previous works is clear.
* This paper uses MFLA to learn multi-index model and consider the effective dimension, which is novel to me.
* This paper achieves optimal sample complexity, albeit leading to exponential dependence of $d_{eff}$ in $\mathbb{R}^d$. But the authors well explain this phenomenon and improve this dependence in Riemannian manifold under some curvature conditions.

### Weaknesses
* Discussion on Assumption 1 is weak.
* Some notations are not clearly stated.
* This paper is short of some simulation.

See more details in Questions below.

### Questions
* The authors claim "the Brownian noise can help escaping from spurious local minima and saddle points" after presenting (2.3). Can you provide some reference of this statement? This is different from SGD which uses inexact oracle (stochastic gradient) leading to different randomness at each iteration. Adding Brownian motion is equivalent to adding randomness with the same magnitude at each iteration.

* In the row between 230 and 231, should the weights be $W\in\mathbb{R}^{(2d+2)\times m}$ instead of $W\in\mathbb{R}^{2d+2}$?

* What is $\mathbb{E}_{\xi}[\rho(\xi)]$ in Thm 3?

* Assumption 1 is restrictive to me. Could you provide some references which also consider this assumption on dataset? Or could you verify this assumption on some dataset? Is this assumption only for the proof? If so, why you need this assumption in your proof should be clearly stated in the main paper. It would be helpful to provide some proof sketch of Thm 3 discussing that.

* Since you establish better sample complexity under subgaussian data assumption, it would be interesting to do more simulation even with artificial mean-zero and subgaussian data for empirical comparison with other results in Table 1.

### Soundness
3

### Presentation
3

### Contribution
3
