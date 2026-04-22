# Diffusion-DFL: Decision-focused Diffusion Models for Stochastic Optimization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 0

## Abstract
Decision-focused learning (DFL) integrates predictive modeling and optimization by training predictors to optimize the downstream decision target rather than merely minimizing prediction error. To date, existing DFL methods typically rely on deterministic point predictions, which are often insufficient to capture the intrinsic stochasticity of real-world environments. To address this challenge, we propose the first diffusion-based DFL approach, which trains a diffusion model to represent the distribution of uncertain parameters and optimizes the decision by solving a stochastic optimization with samples drawn from the diffusion model. Our contributions are twofold. First, we formulate diffusion DFL using the reparameterization trick, enabling end-to-end training through diffusion. While effective, it is memory and compute-intensive due to the need to differentiate through the diffusion sampling process. Second, we propose a lightweight score function estimator that uses only several forward diffusion passes and avoids backpropagation through the sampling. This follows from our results that backpropagating through stochastic optimization can be approximated by a weighted score function formulation. We empirically show that our diffusion DFL approach consistently outperforms strong baselines in decision quality. The source code for all experiments is available at https://github.com/GT-KOALA/Diffusion_DFL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a novel DFL approach that leverages diffusion models to represent the distribution of uncertain parameters. To mitigate the issue raised by using diffusion models, i.e., sequential sampling procedure, the authors develop two algorithms: reparametrization and score function. Experiment results demonstrate that using diffusion models outperforms prior baselines.

### Strengths
- Utilizing diffusion models in decision-focused learning is sound and novel
- Authors successfully mitigate issues raised when using diffusion models in decision-focused learning
- Experiments and further analysis are comprehensive

### Weaknesses
- The authors say one benefit of utilizing diffusion models is capturing a multi-modal distribution. Does the task provided benefit when we capture the multi-modal distribution? It would be nice to persuade readers why we should capture multi-modal distribution in decision-focused learning.

### Questions
- I'm not familiar with decision-focused learning. Are there any other benchmarks for evaluating the performance of decision-focused learning? It seems that the three optimization tasks provided in the paper are not complex enough.

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
3

### Summary
This paper introduces Diffusion-DFL, a first framework that integrates diffusion probabilistic models into decision-focused learning (DFL) — an approach that trains predictive models to directly optimize downstream decision quality rather than prediction accuracy.
Unlike prior DFL methods, which rely on deterministic or simple probabilistic predictors (e.g., Gaussian), Diffusion-DFL uses a conditional diffusion model to capture potentially complex and multimodal uncertainty in the parameters of a downstream stochastic optimization problem. The decision layer then optimizes the expected cost under this learned distribution.

To train the model end-to-end, the authors develop two gradient estimators:
1) Reparameterization estimator: backpropagates through the diffusion sampling process, providing an exact but memory-intensive gradient.
2) Score-function estimator: avoids backpropagating through diffusion by applying the log-trick, estimating gradients via the model’s ELBO  — drastically reducing memory and compute cost with minimal accuracy loss.

Experiments on synthetic manufacturing, power scheduling, and portfolio optimization tasks show that Diffusion-DFL consistently yields better decision outcomes than both two-stage and deterministic DFL baselines, while the score-function version achieves similar performance to reparameterization with over 400× lower memory usage.

### Strengths
The paper introduces the first integration of diffusion models into the decision-focused learning framework, allowing richer uncertainty modeling than deterministic approaches.

- The proposed score-function estimator is a practical contribution: it replaces costly pathwise differentiation with an ELBO-based surrogate, maintaining decision performance while reducing memory and compute by several orders of magnitude.
- The methodology is well-grounded theoretically (via implicit differentiation and the log-trick) and empirically validated on multiple domains, showing consistent improvements in decision quality.
- Clear ablation studies (memory vs. accuracy, variance reduction, decision dimension scaling) demonstrate that the approach is both effective and scalable.

### Weaknesses
- The score-function approximation relies on an implicit assumption of local smoothness between the model’s likelihood and decision-loss landscapes, which is not formally analyzed or guaranteed, although supported empirically. 
- Experimental settings are limited to convex or low-dimensional decision problems; performance on more complex or nonconvex tasks (where diffusion expressiveness would matter most) remains unclear.

### Questions
The score-function estimator seems to rely on local smoothness between the diffusion model’s likelihood landscape and the downstream decision cost; can the authors discuss or empirically validate when this assumption might fail, e.g., in highly nonconvex or discontinuous decision problems? Have the authors tested Diffusion-DFL in nonconvex or combinatorial settings, where diffusion’s multimodality could be beneficial?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a Decision Focused Learning method where the output of the estimator is stochastic, rather than being a point estimate, and the downstream problem is also a stochastic optimization one. As a main advantage, the approach can better address decision problems where no deterministic formulation can result in the correct optimal solution under uncertainty, e.g. because any individual realization leads to an either maximizing or minimizing a variables, while the best value under under uncertainty would be somewhat in the middle.

The authors use a diffusion process to generate their estimate and present two techniques to enable differentiation. The first -- technically more precise, but expensive in terms of computation time and memory -- is based on a reparameterization of each diffusion step and implicit differentiation. The second -- much lighter from a computational standpoint -- is based on Score Function Gradient Estimation and an Evidence Lower Bound. Using a diffusion process enables modeling complex distributions

The approach is tested on selected benchmarks from the literature, and at least in one realistic (but still far from true practical applicability) problem it achieved a very significant advantage in terms of solution quality.

### Strengths
The insight about the limited expressiveness (for lack of a better term) of deterministic DFL is correct and definitely useful, even if not entirely novel.

The empirical results on the stock portfolio problem provide evidence that a combination of stochastic prediction and a complex distribution can improve the effectiveness of the DFL approach very significantly, beyond what is possible for either technique. Remarkably, this is true even when the two-phase approach is rather expressive and should be capable of approximating well the ground truth parameter distribution (the PFL diffusion method).

The suggested differentiation method based on SFGE + ELBO appears to be rather good at improving the computational efficienty, with no major drop in terms of solution quality.

The SFGE/ELBO method also seems to be applicable, with a few minor modifications, to a _much_ more general setting than the one presented in eq. (2). Since the method only requires function evaluation, it seems it could easily account for solution repairs costs (similar to the other SFGE method by Silvestri et al.) and therefore, combinatorial problems, two-stage problems, or decision problems with uncertain parameters in the constraints.

### Weaknesses
It is my opinion that the paper strengths outweigh its weaknesses. That said, I can find some, of which a few are major and a few minor.

The first major weakness I can find is that the approach requires solving a stochastic optimization problem per example, and per epoch. Stochastic optimization problems can be very challenging to address and the corresponding solution methods are well known for their rather poor scalability. Combined with the already computational cost of DFL training, this significantly reduces the practical appeal of the method. That said, given the improvements reported in one benchmark, I think there are definitely practical settings with relatively lightweight optimization problems that can benefit from the proposed technique.

The second major weakness is again related to applicability. If enough data is available and a sufficiently expressive (distributional) ML model is used, the approximation error for the ground truth distribution will eventually converge to zero. Under these conditions, a Prediction Focused Learning (i.e. two phase) pipeline should be capable of reaching optimal or close to optimal results, similarly to what happens in a classical DFL setup with non-misspecified ML models. The empirical results corroborate this intuition, with the Diffusion TS method reaching high quality results in the synthetic benchmark. It would seem that the ideal application case for the proposed method is one where a deterministic decision problem is inadequate, a complex distribution is needed, and the problem cost is not well aligned with the likelihood function used in PFL. While situations like this certainly exists, the applicability of the method is also reduced.

Here is a list of some minor weaknesses:

* The fact that deterministic DFL may not be expressive enough for some optimization problems under uncertainty is also established in [1]. Given that this paper is very recent and that the expressiveness claim is not at the center of this work, this issue is definitely minor. That said, it might be worth including a citation
* The Gaussian method by Silvestri et al. is not stochastic, but deterministic: the Gaussian distribution is used just for smoothing, while the optimization problem works with a single parameter vector (both at training and inference time). By reading Appendix A.5, it seems the authors instead implemented an actual Gaussian stochastic DFL pipeline, based on their own techniques, but using a simpler distribution. This is again a minor issue, since the original method by Silvestri et al. would likely work even worse in this setting (though it would be much more scalable)
* This might be my fault, but I did not easily find the number of samples used in the empirical evaluation

[1] Schutte, Noah, et al. "Sufficient Decision Proxies for Decision-Focused Learning." arXiv preprint arXiv:2505.03953 (2025).

### Questions
* Can you confirm that the Gaussin DFL method is in fact a stochastic method, i.e. using a stochastic optimization problem with multiple samples downstream?
* How many samples were used in the empirical evaluation?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper studies a paradigm termed decision-focused learning, where the goal is to solve a decision-making problem that depends on uncertain parameters that must be learned. The authors develop a computational pipeline to propagate uncertainty from the learning procedure to the decision procedure using diffusion models, and show that incorporating uncertainty results in improved decision-making performance.

### Strengths
I note two key strengths:
* **Writing attention to detail.** This paper is reasonably polished, and it is clear the authors know how to write a paper to an appropriate technical standard. 
* **Reasonably-careful experiments.** The authors not only demonstrate good performance on a key benchmark of interest compared to pure-deterministic methods, they also include various ablation studies to check individual parts of their conclusions.
 
In an ordinary review, I would offer many more details on these points. However, in light of a very serious issue which I will bring to light shortly, I will omit these in the interest of time.

### Weaknesses
This paper has a critical weakness because of which I am confident it needs to be rejected: **a very significant portion of this work's findings are known and covered in standard textbooks, but are presented as novel.** In particular, the paper's actual novelty is limited to introducing diffusion models to well-known decision-making paradigms. If re-written completely in a manner that reflects this contribution, I do believe the authors have done sufficient development to warrant a paper, but the gap between the current manuscript and what would be appropriate is, to say it again, sufficient to warrant a complete rewrite. This is unfortunately far too much to address in the rebuttal phase.

Allow me to elaborate:
* **Decision-focused learning is presented as a novel emerging paradigm, but is not new, because the authors' formulation is equivalent to a (stochastic) contextual bandit.** Let me be completely precise: the authors define their setting via the optimization objective $\mathbb{E}f(y^\star, z^\star(x))$ where $(x,y^\star) \sim D$ and $z^\star$ is parameterized by some vector $\theta$ which is to be optimized. Let us define the notation $\phi^\star(z) = f(y^\star, z)$, so the objective becomes $\mathbb{E} \phi^\star(z^\star(x))$ where $(x,\phi^\star) \sim D$. With this notation change, it is clear that the authors' setting is a (stochastic) contextual bandit: we are given a random context vector x which is correlated with some random objective $\phi^\star$ (either in a known way in the Bayesian case, or unknown way in the non-Bayesian case), and our goal is to choose the optimal map from context to actions. With this notational link made, there is a substantial literature of prior work the authors could consider and compare with, which is not mentioned in this work. Note, in particular, that if D is assumed known, maximizing the objective is equivalent to minimizing regret, which is the usual performance objective considered because it behaves much better mathematically.
* **Using stochastic models inside dynamic programming is not new.** Indeed, this is covered in Chapter 1.5 of "Sequential Decision Analytics and Modeling" by Warren Powell, to name just one book that includes this topic.
* **In the contextual bandit literature, the need for uncertainty is well-understood.** This is the basis not only for classical upper confidence bound algorithms, but also methods based on approximate dynamic programming such as expected improvement algorithms from the Bayesian optimization literature.
* **Even if one is interested purely in practical aspects involving diffusion models, one should benchmark against standard algorithms.** For example, I see no reason why, in light of the above equivalence, any DFL algorithm in the authors' sense cannot directly be benchmarked against contextual kernel UCB.
* **Section 4 consists entirely of known ideas.** In particular, the application of the reparametrization trick to handle objectives defined as expectations of this kind is, at this point, completely standard and need not be introduced except in code. The discussion on differentiating through the solution of a parameterized optimization problem is also standard, though less than reparametrization tricks - this should be deferred to an appendix.
* **In Section 5, the key ideas involving score estimation are essentially the same as the standard REINFORCE trick from reinforcement learning. Moreover, the algorithmic justification for using ELBO as a surrogate to the score is not technically sound.** The authors' argument, essentially, is to state that $f(x) \leq g(x)$ on basis of an evidence lower bound, and argue that this implies $\nabla f(x) \approx \nabla g(x)$, which is not necessarily true even if $|f(x) - g(x)| \leq \epsilon$ for an arbitrary given $\epsilon$. Rather than relying on a theoretical explanation that does not follow, I believe the authors should instead lean on their empirical results in Figure 2, which shows these are close, as this avoids a misleading explanation.

Summarizing, I believe this work's only contribution which is novel is the exploration of diffusion models in the context of stochastic dynamic programming. This contribution is a reasonable fit for ICLR and could make for a good paper, but the paper needs to be re-written in a manner that reflects this.

### Questions
Have I missed some critical contribution in the above points that should justify a different conclusion?

### Soundness
3

### Presentation
4

### Contribution
1
