# Provable Derivative-Free Inference with Score-Based Generative Priors

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
A growing trend in solving inverse problems is to use pre-trained score-based generative models (SGMs) as plug-and-play priors. This paradigm retains the generative power of SGMs while allowing adaptation to different forward models without requiring re-training. In parallel, derivative-free posterior sampling algorithms have gained increasing attention for solving inverse problems where the derivative, pseudo-inverse, or full knowledge of the forward model is unavailable or impractical to compute. Despite their success, these methods lack principled foundations and provide no convergence guarantees to the true posterior distribution or to its $\varepsilon$-accurate approximation. We propose \textit{zeroth-order annealed plug-and-play Monte Carlo (ZO-APMC)}, the first principled derivative-free framework for solving general inverse problems that requires only forward-model evaluations and a pre-trained SGM prior. We derive complexity bounds for obtaining samples with $\varepsilon$-relative Fisher information under a non-log-concave likelihood distribution and, under a Poincar\'e inequality assumption, $\varepsilon$-accuracy in total variation distance, and we establish weak convergence of ZO-APMC to the target posterior. We verify our theory with numerical experiments and demonstrate its performance on both linear and nonlinear inverse problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors aim to sample from $\pi(x|y)\propto p(y|x)p(x)$ with  given diffusion prior $p(x)$ 
and known unnormalized Boltzmann density $p(y|x)$ using an annealed Langevin algorithm 
$\text{d} x_t = [\nabla \log p(y|x_t)-S_\theta(x_t,\sigma_t)] \text{d} t + \sqrt{2} \text{d} B_t$ 
with learned score $S_\theta$. 
The authors consider 
0-th-order annealed plug and-
play Monte Carlo (ZO-APMC) that requires only forward-model evaluations
and a pre-trained SGM prior and derive complexity bounds for obtaining samples
with $\varepsilon$-relative Fisher information under a non-log-concave likelihood distribution (Thm 1)
and, under a Poincar\'e inequality assumption, also $\varepsilon$-accuracy in TV
distance (Cor 1). Weak convergence of ZO-APMC to the target posterior is established in Thm 2.
The ideas of the proofs are based on existing literature.

### Strengths
The paper is  well written. Having explicit bounds on the accuracy with which an algorithm based on a trained diffusion model can approximate the posterior is desirable.

I partially red the proofs which were correct.

### Weaknesses
Realistic estimates for practical applications or just mathematical playground: 
the weakness lies in the strong assumptions 1-5 which seem to be not satisfied for the numerical examples. 
(In Remark 1says that Assumption 1 is not satisfied for Gaussian noise. But all experiments have Gaussian noise likelihood?)
The bounds seem to scale badly: the number of forward model evaluations scales with $\mathcal{O}(d^7L_m^3\epsilon^{-4})$,
where $\epsilon$ is the desired distance between $\nu_t$ and $\pi(x|y)$.

### Questions
*  Proposition 1: $f_\mu$ is not explained.
Why the bound indicates a variance reduction? In other words, why a small variance of $e_k$ implies a small one for $g_k$ and is smaller than the ZO gradient estimator?
* In the experiments I like to see a comparison with the  ZO estimater.
*Have the experments something to do with the  theoretical bounds?
* Could the authors run a Laplacian noise example?
* in Assumption 1, what is $f_2$?
* There is a typo in (24), it should be $B_k=ref$
*  the expression line 889 looks cumbersome: $f_mu$

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose zeroth-order annealed plug-and-play Monte Carlo (ZO-APMC), an algorithm for sampling from a posterior with a pretrained score-based generative model (SGM) prior and a likelihood where only a black-box forward model is given. The key idea of the algorithm is to approximate the gradient of the likelihood with finite differences. The focus of the paper is deriving theoretical bounds and convergence guarantees for the algorithm. The authors provide numerical experiments on several inverse problems verifying convergence and reasonable image reconstructions.

### Strengths
* The proposed algorithm is quite principled, and the theoretical results seem useful.
* The method is validated with solid numerical experiments and baselines.

### Weaknesses
* The algorithmic contribution is quite simple, although I would not consider this a weakness when deciding the final rating. It is just that approximating the likelihood gradient with finite differences seems like an easy and obvious step that one would come up with when making APMC work without access to the forward model gradient. 
* PLEASE REMOVE TABLE 3. I find these types of tables dangerously misleading. Just because a method does not have convergence guarantees YET does not mean it is not possible to derive them or that they do not in fact converge to a more accurate posterior. Table 3 implies that many other methods are not theoretically grounded just because their authors did not derive convergence guarantees, which may lead readers to not give a fair chance to other algorithms. Rather, I urge the authors to spend more time in the paper describing which situations their algorithm is well-suited for and which situations other algorithms would be better suited for.
* In general I find the presentation overly dense without clear intuition. The theoretical results seem impressive, but it is difficult for me to verify them or understand their intuition. It is unclear what the pros and cons of the algorithm are.

### Questions
* The authors say in lines 96-97 that they address the inverse problem with MAP estimation, but all the theoretical results and experiments imply that they are sampling from the posterior. Please clarify which you are actually doing.
* I will increase my rating if the authors agree to remove table 3 and instead provide a more balanced discussion of the pros and cons of their algorithm compared to others for different types of applications.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a Langevin inspired zeroth order annealing method for image restoration. The authors rigorously show that they approximate the ground truth posteriors in their setting, showing the scaling in number of forward evaluations. The approach is evaluated on a toy example, MRI, Black Hole imaging and a Navier Stokes example.

### Strengths
The paper is mostly well written, the motivation is relatively clear, and the math is proved rigorously. The evaluation is extensive.

### Weaknesses
1) I find the description in 3.1 of the algo to be insufficient for understand it, I would rather present non detailed theoretical statements in favor of describing the method properly. In particular the "accept-reject" like step is super unclear and needs elaboration. Also for the small-large batch size an intuitive explanation would help. 

2) The theoretical restrictions are quite large, in particular Gaussians are usually not contained. I am also struggling with the main take away of the theory. If there was "informal" statements in the main paper, one could distill the statements into takeaways for practitioners, ignoring technical constants and looking at the main scaling behavior.

3) assumption 5 bites itself with the usual diffusion assumptions. At least in the diffusion case, one can show that the score needs to explode to "approximate" an empirical distribution. The authors now require that the learned score is bounded, and the difference to the true score is bounded, which I dont think is possible for empirical measures, and explodes for smoothing going to zero. (see for instance "Score based generative models detect manifolds")

4) I am also skeptical of the main motivation. The authors state examples like if the forward operator is a PDE solution, but calculating gradients (if the pde is already implemented) is in O(1). Of course there can incur instabilites, but they might also happen in the usual forward eval without the gradients. If the authors could elaborate on their specific setting, they would apply their algo in, I would be happy. The black hole example, I would be interested on the forward operator.

### Questions
see weaknesses.

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
The paper proposes ZO-APMC, a derivative-free posterior sampling method for inverse problems that uses pre-trained score-based generative models (SGMs) as priors. The main claim is to provide the first formal convergence guarantees for such a "black-box" setting, where only forward-model evaluations are available. The authors combine a zeroth-order (ZO) gradient estimator with a variance reduction technique and an annealing schedule. Theoretical bounds on convergence are derived, and the method is empirically evaluated on several inverse problems, including MRI reconstruction, black-hole imaging, and Navier-Stokes equations.

### Strengths
1. The paper is overall well-written.
2. The author conducted extensive derivations, although most of the formula proofs felt over-engineered in an attempt to complete the proofs.

### Weaknesses
1. The central premise of this work is to establish theoretical guarantees for a highly specific, arguably contrived, problem setting: derivative-free posterior sampling with SGM priors. While technically challenging, the practical motivation for this exact combination feels weak. A significant portion of modern inverse problems, especially in imaging, do have differentiable forward models (e.g., via auto-differentiation through simulators). The community is increasingly moving towards differentiable programming. This paper overstates the prevalence of purely "black-box" scenarios where its method would be uniquely essential, making the direction feel more like a niche theoretical exercise than a solution to a pressing, widespread problem.
2. The paper attempts to tame the notoriously high variance of ZO gradient estimators within a stochastic sampling framework. This involves complex machinery like PAGE-style variance reduction and delicate annealing schedules, creating a "tricky" and highly engineered solution. The resulting algorithm is complex, laden with sensitive hyperparameters.
3. Despite the variety of tasks, the experimental setup feels superficial and fails to convincingly demonstrate that ZO-APMC represents a significant practical advancement over existing methods. The results (Table 1) show that ZO-APMC (PSNR 35.29) barely outperforms a simple baseline like DPG (PSNR 32.17) and is still noticeably worse than the gradient-based APMC (PSNR 36.55). 
4. The paper's main selling point is its theoretical guarantees. However, the derived complexity bounds are astronomical (e.g., O(d⁷...)). Such bounds are often so loose that they provide little practical insight into the algorithm's real-world behavior or how to set its hyperparameters.

### Questions
I'm curious to know, besides publishing papers, what is the practical significance of researching inverse problem solving under black boxes?

### Soundness
2

### Presentation
2

### Contribution
1
