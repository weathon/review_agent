# Proximal Diffusion Neural Sampler

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 4

## Abstract
The task of learning a diffusion-based neural sampler for drawing samples from an unnormalized target distribution can be viewed as a stochastic optimal control problem on path measures. However, the training of neural samplers can be challenging when the target distribution is multimodal with significant barriers separating the modes, potentially leading to mode collapse. We propose a framework named **Proximal Diffusion Neural Sampler (PDNS)** that addresses these challenges by tackling the stochastic optimal control problem via proximal point method on the space of path measures. PDNS decomposes the learning process into a series of simpler subproblems that create a path gradually approaching the desired distribution. This staged procedure traces a progressively refined path to the desired distribution and promotes thorough exploration across modes. For a practical and efficient realization, we instantiate each proximal step with a proximal weighted denoising cross-entropy (WDCE) objective. We demonstrate the effectiveness and robustness of PDNS through extensive experiments on both continuous and discrete sampling tasks, including challenging scenarios in molecular dynamics and statistical physics. Our code is available at https://github.com/AlexandreGUO2001/PDNS.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work proposes the “Proximal Diffusion Neural Sampler”. The authors consider the setting of sampling from distribution with given unnormalized densities on both discrete and continuous state spaces. Rather than solving the SOC/sampling problem in one step, they propose an iterative procedure where an intermediate set of path measures are approximated, starting with a simple path measure and a final target path measure. It is shown that these intermediate path measure correspond to a geometric annealing in path space (proposition 3.1). Every step of the optimization problem is solved with proximal WDCE, a variant of WDCE designed for the proximal setup of this work.

### Strengths
- Novel idea that is well-motivated with a technically good presentation.
- A nice and simple unifying framework of existing neural sampler methods. They consider both discrete and continuous setting in one (instead of writing two separate papers, one general principle is presented).
- Great illustration of mode collapse (section 3.1) motivating the work.
- Experiments show improvements that illustrate the theoretical innovations.

### Weaknesses
- For a less-experienced reader, the paper would potentially be hard to follow. I recommend including more sentences explaining intuitions and motivations at intermediate paragraphs.
- The experiments are only on simple distributions. While this unfortunately common in this line of literature, a more complex setting of higher-dimensional distributions would have been more convincing.

### Questions
- Line 72: “path space, which contains all functions from [0, T] to X”. A path measure is only defined for a subset of functions usually (there is a measurable space, etc.)
- The ESS for the LEAPS algorithm reported here is significantly lower than report in their work. Comparing the setting, you have slightly changed the settings (slightly different grid size) or q=4 instead of q=3 for the Potts model. Why did you change this setting? It would be more compelling if you compare to exactly the same parameters than their work.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed Proximal Diffusion Neural Sampler (PDNS) to learn a neural sampler for sampling from unnormalised density functions.   Both continuous and discrete formulation is proposed and verified in this paper.
The paper argus that the reason behind model collpasing is that the sampling task is too far from the prior and hence PDNS breaks down the entire problem with proximal point method, progressively driving the neural sampler towards the target.
This proposed method enriches the family of neural samplers.

### Strengths
The paper is clearly written and easy to follow. It is complete and well structured. It covers both discrete and continuous domains and presents results for each. The proposed algorithm is interesting and offers insights into mode collapse. The empirical evaluation indicates that the approach is promising.

### Weaknesses
While the paper is motivated by mitigating mode collapse, the experimental section does not convincingly demonstrate this. Several benchmarks (e.g., LEAPS) do not appear to exhibit severe collapse. Do you have any explanation to this?


On line 780: 
> Blessing et al.  (2025) couple CE training... In contrast,PDNS utilize
 proximal algorithm with WDCE bjective,which can be applied uniformly to continuous-time
 and discrete-time diffusions,and is designed to be memory-efficient while retaining the benefits of
 denoising or score-matching objectives. 

The mentioned algorithm has quite similar formulation and has strong connections. Blessing et al.  (2025)  works on continuous. But to my knowledge, it should also be able to work on discrete domain. Is this the main difference between PDNS and their approaches? Or is my understanding wrong?

### Questions
See my weakness

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
3

### Summary
Proximal Diffusion Neural Sampler proposes a method an stochastic optimal control (SOC) sampler that learns a target distribution based on having an energy function alone. It is an extension of the current developments in this area, specifically by applying the proximal point method on the space of path measures. They provide a method that works for both discrete and continuous cases. The end result is an algorithm where one can do a variable amount of optimization problems that empirically (and theoretically) converge to the target distribution without mode collapse. Essentially, a computationally tractable method for "forward-KL" solving with SOC.

In general, the contribution is strong and fairly easy to follow given the complexity (although I suggest improvements below). It seems applicable with fairly good results.

### Strengths
# overall
- problem setting is well identified and known to be a practical issue
- background is well explained and covers both discrete and continuous cases
- In general, the key results from a hard topic are well presented... (The extensions from the paper need some notation work, in my opinion).
- The method introduces a scheme of iterates to help exploration, including a proposed scheduler that dynamically decides the number of iterations based on a bounded KL term. This is nice and well motivated.

# results
- results on non-particle continuous models are very good.
- results seem fairly good on particle models
- results are strong on the discrete models

### Weaknesses
# clarity in actual objective
- From lines 298 - 318 the paper does not make it extremely clear how the actual objective is computed. There are a number of identities used, in particular the right part of (15) that do not seem clearly motived by other equations in the text.
- Similarly, about clarity, You use so many version of $\mathbb{P}$ it becomes difficult to compare them. In particular the most confusing are the non-parametric ones: P^k, P^k*. P^k is a non-optimal, non-parametric path measure? It's not obvious why we even need this thing or how it is defined. Maybe it would be worth introducing alternative notation of P to help keep everything straight.
- ... Similarly, (16) refers to a ratio of path measure derivatives that does not appear in the discrete objective on 322-323. Can you please write it out?

# cost
- can you better explain the cost of training your model versus others? Is it k-times as expensive?

# unclear about some parts of method
- ASBS, for example, is extremely hyperparameter sensitive. How about your method?
- You use a memoryless path, correct? How does this interact with your optimization problem?
- Can you use a non-memoryless distribution, e.g. like ASBS does?

### Questions
# results
- can you please plot the energy histogram for the particle problems? It tells us much more than W2 and E() W2.
- can you please work on notation in the methods you are proposing a bit? I know they're all path measures...
- I know it's a lot to ask, but is there any molecular task you can apply this to? Alanine Dipeptide for example? Strong contender ASBS misses important modes, if your method gets them that would be great!
- Can you better quantify how much excess training cost is required to solve the k proximal problems? Does it really slow everything down by a lot?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses general-purpose sampling by simulating diffusion inference trajectories in both continuous and discrete settings. A central challenge is that fitting the noisy target distribution with a neural network often leads to mode collapse and training instability. To mitigate these issues, the authors augment standard SOC objectives (e.g., CE, WDCE) with a KL-regularized local objective, yielding a proximal point method in path space. They instantiate this “path-measure proximal point” framework (PDNS) for both continuous SDEs and discrete CTMC diffusion samplers, demonstrating its generality.

### Strengths
1. The proximal-in-path-measure perspective is clean and broadly applicable across continuous SDE and discrete CTMC settings. In particular, the proximal WDCE naturally tempers importance weights, directly addressing mode collapse and high-variance gradients. 
2. The paper presents the SOC, CE, WDCE, proximal formulations, and Girsanov/CTMC Radon–Nikodym weights with clarity. Practical guidance (e.g., OU reference bridges, conditional score formulas, and step-size schedulers) facilitates reproducibility.
3. Experiments span synthetic multimodal and high-dimensional mixtures, challenging particle systems (LJ-13/55), discrete Ising/Potts models near criticality, and combinatorial max-cut. PDNS achieves state-of-the-art or near-state-of-the-art results on multiple tasks, measured by domain-relevant metrics (Sinkhorn/MMD, $W_2$, energy $W_2$, magnetization, two-point correlations).

### Weaknesses
1. Proximal WDCE introduces overhead from replay buffers, weight computation (Girsanov/CTMC), and resampling. While results are strong, the paper does not quantify compute and memory costs relative to baselines (e.g., wall-clock, NFE, memory footprint), especially on the largest tasks (LJ-55, Potts). Resource-normalized comparisons would clarify efficiency. 
2. Many derivations rely on independence between initial and terminal distributions (Eq.~(2)). The framework assumes (nearly) memoryless references to obtain closed forms and reciprocal sampling, which may be unavailable for some targets. Although non-memoryless extensions are mentioned in related work, PDNS is not analyzed under such references. 
3. The proximal step size $\eta_k$ (or $\lambda_k$) critically trades off stability and speed. Despite an adaptive KL-based scheduler, the paper lacks systematic sensitivity studies and practical heuristics for choosing $\epsilon$ across tasks and scales. Additional ablations on $\eta_k$, buffer size, and resampling vs.\ weighting would aid practitioners.

If the authors resolve my concerns, I am willing to raise my rating.

### Questions
1. Did authors consider using the Wasserstein two distance as a regularizer to replace the KL divergence between $p^{\theta}$ and $p^{\theta_{k-1}}$, by coupling the JKO scheme, the proximal problem formulation may be more intuitive.

### Soundness
3

### Presentation
3

### Contribution
2
