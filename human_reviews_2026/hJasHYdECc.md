# Diffusion Posterior Sampling for Nonlinear Contextual Bandits

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
We study multi-task nonlinear contextual bandits, where different tasks share the same reward structure but are characterized by distinct model parameters drawn from a common unknown prior distribution. The goal is to leverage information from past tasks to minimize regret on a new task with limited online interactions. Thompson Sampling (TS) is a popular approach for solving contextual bandits, maintaining a posterior over the model parameter that is updated each round using a hand-specified conjugate prior (e.g., Gaussian) and the observed rewards. However, such priors cannot capture the rich cross-task structure in multi-task settings, leading to misspecified posteriors and suboptimal exploration. To address this, we train a diffusion model on data from past tasks to learn a flexible prior distribution over task parameters. In a new bandit task, parameters are estimated via a conditional reverse-diffusion process, where each step combines: (i) an unconditional drift from the diffusion prior, (ii) a likelihood-driven drift from the interaction history, and (iii) a noise term enabling randomized exploration. We instantiate this framework in two ways. DLTS integrates history into the diffusion prior at every reverse step to form a conditional posterior, from which approximate samples are drawn. DPSG first performs unconditional reverse sampling from the pretrained diffusion prior and then applies a single history-guided gradient correction. Both methods adhere to the same framework but differ in how they incorporate interaction history from the new task: DLTS explicitly constructs the conditional posterior, while DPSG provides a lightweight approximation by coupling unconditional sampling with one corrective step. In theory, we formalize oracle TS (OTS) and its diffusion counterpart (ODTS) and prove they are equivalent when the diffusion prior matches the true prior. We bound the per-round expected regret gap between ODTS and OTS by the cumulative score estimation error across diffusion levels. Our empirical evaluation demonstrates that our proposed methods are competitive with specialized baselines in linear settings and outperform baselines benefiting from the diffusion prior in challenging nonlinear bandit environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a diffusion-based posterior sampling framework for multi-task nonlinear contextual bandits. The key idea is to replace the standard conjugate prior in Thompson Sampling (TS) with a learned diffusion prior trained on past tasks. Two algorithms are introduced: **DLTS** and **DPSG** and its variant **DPSG-MP**. Theory shows that oracle diffusion TS coincides with standard oracle TS when the learned prior is exact, and bounds the regret gap (when the learned prior is not exact) via cumulative score-estimation error. Experiments on synthetic bandit settings indicate that DLTS and DPSG-MP perform well.

### Strengths
**S1.** Clear and well-structured exposition; algorithms are easy to follow.

**S2.** First attempt to apply diffusion posterior sampling beyond the generalized linear bandit case.

**S3.** Provides a unified formulation connecting diffusion priors, Langevin updates, and TS exploration.

**S4.** Empirical results show competitive regret versus specialized baselines in both linear and nonlinear regimes.

### Weaknesses
**W1. Novelty.** The paper mainly extends prior work on diffusion-based Thompson sampling for generalized linear bandits [1, 2] to more general nonlinear reward models. As a side note, [2] already considers nonlinear diffusion priors: the linear assumption is made only for theory. Similar to [1], both works rely on the generalized linear models to enable their approximations. The present contribution thus extends those to any form of non-linear rewards.

**W2. Limited theory.**
Theorem 4.4 is a sanity check, confirming equivalence under an exact prior. Theorem 4.5 provides a useful per-round regret bound that highlights an implicit trade-off: smaller diffusion depth $L$ reduces the per-round regret, but might increase the score-estimation error \$\epsilon_{\rm{score}}\$ due to lower expressiveness. However, I think the per-round bound is loose: it does not depend on $t$, relies on TV distance, and a worst-case $f_{\max}$. In particular, it implies linear regret under prior misspecification \$\epsilon_{\rm{score}} \neq 0\$. In addition, the coefficient $\kappa_\ell$ in the bound is never explicitly defined.

**W3. Computational efficiency.**
Proposed approximations are computationally heavy (compared to closed-form approximations), which is undesirable in online decision-making. While the claimed efficiency of DPSG (no inner $K_\ell$ loop) is valid in principle, that variant is never evaluated; the reported DPSG-MP reintroduces a $K_\ell$ loop. It might indeed be faster than DLTS as it might need fewer inner updates due to improved stability, but this should be demonstrated empirically through runtime comparisons. Prior works [1, 2] were explicitly motivated by computational efficiency aspect, avoiding costly approximate-posterior methods such as LMC; hence a fair evaluation should compare not only regret but also wall-clock time to assess the regret–efficiency trade-off.

**W4. Experimental scope.**
All experiments use synthetic priors, short horizons (200 rounds), and only 8 trials. Longer-horizon evaluations (as in [1]) is important. In particular, [1] results showed some interesting behavior of DSP after round 300, given that DPSG relies on a similar idea, it is important to showcase what happens with longer horizons. Moreover, real-world or high-dimensional tasks (e.g., MovieLens as in [1]) would further validate the approach.

**W4-bis. Missing ablations.**
Ablations are absent. In particular, the paper does not compare DPSG and DPSG-MP, nor analyze sensitivity to the number of diffusion steps $L$ or inner iterations $K_\ell$.


**References.**

[1] https://arxiv.org/pdf/2410.03919 (Neurips 2024)

[2] https://arxiv.org/pdf/2402.10028 (Neurips 2025)

### Questions
**Q1.** Can the authors report computational/space complexities as well as empirical runtime comparisons between all baselines?

**Q2.** Why was plain DPSG omitted from experiments? Does projection alone explain the observed gains?

**Q3.** How sensitive are results to hyperparameters like the number of diffusion steps $L$, inner updates $K_\ell$, etc?

**Q4.** How sensitive are results to diffusion prior quality (authors can vary the number of tasks used to train the prior, vary $L$, etc.)?

**Q5.** Can the authors share real-world (e.g., MovieLens) semi-synthetic experiments with longer horizons (e.g., 500 for 50 trials for example)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Thompson sampling (TS) is a widely used bandit algorithm that maintains a Bayesian posterior over the parameters of a task reward function. A challenge for TS is that it requires a prior distribution over the parameters that admits a tractable update. This presents a difficulty for domains with complex prior distributions. The main contribution of this paper is to show how TS can be implemented with diffusion models so as to provide a flexible prior. The paper considers a multi-task learning setting in which multiple samples of true task reward parameters are available to train a prior distribution. The method introduced uses a diffusion model to train this prior. The model is constructed such that it can be updated as more actions are taken (and rewards observed) while running TS. The paper shows empirically that this approach lowers cumulative regret compared to a well-chosen set of baselines.

### Strengths
- Thomson Sampling is a very widely used algorithm but stuggles with complex priors. This paper shows how modern generative models can be used to address this limitation.
- The empirical study convincingly supports the central claims of the paper.
- The paper is generally well-written and the authors have made an effort to provide both intuitive high-level descriptions and precise description of details.

### Weaknesses
- The technical novelty of the theoretical results is not clearly highlighted. It seems that the assumption is that the oracle diffusion approach can learn the true prior and posterior update. Under this assumption, it seems trivial to claim that the diffusion approach matches the oracle.
- Why does "Can we design diffusion posterior sampling algorithms for general nonlinear contextual bandits?" matter as a research question? I buy that we care about solving general nonlinear contextual bandits. I am less sure why diffusion posterior sampling is an essential part of doing that.
- Equation 3.1: while I appreciate the effort to make equation understandable, there do not appear to be any formal definition of the terms in this equation. This makes it difficult to understand precisely what each term means.
- The experiments only consider synthetic, low-dimensional domains. There are two sets of experiments: the first using linear reward functions and the second using non-linear reward functions. While the second one uses neural network reward approximations, the underlying domain is still fairly simple. Evaluation on higher dimensional or real-world domains would further solidify the paper's contribution.

### Questions
The following set of questions are supplemental to the weaknesses mentioned above. Addressing the weaknesses is the most important issue for the discussion phase.
- What is the prior for LinTS?
- Figure 3 shows samples from the prior which is over neural network parameters. How is this just shown in two dimensions?
- Can you elaborate on the key technical novelty of the theoretical results?
- How does the learning history affect the posterior in Algorithm 1? I don’t see the learning history being used, only updated.
- Same question for Algorithm 2.
- This work seems closely related to works that learn algorithms in multi-task settings. For instance: "Supervised Pretraining Can Learn In-Context Reinforcement Learning," ""In-context Reinforcement Learning with Algorithm Distillation," and "Pretraining decision transformers with reward prediction for in-context multi-task structured bandit learning" (among many others). It could be interesting to benchmark the proposed method against these methods.

### Soundness
3

### Presentation
3

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
This paper studies multi-task nonlinear contextual bandits where task parameters are drawn from a shared unknown prior. The authors propose an approach that learns the prior with a diffusion model trained on past tasks, then performs posterior sampling on new tasks via a conditional reverse-diffusion process that combines an unconditional drift from the diffusion prior, a likelihood-driven drift from the interaction history, and a noise term for randomized exploration. They offer two concrete algorithms: DLTS, which runs LMC at each diffusion level, and DPSG, which takes unconditional DDPM steps plus a single Tweedie-guided likelihood step (with a stabilized DPSG-MP variant). The paper argues that oracle diffusion Thompson sampling matches standard oracle Thompson sampling when the learned score is exact, and it bounds the per-round regret gap by the cumulative score-estimation error across diffusion levels. Experiments on synthetic data show competitive results in linear settings and gains in nonlinear ones.

### Strengths
I think combining multi-task bandits with a learned generative prior is a reasonable idea. The per-level decomposition is a clean way to view conditional diffusion for decision making, and both instantiations of the algorithm follow naturally from this view.
The experiments also empirically reveal the effectiveness of the approach. The visualizations make sense and help illustrate how the diffusion prior behaves in practice.

### Weaknesses
- The methods treat the diffusion prior largely as a plug-in prior/score oracle with path-wise conditioning. The per-level update separates an unconditional diffusion drift from a likelihood drift, and the theory hinges only on score accuracy. Thus, while the algorithms do use diffusion-time dynamics, the benefits appear model-agnostic and could likely be obtained with other generative oracles offering similar score or denoising interfaces. It would be more important to see how the diffusion denoiser can be trained or adapted online on the new task.

- The justification of DPSG-MP is mainly empirical. The multi-step projection variant is introduced to avoid collapse toward a MAP-like update, but there is little quantitative analysis of its bias or diversity trade-offs.

### Questions
- Can the authors elaborate on why a diffusion prior is preferred compared to other generative-prior choices (for example, normalizing flows or score-only EBMs) in this Thompson Sampling setting?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies using posterior sampling to solve contextual bandits with general function approximation. Specifically, it assumes that the prior of the reward function parameter is represented by a diffusion model, which is in turn represented by a diffusion denoiser \epsilon_{\phi^*} (this can be obtained by e.g., domain knowledge or meta-learning), and designs efficient posterior sampling methods for the posterior distribution after some (action, reward) interaction history is seen. 

It proposes two methods, Diffusion Langevin Thomspon Sampling (DLTS), and Diffusion Posterior Sampling with Guidance (DPSG) for this. DLTS's main idea is to use Langevin Monte Carlo to sample p(\theta_{l-1} | \theta_l, H_t). For DPSG, it is motivated by DPS (Chung et al, 2023), which approximates the likelihood-based score; the paper also proposes the DPSG-MP algorithm using multi-step projection as a more stable heuristic. 

The paper gives some theory that when the quality of the initial diffusion denoiser \epsilon_{\phi^*} is imperfect (Assumption 4.3), and when the per step conditional reverse process p(\theta_{l-1} | \theta_l, H_t) is sampled exactly, then the proposed approximate Thompson sampling algorithm has Bayesian regret close to the exact Thompson sampling algorithm. 

Experiments show that the proposed algorithms, DLTS and DPSG-MP, have good performance in linear and nonlinear bandits problems.

### Strengths
- Using diffusion models to model general prior distributions on reward parameters is a natural idea that can capture multimodality of the prior

- The proposed algorithm can handle general function approximation, which is in contrast with (Kveton et al, 2024) that can only handle linear and generalized linear models using special closed-form updates

- The experimental results are impressive

### Weaknesses
- The quality of approximation in DLTS (line 227), and DPSG (line 263) is not very clear to me. Can the authors comment on this? 

- Continuing the last comment, Theorem 4.5 is proved in the ideal setting where p(\theta_{l-1} | \theta_l, H_t) is exact. How easy can it be extended to the setting where it is inexact?

- Can the author provide ablation study on DPSG vs DPSG-MP? Also, how would you recommend choosing K_l in practice?

- I think it would be nice to experiment on a bandit setting with reward function beyond the generalized linear form (line 422), so DiffTS is not runnable, to show the versatility of the proposed algorithm.

### Questions
- Is figure 3 showing the prior distribution of the MLP parameters? Then it is a high-dimensional distribution, so it is showing a projection of the samples from the distribution? How is that related to the distribution of \theta in the model r = f(x^T theta) + \epsilon in line 422?

### Soundness
3

### Presentation
3

### Contribution
3
