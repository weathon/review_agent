# Adaptive Symmetrization of the KL Divergence

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 2, 8, 6

## Abstract
Many tasks in machine learning can be described as or reduced to learning a probability distribution given a finite set of samples. A common approach is to minimize a statistical divergence between the (empirical) data distribution and a parameterized distribution, e.g., a normalizing flow (NF) or an energy-based model (EBM). In this context, the forward KL divergence is a ubiquitous due to its tractability, though its asymmetry may prevent capturing some properties of the target distribution. Symmetric alternatives involve brittle min-max formulations and adversarial training (e.g., generative adversarial networks) or evaluating the reverse KL divergence, as is the case for the symmetric Jeffreys divergence, which is challenging to compute from samples. This work sets out to develop a new approach to minimize the Jeffreys divergence. To do so, it uses a proxy model whose goal is not only to fit the data, but also to assist in optimizing the Jeffreys divergence of the main model. This joint training task is formulated as a constrained optimization problem to obtain a practical algorithm that adapts the models priorities throughout training. We illustrate how this framework can be used to combine the advantages of NFs and EBMs in tasks such as density estimation, image generation, and simulation-based inference.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a method to minimize the symmetric Jeffreys divergence (the sum of forward and reverse KL divergences) to learn a probability distribution from data. The key challenge is that the reverse KL term depends on the unknown data distribution. To address this, the authors introduce a proxy model to approximate the data distribution and formulate the problem as a constrained optimization task. A central contribution is an "adaptive symmetrization" mechanism, implemented via a resilient optimization framework (P-DYN), which dynamically adjusts the emphasis on the forward KL, reverse KL, and proxy model fidelity during training. The authors develop a primal-dual algorithm based on non-convex duality theory to solve this problem and propose a synergistic combination of a Normalizing Flow (NF) as the primary model and an Energy-Based Model (EBM) as the proxy. Experimental results on synthetic 2D data, latent-space image sampling (CelebA), and simulation-based inference (SBI) benchmarks are provided, claiming improved stability and performance over baselines like NF, WGAN, and a fixed-weight penalty method.

### Strengths
The core idea of using a collaboratively trained proxy model to enable the minimization of the Jeffreys divergence is novel and interesting. Moving away from the adversarial setup of GANs towards a collaborative, constrained optimization framework is a worthwhile direction.

### Weaknesses
The paper lacks a critical discussion on the fundamental tension between the forward and reverse KL divergences. It is well-established that their minima can be contradictory (mode-covering vs. mode-seeking). The proposed adaptive symmetrization aims to balance them, but the theoretical conditions under which minimizing their sum leads to a desirable solution are not analyzed. The claim that the method avoids the issues of both extremes needs deeper justification.

The transition from the idealized problem (PI) to the dynamically constrained problem (P-DYN) is presented as a solution for infeasibility, but the approximation gap between these two formulations is not quantified or analyzed. It remains unclear how the solutions of (P-DYN) relate to the original goal of minimizing the Jeffreys divergence.

The transition from Eq. (P-DYN) to the empirical dual problem in Eq. ($\hat P$-DYN) is flawed. The entropy of the data distribution is ignored. This oversight invalidates the equivalence claimed in this step.

The experiments are primarily conducted on low-dimensional, synthetic 2D datasets. While useful for illustration, they are insufficient to demonstrate the scalability and practical utility of the method for modern machine learning problems. The claim that the method is "more accurate on a variety of datasets" is overstated.

The comparisons, while including NF and WGAN, lack benchmarks against other state-of-the-art generative models (e.g., diffusion models, VAEs) or other methods for symmetric divergence minimization.

There are related work on combining forward and reverse KL, such as the $\alpha$-bridge [1]. Discussions on the pros and cons of the proposed method over these existing ones are necessary.


[1] Zhao, Miaoyun, et al. "Bridging maximum likelihood and adversarial learning via α-divergence." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 04. 2020.

### Questions
Please refer to the Weaknesses.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a new way to minimize the Jeffreys divergence by introducing a " proxy mode". The final objective is weighted combination of three KL divgerneces. The proposed method is then applied to task like: density estimation, image generation, and
simulation-based inference.

### Strengths
The experiment section covers different potential use cases of the proposal method, which is good.

### Weaknesses
1. **Motivation is unclear**
The motivation for using Jeffreys divergence is "training on discrete samples may lead to a
mismatch between the modelled distribution and the data distribution (illustrated in Figure 1a). Ordinarily, minimizing a symmetric divergence would alleviate this issue," which is unclear to me. I am not sure why minimising a symmetric divergence will make the mode and target distribution more matched. They are all valid divergences; any valid divergence will make two distributions equal when the divergence goes to zero. Need more explanation on the motivation, practical evidence or reference to illustrate this problem.

There are some benefits of combining forward KL and reverse KL for training, for example, in this paper https://arxiv.org/pdf/1907.11891, the motivation is that Reverse KL will lead to mode collapse but will get sharper mode estimation, forward KL will have better mode covering ability, so adding FKL to RKL will improve the diversity. This is one example of a valid motivation. This is the most important question to answer when starting research.

2. **The experiment results are too bad** The FID for CelebA is too high, and only on low low-dimensional latent space is too out of date. A valid paper either has a new method with a good motivation or can improve some current best methods. This FID is too high, couldn't show the proposed method is effective in high dimensions. Other 2d experiments are too trivial.

3. **Idea is not inspiring** The proposed method requires introducing another proxy model, which needs to be as powerful as the main model, which is unaffordable in current machine learning world.

### Questions
See above for the weekness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a new approach to generative modeling by minimizing the Jeffrey's divergence.  The forward KL $D(\pi|p_\theta)$ is trained in the usual MLE way using samples of $\pi$ while the reverse KL $D(p_\theta|\pi)$ which is not directly accesible is trained by using a surrogate distribution $q_\psi$.   Instead of minimizing $D(p_\theta|\pi)$ directly one minimizes instead $D(p_theta|q_\psi)$ (which is then computed "explictly" or via MC) while keeping $D(\pi|q_\psi)$ small.  

The main idea is to avoid the adversarial (and sometime brittle) approach to GANs which is use the dual formulation of the KL divergences and neural newtowkr architectures. In the current approach the family $p_\theta$ and $q_\psi$ are parametrized directly a combination of neural flows and energy models to ensure expressivity. 

The resulting objective functional is not convex so it is treated by dual optimization and a control on the difference bewteen solutions of the original and dual functional.

### Strengths
1) The paper is very well written, the main ideas and concepts are presented with clarity and in a nuanced manner.

2) The ideas in the paper are novel and original.  As far as the reviewers knows,  this is a completely new approach to generative model and a new way to avoid the adversarial training of GANs.   In some way the introduction of a surrogate  replaces the back-forward training commong in flow models (such as diffusion models or normalizing flows)  by the introduction of the surrogate. The combined use of neural flows and energy model is also interesting.  

3) The experiments are overall sufficient to demonstrate the effectiveness of the training.

4) The reviewer appreciate the thoughtful discussion  about the limit of the methods in high-dimension, and maybe the need for other divergences

### Weaknesses
1)  The fact that generative/surrogate  divergence is handled via importance sampling  MC is a little bit worrying, especially if the target has a complex structure with metastable behavior.   

2) The dual optimization framework seems super interesting.  The reviewer would have appreciated a bit more background and intuition about why this works and why this apply here.  In particular the assumption about closeness in total variation (which is a very strong norm) is not very likely to be true in practice.

### Questions
My two questions would be to adress the two weakness noted above.

1) Can you explain where your method starts to fail?

2) How can one understand the theory behind the optimization problem better?  It seems from the experiments that that the gap between primal and dual solutions is very small.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper proposes a collaborative alternative to adversarial training for minimizing the Jeffreys divergence. The key idea is to introduce a proxy model \(q_ψ\) that both fits the data and serves to approximate the reverse KL term \(DKL(p_θ || π)\) via \(DKL(p_θ || q_ψ)\).

### Strengths
This study proposes an adversarial training method for minimizing the Jeffreys divergence. The method is well motivated and theoretically supported. I find no critical flaws in the derivation. However, I am not very familiar with energy-based models and could not fully appreciate the significance of the contributions.

### Weaknesses
See above.

### Questions
- As I understand it, the KL divergence is preferred because it is connected to maximum likelihood estimation, which yields asymptotically efficient estimators. What is the statistical advantage of symmetrizing the KL divergence?

### Soundness
3

### Presentation
3

### Contribution
3
