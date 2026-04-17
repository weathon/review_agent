# Flow Matching Policy Gradients

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Flow-based generative models, including diffusion models, excel at modeling continuous distributions in high-dimensional spaces. In this work, we introduce Flow Policy Optimization (FPO), a simple on-policy reinforcement learning algorithm that brings flow matching into the policy gradient framework. FPO casts policy optimization as maximizing an advantage-weighted ratio computed from the conditional flow matching loss, in a manner compatible with the popular PPO-clip framework. It sidesteps the need for exact likelihood computation while preserving the generative capabilities of flow-based models. Unlike prior approaches for diffusion-based reinforcement learning that bind training to a specific sampling method, FPO is agnostic to the choice of diffusion or flow integration at both training and inference time. We show that FPO can train diffusion-style policies from scratch in a variety of continuous control tasks. We find that flow-based models can capture multimodal action distributions and achieve higher performance than Gaussian policies, particularly in under-conditioned settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Flow Policy Optimization (FPO), a novel on-policy reinforcement learning algorithm for training flow-based generative models, particularly diffusion models, as policies. The core innovation is reformulating the policy gradient objective by replacing exact likelihood computations with a ratio derived from the conditional flow matching (CFM) loss. Specifically, FPO uses  a surrogate loss (See Sec. 3.3, 3.4) for the standard PPO likelihood ratio, enabling integration into the PPO-clip framework. The authors demonstrate that this ratio corresponds to optimizing an advantage-weighted evidence lower bound (ELBO), making FPO theoretically grounded while computationally tractable. Importantly, FPO is agnostic to the choice of sampling method during both training and inference, unlike prior denoising MDP approaches. Experiments across GridWorld, 10 MuJoCo Playground tasks, and high-dimensional humanoid control show that FPO successfully trains flow-based policies from scratch. The method demonstrates particular advantages in under-conditioned settings where multimodal action distributions are beneficial, outperforming Gaussian policies. The paper provides both theoretical analysis connecting the CFM objective to ELBO maximization and empirical validation showing FPO achieves competitive or superior performance compared to standard PPO and DPPO baselines.

### Strengths
The paper presents a genuinely novel approach to combining flow matching with policy gradients. The key insight—using the CFM loss differential as a surrogate for log-likelihood ratios—is elegant and theoretically motivated. Unlike prior work (DDPO, DPPO) that treats the denoising process as an MDP, FPO directly integrates flow matching into the policy gradient framework, avoiding the artificial expansion of the horizon and observation space. The connection to ELBO optimization through existing framework is well-established, and the observation that gradient estimates remain unbiased despite upward bias in the ratio (Equation 18-20) is insightful.


Moreover, the paper is well-written with clear motivation. Algorithm 1 provides a concise implementation overview, and the progression from standard PPO to FPO is logical. The GridWorld visualization (Figure 1) effectively demonstrates the multimodal behavior learned by FPO, showing the learned bimodal distribution at the saddle point. The humanoid control results clearly illustrate FPO's advantage in under-conditioned scenarios.

### Weaknesses
1. **Bias analysis incomplete**: While the paper shows gradient estimates are unbiased (Eq. 20), the impact of ratio overestimation on actual policy updates is not thoroughly analyzed. How does this bias interact with PPO clipping in practice? The claim that "clipping mechanism controls magnitude" needs more rigorous justification—does the clip threshold need adjustment to account for systematic overestimation?

2. **Missing analysis of multimodality**: While GridWorld demonstrates multimodal learning, there's no quantitative analysis. How does the learned distribution compare to the true optimal distribution at saddle points? For humanoid control, are the policies actually multimodal or just higher variance? Entropy measurements or explicit distribution visualization would strengthen these claims.

3. **Reproducibility concerns**: While code release is promised, many implementation details are missing. What network architecture is used for flow models? How is the timestep encoded?

### Questions
1. The paper claims sampling method agnosticism, but experiments only use 10-step Euler integration. Have you validated with DDIM, DPM-Solver, or higher-order methods? Does performance change significantly?

2. In Eq. 12, you decompose $r_{FPO}$ into likelihood ratio × inverse KL gap. Can you provide empirical measurements of how each term evolves during training? Does the KL gap actually decrease?

3. Why is the entropy coefficient set to 0 for FPO (Table A.2) but 0.01 for Gaussian PPO? Doesn't this disadvantage exploration for FPO?

4. For the humanoid under-conditioned experiments, can you provide quantitative measures of policy multimodality (e.g., entropy, number of effective modes) rather than just success rates?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces flow matching policy gradients, a method for optimizing a policy using flow matching.
The benefit of this framework is in enabling a more general class of generative policies that do not require a pre-defined policy class (e.g., unimodal gaussian/beta distribution).

### Strengths
The benefit of this framework is in enabling a more general class of generative policies that do not require a pre-defined policy class (e.g., unimodal gaussian/beta distribution).

I see the main benefit of this work shown in the humanoid control section.
We know that large NNs generally converge to global optima ("there's always some decent direction so long as we have some random noise in the system"). It isn't clear whether continuous control exhibits similar properties or not. 

The mujoco playground experiments are low degrees of freedom and might indeed suffer from local optima, however experiments on the humanoid shows PPO outperform FPO -- this suggests that maybe in the case of very large action spaces we do not suffer from sub-optimal local minima.

The interesting result is then the ability of FPO to converge to multi-modal solutions when needed. This is shown in the under-specified humanoid problem where the policy receives rewards for the full-body pose but is only conditioned on a subset of these constraints.

### Weaknesses
I believe the work should focus more on the generative aspects of the method, as in the humanoid control effort, and less on toy problems.
The mujoco playground results are "nice to have" they show the method generally works. But the main strength of generative methods is in their ability to model a distribution.

### Questions
Does FPO also work in a sparser setting?
For example in Tessler 2024 they go beyond root/hands conditioning and also show cases where the constraints are multiple frames into the future. This is a very underspecified and hard problem to solve and would be very impressive if FPO can still tackle it.

### Soundness
4

### Presentation
4

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
This paper introduces Flow Policy Optimization (FPO), an on-policy reinforcement learning algorithm that enables training of flow-based generative models (including diffusion models) as policies within the policy gradient framework. The key innovation is replacing the intractable likelihood ratio in PPO with a surrogate ratio computed from conditional flow matching (CFM) losses.

### Strengths
1, The motivation is clear and significant. Training flow matching models directly from rewards can greatly popularize their usage to robotics.

2, The evaluation is comprehensive to show the effectiveness of the proposed method on simple robotic tasks (with simulation).

### Weaknesses
1, The baseline is limited. There are existing methods which use direct rewards to weight the trajectory and are agnostic to sampling methods, although most of them are applied to text-to-image generation and other generation tasks, for example, [A]. The author should also implement some of these methods on robotics tasks and conduct simple evaluation, or at least include them as related works and describe the difference.

[A] Online Reward-Weighted Fine-Tuning of Flow Matching with Wasserstein Regularization

### Questions
1, What is the main difference of your method compared to other reward-weighted methods, empirically speaking?

2, Can the proposed method generalize to scopes other than robotics? Or what could be the domain-specific point?

I am willing to raise my score if all my concerns are well solved.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
They propose Flow Policy Optimization (FPO): swap PPO’s likelihood ratio with an ELBO-ratio computed from conditional flow matching (CFM) losses. This lets you train flow/diffusion policies in a PPO-style loop without evaluating exact log-likelihoods. Empirically, they beat Gaussian-PPO and a diffusion-PPO baseline on most MuJoCo Playground tasks, and show robustness on a humanoid control benchmark.

### Strengths
Pros:
- Ratio-as-difference-of-CFM-losses is simple to implement and keeps GAE/GRPO compatibility
-Clear ablations: effect of #MC samples, ω- vs u-parameterization, clipping sensitivity. Shows robustness under sparse goal conditioning in humanoid.

### Weaknesses
Cons:
- ELBO is not exact likelihood. The ratio decomposes into true likelihood ratio times an inverse KL-gap factor. That second term is policy-dependent and unknown, so the proxy ratio is biased w.r.t. the true PPO ratio

### Questions
Questions:
- Is there any ablations for choosing different weightings?
- while you mentioned the method is agnostic to sampler choice, do you observe empirically any difference between SDE/ODE samplers? Does it affect the ratio variance?

### Soundness
3

### Presentation
3

### Contribution
3
