# Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
We study diffusion-based world models for reinforcement learning, which offer high generative fidelity but face critical efficiency challenges in control. 
Current methods either require heavyweight models at inference or rely on highly sequential imagination, both of which impose prohibitive computational costs. 
We propose Horizon Imagination (HI), an on-policy imagination process for discrete stochastic policies that denoises multiple future observations in parallel. HI incorporates a stabilization mechanism and a novel sampling schedule that decouples the denoising budget from the effective horizon over which denoising is applied while also supporting sub-frame budgets.
Experiments on Atari 100K and Craftium show that our approach maintains control performance with a sub-frame budget of half the denoising steps and achieves superior generation quality under varied schedules. 
Code is available at https://github.com/leor-c/horizon-imagination.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
To address the low sampling efficiency of existing diffusion world models, especially the quadratic computational cost arising from the product of environment steps and diffusion steps, the authors introduce Horizon Imagination, an algorithm that enables more efficient on-policy sampling of discrete stochastic policies within diffusion world models. The authors make two key improvements over prior diffusion model sampling methods. First, they introduce a horizon sampling schedule that separates the total denoising budget from the decay horizon, allowing flexible and effective configuration. They also propose a novel sampling strategy that stabilizes policy output actions during denoising and prove it preserves the original sampling distribution. Experiments on Atari 100K and Craftium show that Horizon Imagination greatly reduces denoising steps and outperforms previous schedules like pyramidal schedule in generation quality.

### Strengths
1. The diffusion-based world model studied in this paper is a rapidly growing research area with significant value in both offline data generation and online policy learning. It holds great promise for substantially reducing the cost of real-world interactions.

2. The authors effectively resolve the instability that occurs when diffusion models and policies interact to jointly sample multi-step trajectories by introducing a theoretically grounded stable action sampling method. This approach significantly enhances the quality and stability of long-horizon trajectory generation.

3. Compared with the previous pyramidal schedule, the proposed horizon schedule provides a more general formulation. By decoupling the decay horizon from the denoising budget, it enables more consistent denoising schedules and achieves higher-quality generation.

### Weaknesses
1. The stable action sampling algorithm proposed by the authors is only applicable to discrete action spaces, which limits the applicability of the Horizon Imagination framework in more general environments with continuous action spaces.

2. A key weakness is the limited scope of the experimental comparisons. The control performance results in Section 5.2 are structured as an internal ablation study, comparing the proposed parallel method only against an autoregressive baseline within their own framework. The paper does not benchmark its end-to-end performance against other established world model agents, such as those discussed in the related work. The absence of direct, end-to-end SOTA comparisons makes it difficult to fully assess the proposed method's relative performance and efficiency in the field.

3. Training for 19 or 27 hours on A100 for relatively simple games like Atari seems somewhat costly. Conducting experiments in more challenging environments could make the proposed method more practically valuable.

4. There is an extra quotation mark at the end of line 446.

### Questions
1. Are there any possible exploratory directions for extending the idea of Horizon Imagination to continuous action spaces?

2. Algorithm 1 only includes the world model sampling process. Could the authors provide a complete pseudocode of the entire training pipeline, including world model training and RL algorithm updates? This would help readers gain a clearer understanding of the overall framework.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work proposes Horizon Imagination that tackles a common problem in model-based reinforcement learning with diffusion world models. In each step, the policy receives an observation from the world model, which is subsequently used to generate an action. The world model then predicts the next observation. This sequential dependency on the policy leads to a long inference time that is a bottleneck in fast training. Horizon Imagination, therefore, proposes denoising multiple future observations in parallel in conjunction with a stabilization mechanism to account for the changes in the actions and a sampling schedule. The effectiveness of the proposed method is shown on several benchmark environments.

### Strengths
- The paper tackles an interesting and important problem in world models for policy learning.  I think the proposed method is quite important for the RL field.

### Weaknesses
- I think the presentation of the paper can be improved. 
 
 - Please also see my questions.

### Questions
- I found it a bit difficult to follow. As a non-expert in Flows/Diffusion models, I was a bit confused with the notation. Is it for every z several denoising time steps sampled in Eq. 1, or is it only one denoising time step per sample?

- Section 4.2 states that Eq. 1 requires knowing all actions; however, the velocity field v_\theta is expecting actions $a_{<t}$, which are all actions in the history starting from the current time-step $t$. Doesn't this mean that there is actually no need for knowing future actions, as stated in Section 4.2?

- From my understanding, the paper proposes a method that turns the strict autoregressive structure of the world model (i.e., policy gives action, world model predicts observation, ...) into a parallel inference version. However, I don't understand how this autoregressive structure is still retained in the current version? I think the autoregressive prediction structure still needs to hold in general for world models. 

- Given that the policy is queried for noisy variables z, aren't there superscripts to the actions in Eq. (1) missing?

- Section 4.3 mentions using the REINFORCE algorithm. It is well-known that the REINFORCE has high variance in the gradient estimates, whereas the reparameterization trick has smaller variance in the gradients. Is there a reason why the reparameterization trick was not used, e.g., in connection with the Gumbel softmax [1] policy representation that allows using the reparameterization trick? 

- Is the proposed method also applicable to continuous state-action spaces? What would be necessary in this case? 

[1] E. Jang, et al. Categorical Reparameterization with Gumbel-Softmax. ICLR 2017.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Horizon Imagination, an on-policy training procedure for reinforcement learning with diffusion world models that denoise multiple future observations in parallel. This is done by introducing a stable discrete-action sampling mechanism to avoid spurious action flips during denoising, and a novel Horizon schedule that decouples the denoising budget from the decay horizon ν, enabling sub-frame budgets and finer control of compute–quality trade-offs. Experiments on Atari 100K and Craftium show the method maintains control performance with only half the denoising budget and improves generation quality under parallel configurations.

### Strengths
The proposed horizon schedule is a neat design: by fixing ν while varying B, it breaks the tight coupling seen in pyramidal schedules, allowing consistent temporal denoising behavior across budgets and enabling sub-frame B < h operation.
 
The stable action sampling a(π,ω) for discrete policies is elegant and theoretically justified: action changes between denoising steps are bounded below by total variation distance and above by a derived l_1  term, greatly reducing unnecessary flips during denoising. 

Solid empirical analysis: (a) action-consistency experiments demonstrate near-optimal behavior vs. TV lower bound and strong improvements vs. naïve sampling; (b) control performance comparison across ν/B settings; (c) generation quality vs. ν/B via FVD and MSE Clarity.

### Weaknesses
The proposed method, especially the action sampling, is only for discrete action spaces. This limits applicability to many continuous-control tasks 

The paper would benefit if a per-stage runtime analysis and real-time control throughput (fps) comparison were presented to show the improvement. 

It would help to connect the theoretical bound to returns—e.g., does reduced action-flip rate correlate with improved advantage estimates or policy gradient variance?

### Questions
Do you foresee a principled extension of a(π,ω)to continuous spaces (beyond discretization)?

How sensitive are returns to policy entropy regularization when using a(π,ω)? Is there a sweet spot where action stability helps most?

FVD/MSE trends favor certain ν/B regimes; can you relate those trends directly to control performance (e.g., correlation analyses across seeds)?

### Soundness
3

### Presentation
3

### Contribution
3
