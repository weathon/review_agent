# Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6

## Abstract
Generative models such as diffusion and flow-matching offer expressive policies for offline reinforcement learning (RL) by capturing rich, multimodal action distributions, but their iterative sampling introduces high inference costs and training instability due to gradient propagation across sampling steps. We propose the \textit{Single-Step Completion Policy} (SSCP), a generative policy trained with an augmented flow-matching objective to predict direct completion vectors from intermediate flow samples, enabling accurate, one-shot action generation. In an off-policy actor-critic framework, SSCP combines the expressiveness of generative models with the training and inference efficiency of unimodal policies, without requiring long backpropagation chains. Our method scales effectively to offline, offline-to-online, and online RL settings, offering substantial gains in speed and adaptability over diffusion-based baselines. We further extend SSCP to goal-conditioned RL (GCRL), enabling flat policies to exploit subgoal structures without explicit hierarchical inference. SSCP achieves strong results across standard offline RL and GCRL benchmarks, positioning it as a versatile, expressive, and efficient framework for deep RL and sequential decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Single-Step Completion Policy (SSCP), a flow-based generative policy framework for reinforcement learning that achieves single-step action generation while maintaining the expressiveness of multi-step generative models. The key innovation is training a completion model (instead of a shortcut model[1]) that predicts normalized completion vectors from intermediate flow points directly to target actions, bypassing iterative sampling. The authors demonstrate SSCP's effectiveness across three settings: (1) offline RL with behavior-constrained policy gradients (SSCQL), (2) goal-conditioned RL where hierarchical policies are distilled into flat inference (GC-SSCP), and (3) behavior cloning. The method achieves competitive or superior performance compared to diffusion-based baselines while offering substantial computational advantages.

[1] Frans, Kevin, et al. "One step diffusion via shortcut models." arXiv preprint arXiv:2410.12557 (2024).
[2] Park, Seohong, Qiyang Li, and Sergey Levine. "Flow q-learning." arXiv preprint arXiv:2502.02538 (2025).
[3] Espinosa-Dice, Nicolas, et al. "Scaling Offline RL via Efficient and Expressive Shortcut Models." arXiv preprint arXiv:2505.22866 (2025).
[4] Sheng, Juyi, et al. "MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation." arXiv preprint arXiv:2507.10543 (2025).

### Strengths
1. Novel and well-motivated approach: The completion vector formulation elegantly addresses a fundamental limitation of diffusion/flow policies—the need for iterative sampling—while maintaining expressiveness for multimodal action distributions. Unlike bootstrap-based shortcut methods [1], SSCP uses ground-truth targets from the dataset, avoiding early training instability.
2. A significant practical advantage is that SSCP enables training generative policies without backpropagating through iterative generation chains, removing the requirement for distillation as shown in FQL
3. The paper demonstrates consistent improvements across diverse benchmarks:
4. The extension to goal-conditioned RL (GC-SSCP) is particularly innovative, showing that multi-level hierarchical reasoning can be compressed into a single flat policy without explicit hierarchical inference. This challenges the assumption that hierarchical structure is necessary for long-horizon tasks.
5. The paper provides extensive ablations on key hyperparameters (α₁, α₂), bootstrap targets vs. completion loss, and demonstrates the learned completion field can support both single-step and multi-step rollouts (Table 9).
6. Extensive comparisons with FQL in Appendix A.7

### Weaknesses
1. While the paper compares against FQL [Park et al., 2025], there are other recent few-step policy methods [3-4] that should be discussed and compared
2. The paper doesn't provide clear guidance on when SSCP is expected to outperform alternatives
3. While Table 9 shows multi-step rollout results, the analysis is limited
4. In Figure 7, the performance change could be quite large depending on hyperparameters chosen

### Questions
1. Why are multi-step actions worse in some cases, as shown in Table 9? Could you visualize the performance difference with x-axis = # of steps and y-axis = performance 
2. How does SSCP compare with other one-step policies proposed recently, like MP1?
3. Can you explain why GC-SSCP fails as you scale up pointmaze-large-stitch to a larger setup?
4. It has been claimed by many papers extending FQL that distillation of the policy is the bottleneck, while no one has verified this approach. Can you verify this by simply training a flow policy and performing BPTT to see if we can achieve stronger performance by just directly optimizing the Q-value of the flow policy? If that is the case, it will strongly support the necessity of having a stronger policy than a distilled one.

### Soundness
3

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
1

### Summary
The authors propose single-step completion in flow matching, enabling transitions between arbitrary intermediate states rather than only predicting instantaneous velocity (the zero-jump case). While the standard self-consistent shortcut model relies on bootstrapped, potentially inaccurate targets—risking drift and unstable exploration—the authors address this by fixing the target to the final sample and learning one-step completions from any time step to the final one

This leads to the Single-Step Policy Completion (SSPC) objective, which learns complex, multimodal policies in a single step. The method achieves competitive performance with significantly faster training and inference. They further extend it to goal-conditioned RL, where a shortcut-based flat policy replaces hierarchical structures, improving efficiency while maintaining strong results.

### Strengths
- Big efficiency gains while maintaining or exceeding baseline performance.
- The paper is well-written and the method is well documented.
- Figure 8 in the appendix shows the strength of SSCP against shortcut models and makes the case that relying on bootstrap targets induces instability in the training. **I think that this figure is important since it clearly motivates the use of SSCP and thus would like to see it in the main text.**

### Weaknesses
N/A

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work considers generative modeling in RL, like diffusion models and flow-matching. Although these models have made great progress, they always rely on high inference costs and training instability caused by iterative sampling. To address this, this work proposes the Single-Step Completion Policy (SSCP), with an augmented flow-matching objective to predict direct completion vectors from intermediate flow samples, enabling accurate, one-shot action generation. This method can be extended into offline RL, offline-to-online RL, and online RL. This method is verified across various settings and different environments.

### Strengths
- This paper is well written and easy to follow.

- Stable training and efficient sampling are core concerns in diffusion policies in RL.

- The proposed method is flexible to different settings, like offline RL, online RL, and offline-to-online RL.

### Weaknesses
- The Q update function (4) utilizes the standard TD error. However, various works in offline RL propose that there is an overestimation error of the Q function caused by the distribution shift. Thus, several works will choose conservative Q learning techniques like IQL. What about the performance of using IQL in the offline setting?

- What is the difference between online RL and offline RL when applying SSCP?

- In offline-to-online experiments like Fig.4 and Fig.5, it seems that online fine-tuning in various environments can not improve the performance. Is there any explanation?

- It is better to add offline-to-online and online RL experiments to the main text.

- There are still some diffusion policies for RL that need to be discussed, including online fine-tuning [1-3] and offline diffusion planners [4-6].

Ref:

[1] Policy agnostic RL: Offline RL and online RL fine-tuning of any class and backbone

[2] Exploratory Diffusion Model for Unsupervised Reinforcement Learning

[3] Efficient Online Reinforcement Learning for Diffusion Policy

[4] What makes a good diffusion planner for decision making?

[5] Simple hierarchical planning with diffusion

[6] Latent diffusion planning for imitation learning

### Questions
See weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2
