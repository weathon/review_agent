# Mean Flow Policy with Instantaneous Velocity Constraint for One-step Action Generation

- Avg Score: 7.00
- Decision: Accept (Oral)
- Scores: 8, 4, 8, 8

## Abstract
Learning expressive and efficient policy functions is a promising direction in reinforcement learning (RL). While flow-based policies have recently proven effective in modeling complex action distributions with a fast deterministic sampling process, they still face a trade-off between expressiveness and computational burden, which is typically controlled by the number of flow steps. In this work, we propose mean velocity policy (MVP), a new generative policy function that models the mean velocity field to achieve the fastest one-step action generation. To ensure its high expressiveness, an instantaneous velocity constraint (IVC) is introduced on the mean velocity field during training. We theoretically prove that this design explicitly serves as a crucial boundary condition, thereby improving learning accuracy and enhancing policy expressiveness. Empirically, our MVP achieves state-of-the-art success rates across several challenging robotic manipulation tasks from Robomimic and OGBench. It also delivers substantial improvements in training and inference speed over existing flow-based policy baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes builds on the use of generative modelling for policy learning by introducing a new policy learning framework with MeanFlow, a one-step alternative to popularly used flow matching based approaches. Additionally, to improve the accuracy of the policy learning, the authors propose a new constraint on the MeanFlow generation framework that ensures the consistency of the instantaneous velocity in modelling the velocity field of flow matching. This is proven both theoretically and experimentally. The policy is then trained in an offline-to-online reinforcement learning setting wherein the Q-values from the critic are used to select the action targets in a best-of-N manner to train the MeanFlow policy network.

### Strengths
The proposed approach shows a novel way of modelling policies using MeanFlow rather than a naive approach that just shows the use of a new generative modelling method to policy learning. The authors show how their instantaneous velocity constraint helps improve the policy learning. The results show a good improvement over previous approaches that use different combinations of flow matching and a best-of-N selection. The presentation clarity of the paper is easy to follow, and the contributions are highlighted well both in terms of the writing and in terms of experimentation.

### Weaknesses
Given the growing use of generative modelling in Imitation Learning settings, and since the authors explore the offline-to-online RL case, a comparison in a simpler behavioral cloning paradigm can help show the advantage of the proposed approach better, which in general can be useful for learning much more complex tasks.

### Questions
- How well does the approach perform with an image-based backbone?
- Since the authors train their model from offline data as well, does similar performance hold for Imitation Learning as well?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Mean Flow Policy (MFP), a reinforcement learning approach that adapts the mean-flow generative modeling framework for efficient one-step action generation. Instead of learning the instantaneous velocity field as in standard flow matching, MFP models the mean velocity field, enabling direct sampling without multi-step integration. To address the underdetermined mean-flow ODE, the authors add an Instantaneous Velocity Constraint (IVC) enforcing equality between mean and instantaneous velocities at the boundary r=t. Combined with a best-of-N Q-guided action selection scheme, MFP achieves competitive or superior results to existing flow-based RL methods (FQL, BFN, QC) on Robomimic and OGBench, while significantly improving training and inference speed.

### Strengths
- Clear theoretical presentation with formal analysis of the mean-flow ODE’s non-uniqueness and the effect of the IVC boundary constraint.
- Empirical results show consistent efficiency gains over multi-step flow policies.
- The IVC ablation confirms its stabilizing effect during training.
- Writing and experimental setup are clear and reproducible.

### Weaknesses
- The mean-flow formulation itself is taken directly from prior generative modeling work (*Geng et al., 2025a*); this paper mainly applies it to RL.
- The proposed IVC is conceptually implied by the mean-flow definition when r\!\to\!t, so it should not be considered a new theoretical contribution.
- The best-of-N action-selection mechanism is standard in prior RL methods (EMaQ, BFN, FQL).
- The experimental scope is limited: only a small number of state-based manipulation tasks are used, with no visual-policy or diverse-domain evaluation. This constrains the empirical strength of the claims.

### Questions
The IVC condition appears to be mathematically implied by the mean-flow definition (Eq. 6) and the mean-flow loss (Eq. 9) as $r \rightarrow t$. The added IVC term seems to primarily emphasize the boundary condition rather than address a missing component of the mean-flow formulation. Could the authors clarify whether this term introduces any new theoretical insight beyond reinforcing an already implied constraint?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the Mean Flow Policy (MFP), a novel generative policy for reinforcement learning that is both highly expressive and computationally efficient.

The core idea is to represent the policy using MeanFlow [1]. As MeanFlow is trained via supervised learning, it is incompatible with standard reinforcement learning frameworks. To bridge this gap, the proposed method samples N actions from the policy, selects the action with the highest Q-value, and uses this optimal action as a supervised target to train the MeanFlow policy.

Furthermore, to ensure stable and accurate learning, an Instantaneous Velocity Constraint (IVC) is introduced. This constraint imposes a necessary boundary condition on the underlying ordinary differential equation, which stabilizes the training process and enhances model fidelity.

Empirically, MFP achieves state-of-the-art success rates on several challenging robotic manipulation benchmarks (Robomimic and OGBench), while demonstrating significant computational speedups over existing multi-step, flow-based baselines.

[1] Geng, Zhengyang, et al. "Mean flows for one-step generative modeling." arXiv preprint arXiv:2505.13447 (2025).

### Strengths
- It has always been a question for me how diffusion / flow-matching policies can be used in RL. The “generate-and-select” mechanism used in this paper seems simple and straightforward, yet the authors demonstrate that it works remarkably well. I really appreciate this idea.

- By incorporating recent advances from generative AI (specifically, MeanFlow), the method enables fast online RL training while preserving the generative model’s ability to represent complex, multimodal action distributions.

- The approach achieves the highest or second-highest success rates across nine challenging robotic manipulation benchmarks, showcasing its effectiveness in solving complex, long-horizon tasks.

- The authors provide rigorous mathematical proofs to justify their framework. They show that the implicit value constraint (IVC) ensures a unique solution for the mean flow identity and that their overall algorithm guarantees consistent policy improvement.

### Weaknesses
- The paper emphasizes its one-step inference speed but understates the associated training cost. The core training loss (Eq. 9) necessitates a Jacobian-vector product (JVP) to compute the $\frac{d}{dt} \mathbf{u}_\theta$ term. This operation is often incompatible with optimized attention implementations like FlashAttention, potentially limiting the method's training efficiency and scalability.

- The paper repeatedly claims suitability for "real-time control systems" and "real-time deployment," yet these claims lack empirical validation on physical robotic hardware, as all experiments are conducted in simulation.

- The theoretical proofs for policy improvement (Theorem 1) and solution uniqueness (Theorem 3) are elegant but rely on strong assumptions that are often violated in practice. For instance, Theorem 1 assumes bounded Q-function and mean flow matching errors (epsilon_Q, epsilon_A). In deep RL, these errors can be large and unstable. Similarly, the proof for IVC's effectiveness assumes the mean flow loss (L_MF) is perfectly minimized, which is never the case with non-convex optimization. While the theory provides intuition, its practical guarantees are much weaker than presented.

### Questions
- In the MeanFlow paper [1], the authors train the MeanFlow model using the MeanFlow objective for only 25% of the training time, while employing the Flow Matching objective (where r=t) for the remaining 75%. Similarly, FACM [2] incorporates the original Flow Matching loss into its training objective. How does the Instantaneous Velocity Constraint (IVC) proposed in this paper differ from them?


[1] Geng, Zhengyang, et al. "Mean flows for one-step generative modeling." arXiv preprint arXiv:2505.13447 (2025).

[2] Peng, Yansong, et al. "Flow-anchored consistency models." arXiv preprint arXiv:2507.03738 (2025).

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the Mean Flow Policy (MFP) which is an improvement for flow-matching based policies using the idea of mean flows. The key insight is that in traditional flow policies, one has to integrate the instantaneous velocity from 0 to 1 to transform from the standard normal distribution to the output distribution which selects the action. Instead of learning instantaneous velocities (v(t)) and then performing numerical integration, this paper shows that you can learn a mean-flow (u(t,r)) which is the velocity from t to r. This allows one to get the target distribution in one step as opposed to the several needed in typical flow matching papers. To ensure accurate learning of this mean velocity, the authors propose an Instantaneous Velocity Constraint (IVC), which serves as a boundary condition to eliminate solution ambiguity in the mean flow ordinary differential equation. Theoretically, the paper proves that IVC ensures a unique and accurate solution and that MFP guarantees policy improvement under bounded fitting errors. Empirically, MFP achieves state-of-the-art success rates but provides training/inference speedups on two standard simulation benchmarks.

### Strengths
This is a strong paper that introduces a new formulation for flow matching, which is prevalent tool in robot learning. A major pain point for flow-matching or diffusion style approaches is the slow inference times. We need several denoising steps or numerical integration, that greatly slows down inference and the rate at which such policies can be applied. Unlike "hacks", this paper presents a clean reformulation using the ideas of mean-flow to reduce the inference time. The theoretical analysis is strong. I expect this to be the starting point for several follow-up ideas that can greatly improve the empirical performance as well.

### Weaknesses
An unfortunate (IMO) weakness of the paper is the limited empirical results. It uses only two simulation environments and the improvements are minimal over the baselines. In most cases, it matches or only slightly exceeds the baseline. In a way that's not surprising to me. The claim isn't that this is a better method (in terms of higher success rates) than baselines but that it is a faster method (during inference). However, the inference time comparisons are only in the appendix with some short description in the main body. I would hav liked to see more discussion on that aspect. 

Another issue is that the two simulation seem "saturated" in that the baselines already perform pretty well. It would have been nice to look at more complex benchmarks.

It would have been especially nice to see how fixing the inference time (for example) results in better/worse performance for the several baselines. For diffusion style methods, it is not uncommon to perform comparisons with varying the number of denoising steps (which controls for the inference time). An equivalent study here would have really benefitted the paper.

### Questions
1. The paper’s main claim is that Mean-Flow Policy achieves faster inference than standard flow or diffusion policies. However, the runtime comparisons are only briefly mentioned in the appendix. Could you provide more details on the experimental setup for these timing results e.g., hardware used, batch size, and number of integration or denoising steps for baselines — and consider moving a quantitative runtime table into the main text?

2. The reported environments (Robomimic and OGBench) seem relatively saturated, with strong baseline performance. Have you considered evaluating on more complex or long-horizon benchmarks to better stress-test the method’s benefits? If not, could you explain the rationale behind choosing these two domains?

### Soundness
4

### Presentation
3

### Contribution
3
