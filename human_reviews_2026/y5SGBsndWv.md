# Leave No Observation Behind: Real-time Correction for VLA Action Chunks

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
To improve efficiency and temporal coherence, Vision-Language-Action (VLA) models often predict action chunks; however, this action chunking harms reactivity under inference delay and long horizons. We introduce Asynchronous Action Chunk Correction (A2C2), which is a lightweight real-time chunk correction head that runs every control step and adds a time-aware correction to any off-the-shelf VLA's action chunk. The module combines the latest observation, the predicted action from VLA(base action), a positional feature that encodes the index of the base action within the chunk, and some features from the base policy, then outputs a per-step correction. This preserves the base model’s competence while restoring closed-loop responsiveness. The approach requires no retraining of the base policy and is orthogonal to asynchronous execution schemes such as Real Time Chunking (RTC). On the dynamic Kinetix task suite (12 tasks) and LIBERO Spatial, our method yields consistent success rate improvements across increasing delays and execution horizons (+23% point and +7% point respectively, compared to RTC), and also improves robustness for long horizons even with zero injected delay. Since the correction head is small and fast, there is minimal overhead compared to the inference of large VLA models. These results indicate that A2C2 is an effective, plug-in mechanism for deploying high-capacity chunking policies in real-time control.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address latency problems with robotic policies at inference time by training an additional correction policy. This policy predicts a residual correction to the action chunk sampled from the base policy (e.g. a VLA policy). The correction head conditions on the observation a timestep within the prior chunk, the action predicted by and latent features from the base policy, the time feature, and the language instruction.

### Strengths
1. The motivation for this paper is quite clear, especially due to Figure 1 which clarifies the latency issues with robotic policies. The method is clearly written and is accompanied by several figures (Figure 2, Figure 3 and Figure 4) that clarify the complete system adequately. 
2. Residual policies are not completely new, but A2C2 proposes a very simple method for training one that does not require either reinforcement learning or additional demonstrations (as in the style of DAgger).  
3. The paper has tested their proposed method on a range of environments (12 tasks in Kinetix and 10 tasks in LIBERO).

### Weaknesses
1. The paper does not compare A2C2 with a few important baseline methods like temporal ensemble (Zhao et. al.) and bidirectional decoding (Liu et. al.). The latter would be a valuable comparison because they also use a weaker policy to guide the action selection. These are standard baseline methods as used in papers such as real-time chunking (Black et. al.). 
2. No comparison against real-time chunking (RTC) on LIBERO Spatial. This is confusing because RTC was used for comparison in Kinetix but not on LIBERO Spatial. Unlike in Kinetix, you use a significantly larger correction head (of 32M params) in LIBERO Spatial, which makes the delay $d > 0$ when $e = 40$. This makes the comparison against RTC very crucial to understand how having an additional correction head fares against the guidance provided by inpainting. 
3. One of the benefits of using action chunking is the ability to capture temporal consistency in actions executed (Liu et. al., Torne et. al., Black et. al.) which is captured by either conditioning on a longer context of observations (generally not preferred because of overfitting issues), longer context of actions or by predicting the chunk of actions. The correction head of A2C2 only conditions on the most recent observation, the most recent base action and the most recent latent representation. It is not clear to me how the correction head ensures temporal consistency across actions in this situation. As an illustration, the correction head’s residual action could cause the policy to switch from high velocity to low velocity and so on. 
4. The need for ablations here is quite large to help us understand whether the performance gains of A2C2 come from greater closed-loop responsiveness or whether it comes from residual action correction or whether it comes from being able to use action correlations across time-steps (Torne et. al.). In particular, I am interested in ablations over what the correction head can condition on i.e., how important is it that the correction head conditions on base policy’s action, latent representations, current observation, and time signature. 

Minor: 
1. It would be good to indicate that the action is sampled from a stochastic policy i.e., $A_t \sim \pi(\cdot \mid o_t, l)$, since the current notation almost suggests that the policy is deterministic. 
2. The paper could benefit from a rigorous check on spelling (such as capitalizations in phrases like “Average Success rate are…”) and polishing the syntax (such as using $\pi_\mathrm{base}$ instead of $\pi_{base}$).

### Questions
1. The importance of conditioning on the time feature is unclear to me. In line 199, you claim that it brings about greater “chunk level smoothness”. Some ablations could be helpful here since, intuitively, it almost feels like conditioning on the observation, base action and base policy’s latent representations might be sufficient? 
2. The central assumption here seems to be that the correction head can be made significantly smaller than the base policy. This is, certainly, an interesting scientific question, and it requires more empirical support. For generalist policies, it is not clear whether this assumption is fair. The language instructions, observations, etc. become more complex and diverse. How large the correction head needs to be able to learn this complexity is unclear. This question is not sufficiently well-answered since the complexity of the tasks (and the fact that the policy is not conditioned on image observations) in Kinetix is quite low and the tasks in LIBERO Spatial do not consist of a sufficiently large diversity. 
3. Extending on (1), the need for the correction head to be strong is greater when the policy needs to both generalize and react to environmental stochasticity. In the case of A2C2, even though you may benefit from a robust base policy, it is not clear to me whether you can get a correction head that also generalizes while also not suffering from large inference-time. Similarly, if the need for reactivity is such that the correction head does not merely need to add a residual correction to the base action but change the mode in the distribution altogether, then we might require a much stronger correction head and it’s not clear to me why the inference-time of such a model would be sufficiently small for A2C2 to be able to handle delays. 
4. Figure 5 shows the average over all tasks but a separate comparison within each task would be very important since some tasks in Kinetix require significantly greater closed-loop responsiveness over others. Perhaps this could be done in the appendix? Furthermore, it would be good to show that standard deviation/error within the average result plot.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a novel method, Asynchronous Action Chunk Correction (A2C2), to deal with the drawbacks of traditional action chunks, where the chunk length is fixed. Normal action chunks could leave actions out-of-date, because the action really executed in the environment is predicted with observations of several frames ago, and this is not wanted for a reactive behavior to run VLA in a dynamic situation. The proposed A2C2 learns a correction head and it outputs a residual action to refine the original action predicted by the base policy. This kind of method is orthogonal to the real-time chunking. The experiments in Kinetix and LIBERO verify the effectiveness of A2C2.

### Strengths
- A2C2 is orthogonal to RTC, bringing new aspects of improving qualities of action chunks, this is interesting.
- The experiments in Kinetix and LIBERO verify the effectiveness of A2C2.

### Weaknesses
- Figure 1 could be better for presentation, now it is difficult to understand its meanings. For example, the two emoji faces are not shown anywhere; the color of the emoji is unclear; 4 different action chunks look alike with the only difference being the border color, but what does the color represent?
- A2C2 learns the residual action to fill the gap between ground truth actions and base policy predicted actions. Here the base policy plays a critical role, but there are not adequate ablation studies on the model capacity of the VLA models.
- Learning a residual head for action improvement has been proved effective in many robotics reinforcement learning papers, like PLD or Policy Decorator. What A2C2 has done is to learn a residual refinement head by imitation learning objective, which are alike.
- A2C2 assumes to have access to the training dataset, which is not always possible (for example, one wants to deploy a pretrained VLA model to his environment zero-shot). While RTC is a rule-based refinement technique and plug-and-play, which is more intuitive. 

reference:

[1] Self-Improving Vision-Language-Action Models with Data Generation via Residual RL

[2] Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model

### Questions
- In the introduction, the author argues that "We first formulated delays in policy inference with VLAs that generate action chunks.". However, RTC also defines $d$ as inference delay. Could the author make this contribution statement more detailed and clear?
- In LIBERO setting, A2C2 uses the latent representation from smolVLM $z_t$. What about training the correction head without this feature? thus making the correction part agnostic to the base policy.
- RTC shows its effectiveness by doing hard tasks in the real world, e.g. strike a match and light the candle. What about the performance of A2C2 in the real world?
- The correction head is indeed a residual head, learning the difference between base policy and the ground truth data. What if using a more advanced VLA model like $\pi_{0.5}$ or X-VLA? I think the residual head works just because of the poor model capacity.

reference:

[1] π0.5: a Vision-Language-Action Model with Open-World Generalization

[2] X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical problem of inference latency in large Vision-Language-Action (VLA) models used for robotic control. The authors identify that the common practice of "action chunking," where a sequence of actions is predicted at once, leads to performance degradation in dynamic environments because the executed actions are based on stale observations. To mitigate this, they propose A2C2 (Asynchronous Action Chunk Correction), a lightweight add-on module that refines the base policy's pre-computed actions in real-time using the most recent observation. A2C2 is trained to predict a residual action, which is added to the base action, effectively closing the loop and providing reactivity without retraining the large, expensive base VLA. The method is evaluated on two distinct benchmarks: the highly dynamic Kinetix suite and the manipulation-focused LIBERO Spatial benchmark. Results demonstrate that A2C2 consistently and significantly improves success rates over naive and other baseline methods (like RTC) under various inference delays and long execution horizons, with minimal computational overhead.

### Strengths
1. The paper clearly formulates a critical, real-world problem—the loss of reactivity due to inference latency in large VLAs—that becomes more pressing as models scale. The framing of the problem, including the definitions of delay (d), horizon (H), and execution horizon (e), is precise and insightful.

2. The proposed A2C2 method is conceptually simple, architecturally lightweight, and highly practical. Its design as a plug-and-play residual correction head that can be attached to any pre-trained chunking policy is a major strength, as it avoids the prohibitive cost of fine-tuning large foundation models.

3. The paper provides a thorough empirical validation across two diverse simulation environments (Kinetix and LIBERO), which have different characteristics (highly dynamic vs. spatial reasoning). The experiments systematically test the method under varying delays and horizons, convincingly showing that A2C2 outperforms baselines across the board. The inclusion of inference time benchmarks (showing a 20x speedup over the base policy) solidifies the claim of low overhead.

4. The performance gains are substantial. For instance, on LIBERO with a significant delay (d=10, e=40), A2C2 improves the success rate from 67% to 84%. The results in the Kinetix environment (Table 4) are particularly striking, where A2C2 maintains high success rates even under large delays where other methods fail catasthetically.

5. The authors provide extensive implementation details, hyperparameters in the appendix, and have released source code for both experimental setups, which greatly facilitates reproducibility and future research.

### Weaknesses
1. While the core method is validated, a more detailed ablation study would strengthen the paper. For example, the contribution of the specific input components to the correction head (e.g., the temporal feature τ, the base policy's latent representation z, the base action itself) is not thoroughly isolated and analyzed. Understanding which components are most critical would provide deeper insight.

2. The paper focuses on success rates but provides little discussion or qualitative analysis of how the correction head fails when it does, or in what scenarios it might not help. A discussion of the limitations and potential failure modes (e.g., in extremely out-of-distribution states) would provide a more balanced view.

3. The correction head uses different architectures for the two benchmarks (a simple MLP for Kinetix and a more complex transformer encoder for LIBERO). The rationale for this difference, and an ablation showing whether the simpler MLP could suffice for LIBERO (or vice versa), is missing. This leaves some questions about the general architectural requirements for the correction head.

4. The evaluation in LIBERO, while valid, uses a set of instructions that are structurally very similar (all variations of "pick up the black bowl and place it on the plate"). It is unclear how well the method generalizes to more diverse and complex language instructions, which is a key promise of VLAs.

### Questions
1. The paper demonstrates A2C2 on a flow-matching policy (Kinetix) and SmolVLA (LIBERO). How sensitive is the performance of the correction head to the type of base policy (e.g., a diffusion policy or a much larger VLA like RT-2)? Is any fine-tuning of the correction head required when switching base models?

2. The method adds a residual to the base action. Were there any observed issues with the correction head producing overly large or unstable corrections that could lead to jerky or unsafe motions, especially at the boundaries between chunks? Is the smoothness purely emergent from the data, or was it explicitly encouraged?

3. The results show that A2C2 helps even in the absence of delay (d=0) on long horizons (e.g., H=50). Does this suggest that the correction head is also compensating for the base policy's own errors in long-horizon open-loop prediction, beyond just compensating for latency? Could A2C2 be seen as a general tool for improving the closed-loop performance of any chunk-based policy?

4. The experiments are in simulation. What are the anticipated biggest challenges in transferring this method to a real-world physical system, where observation noise and dynamics mismatches are significant? Does the correction head risk overfitting to the specific physics of the simulation?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Asynchronous Action Chunk Correction (A2C2), a lightweight add-on module for vision-language-action (VLA) models that corrects action chunks in real time using the latest observations. The method aims to mitigate performance degradation caused by inference delays and long execution horizons in dynamic robotic tasks. Experiments on Kinetix and LIBERO Spatial benchmarks show improved success rates under varying delays and horizons compared to naive and real-time chunking baselines.

### Strengths
1. The work addresses the critical and practical challenge of inference latency in large-scale robotic policies, a key obstacle to real-world deployment.

2. The proposed correction head is modular and can be applied to any pre-trained VLA without retraining, which is appealing for deployment.

### Weaknesses
1. No demo support: Without a demo, it is impossible to verify claims about mitigating "jerky motion" or to visually assess the failure modes of baselines. The large quantitative gains would be far more convincing if supported by qualitative evidence showing how A2C2 succeeds where others fail. 

2. No real-world experiments: As a robot learning paper, it is difficult to be highly convinced of the method's real-world effectiveness without physical experiments.

If I have misunderstood any points, I am open to discussion.

### Questions
1. How does the total inference time (base + correction) compare to the base policy alone? If the correction head runs in parallel, what is the end-to-end latency?

2. The training of $D_{cor}$ uses base policy inferences on the same dataset. Does this lead to overfitting? Have you tested on out-of-distribution tasks or environments?

3. The paper claims the method is “complementary to RTC,” but no experiments combine A2C2 with RTC. Have you tested this combination, and if so, what were the results?

4. Why not directly train a diffusion policy (or other policy) using the ​$D_{cor}$ data format? Such an approach might learn to generate actions conditioned on the latest observations and base actions, potentially solving latency without a separate correction head. Have you explored this? If not, why? If so, how does it compare to A2C2?

5. Why not plug the module in the large-scale VLA (e.g., $\pi_0$ or RDT)? because the latency phenomenon is more pronounced on large VLAs.

### Soundness
3

### Presentation
2

### Contribution
2
