# Flow Policy Gradients for Legged Robots

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
We study robot control with flow policy optimization (FPO), an online reinforcement learning algorithm for flow-based action distributions. We demonstrate how flow policy optimization can succeed for more difficult continuous control tasks than shown in prior work, using a set of design choices that reduce gradient variance and regularize entropy. We show that these design choices mitigate policy collapse challenges faced by the original FPO algorithm and use the resulting algorithm, FPO++, to train flow policies for legged robot locomotion and humanoid motion tracking. We find that FPO++ is stable to train, interpretably models cross-action correlations, and can be deployed to real humanoid robots.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of learning control policies for high degree of freedom robots with complex dynamics, such as humanoid robots. The paper shows that a prior state of the art algorithm FPO suffered from problems of mode collapse, or gradual decay in performance. The authors propose a modified algorithm called FPO++ that contains four improvements to FPO, and show that FPO++ allows some complex behaviours to be learned.

### Strengths
The paper shows the limitations of FPO, and empirically delivers on the promised performance. The ideas are relatively clearly spelled out.

### Weaknesses
There isn't anything specifically wrong with this paper, but the contribution is somewhat limited, and there is not sufficient analysis of trade-offs presented by the four modifications. The four main changes from FPO to FPO++ are:

1. Ratios are computed per-sample rather than per-action, increasing effective
batch size.

The authors state that this modification decreases the gradient variance, but "it comes at the cost of bias.". The performance improvement is empirical, and while it may lead to more stable final policy returns, it would be important to perform analysis that says when this bias is problematic -- for example, if the samples are themselves high variance and have high dynamic range, then the gradient may actually vanish. Are there problems with this characteristic?

2. The SPO surrogate objective is used for negative advantages.

This objective "also reduces gradient variance" and empirically improves stability but again -- when is this problematic? Could the gradient vanish in this case too?

3. The CFM loss uses a Huber kernel with clamping to improve numerical
stability.

How did the Huber kernel get chosen? How easy is it to choose? Is there an interaction between the Huber kernel size and the trust region size?

4. At test-time, actions are sampled by initializing flow integration from
\epsilon = 0.

This seems to be a fairly minor step.

Given that these are the contributions, it would be important for the authors to give insight as to whether these changes in general are appropriate for most applications (should all applications of FPO use these modifications) or if they really are specific to legged locomotion in some way. This is especially important because the authors attribute the two main failures of FPO to numerical issues ("floating-point overflow" and "a small approximation error from Euler integration"). These would presumably be present in many applications.

The experimental results demonstrate generality across robots, but the paper is not precise about what the benchmarks are -- the cited paper is for Orbit which is a simulation environment paper, not a benchmark paper. Presumably the episode return in figures 2-6 are forward progress in some manner -- this is not stated.

The results in figures 7-9 are impressive, but do not necessarily add much to the overall paper in terms of convincing us that the authors have overcome some issues with numerical conditioning.

### Questions
Please see the questions in my above review.

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
This paper addresses challenges with the existing Flow Policy Optimization (FPO) and proposes improvements for successful deployment in real-world robotic experiments. The authors also include strong supplementary material demonstrating the policy’s performance on physical robots, a commendable and non-trivial effort.

They employ a "per-ratio clipping" mechanism and replace standard PPO with an entropy-preserving trust region (ASPO), claiming that these design choices are crucial for stable performance.

While the authors note that outperforming Gaussian PPO is not the primary goal, it is nonetheless used as the main baseline. This choice is somewhat questionable. Why not include more relevant baselines such as DIME [1] or DDPO [2]?

[1] DIME: Diffusion-Based Maximum Entropy Reinforcement Learning (Celik et al.)

[2] Training Diffusion Models with Reinforcement Learning (Black et al.)

### Strengths
- Impressive results on real robot experiments – a significant and practical achievement.  
- The topic is relatively new; related work is mostly well-covered (though missing DIME).
- The paper is clearly written and easy to follow.

### Weaknesses
- The contribution is largely a combination of engineering and design choices rather than a fundamentally new method.  
- The experimental analysis is underdeveloped. For example, Figure 3 shows that the per-sample ratio has a large effect, but the paper lacks further investigation into vanilla FPO (e.g., varying batch sizes, the number of monte carlo samples, gradient clipping, etc.) aside from a brief description in section 2.2.  
- The claim of reduced gradient variance is made without theoretical or empirical evidence. The argument using Jensen’s inequality on averaged importance weights is questionable: maximizing an upper bound (on the weighted advantage) does not seem well-justified and may even be problematic from a theoretical standpoint. Since this is listed as a key contribution, it should be justified more rigorously.  
- The paper seems to straddle two goals: (1) investigating methods for improving FPO, and (2) showcasing real-world robotics results. It would be stronger if it leaned more clearly toward one of these directions or made more comprehensive use of the appendix to address weaknesses.  
- The relative importance of the three main components in Section 2.4 (“Final FPO++ Objective”) is not adequately analyzed in the results. As a smaller note, item 4 (“epsilon starting at zero”) does not seem distinctive enough to be listed as a key contribution.

### Questions
- Figure 5 (single action dimension) is not very informative, and the specific action dimension is not discussed. Showing the full action space would be more meaningful (e.g., including additional DoFs in the appendix, or projecting all dimensions to a 2D space somehow). The resulting distribution is unimodal, which weakens the motivation for using a more expressive flow-based policy over a Gaussian one. Do the authors observe any cases where a more complex action distribution emerges?  
- Did the authors attempt adaptive learning rates (e.g., based on the trust region) or smaller Monte Carlo sample sizes (smaller N) to address FPO instability?  

## Minor Comments
- Unclear notation in the legends of Figures 3 and 4. I assume “w/” is meant to indicate “without” (“w/out” would be clearer).  
- The action correlation visualization is an interesting idea but lacks clear practical value. Similar correlations could likely appear with Gaussian PPO, so this figure might be better suited for the appendix.  
- Reported return and episode length are difficult to interpret without context on the environment horizon or reward scale. Providing this context would improve clarity.  


## Overall Assessment
The paper is valuable for its applicability and engineering contributions to real-world robot learning. However, its scientific claims are weakly supported, particularly regarding gradient variance reduction and the claimed expressivity benefits of flow-based policies. As a result, the theoretical and empirical analyses require further development. However, I commend the authors’ effort in real-world validation and supplementary videos, which are the clear strengths of the paper.

### Soundness
2

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
The paper proposes modifications to Flow Policy Optimization (FPO) to prevent instability and entropy collapse in robot control, and shows that the resulting FPO++ learns legged locomotion and motion tracking in simulation and transfers zero‑shot to two humanoid robots, albeit with limited baselines and generalisation.

### Strengths
- **Extensive Benchmarks:** The paper provides thorough experimental validation across four challenging locomotion tasks (Go2, Spot, H1, G1) in IsaacLab. Also, includes detailed ablation studies isolating each proposed component and direct comparisons against Gaussian PPO.  

- **Empirical Results:** FPO++ demonstrates stable training without entropy collapse and consistently achieves higher or comparable returns relative to Gaussian PPO,.

- **Sim-to-Real Transfer:** Policies trained entirely in simulation transfer successfully to real humanoid robots, exhibiting robust walking and motion-tracking behaviors without additional fine-tuning.

### Weaknesses
- **Limited Baselines:** The paper mostly compares FPO++ with Gaussian PPO. It does not include other baselines like FPO or its variants. This makes it hard to know if the improvements come from the modifications to FPO.  

- **Limited Evidence for Claimed Mechanisms:** The paper suggests FPO++ reduces variance and preserves entropy but does not provide direct measurements (e.g., gradient variance, policy entropy curves). The evidence is indirect, so the explanation is not fully convincing.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

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
This paper extends flow policy optimization (FPO) to complex continuous control tasks like legged locomotion. Specifically, the authors empirically find 2 challenges in the existing FPO method. They observe that as the training progresses, the policy either collapses during training or there's a gradual decay once it reaches the peak. To address this, the authors propose to reduce the variance of gradients during training by computing the surrogate ratio per-sample instead of per-action, regularizing the entropy of actions with Asymmetric SPO. They show strong sim2real experiments on two humanoid robots for locomotion and whole body motion tracking.

### Strengths
1. The proposed method shows strong Sim2Real results (on multiple embodiments - Booster and Unitree G1) when compared to the FPO algorithm, which cannot be transferred to the real world for legged robots.

### Weaknesses
1. The motivation of "Policy Collapse during training" is lacking concrete experimentation. The authors on L124 say "even after tuning hyperparameters like learning rate, clip threshold, ..." but provide no plots backing up their claim in the main paper / in the appendix.

	- What normalization strategies have the authors evaluated to back up this claim?


2. I would like to see a comparison with state-of-the-art practices for Deep RL. Specifically:
	- Instead of an MLP, use a more BroNet/SimBa(v1/v2)-like architecture, which has skip connections + normalizations. There is strong evidence to suggest that adopting such architecture leads to much stable training.
	- Adopt RSNorm for the observation to be able to deal with non-stationarity in Deep RL appropriately.

3. [Unverified Huber loss experiments]: On L189, the authors mention that "We empirically verify the importance of both the Huber kernel and clamping in our experiments." However, there are no experiments showing this.
	- Does the same value of $\delta$ work across different environments?
	- How sensitive is the method ot $\delta$?

4. Figure 4 shows that FPO++ without the ASPO objective -- either with PPO or SPO-- is as performant as the proposed FPO++. It is unclear, then, as to what ASPO's contribution really is. Can the authors elaborate on this?


5. The hyperparameters section in the Appendix is quite small and makes me question the reproducibility of this paper. No mention of the huber and clamp parameters (unless they are the same as the clip parameters -- but then they also need to be validated with a hyperparameter sweep), number of environments used, mini-batch size during the training, etc.

### Questions
6. How significant is the difference between FPO++ and PPO when it comes to Table 1 (Motion tacking)?

### Soundness
3

### Presentation
2

### Contribution
2
