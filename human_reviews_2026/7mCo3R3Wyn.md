# TEMPFLOW-GRPO: WHEN TIMING MATTERS FOR GRPO IN FLOW MODELS

- Decision: Accept (Poster)
- Scores: 10, 8, 6, 6

## Abstract
Recent flow matching models for text-to-image generation have achieved remarkable quality, yet their integration with reinforcement learning for human preference alignment remains suboptimal, hindering fine-grained reward-based optimization. We observe that the key impediment to effective GRPO training of flow models is the temporal uniformity assumption in existing approaches: sparse terminal rewards with uniform credit assignment fail to capture the varying criticality of decisions across generation timesteps, resulting in inefficient exploration and suboptimal convergence. To remedy this shortcoming, we introduce TempFlow-GRPO (Temporal Flow-GRPO), a principled GRPO framework that captures and exploits the temporal structure inherent in flow-based generation. TempFlow-GRPO introduces three key innovations: (i) a trajectory branching mechanism that provides process rewards by concentrating stochasticity at designated branching points, enabling precise credit assignment without requiring specialized intermediate reward models; (ii) a noise-aware weighting scheme that modulates policy optimization according to the intrinsic exploration potential of each timestep, prioritizing learning during high-impact early stages while ensuring stable refinement in later phases; and (iii) a seed group strategy that controls for initialization effects to isolate exploration contributions. These innovations endow the model with temporally-aware optimization that respects the underlying generative dynamics, leading to state-of-the-art performance in human preference alignment and text-to-image benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper proposed TempFlow-GRPO, a framework that makes the optimization process temporally aware to address the key limitation of temporal uniformity in previous RLHF works. The paper introduces a mixture of ODE and SDE sampling, along with a noise-aware policy weighting scheme, to balance exploration and reward exploitation. Experiments demonstrate that TempFlow-GRPO achieves state-of-the-art performance, yielding higher rewards than standard GRPO approaches.

### Strengths
- The paper pinpoints temporal uniformity as the primary limitation of existing flow-based GRPO methods and proposes TempFlow-GRPO to solve it with precise credit assignment and noise-aware optimization. The authors demonstrate this non-uniformity well with empirical evidence from rewards, supporting the need for temporal information.
- The paper introduces the core mechanisms of trajectory branching and noise-aware reweighting to create temporally-structured policies that respect the dynamics of the generative process. The authors also provide a theoretical justification from the policy gradient perspective, further supporting the use of noise-aware reweighting.
- The proposed TempFlow-GRPO achieves state-of-the-art performance compared to the existing vanilla GRPO approach, demonstrating the effectiveness of the method. The authors also include comprehensive ablation studies to better understand the dynamics of this model.

### Weaknesses
- The computational cost, as thoroughly analyzed in Appendix A.6, will be higher than the vanilla GRPO models due to the branching process. Nonetheless, this is more like a trade-off between quality and time, given the superior quality metrics.

### Questions
- How is the performance affected by the number of branches (K) at each step, the specific timesteps chosen for branching, or the exact function used for noise-aware weighting? The ablation study (Fig. 8) shows that the 4x6 (seed x branch) configuration was chosen, but it's unclear how much tuning is required to find the optimal setup for a new model or dataset. A discussion on how to choose these hyperparameters will be useful for general applications of the proposed framework.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents TempFlow-GRPO, a new reinforcement learning framework that addresses the limitation of uniform credit assignment across timesteps. The method introduces trajectory branching, which switches from ODE to SDE sampling at selected timesteps to generate exploratory branches and assign their rewards to intermediate states. This paper further proposes noise-aware policy weighting, prioritizing optimization at high-noise early stages over low-noise refinement phases. Experiments show that TempFlow-GRPO achieves substantially improved efficiency and final performance compared to the baselines.

### Strengths
- The paper is overall well-written and easy to follow.
- The motivation and the proposed method are clear and straightforward: addresses the temporal inhomogeneity and credit assignment problems through intermediate resampling for intermediate value estimation and noise-aware reweighting.
- The proposed method shows strong empirical performance in both efficiency and end-level performance, with comparisons that include GPU time.

### Weaknesses
- Theorem 1 is intuitively reasonable, but labeling it as a Theorem feels overstated since the underlying assumptions and proof sketch are insufficiently formalized. The analytical depth is also somewhat limited.
- The explanation around line 847 (regarding why the average number of branches is 4.5× when K = 10) is unclear. It is not obvious how this factor arises or how the branching schedule operates, and the paper does not explicitly describe it.
- Adding more algorithmic details or pseudocode would improve readability and make the proposed procedure easier to follow.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

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
This paper addresses the sparse terminal reward and uniform credit assignment problem in GRPO training of flow models. The authors propose TempFlowGRPO, which includes: (1) Trajectory Branching, where only one step of SDE is used at timestep k; (2) Noise-Aware Policy Weighting by reweighting according to noise level; and (3) a seed group strategy. The method achieves state-of-the-art performance in human preference alignment and text-to-image benchmarks.

### Strengths
1.	The authors astutely identify that the FLOW-GRPO algorithm treats all timesteps equally, and tackle this issue via single-timestep SDE optimization.
2.	The noise reweighting method is shown to be effective through both soild theoretical analysis and experiment results.
3.	The paper is generally well written with a clear logical structure.

### Weaknesses
1.	The contribution of seed group strategy is relatively small to other parts of the work, and the paper should provide additional details of the seed group strategy.
2.	Similarly, MixGRPO [1] proposes a training window of SDE time steps that also tackles the issue of treating all timesteps equally. However, there is limited discussion comparing with MixGRPO.
3.	The paper does not discuss the phenomenon of reward hacking, which is an inevitable problem for the GRPO method.

[1] Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde

### Questions
1.	The trajectory branching mechanism appears similar to MixGRPO limited with a single-timestep window. How do their efficiency and effectiveness compare?
2.	The paper claims that Flow-GRPO (Prompt) is an improved baseline with group standard deviation stabilization, but does not provide much detail. Could the authors elaborate on this improved method?
3.	Why are the Pickscore curve trends by steps and GPU hours on the left of Figure 3 inconsistent?
4.	Compare to FlowGRPO, the experiment of Visual Text Rendering is not addressed. How well does TempFlow-GRPO perform on this particular task?

### Soundness
4

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
TempFlow-GRPO is a temporally-aware reinforcement learning framework for flow matching models that improves human preference alignment by introducing trajectory branching, noise-aware weighting, and seed grouping to achieve precise credit assignment and efficient optimization across timesteps.

### Strengths
For reinforcement learning tasks, dense rewards are crucial for effective credit assignment. The proposed Trajectory Branching mechanism provides an elegant and effective way to obtain dense rewards along the denoising trajectory.

The introduced reweighting mechanism offers a valuable analysis of how gradients evolve across steps in baseline algorithms and presents a solution to mitigate the identified issues.

### Weaknesses
The proposed method involves numerous ODE denoising steps, which substantially increase computational overhead. However, the paper lacks a comparison against the baseline method using training time as the horizontal axis to illustrate efficiency trade-offs.

The authors should evaluate the performance of the reweighting mechanism under different $\sigma_t$ schedulers rather than relying solely on the one used in Flow-GRPO, to examine how the choice of scheduler influences its effectiveness. It remains unclear whether simply reweighting the coefficients in the earlier part to 1 would yield good results under different schedulers.

### Questions
The comparison between batch std and global std is only evaluated on PickScore. How does this observation generalize to other tasks?

Can the proposed reweighting mechanism be applied to hybrid variants (FlowGRPO-Fast/MixGRPO) where only a subset of steps follows an SDE formulation?

### Soundness
3

### Presentation
3

### Contribution
3
