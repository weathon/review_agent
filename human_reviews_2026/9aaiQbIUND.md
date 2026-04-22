# Leveraging Generative Trajectory Mismatch for Cross-Domain Policy Adaptation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Transferring policies across domains poses a vital challenge in reinforcement learning, due to the dynamics mismatch between the source and target domains. In this paper, we consider the setting of online dynamics adaptation, where policies are trained in the source domain with sufficient data, while only limited interactions with the target domain are allowed. There are a few existing works that address the dynamics mismatch by employing domain classifiers, value-guided data filtering, or representation learning. Instead, we study the domain adaptation problem from a generative modeling perspective. Specifically, we introduce DADiff, a diffusion-based framework that leverages the discrepancy between source and target domain generative trajectories in the generation process of the next state to estimate the dynamics mismatch. Both reward modification and data selection variants are developed to adapt the policy to the target domain. We also provide a theoretical analysis to show that the performance difference of a given policy between the two domains is bounded by the generative trajectory deviation. More discussions on the applicability of the variants and the connection between our theoretical analysis and the prior work are further provided. We conduct extensive experiments in environments with kinematic and morphology shifts to validate the effectiveness of our method. The results demonstrate that our method provides superior performance compared to existing approaches, effectively addressing the dynamics mismatch. We provide the code of our method at https://anonymous.4open.science/r/DADiff-release-83D5.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DADiff, a diffusion-based framework that addresses the challenge of transferring reinforcement learning policies across domains with different dynamics. By leveraging the generative trajectory discrepancy between source and target domains, DADiff estimates dynamics mismatch and adapts policies through either reward modification or data selection strategies. Supported by theoretical analysis showing the performance difference is bounded by generative deviation, the method demonstrates superior effectiveness in experiments with kinematic and morphology shifts compared to existing approaches.

### Strengths
# Strengths

- This paper is well-motivated and mostly well-written
- This paper is easy to follow, and the studied topic is of importance in the context of the reinforcement learning community. It is always important to develop more general and stronger transfer algorithms in RL, especially considering the fact that online off-dynamics RL papers have rarely appeared in recent years
- The authors include theoretical analysis to provide better guarantees for the proposed method (despite the fact that some of the theoretical results resemble those in prior works, they are still interesting and bring some insights into the cross-domain reinforcement learning). I appreciate that the authors include a detailed discussion about the connections between the theoretical bounds of their method and those of PAR
- The presentation is good, and I like the way the authors tell the whole story
- The parameter study is extensive, covering numerous tasks in the main text and the appendix.

### Weaknesses
# Weaknesses

- The authors propose to address the online policy adaptation problem from the perspective of generative modeling; however, the downstream methods still rely on reward modification or data filtering, which resembles DARC, PAR, and VGDF
- The evaluations are limited to kinematic shift and morphology shift. As far as the reviewer can tell, ODRL provides other dynamics shifts like gravity shift, friction shift, etc. This paper can benefit greatly from extending its experimental scope
- The authors mention flow matching in the main text. This raises questions that there are numerous generative modeling methods other than diffusion models. This paper lacks a comparison between different generative modeling methods.

Overall, I would recommend a "weak accept" of this paper.

### Questions
# Questions

1. As a generative modeling method, diffusion can also be used for data augmentation, e.g., generating samples that lie in the scope of the target domain. What is the insight in using diffusion model for *Generative Trajectory Mismatch* rather than target domain data augmentation?
2. How diffusion models compare against other generative modeling methods like flow matching, VAE?
3. The diffusion steps seem to have a significant impact on DADiff. Can authors provide more insights on how to select this parameter and why different diffusion steps can have such significant impacts on DADiff?

### Soundness
3

### Presentation
3

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
This paper proposes DADiff, an online dynamics adaptation method for RL that measures source–target dynamics shift via generative trajectory deviation from diffusion models. It developed two variants: reward modification and data selection. A performance bound links return gap to KL terms along a shared latent trajectory. Experiments on 4 MuJoCo envs report competitive or superior performance to DARC, VGDF, PAR, SAC‑tune, and SAC‑tar.

### Strengths
1. Clear theoretical link from generative trajectory discrepancy to performance, with clean proof and recovery of PAR as a special case.
2. Consistent improved empirical performance on many tasks. DADiff‑modify often leads; DADiff‑select is strong when penalties mis‑shape rewards.
3. Parallel latent sampling avoids reverse‑chain cost; runtime comparable to model‑free baselines and far below VGDF.

### Weaknesses
1. Baseline fairness. SAC‑tar is trained for 10^5 target steps, while DADiff and SAC‑tune use 1M source steps + 10^5 target steps. This probes a target‑only‑from‑scratch regime but does not compute‑match total experience. Please add a compute‑matched target‑only control with comparable total environment interactions and gradient updates

2. Insufficient analysis: The text narrates Fig. 2 but provides little analysis in Sec 5.2. Please also quantify the deviation differences between the two generative trajectories, since the paper only covers the computational effeiciency. There should exist a deviation difference between these two trajectories.

3. Writing quality: Multiple typos, symbol switches, and undefined or late‑defined notation reduce clarity. For examples: Fig 4(a) using $\gamma$ while Eqn 11 using $\lambda$. $\phi_i$ is undefined in Eqn 14 until I found out the algorithm is based on SAC. Sec 5.3 states optimal $\lambda$ is task-dependent while Sec E.2 (line 1019) says $\lambda$ is task-independent.

### Questions
Same as weakness

Additional questions:
1. Is there a missing square in the Eqn 12 and 13? If not, justify using $E[Q−TQ]$ rather than MSE. If yes, re‑run results with corrected losses and report any deltas.
2. Can you extend the analysis of why "directly filtering for transitions with low dynamics mismatch is a more effective strategy than modifying rewards." in your Sec 5.2 (line 352). Provide mechanism‑level reasoning and ablations that include filtering only vs reward‑shaping only vs both. Maybe analysis from the perspective of probabilistic trajectory in diffusion model could explain why filtering is better.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a diffusion-based method to obtain the domain gap and provides a reward modification and data filtering method for policy learning. They provide a theoretical guarantee of the policy $\pi$'s performance on the two domain.  Theoretical results and empirical results shows performance improvement of the method.

### Strengths
The paper is well written and easy to follow. They propose a new diffusion-based domain gap measure method similar to DARC and PAR. Similar to DARC and PAR, they identify a performance gap in policy $\pi$ between the source and target domains， defined by the KL divergence of the latent representation.

### Weaknesses
1, the odrl benchmark has both gravity shift and friction shift, which are not included in the experiments. Also, what is the shift level of the experiment? Is it easy, medium or hard? 

2, the novelty of the paper seems not significant to me. The high-level idea of it is to obtain a more fine-grained shift measurement compared to DARC and PAR, and the theoretical analysis is actully similar to those paper, except with sligtly different notation of the shfit measurement. Also, the performance doesn't have a significant improvement compared to them. 

3, DARC and PAR have been shown to be ineffective when the shift is large. What is your performance on a large shift case? 

4, The performance of your method seems to rely on an assumption that the domain shift is not that large. If the shift is too large, the KL in Eq. 5 will grow extremely large, or even infinity. The performance is bounded only when the KL of the source and target is bounded. Also, as stated in [1], the KL can be ill defined when the shift is large because there is no support of some target transition in source domain. 

In summary, I am questioning whether the reward modification method is still a valid method to solve the off-dynamcis RL problem as many previous work [1,2] has shown the limitation of it and when the shift is large (the joint distribution is small), the reward modicication method always fails, performing good in the source but poorly in the target.   

[1] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data
[2] Off-Dynamics Reinforcement Learning via Domain Adaptation and Reward Augmented Imitation

### Questions
See weakness.

### Soundness
2

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
4

### Summary
This paper addresses the problem of online dynamics adaptation in reinforcement learning, where a policy is pre-trained in a source domain (e.g., a simulator) and must be adapted to a target domain (e.g., the real world) with only limited interactions. The authors propose DADiff, a novel framework that leverages generative models, specifically diffusion models, to capture the dynamics mismatch between domains. The core idea is to interpret the state transition as a conditional generative process and to measure the "generative trajectory deviation"—the discrepancy between the latent state trajectories of the source and target domains during the diffusion generation process. The paper provides a theoretical performance bound linking this deviation to the policy's performance gap and proposes two practical variants: DADiff-modify (which penalizes source-domain rewards based on the deviation) and DADiff-select (which filters source-domain data). The method is also extended to the Flow Matching framework. Experiments on MuJoCo environments with kinematic and morphology shifts show that DADiff outperforms several strong baselines, including PAR.

### Strengths
The paper offers a fresh and principled perspective on dynamics adaptation by framing it as a problem of generative trajectory mismatch. This is a significant conceptual shift from prior work that often relies on domain classifiers or single-step representation learning.

### Weaknesses
The primary concern is the justification for the added complexity of using a diffusion model for dynamics modeling. While the results are strong, the performance gain over the strongest baseline, PAR, is sometimes marginal (e.g., in `ant(broken hips)` or `walker(broken right foot)`). The paper acknowledges that VGDF, a model-based method, is significantly slower, but it does not provide a detailed analysis of DADiff's own computational cost (e.g., training/inference time, memory footprint) compared to PAR, which is a crucial factor for real-world applicability. The increased complexity needs a more compelling justification in terms of capability.

The experiments are conducted on standard MuJoCo locomotion tasks, which, while common, have relatively simple and deterministic dynamics. The paper’s core claim is about capturing complex dynamics mismatches via diffusion models. To truly validate the advantage of modeling the full generative trajectory, experiments on tasks with more complex, high-dimensional, or highly stochastic dynamics would be far more convincing. The current experiments, which largely follow the setup of PAR, do not fully showcase the potential benefits of the proposed method in more challenging scenarios.

### Questions
The paper mentions an extension to Flow Matching (Appendix C). Could the authors elaborate on the specific modifications required in the algorithm? In the diffusion framework, the deviation is calculated using the noise prediction model `ϵ_θ`. What is the direct analogue in the Flow Matching framework? Is it solely based on the vector field prediction `v_θ`, and if so, how does the continuous-time nature of the trajectory in Flow Matching affect the calculation and interpretation of the "generative trajectory deviation" compared to the discrete steps in diffusion?

### Soundness
3

### Presentation
3

### Contribution
3
