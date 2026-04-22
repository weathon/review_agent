# Model-Based Diffusion Sampling for Predictive Control in Offline Decision Making

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Offline decision-making requires synthesizing reliable behaviors from fixed datasets without further interaction, yet existing generative approaches often yield trajectories that are dynamically infeasible. We propose Model Predictive Diffuser (MPDiffuser), a compositional model-based diffusion framework consisting of: (i) a planner that generates diverse, task-aligned trajectories; (ii) a dynamics model that enforces consistency with the underlying system dynamics; and (iii) a ranker module that selects behaviors aligned with the task objectives. MPDiffuser employs an alternating diffusion sampling scheme, where planner and dynamics updates are interleaved to progressively refine trajectories for both task alignment and feasibility during the sampling process. We also provide a theoretical rationale for this procedure, showing how it balances fidelity to data priors with dynamics consistency. Empirically, the compositional design improves sample efficiency, as it leverages even low-quality data for dynamics learning and adapts seamlessly to novel dynamics. We evaluate MPDiffuser on both unconstrained (D4RL) and constrained (DSRL) offline decision-making benchmarks, demonstrating consistent gains over existing approaches. Furthermore, we present a preliminary study extending MPDiffuser to vision-based control tasks, showing its potential to scale to high-dimensional sensory inputs. Finally, we deploy our method on a real quadrupedal robot, showcasing its practicality for real-world control.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Model Predictive Diffuser (MPDiffuser), a model-based diffusion framework for offline decision-making. It integrates three modules as a self-loop refinement on rollout state-action sequences: a planner for generating task-aligned trajectories, a dynamics model for enforcing feasibility, and a ranker for selecting high-quality trajectories, combined via an alternating diffusion sampling process. 

The method is evaluated on both unconstrained (D4RL) and constrained (DSRL) benchmarks, as well as a real quadruped robot, showing consistent performance improvements over prior diffusion-based methods (e.g., Decision Diffuser, Diffuser, D-MPC).

### Strengths
1. The modular composition of planner, dynamics, and ranker is simple and practical, allowing separate training yet coherent sampling through alternating diffusion.
2. Alternating updates effectively enforce dynamics consistency, an issue often ignored in prior diffusion planners like Decision Diffuser or Diffuser.
3. Solid empirical sweep across unconstrained and constrained offline RL, plus a real robot showcase.
4. The paper is well written and easy to follow.

### Weaknesses
1. Experiments: (1) There are some major diffusion-based paradigms that are missed; quantitative comparison and difference clarification to, e.g., latent diffusion planning [1] and IDQL [2] are necessary. (2) D-MPC is a close cousin; more controlled ablations to isolate alternation vs. two-model composition vs. one-shot joint denoising would be valuable.

2. Feasibility is argued indirectly. Results emphasize returns/costs and success rates, but I’d like explicit dynamics-consistency metrics (e.g., simulation rollout deviation between predicted v.s. realized states across datasets, beyond the Car U-Maze illustration in Fig. 2, p.4).

3. Ranking depends on learned models. The ranker uses learned reward/cost estimators (p.5)—this introduces model bias that could favor trajectories that “look good” under misspecified critics. Some calibration or OOD-robustness analysis would increase confidence.

4. Theoretical results are heuristic. I understand the author’s point, but adding a formal one in the appendix or clarifying which prior work could be reused would be better.

5. **Novelty**: I am mainly concerned about this paper because it is indeed an inverse dynamic model; There are so many works proposing similar variants in recent years. Also, the advantages of this design are unclear, even considering it as a recurrent paradigm, compared to a time-iterative paradigm, e.g., diffusion forcing.

Minor: 
1. Sensitivity to guidance strength $\omega$, number of alternations, and sample count $N$ (beyond Table 5) could be expanded.
2. There’s an appendix note about initial-state conditioning choices; a quick pointer to which is used in the main paper would help clarity.
3. Using \citet or \citep could improve readability and match the submission requirements.

Generally, this paper presentation is good and the improvement is impressive, so I would consider raising the score by 2-4 accordingly, if the authors can convince me, especially in novelty.

[1] Xie, Amber, et al. "Latent diffusion planning for imitation learning." arXiv preprint arXiv:2504.16925 (2025).

[2] Hansen-Estruch, Philippe, et al. "Idql: Implicit q-learning as an actor-critic method with diffusion policies." arXiv preprint arXiv:2304.10573 (2023).

### Questions
1. **Concerns on Self-loop/Overfitting**: Does the self-refinement via the dynamics model risk confining samples to a specific data manifold or its original dynamics, reducing robustness to OOD or novel environments? Is there any mechanism, such as unsupervised diversity injection or additional constraints in planning, to mitigate this? 

2. The dynamics diffusion model only predicts states, but does not explicitly model the control mechanism or policy uncertainty involved with action. Could this limit safety or adaptability when control distributions shift (e.g., actuator latency or unmodeled forces)? How might the framework be extended to capture such control-level variations? Is it a bottleneck in experiments?

3. When does alternation hurt (e.g., highly stochastic or chaotic dynamics)? Any examples where the dynamics model over-regularizes and harms task performance?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed model predictive diffuser, where generated trajectory from diffuser planner is further grounded with learned dynamics model and improve it with rank model. The combination of planner and dynamics update through alternating diffusion sampling ensure task alignment and dynamical feasibility. MDPiffuser also shows strong performance on common benchmarks and the author also provide real world result on unitree go2 robot.

### Strengths
- the paper provide comprehensive comparison between the proposed method with baselines on common benchmarks with application on real world robots.
    
- by learning a dynamics model, the proposed method can better utilize the low quality data to further improve the sample efficiency and generation quality

### Weaknesses
- although author provides comprehensive study on baselines like d-mpc and decision diffuser, further clarification on why proposed mdpdiffuser is better than others method in terms of dynamical feasibility is still unclear. assuming all methods learns a correct dynamical model, then all generated trajectory should be feasible.
    
- the role of ranker module and its contribution to final performance is missing. It would be helpful to comment on ranker module’s difference and connections with other method which use reward as extra guidance.

### Questions
- can author comment on what if the learned dynamics model is off and if that would further degrade the performance of the planner itself?
    
- how does proposed alternating update scheme compared to classifier-guidance -style score combination? the latter would be similar to Lagrangian method update in constrained optimization. further justification would be appreciated.

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
4

### Summary
This paper proposes an interleaved-stage optimization scheme for diffusion-based control, where the planning model (which generates state–action sequences) and the dynamics model (which generates states) are updated alternately. Candidate trajectories are then ranked using task-conditioned targets. The idea is to ensure that (i) the dynamics model stays faithful to the real environment, and (ii) the planner actually produces task-solving trajectories that also satisfy constraints and remain dynamically consistent. The method is clearly presented, and the experiments on several continuous-control tasks plus a real-world setup show improvements in policy learning and dynamics consistency, including in constrained MDP settings.

Overall, the motivation is reasonable and the idea is neat. The empirical results do support the main direction. That said, I still have some concerns about the method and the experimental evidence, especially regarding how strongly we can claim “dynamics consistency” from the current setup. So at this point I would rate it a 4, but I’m open to revising this after clarification and discussion

### Strengths
1. The overall motivation is clear, which aims for planning and dynamics consistency is an important goal, especially in offline RL settings.


2. The experiments are fairly comprehensive. While some important baselines are missing, the current results are still sufficient to support the main claim that this iterative learning approach can improve policy performance.


3. The overall presentation is clear and easy to follow.

### Weaknesses
I list both weaknesses and questions together here, since many of them overlap.

1. My main question is whether the method truly achieves dynamics consistency. As I understand it, the dynamics model is also conditioned on the target variables $y$ . This can make the generated trajectories biased toward those targets, effectively learning “goal-conditioned” dynamics rather than the true environment dynamics. Unless the dataset has good coverage over target variables, the learned dynamics may still be biased.

2. Related to that, dynamics are inherently temporal. For real dynamics consistency, the model should condition on temporal information; otherwise, denoising all states across time independently can break the Markov structure. You may want to look at temporal denoising approaches such as Diffusion Forcing [1], which explicitly denoise along the time axis and better preserve the actual dynamics.

3. This is minor, but given the above, if the planning step produces noisy / off-manifold actions, the dynamics model might also become flawed.

4. Following the previous points, it would be helpful to show whether the dynamics stage can actually correct or pull back bad / inconsistent plans produced by the planning stage (correct me if I missed anything).

5. On evaluation, a few relevant baselines are missing (not necessarily all needed for rebuttal), especially Diffusion Forcing [1] and Safe-Diffuser [2] (for the constrained MDP setting).

6. I’m also curious whether the interleaved optimization is sensitive to initialization, e.g., whether an early bias in the planner will steer the dynamics updates in a suboptimal direction.


[1] Chen, Boyuan, et al. "Diffusion forcing: Next-token prediction meets full-sequence diffusion." Advances in Neural Information Processing Systems 37 (2024): 24081-24125.

[2] Xiao, Wei, et al. "Safediffuser: Safe planning with diffusion probabilistic models." The Thirteenth International Conference on Learning Representations. 2023.

### Questions
Other than the points above, I’m also curious whether enforcing consistent dynamics brings additional empirical benefits. Specifically, does it allow the planner to operate over longer horizons than the baselines, improve the accuracy of inverse dynamics models, or lead to faster convergence during training?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes MPDiffuser, a model-based diffusion framework for trajectory generation in offline RL. Unlike previous decision diffusion models that generate trajectories purely from a data-driven denoising process, MPDiffuser alternates between two denoising operators:
a task-oriented planner (encouraging reward-optimal trajectories) and
a dynamics diffusion model (projecting trajectories back onto the manifold of physically feasible transitions).
This “alternating diffusion” is theoretically supported using KL projection and operator splitting, suggesting that the method approximates constrained optimization over both reward and system dynamics. Additionally, a ranker is included to enforce safety constraints and trajectory preferences. Experiments on D4RL, DSRL, and a real Unitree Go2 robot show improved performance and safety compliance over standard Decision Diffuser and D-MPC.

### Strengths
Addresses a clear weakness of current decision diffusers
Most diffusion-based planners ignore system dynamics and often generate trajectories that are not physically realizable. Alternating between planning and dynamics correction feels like a natural but powerful extension.

Elegant modular design
The architecture splits the problem into three parts: planner, dynamics module, and a ranker. Each is separately trained and conceptually clear, which improves interpretability and reproducibility.

Theoretical grounding
The explanation using KL-constrained optimization and operator splitting helps justify why alternating diffusion steps should converge to valid and optimal trajectories. It’s not just a heuristic.

Robust experimental results
MPDiffuser achieves consistently higher returns than prior diffusion RL methods (Diffuser, Decision Diffuser, D-MPC) on locomotion and manipulation benchmarks. I also appreciate that they test both reward maximization and constrained (safety) optimization settings.

Real robot deployment
The fact that the method runs on a real quadruped robot adds credibility. Many diffusion RL papers remain purely in simulation.

### Weaknesses
Still limited to state-based tasks — no vision input
All experiments assume full-state observations. It is unclear if the method can scale to high-dimensional visual inputs or work jointly with latent diffusion policies.

No direct comparison to strong model-based RL or world models (Dreamer, TD-MPC2)
Since this is a model-based method, I expected comparisons to world-model-based planners, not just decision diffusers.

Scalability and inference cost not fully discussed
Alternating between two diffusion models during sampling is clearly more expensive than a single denoiser. There is no clear measurement of planning time (ms per trajectory), nor GPU/compute cost.

Dynamics model accuracy is critical but not analyzed
If the learned dynamics model is slightly wrong or trained on limited data, does the system collapse? No robustness or ablation is reported in this direction.

Real-world testing could be more diverse
The Unitree Go2 experiment is appreciated, but it is a single locomotion task. More challenging conditions (e.g., external perturbations, uneven ground, changing goals) would make the evaluation stronger.

Ranker requires manual task-specific design
Safety constraints and preference scores are hand-crafted per domain. This reduces automation and may require significant tuning when changing tasks.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
