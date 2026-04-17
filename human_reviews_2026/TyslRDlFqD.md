# Balancing Safety and Return: Region-based Reward Penalty over Action Chunks For Offline Safe RL

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
In-sample learning has emerged as a powerful paradigm that mitigates the Out-of-Distribution (OOD) issue, which leads to violations of safety constraints in offline safe reinforcement learning (OSRL). Existing approaches separately train reward and cost value functions, yielding \textit{suboptimal} policies within the safe policy space. To address this, we propose the \textit{Region-Based Reward Penalty over Action Chunks} (R2PAC), a novel method that trains $h$-step optimal value function within the safe policy space. By penalizing reward signals over action chunks that may potentially lead to unsafe transitions, our method: (1) integrates cost constraints into reward learning for constrained return maximization; (2) improves joint training stability by accelerating the convergence speed with unbiased multi-step value estimation; (3) effectively avoid unsafe states through temporally consistent behaviors. Extensive experiments on the DSRL benchmark demonstrate that our method outperforms state-of-the-art algorithms, achieving the highest returns in 13 out of 17 tasks while maintaining the normalized cost below a strict threshold in all tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes R2PAC, an offline safe reinforcement learning method based on reward penalty and action chunking. R2PAC updates the reward–action value function using a reward penalty, introduces action chunks to improve the accuracy of value estimation, and finally performs policy updates through conditioned flow matching. The method achieves strong reward performance on the DSRL benchmark.

### Strengths
- The paper is clear and well-written.
- Applying action chunking in offline safe RL to improve value estimation accuracy and enable implicit prediction of future outcomes is a reasonable and effective approach.
- The paper provides a comprehensive set of ablation studies.

### Weaknesses
- The proposed approach appears to be a combination of several existing methods, including the reward penalty from Safe MBPO, action chunking, the expectile value learning used in IQL and FISOR, and FISOR’s safe/unsafe region-specific policy update objectives together with its generative model–based policy extraction. As a result, the novelty of the contribution is somewhat limited, and the work feels more incremental in nature. That said, the application of action chunking in this context is, in my view, a notable and valuable contribution.
- There is some ambiguity between the hard-constraint and soft-constraint settings. The overall framework seems intended to address hard constraints, yet the reward penalty introduces a safety threshold hyperparameter $\mathcal{l}$ that is determined by the quantile of dataset values, rather than being a fixed threshold manually specified. This design choice appears inconsistent with the motivation.
    - If the goal is to handle hard constraints, $\mathcal{l}$ should ideally be set to zero.
    - If the setting is not hard-constrained, then $\mathcal{l}$ should instead correspond to the externally defined safety threshold κ.
    
    Although introducing a tunable hyperparameter may improve empirical performance, it reduces the conceptual clarity and persuasiveness of the method’s formulation.
    
- During policy extraction, the method employs conditioned flow matching. It is not entirely clear why flow matching was chosen over diffusion-based approaches. Furthermore, the paper replaces FISOR’s AWR with a conditioning mechanism whose design seems somewhat strange: it only distinguishes whether the reward and cost advantages in safe and unsafe regions are ≥ 0 or ≤ 0. However, if multiple actions satisfy these conditions, how are better actions selected among them? Is this achieved solely through rejection sampling based on cost values?
- I have some concerns about the evaluation criteria. If the paper is positioned as addressing a hard constraint setting, then safety satisfaction should be considered more important than reward maximization—that is, one should minimize cost first and then maximize reward under that constraint. However, the paper seems to focus primarily on reward maximization when cost < κ. Based on the results shown in the appendix, if the objective were to minimize cost, the flow AWR variant would actually perform better than the proposed conditioned version.
- The authors should include more experiments on Safety Gymnasium Navigation tasks (e.g., PointXXX environments) to provide a more comprehensive evaluation of the proposed method’s generality and robustness.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes R2PAC (Region-Based Reward Penalty over Action Chunks), an offline safe RL method that integrates cost constraints directly into reward learning to improve safety and stability. By penalizing rewards over action chunks that may lead to unsafe transitions, R2PAC enables multi-step value learning within the safe policy space. Experiments on the DSRL benchmark show that it achieves higher returns and lower costs than state-of-the-art methods.

### Strengths
The authors provide theoretical analysis showing that the proposed reward penalty framework preserves policy optimality and ensures convergence guarantees.

### Weaknesses
The identification of safe and unsafe sets relies on the learned value functions (Eq.(17)), which may be inaccurate during training. As a result, the training process appears unstable, with both reward and cost exhibiting noticeable oscillations in the reported figures (Figures 9, 10, 11).

### Questions
1. What are the specific assumptions required for the theoretical propositions (e.g., Propositions 1 and 2) to hold? Do they rely on perfect estimation of $V_c$ and $V_r$, accurate separation of safe and unsafe regions, or bounded approximation errors?

2. In Eq. (4), could the authors clarify why the $Q$-function takes only $s_t$ and the action chunk as inputs? Since $V(s_{t+h})$ depends on the future state $s_{t+h}$, if $Q$ were instead conditioned on both $s_t$ and $s_{t+h}$, how would the corresponding value function be defined? Does the proposed formulation imply that the value function also depends on both states?

3. In the ablation study, the authors mention that “satisfactory evaluation results can still be achieved using model checkpoints” (lines 418–419). Could the authors elaborate on what this statement means in practice? Specifically, are the reported results based on the best-performing checkpoint during training, and how is fairness ensured in such comparisons? For the main results, are the evaluations conducted using the best checkpoint or the final model parameters?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
R2PAC addresses a key limitation in offline safe RL: separately trained reward and cost value functions produce conflicting policies. The method trains a single h-step value function within the safe policy space by penalizing rewards for unsafe action chunks. Key contributions: (1) region-based reward penalty integrating cost constraints, (2) action chunking for stability and temporal consistency, (3) flow-matching policy extraction. Results: highest returns in 13/17 DSRL tasks while maintaining safety across all tasks.

### Strengths
- Figure 1 effectively illustrates how decoupled value functions lead to suboptimal trajectories. The problem is real and well-articulated.

- First to integrate action chunking into in-sample OSRL. Region-based penalty (Eq. 7, 12-13) elegantly incorporates safety into reward learning. Proposition 1 provides theoretical guarantees.

- Normalized cost < 1.0 in all 17 tasks, highest rewards in 13 tasks. Substantially outperforms FISOR and other baselines.

### Weaknesses
- Safety threshold ℓ (α-quantile) varies from 0.5-0.99 across tasks (Table 4)
- Proposition 1 assumes binary costs but experiments use continuous costs
- Only two C values (1.0, 10.0) used; claims "insensitivity" without evidence
- Flow-matching with 32 steps + 16 candidates likely expensive

### Questions
1. How sensitive is performance to h > 15? What's the relationship to task horizon?

2. Computational cost vs baselines (time, memory)?

3. Why does C differ by 10× between task families? How to select for new tasks?

4. Figure 3 shows V assigns similar values to unsafe states—doesn't this contradict sufficiency claims and necessitate Eq. 17?

5. How do V_c estimation errors affect safe region identification and final safety?

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
The paper tackles offline safe reinforcement learning’s tendency to violate constraints and the mismatch that arises when reward and cost critics are trained separately, and proposes R2PAC, which embeds safety directly into reward learning via a region-based reward penalty and uses in-sample (IQL-style) learning.  It converts any transition that would enter an unsafe region into an absorbing state with reward −C. To stabilize and speed up training, R2PAC introduces action chunking, h-step critics that reduce bootstrap-error propagation and improve numerical stability compared to 1-step IQL. On the DSRL benchmark, the method reports highest returns on 13 of 17 tasks while keeping costs below strict thresholds on all tasks.

### Strengths
- The method demonstrates strong empirical performance, reportedly achieving the highest returns in 13 out of 17 DSRL benchmark tasks while successfully satisfying safety constraints in all of them.
- It identifies and addresses the OSRL problem of the separate training of reward and cost value functions, which can lead to conflicting policy recommendations and suboptimal or unsafe behavior.
- The integration of action chunking and flow-matching enhances the method's performance.

### Weaknesses
* The paper's central claim of using a "single value function" is contradictory. It criticizes separating reward and cost functions but then explicitly uses a decoupled objective that relies on both its new penalized reward value,  $\bar{V_{r}}$, and the standard cost value, $V_{c}$ for policy extraction and recovery from unsafe states.

* The R2PAC method relies on per-task tuning of the safety threshold $l$. While acknowledged as a limitation, it makes the achieved performance gains suspect to the tuning. Fisor, for instance, doesn’t rely on such an extensive tuning. 

* The paper's reliance on setting the critical safety threshold $l$ as an arbitrary $\alpha$-quantile of the cost values is flawed because this threshold is not shown to have any relationship with the task's true, fixed cost limit (e.g., 5 or 10). If the goal were purely to find the maximally safe policy, the threshold should ideally be set toward zero rather than tuned based on a data percentile.

* The paper's experimental evaluation is lacking as it omits comparisons to many relevant and recent methods.  Please check missing  related work below. A relevant baseline is CAPS. The Figure 1 motivates R2PAC approach by showing a failure mode where myopically selecting the highest-reward, non-unsafe action leads the agent toward a suboptimal, complicated trajectory. CAPS seems to do that by selecting the reward-maximizing action from that safe set at each step. 


* The paper's novelty appears incremental. Its core logic (maximize reward in safe states, minimize cost in unsafe states) and components like rejection sampling are conceptually very similar to FISOR, suggesting a "reworked FISOR" that swaps components.

* The claim of "simplicity" for policy extraction is contradicted by the use of a complex flow-matching model (CFG).

* The justification for action chunking is weak. The plots show it changes $Q$-value scales and convergence speed, but this does not prove it produces more *accurate* value estimates or is essential for safety, especially since FISOR performs well without it (in terms of safety).

* The paper appears to copy baseline results directly from the FISOR paper rather than re-running them.

* The abstract's claim that the method is a "drop-in replacement within existing offline RL pipelines" is misleading. The paper focuses almost exclusively on the in-sample learning paradigm and only demonstrates its method as a deep modification of the IQL algorithm. The method is highly integrated with one specific algorithm (IQL) and not shown to be easily applicable to other offline RL methods.

- “we employ rejection sampling to select the action with lowest $Q_c$,” but that tells you if the action is safe by following the safest policy afterwards. It is not the $Q_c$  of the current policy.
- While the authors used the results from the FISOR paper for baselines, they omitted results for the MetaDrive benchmark.
- The literature review also appears incomplete, omitting several relevant OSRL publications, including:
  - CAPS: Constraint-Adaptive Policy Switching [https://www.arxiv.org/pdf/2412.18946]
  - OASIS [https://arxiv.org/pdf/2407.14653]
  - Latent Safety-Constrained Policies [https://arxiv.org/pdf/2412.08794]
  - Trajectory Classification for Safe RL [https://arxiv.org/pdf/2412.15429?]
  -  Constraint-conditioned actor-critic for OSRL [https://openreview.net/pdf?id=nrRkAAAufl]
  - A Similar relabeling idea for the online case. Safe Exploration in Reinforcement Learning.  [https://arxiv.org/pdf/2310.03225]

### Questions
- Please check weaknesses.
- "although satisfactory evaluation results can still be achieved using model checkpoints." what does this mean? 
- Figure 4 : Why do the "Cost Training Curves" not align with the "Final Cost Scores" bar chart for BallRun? For instance, the $\alpha=0.4$ curve shows costs near 0, but the bar reports a normalized cost of 0.22. Conversely, the $\alpha=0.6$ curve oscillates up to 5, but the bar reports a final normalized cost of only 0.1.
- Figure 10: Is the cost axis in the training curves in Figure 10 (e.g., CarGoal1) showing absolute cost or normalized cost? If the curve shows values between 0.0 and 1.0, how does this reconcile with the normalized cost of 0.57 reported in Table 1, which implies an absolute cost of 5.7?

### Soundness
2

### Presentation
2

### Contribution
2
