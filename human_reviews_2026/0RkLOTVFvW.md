# Reparameterization Proximal Policy Optimization

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Reparameterization policy gradient (RPG) is promising for improving sample efficiency by leveraging differentiable dynamics. However, a critical barrier is its training instability, where high-variance gradients can destabilize the learning process. To address this, we draw inspiration from Proximal Policy Optimization (PPO), which uses a surrogate objective to enable stable sample reuse in the model-free setting. We first establish a connection between this surrogate objective and RPG, which has been largely unexplored and is non-trivial. Then, we bridge this gap by demonstrating that the reparameterization gradient of a PPO-like surrogate objective can be computed efficiently using backpropagation through time. Based on this key insight, we propose Reparameterization Proximal Policy Optimization (RPO), a stable and sample-efficient RPG-based method. RPO enables stable sample reuse over multiple epochs by employing a policy gradient clipping strategy tailored for RPG. It is further stabilized by Kullback-Leibler (KL) divergence regularization and remains fully compatible with existing variance reduction methods. We evaluate RPO on a suite of challenging locomotion and manipulation tasks, where experiments demonstrate that our method achieves superior sample efficiency and strong performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a PPO-resembling mechanism enabling reusing collected rollouts for multiple policy update steps in reparameterization policy gradient methods (policy gradients estimated in a model-based fashion, e.g., using a differentiable simulator).

### Strengths
- The problem being solved is meaningful and practically useful, i.e., sample reusing for RPG methods. 
- Empirically, the proposed approach is among the leading methods across all the evaluated tasks.

### Weaknesses
- The proposed approach seems to be a bit incremental—mostly a direct map from PPO to the RPG world. Although caching action gradients for restoring policy gradients for the updated policy does look like a smart engineering workaround. 
- Line 243 “We take the gradient of the discounted cumulative rewards with respect to each sampled action via BPTT.” - This assumes a known reward function (and smoothness as well as differentiability w.r.t. states and actions)
- Presentation can be organized in a better way. Right now, section 4.1.2 is describing the main idea while referencing a bunch of other sections and jumping back and forth; then, section 4.2.1 is repeating things already described above. I feel like readers can benefit from a more focused presentation. 
- “RPO significantly improves sample efficiency over baselines.” — This sounds a bit like overclaiming, as the proposed approach is on par with strong baselines on a good number of tasks
- The ablation seems insufficient; also, there does not seem to be a big difference in all three subplots in Figure 4. I would suggest a more detailed ablation study to better understand the necessity and benefit of each component.

### Questions
- Line 242: How do the authors compute the infinite sum given collected finite-length rollouts?
- Algorithm 1: If all the collected rollouts are short-horizon, how does the learned value function know about longer-term returns?

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
This paper identifies that Reparameterization Policy Gradient (RPG) methods, though with a low-variance, suffer from the instability when long-horizon back-propagation or sample reuse is attempted. Inspired from Proximal Policy Optimization (PPO), this paper shows that the reparameterization gradient of a PPO-like surrogate objective can be computed efficiently using Back-Propagation Through Time (BPTT). Leveraging this connection, the authors propose Reparameterization Proximal Policy Optimization (RPO), which re-uses cached action-gradients for multiple epochs while keeping updates conservative via (i) asymmetric clipping of importance weights tailored to RPG and (ii) an explicit KL-penalty that activates only during off-policy epochs.

### Strengths
1) The paper is well-written and the ideas are presented clearly, making it accessible and easy to follow.

2) The experimental part provides sufficient verification on multiple tasks, and well-chosen baseline methods, clearly demonstrating the advantages of RPO in terms of the sample efficiency and training stability.

3) Ablation experiment is well-organized, which separately describes the contributions of three algorithm improvements: KL regularization, sample reuse and clipping mechanism.

### Weaknesses
1) Introducing the clip and KL divergence penalty operations from PPO into model-based methods is a relatively intuitive idea, and it is not novel in terms of the algorithm improvement. If a theoretical analysis can be provided to prove that the improved training algorithm can lead to a better final performance and more stable training, the contributions of this paper will be enriched.

2) During the design of the PPO algorithm, it was observed that there is a certain degree of redundancy between the KL regularization and the clipping mechanism, and only one of them was used. In the ablation study of this paper, the results of “no KL loss” and “no clipping” were very similar. This raises the suspicion that in model-based methods, the redundancy seems to still hold, and there is no need to jointly use them both.

### Questions
1) A theoretical analysis can be provided to prove that the improved training algorithm can lead to a better final performance and more stable training.

2) It is hoped to see an additional section, to discuss (from a theoretical/experimental perspective) that whether the redundancy still exists between KL and clipping in model-based methods.

3) Among the experimental results of all tasks, PPO showed a very obvious, even counterintuitive performance disadvantage. Can the authors analyze the reasons behind this phenomenon? Is it caused by the task selection?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Reparameterization Proximal Policy Optimization (RPO), which integrates the PPO surrogate objective into the reparameterization policy gradient (RPG) framework.
By caching action gradients from a single BPTT pass and reusing them through importance-weighted clipping and KL regularization, RPO enables stable and sample-efficient updates in differentiable simulators.
Experiments on DFlex and Rewarped show improved stability and efficiency compared with SHAC and SAPO.

### Strengths
Generally, I like the idea of the paper. Using the PPO's idea of surrogate function to update the computation graph and reuse the previous samples is smart.

In addition, the paper explores an important problem — stabilizing reparameterization-based RL in differentiable simulators — and provides an algorithm that is simple, general, and compatible with existing frameworks.
The empirical results on DFlex and Rewarped demonstrate that the method can improve stability and efficiency across multiple continuous-control tasks.

### Weaknesses
Though the idea of the paper looks good, its writing is not very good to me.

### The clearness of the presentation:
I found the paper hard to follow and understand. The derivation of the formulas is not very clear and solid, which need largely polish.
1. Eq. (8) and the following text are confusing to me. They use $A(s,a)$ in Eq. (8), and then in line 283 they use $\nabla R(\tau)$. It would greatly increase readability if Eq. (8) could be written in the form of $R(\tau)$.
2. In Eq. (3), the expectation is over the whole trajectory ($s_t,\epsilon_t$), while in Eq. (8) it would be clearer to express it in trajectory form as well. Otherwise it is unclear to the readers. 
3. Eq. (10) should give a reference to Proposition 1. Otherwise readers cannot understand why it takes this form.

### The experiments are not very sufficient:
1. Regarding fairness: it seems that the method updates the policy for additional epochs (while using the same samples). It is unclear what happens if the compared methods are also trained with more epochs. I do not necessarily expect the proposed method to achieve higher scores, but it may save computation time. The paper should include such results to demonstrate the practical benefit.
2. The ablation study on the clipping hyperparameters $c_{low}, c_{high}$ is missing. 
3. From the experiments, KL and clipping appear to have less effect, while the number of update epochs has more effect. This may reduce the necessity of the proposed method.

### Some minor problems:
1. The method relies on differentiable transitions; this should be stated clearly in the Preliminaries. I am also uncertain whether the method requires the transition to be deterministic.
2. The y label in Figure 1 should be cleared shown.

### Questions
1. the clipping hyperparameters $c_{low}, c_{high}$. I check Table 4, they use very large clipping range (0.8), while in PPO they usually use 0.2. Can you tell the reasons? How to choose them in practice?

2. Figure 1 only shows the result of one seed. Is that also common in other seeds?

3. Also see my concerns in Weakness.

### Soundness
4

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper enhances the Reparameterization Policy Gradient (RPG) method through two key modifications. The method improves sample efficiency by substituting the policy gradient objective with that of PPO, and it ensures training stability by employing the reparameterization trick, which facilitates gradient computation via standard Backpropagation Through Time (BPTT). Experiments validate that the proposed method outperforms existing RPG baselines.

### Strengths
1. replacing the policy gradient-based objective of RPG with an PPO objective is interesting in terms of sample reuse.
2. using the old backpropagation through time (BPTT) trick to train the RPG model has not been done before.

### Weaknesses
1.The primary contribution of this work lies in the reuse of trajectory samples. Given this, a comparison with off-policy evaluation and multi-step Q-learning methods is warranted, as they similarly grapple with the risk of high variance or inaccurate estimation caused by products of importance ratios. Although the authors employ a clipping trick to mitigate numerical instability, this approach comes at the cost of sample effectiveness, potentially preventing a significant portion of trajectories from being reused.

2. The paper introduces a design choice in Eq. (9) to explicitly align the new policy with the old trajectory. We argue that the method should also be feasible without this alignment. Specifically, one could directly assume the actions from the old trajectory are executed by the new policy—given the identical action space—and then calculate the importance sampling ratio conventionally, similar to PPO. The authors should arguably include this approach as a necessary baseline and present experimental comparisons to justify the superiority of their proposed design.

3. The experimental validation presented is incomplete. The authors have not adequately discussed or analyzed the method's key hyperparameters, such as by providing a sensitivity analysis on how different parameter values affect final performance. This omission makes it difficult for readers to assess the method's stability and its generalization capability under different settings.

### Questions
see above section.

### Soundness
2

### Presentation
3

### Contribution
2
