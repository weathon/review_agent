# Stealthy World Model Manipulation via Data Poisoning

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Model-based learning agents that use a world model to predict and plan have shown impressive success in solving diverse, complex tasks and adapting to new environments. However, the process of exploring open environments and updating the model with collected experience also exposes them to adversarial manipulation. In this paper, we propose SWAAP, the first scalable and stealthy data poisoning method for world models, designed to benchmark their adversarial robustness. SWAAP uses a novel two-stage approach. In the first stage, the attacker identifies a target world model that deviates only slightly from the true environment but significantly degrades agent's performance when used for planning. This is achieved via a first-order bilevel optimization and a new transition gradient theorem. In the second stage, the attacker then performs the actual attack by perturbing a small subset of fine-tuning data to steer the fine-tuned world model toward the target model. Evaluations using diverse tasks show that our approach induces a substantial performance drop and remains effective even under robust training and detection, underscoring the urgent need for stronger protection in world modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SWAAP, the first scalable, stealthy data poisoning attack targeting world models in model-based reinforcement learning (MBRL) agents. World models (e.g., DreamerV3, TD-MPC2, DINO-WM) enable agents to simulate environments for planning and adaptation in complex domains like robotics and autonomous driving. However, their reliance on fine-tuning with collected trajectories exposes them to adversarial manipulation. SWAAP degrades agent performance by subtly altering fine-tuning data, while keeping the poisoned model close to clean dynamics to evade detection. Evaluations across DMControl, MyoSuite, and MetaWorld show substantial return drops (e.g., up to 50% in some tasks) with just 10% poisoned data, persisting under defenses like TRIM and detection methods.

### Strengths
- SWAAP's decomposition of the bilevel poisoning problem into target model identification and gradient-matching data poisoning is elegant and practical. It avoids intractable direct differentiation during training, making it scalable to complex, high-dimensional environments—unlike prior RL attacks, which are limited to simple or policy-focused scenarios.

- Addresses emerging risks in high-stakes domains (e.g., robotics via Nvidia 2025b), urging certified defenses for world models. By focusing on fine-tuning—a common practice in continual learning—it exposes a "fundamental vulnerability" with actionable insights.

- The paper is well-written and easy to follow.

### Weaknesses
- SWAAP assumes that the attacker has full knowledge of the world model architecture (white-box). However, in real-world deployments (e.g., foundational world models like OpenAI 2025 or DeepMind 2024), model details may be black-box or partially black-box. This limits the practical feasibility of the attack, and the paper does not discuss gray-box or black-box variants.

- The first stage of the bi-level optimization employs first-order dynamic barrier gradient descent (Eq. 4), which avoids computing the Hessian but requires $N=16$ mini-updates and extensive buffer sampling (Dall). This makes the approach computationally intensive in high-dimensional continuous environments. The paper does not quantify actual runtime, raising concerns about its applicability to real-time attack scenarios.

- The theorem relies on the gradient of the log-likelihood of continuous transitions, but this assumption may break down under discrete actions or noisy transitions. Furthermore, the paper does not provide guarantees for generalization in non-stationary MDPs with evolving dynamics.

- The approach relies on the dynamic barrier method (Liu et al., 2022), but Fig. 7 only illustrates convergence without proving global optimality or providing bounds under the non-convex objective $J(P_\psi, \pi_\theta)$. Moreover, the constraint $q(\psi, \theta) \leq 0$ may lead to suboptimal target models.

- The experiments are limited to small datasets, where the method shows strong performance under low rp. However, in large-scale fine-tuning scenarios (e.g., millions of trajectories), a higher rp may be required, making the approach easier to detect. The paper does not explore the trade-offs when rp exceeds 20%. 

- The method aligns $GrealG_{\text{real}}$​ and $GtargetG_{\text{target}}$​ using cosine dissimilarity, but in high-dimensional embeddings this can lead to gradient explosion or vanishing. The paper does not discuss the sensitivity of the balancing parameter $\alpha$. Thus, how about the impact of $\alpha$?

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents the first, to my knowledge, study of poisoning attacks against model based reinforcement learning (MBRL). The attacker's objective in the authors' formulation is to minimize the agent's test time return while producing a world model that is less detectable, according to some reasonable metrics like residuals. 

The attack has two phases. In the first phase the attacker tries to find a malicious world model $\hat{\psi}$ such that maximizing return under the dynamics of $\hat{\psi}$ also minimizes return in the true task. To achieve this the authors perform some fairly intricate derivations make their bilevel optimization objective more computationally viable. In the second phase the attacker poisons the fine-tuning dataset of the agent's world model such that they learn dynamics close to those of $\hat{\psi}$. In both phases regularization terms are used to minimize the distance between $\hat{\psi}$ and the true dynamics along with reducing the number of poisoned samples.

The attack is evaluated over 9 continuous control tasks from 3 libraries where it achieves respectable results. The detectability of the attack by defenses is also evaluated, showing that some naive defense techniques may not work.

### Strengths
* As said in the summary, this is the first paper I'm aware of that studies poisoning attacks in MBRL.

* The proposed method seems to come from reasonable foundational principals -- starting from a reasonable but computationally complex objective and then iteratively approximating or reducing the problem to be more feasible.

* The authors make decent attempts to study and consider defense techniques against their method which is always appreciated.

* The method is designed in a way to not fundamentally violate the dynamics of the true environment, making the attack objective harder but more realistic.

* Given this is the first paper studying this problem, the author's chosen baselines are reasonable and their results are respectable.

### Weaknesses
### Attack Objective

In general I find the focus of RL poisoning attack papers on universal availability attacks to be a bit questionable. Here the objective of the attack is to simply minimize the agent's return across all states (subject to some reasonable constraints). As previously stated, this means SWAAP is an availability attack (minimizing utility) and is also universal (applies to all inputs). 

The problem with this objective formulation, in my opinion, is that it has a fundamental detectability problem that this paper doesn't really address directly. Let's imagine the "perfect attack" in this setting: the residuals of the poisoned data are indistinguishable from those of the clean data and yet the agent's return is completely minimized, receiving the lowest possible return. In this case the practitioners designing and training the system will for sure not deploy the agent and will try to investigate the issue. In my opinion the same holds for results like those presented in this paper. A 32% performance decrease, on pen-twirl-hand for instance, will certainly be visibly noticable by practitioners as a failed training run. Real deployed systems, e.g. robotic agents, undergo rigorous testing before they're deployed to ensure safety and performance stability. Therefore an objective of universally minimizing the agent's return seems unreasonable to me. 

The objective also doesn't really tell me anything new or unique about MBRL. Universal availability attacks have been known and understood in ML in general for a long time, so it is unsurprising that they also work here.

That being said, this universal objective is (unfortnuately) common within the poisoning attack space in RL so I cannot peanilize the authors too heavily. Though I would have much prefered seeing a more targeted objective. 

### Minor Weaknesses

* Figure 1 is a bit hard to follow as currently presented. It would be better if "stage 1" appeared at the top left of the figure instead of towards the middle. Currently it takes some effort to figure out the exact flow of the diagram or where to start in my opinion. 

* The threat model is still white box and therefore fairly unrealistic. However, since this is the first paper in this area, the threat model is excusable.

### Questions
* What is the authors' motivation or rationale for chosing an adversarial objective of minimizing return?

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
3

### Summary
This paper explores poisoning agent's world models in model-based reinforcement learning. First, the attacker identifies the world model that would lead to the worst-case performance of a learning agent while remaining close to the original environment. In the second stage, the attacker perturbs the model so that it matches the target world model from the first stage. This attack was tested on three common RL benchmarks.

### Strengths
1. The paper addresses a relevant problem, by extending existing poisoning attacks against RL-agents to the poisoning of world models.
2.  The assumptions of the attack are clearly described and generally realistic. The method itself is formally well-defined.
3. The inclusion of defense mechanisms is a useful addition.

### Weaknesses
1. The assumed poisoning rate of 10% is overly high. This is especially problematic, since there appear to be no ablation studies varying the amount of poisoned samples.
2. The results of the proposed SWAAP method show high variance compared to the other baselines, particularly in the clean setting. This makes it difficult to assess the quality of the poisoning attack.
3. It would have been helpful to have more environments, especially since the Myosuite results are extremely volatile.
4. Given the two stage nature of the algorithm, it would be nice to see how well the individual stages work, i.e. is the adversary actually able to discover a good worst-case model, and is the adversary able to poison the model such that it becomes close to the target model?

### Questions
1. How does the attack perform at various levels of poisoning?
2. See W4

### Soundness
2

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
This paper proposes SWAAP, a novel data poisoning attack against the world model. SWAAP is a two-phase attack strategy, comprising the identification of a worst-case target world model via bi-level optimization, and the data poisoning process that guides the benign world model toward the target world model through gradient matching. Experiments show SWAAP can effectively degrade the TD-MPC2 agent's performance across various environments and tasks.

### Strengths
1. This paper is well written and easy to follow.
2. The theoretical part is solid. Although both the bi-level optimization algorithm (Liu et al., 2022) and the gradient matching algorithm (Geiping et al., 2021) are not novel, the joint application of these two algorithms for data poisoning is noteworthy.
3. Experiments show SWAAP is effective for degrading the RL agent's performance via only poisoning a small subset of the fine-tuning dataset.
4. Various defenses are discussed, and SWAAP can still work even under defense.

### Weaknesses
1. The backbone world model algorithm is limited. Comparing different backbone algorithms (e.g., DreamerV3 and DINO-WM, as mentioned by the authors) would benefit this paper.
2. The results in Table 1 and Figure 2 are not consistent. In Table 1, SWAAP obtains 776 in DMControl humanoid-walk, while in Figure 2, SWAAP obtains a score within [400-600] under $\alpha=0.9$. The authors should clearly explain this difference to ensure the reliability of the results.
3. There is no discussion on the computation resources and costs. Since SWAAP is a two-stage complex optimization algorithm, it is beneficial to illustrate the efforts required by the attacker to launch a successful attack, such as the time.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
