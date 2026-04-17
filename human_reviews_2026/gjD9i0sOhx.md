# MAS$^3$AC: A Learning Framework for General Multi-Agent Safe and Stable Control with State-Wise Guarantees

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Ensuring both safety and stability is essential when deploying reinforcement learning to control safety-critical systems, including multi-agent ones. However, many existing safe multi-agent reinforcement learning (MARL) studies focus only on cooperative tasks and adopt the constrained Markov decision process setting that only enforces expectation-based constraints, limiting their applicability to domains requiring strict state-wise guarantees, while stability remains largely underexplored. To address these challenges, we propose Multi-Agent Safe and Stable Soft Actor-Critic (MAS$^3$AC), a model-free framework that incorporates state-wise safety and stability constraints into MARL for general multi-agent tasks where each agent has its own objective. Our approach uses neural barrier functions to enforce safety, supported by a theoretical analysis of its convergence to a feasible local Nash equilibrium. It then uses the concept of input-to-state stability to guarantee stability for the multi-agent system, together with an analysis of the issue of infeasibility arising from conflicting state-wise safety and stability requirements. Empirically, we introduce a suite, spanning both cooperative tasks with global information and non-cooperative tasks with local observation, for benchmarking safe and stable MARL algorithms. Experimental results show that MAS$^3$AC consistently achieves a favorable balance between reward maximization and constraint satisfaction, delivering competitive or superior rewards while maintaining fewer safety violations compared to baselines on benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers the problem of ensuring both state-wise safety and stability in multi-agent reinforcement learning. The paper proposes the multi-agent safe and stable soft actor-critic (MAS$^3$AC) method, which uses neural barrier functions for safety, and the concept of input-to-state stability to guarantee stability. Theoretical analyses are provided to show the convergence and feasibility of the proposed method. Empirical results on the safe multi-agent MuJoCo and an electrical bidding task show that MAS$^3$AC achieves a better balance between reward maximization and constraint satisfaction than baselines.

### Strengths
1. The paper considers an important problem. 

2. The theoretical analysis is adequate. 

3. The overall writing of the paper is good. 

4. Both fully observable and partially observable environments are considered.

### Weaknesses
1. The paper includes many important results in the appendix (like Figures 8 and 11), and the main pages repeatedly refer to the appendix, making the paper difficult to read. Generally, the main pages should be **self-contained**, with only additional details provided in the appendix.

1. The baselines for safe MARL are not appropriate enough. All the safe MARL baselines in the paper consider the **CMDP** setting, while the paper considers **state-wise** safety. It would be better if other works addressing state-wise safety are considered, e.g., the works discussed in the related work section. 

1. Some of the paper's claims are not fully justified. Please see the Questions section (Q1 - Q4). 

1. Some important ablation studies are missing. Please see the Questions section (Q8 - Q10).

### Questions
1. The paper claims that the stability is **guaranteed** using the proposed method. How can this be true with learned value functions?

1. Can you prove that the safety is guaranteed with Definition 3?

1. The stability is defined as convergence in **states** in the main pages. How is this realized in the experiments? It seems that the experiments define cost functions as $|d_t - d^*|$. 

1. Can you show some empirical analysis about the stability of the learned policies? For example, plot $\\|x_t - x_\mathrm{desired}\\|$.

1. Does the stochasticity in the problem formulation only come from the policy? This needs to be clearly stated. 

1. What kind of convergence do you consider with a stochastic ${x}_t$?

1. Some previous works are criticised for "focusing on tasks with a joint team reward/cost". Why is this a problem, as it seems easy to adapt these works to the proposed framework (for example, MAPPO has already been adapted by the authors)? 

1. In the Hamilton-Jacobi reachability community, it is popular to use a max-over-time safety signal as a neural barrier function. In contrast, this paper proposes to use a sum-over-time barrier signal as the neural barrier function. What is the motivation for doing this, and does this achieve better results? If so, it would be great to see some empirical results. 

1. The efficacy of the ISS stability part is unclear. Can you consider completely removing this part and comparing the results? Also, maybe consider completely removing the reward signal and only using the ISS stability term to guide the training. 

1. The paper considers an extra squared term to more strongly penalize safety violations. This needs an ablation study to demonstrate the improvement. 

1. It seems that the learning rates are missing in Table 1. Can you provide this important hyperparameter? I am also curious about the initial and end values of the Lagrange multipliers (curves can be better).

1. Why can MAS$^3$AC achieve higher results than algorithms that do not consider safety?

### Soundness
1

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
4

### Summary
This paper presents MAS3AC, a multi-agent learning framework that unifies state-wise safety and system stability on top of SAC. The method employs learned barrier certificates to transform step-wise violation risks into inequality penalties, and enforces stability via ISS/Lyapunov inequalities, both optimized jointly with returns using a Lagrangian with squared penalties and multi-time-scale updates under decentralized, low-communication settings. Theoretically, under stated assumptions, the algorithm converges to a feasible local Nash equilibrium and includes an analysis to detect and circumvent infeasible regions when safety and stability conflict. On Safe-MuJoCo and other cooperative/decentralized tasks, MAS3AC achieves a superior reward–safety trade-off over strong baselines, with ablations (e.g., neighborhood radius) corroborating its low-communication design. By making “state-wise safety + stability” trainable within a MARL backbone, MAS3AC offers a practical route to safe and stable control for resource-constrained multi-agent systems.

### Strengths
1. This paper proposes a decentralized and implementable MARL framework that unifies state-wise safety and ISS-based stability within a single learning paradigm.
2. It proves convergence to a feasible local Nash equilibrium under given assumptions and provides an infeasibility analysis when safety and stability objectives conflict.
3. The methodology section is written in a clear and well-structured manner, making the technical ideas accessible and coherent.

### Weaknesses
1. Related Work Section: The section only touches on domain-specific challenges at a high level and lacks clear, structured articulation. Please add a dedicated “Challenges” subsection that explicitly enumerates the key difficulties this paper addresses.

2. Method Section: In the multi–time-scale training of neural barrier certificates, the learning-rate decay coefficients for the Critic/Barrier, Policy, and Lagrangian multipliers (e.g., $η_1(e)>η_2,_i(e)>η_3,_i(e)$) are presented without a clear rationale. Please explain why this ordering is preferable and provide justification for these choices to enhance the method’s interpretability.

3. Experiments Section: The comparison baselines are missing recent decentralized constrained MARL methods, such as GCBF-PPO [1], which makes it difficult to fully demonstrate the advantage of MAS³AC over methods in the same category. It is recommended to include such baselines so that the claimed stability guarantees become more convincing. In addition, the only non-cooperative task evaluated is Coupled HalfCheetah. Please clarify whether this is sufficient to justify the “general” claim in the title, or provide additional non-cooperative benchmarks.

**Reference**

[1] Songyuan Zhang, Oswin So, Mitchell Black, and Chuchu Fan. Discrete GCBF proximal policy optimization for multi-agent safe optimal control. In The Thirteenth International Conference on Learning Representations, 2025

### Questions
Please refer to the “Weakness” section for related questions.

### Soundness
2

### Presentation
2

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
Proposes MAS³AC, a model‑free MARL framework that jointly enforces state‑wise safety via a learned neural barrier certificate and stability via input‑to‑state stability (ISS) constraints, trained with a multi‑timescale actor–critic + Lagrangian scheme. Convergence to a feasible local Nash equilibrium is proved under assumptions. Experiments on cooperative Safe‑MuJoCo variants and non‑cooperative, decentralized tasks (incl. an electrical bidding market) show higher reward with fewer safety violations than baselines.

### Strengths
1. Unified treatment of state‑wise safety + stability in general‑sum MARL (not only cooperative), with clear formalization of barrier certificates (Def. 3) and ISS conditions.
2. Practical training objective: penalizes differences of consecutive barrier values (rather than CMDP expected costs) with an extra squared penalty for violations (Eq. (3)), which the authors argue improves learning stability.
3. Theoretical story: multi‑timescale updates to convergence to feasible local NE (Thm. 1), monotone non‑shrinking safe set (Prop. 1), and a global barrier via the minimum over agents.
4. Empirical breadth & clarity: consistent “high reward + low violations” across tasks; the electrical bidding benchmark broadens beyond robotics.

### Weaknesses
1. Safety is only guaranteed after convergence. The paper motivates “zero violations at every step” for real systems, but the formal guarantees apply to the limit policy, not to learning‑time behavior; exploration can still violate constraints (no model/backup/shield). The paper emphasizes the need for per‑step safety, yet Corollary 1 removes disturbances and applies post‑convergence.
2. Assumptions are strong and hard to check. Assumption 4 effectively requires that, from any safe state, there exists a policy for agent i satisfying the barrier inequality against any fixed policies of others, and that all initial states are safe; plus multiple boundedness/Lipschitz assumptions (Assumptions 1–6). Practical verification is unclear.
3. Barrier signal availability. The method relies on a known state‑wise violation signal $b_i$ (set to 0 or a negative constant; Lemma 1) to train the certificate. In many real tasks, safety sets are unknown or non‑Markov, making this labeling non‑trivial.
4. Infeasibility resolution is asymptotic. Proposition 2 assumes positive‑probability satisfaction of stability inside the infeasible set and infinite updates; it does not shield against violations during training.

### Questions
1. How can MAS³AC deliver state‑wise zero‑violation behavior during learning under unknown dynamics, without a model, shield, or backup controller? If this is impossible in general, please (i) clarify the claim/scope to “post‑convergence guarantees,” and (ii) outline a practical safety mechanism (e.g., action filtering, reachability shield, offline‑to‑online warm‑start) that would integrate with MAS³AC.
2. Barrier signal acquisition: When the unsafe set is unknown or partially observed, how do you obtain reliable $b_i$ labels? Can your certificate be trained from implicit signals (e.g., collision proximity, rule sets) or stochastic risk estimators?
3. Sample efficiency & scaling: Multi‑timescale updates can be sample‑hungry; can authors provide the report wall‑clock / environment steps and scaling to larger N (>10) with partial observability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In the paper under review, the authors propose a model-free MARL framework for safe and stable control of general multi-agent tasks. The state-wise safety constraint is handled by neural barrier functions. The stability of the multi-agent system is ensured based on control theory. Both the convergence and feasibility of the proposed framework are established theoretically. Based on the framework, a practical MARL algorithm is proposed, whose effectiveness in dealing with reward maximization and constraint satisfaction is empirically demonstrated through extensive simulation experiments.

### Strengths
(1) The proposed safe MARL framework can deal with state-wise safety constraints, while also addressing the stability issue in multi-agent systems.

(2) Extensive simulation experiments are conducted, and advanced baselines are compared.

### Weaknesses
(1) The problem setting of this work is unclear.

(2) The contributions of this work are unclear.

### Questions
(1) Regarding the first contribution, there exist other neural barrier certificate-based safe RL methods which are also model-free (e.g., [1]). In addition, it seems that your method also relies on the precise labeling of safe and unsafe states. What is the novelty of using neural barrier certificates in this work?

(2) It is mentioned in the related work section that one drawback of existing MARL works is the assumption of the existence of a stationary distribution. Nevertheless, the stochastic approximation theory used in this work for convergence analysis also requires this assumption.

(3) It seems that Definition 1 is only used in Problem 1. What is the relationship between Definition 1 and Definition 3?

(4) Could the authors provide a task example to explain the notations including $\delta_{ij}$, $x_{\text{desire}}$, and $d_t$?

(5) In (2a)–(2b), it seems that all agents need to update their local policies in a certain order, which is similar to HATRPO. Could the authors provide some insights here? In addition, does the feasibility of (2a)–(2b) hold naturally?

(6) Under the setting of coupled dynamics in Section 3, it is apparent that $V^i$ is a function of the joint policy instead of the local policy. Hence, it is not straightforward to obtain the results in Section 4.2, and whether the proposed framework is suitable for the decentralized training setting is questionable.

(7) It is strange that the term $\kappa(\Vert d_t \Vert)$ only appears in (4), which is not considered in the rest of the manuscript because external disturbances are assumed to be negligible.

(8) The simulation result shown in Fig. 1 is not normal. It can be found that HASAC shows similar safety satisfaction ability as safe MARL algorithms like MAPPOL and MAFOCOPS, which do not consider safety in fact. In addition, it is reported in [2] that MAFOCOPS performs well in the ant task. However, it shows the worst learning performance in this figure.

References:

[1] Yang, Y., Jiang, Y., Liu, Y., Chen, J., & Li, S. E. (2023). Model-free safe reinforcement learning through neural barrier certificate. IEEE Robotics and Automation Letters, 8(3), 1295–1302.

[2] Zhao, Y., Yang, Y., Lu, Z., Zhou, W., & Li, H. (2023). Multi-agent first-order constrained optimization in policy space. Advances in Neural Information Processing Systems, 36, 39189–39211.

### Soundness
3

### Presentation
3

### Contribution
2
