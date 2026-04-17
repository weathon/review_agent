# Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 4

## Abstract
Multi-Agent System (MAS) and Reinforcement Learning (RL) are both widely adopted to improve large language model (LLM) agentic performance. MAS strengthens task-specialized performance via role-based orchestration; RL leverages environment rewards to train stronger policies, such as Group Relative Policy Optimization (GRPO)-style optimization. Yet applying on-policy RL training to MAS is underexplored. While promising, it poses several challenges. On the algorithm side, Standard GRPO grouping assumptions fail in MAS because prompts differ by role and turn. On the system side, the training system needs to support MAS-workflow-based rollouts and on-policy updates for both single and multiple policy models. To address these issues, we introduce AT-GRPO, consisting of (i) an Agent- and Turn-wise grouped RL algorithm tailored for MAS and (ii) a system to support both single-policy and multi-policy training. Across game, plan, coding, and math tasks, AT-GRPO demonstrates substantial performance gains across diverse domains. Especially on long-horizon planning tasks, AT-GRPO boosts accuracy from a 14.0–47.0% single-agent RL baseline to 96.0–99.5%. Furthermore, it improves reasoning performance, with an average gain of 3.87–7.62% on coding and 9.0-17.93% on math. The code are available at https://github.com/pettingllms-ai/PettingLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes AT-GRPO, which addresses the problem of applying on-policy RL to MAS with LLMs to improve their collaborative capabilities. This method achieves significant improvements on the Qwen3 model across four tasks: game playing, planning, coding, and mathematics.

### Strengths
Overall, this paper addresses the question of 'why standard GRPO fails in MAS' and proposes a novel sampling and grouping strategy based on the GRPO algorithm, which is applicable to MAS and provides a supporting engineering implementation.

The experiments are logically clear and demonstrate the advantages of the algorithm and the effectiveness of its components under different parameter base models and task domains.

### Weaknesses
In this paper, AT-GRPO uses K sampling and greedy selection of the optimal action in the rollout phase. This effectively provides a search budget of K for each decision step. In contrast, the baseline does not appear to have this budget. 

I have doubts about the fairness of this comparison. How much of the performance improvement is due to AT-GRPO's RL update algorithm, and how much is due to the K-fold increase in the search budget during rollout?

Furthermore, the paper relies on GRPO's variance reduction properties, but does not theoretically analyze whether the new "tree sampling + greedy execution" strategy introduces new biases, or discuss the stability of its advantage estimate as T and K increase.

### Questions
1.	All current experiments are based on two agents. Has the scalability to the number of agents been examined, including performance and computational overhead?

2.	Similarly, the paper repeatedly emphasizes the significant gains on "long-horizon planning tasks" in the abstract and conclusions. However, the turn horizon in the experimental setting is set to T=4.

3.	There is a lack of strong comparison with other related work on MAS-RL, such as MAPoRL and CURE.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the significant challenge of applying on-policy RL to Multi-Agent Systems composed of Large LLMs. The authors identify two primary bottlenecks. First, standard group-based RL objectives, such as GRPO, rely on an assumption of identical prompts for all samples within a comparison group. This assumption is fundamentally violated in MAS, where prompts are heterogeneous, differing by agent role and interaction turn. Second, existing RL-for-LLM training frameworks are predominantly designed for single-agent, single-policy optimization. They lack the architectural support for the complex, multi-agent rollouts and concurrent on-policy parameter updates for multiple, distinct policies required by MAS. To overcome these challenges, the authors propose a two-part solution, the AT-GRPO. This method creates valid GRPO comparison groups by branching $K$ samples at each agent's specific turn in the interaction. The algorithm is characterized by (i) tree-structured sampling, (ii) agent-and-turn-wise grouping, and (iii) an agent-wise credit assignment mechanism that utilizes a mix of global (team) and local (agent-specific) rewards.

### Strengths
1. The paper correctly identifies a key frontier in AI research. The field is actively moving beyond single-agent LLM fine-tuning and prompt-only MAS frameworks. The challenge of applying on-policy RL—which is often crucial for stable learning in complex, non-stationary environments—to heterogeneous, multi-policy MAS is both a real and difficult problem. The paper's attempt to provide a principled solution is commendable.

2. The paper provides a nuanced and well-reasoned analysis of the trade-off between using a single, role-sharing policy versus multiple, role-specialized policies. The finding that specialized policies are superior for roles with highly distinct functions (e.g., the Coder and Tester in the code domain) is intuitive but empirically validated. Conversely, the finding that a shared policy can sometimes be superior for roles with overlapping skills (e.g., the Reasoner and Tool agent in the math domain, where the Tool agent still needs to reason) is a practical and useful takeaway for researchers and practitioners in this field. 

3. The practical engineering of the MAS training system, while derivative of prior work (HybridFlow), is a non-trivial and necessary contribution to enable this line of research. More importantly, the Appendix is exemplary in its thoroughness. It provides exhaustive details on the complex, task-specific reward functions for all five domains (A.1.1-A.1.5) and the complete prompt templates for every agent, role, and interaction phase (A.2.2). This level of transparency is critical for reproducibility and is highly commendable.

### Weaknesses
1. The paper's headline claim—that AT-GRPO boosts planning accuracy from ~14-47% to 96-99%—appears to be built on a comparison between two fundamentally different tasks.

- Baseline Definition: The single-agent (SA) baseline for Plan/Game tasks is defined as "one agent outputs a plan (same termination condition)". For the Code baseline, it is "one agent emits code; single-turn termination". This implies a one-shot, open-loop plan generation.
- MAS Workflow Definition: The Multi-Agent System, by contrast, is an iterative feedback loop: "Planner proposes actions; Executor calls tools and returns effects/observations... termination when the goal is met or turns reach K".

This is a comparison between apples and oranges. The SA baseline is given an exponentially harder open-loop planning problem, while the MAS method is given a standard, iterative, closed-loop planning problem where it receives state feedback at every turn. The massive performance gain (e.g., 14.0% -> 96.0% on Plan-Path ) is likely dominated by this difference in task workflow, not the RL algorithm. The paper is currently attributing the gains of the MAS workflow to the AT-GRPO algorithm. 

2. The paper's remarkable achievements on planning tasks (96-99% accuracy) are fatally confounded. These results are not the product of the AT-GRPO algorithm, but rather of large-scale, complex, task-specific reward functions. These reward functions (see Appendix A.1 for details) provide the agent with "oracle-level" guidance. The algorithm is merely learning how to optimize a pre-computed solution.
- Plan-Path $s_{sp,k}^{Planner}$: The reward component is based on a function called SPNEXT. A positive reward is given if and only if the agent's action $a_k$ is "on at least one shortest path from $s_{k-1}$ to the goal". This is, by definition, a shortest path oracle. The agent is not "learning to plan"; it is simply rewarded at each step for following "breadcrumbs" provided by an external oracle that has already solved the problem.
- Sokoban $s_{dlk,k}^{Planner}$: This component is based on a DEADLOCKFREE heuristic, rewarding the agent for moves that "avoid standard static corner deadlocks." The whole difficulty of Sokoban lies in identifying and avoiding such deadlocks. It provides dense, step-by-step "breadcrumbs"—which, by definition, is a shortest path oracle—for each step. The agent isn't "learning to plan"; it's simply rewarded at each step for following "breadcrumbs" provided by an external oracle that has already solved the problem. This step-by-step reward makes the long-term credit allocation problem insignificant.
- Code $s_{cov,k}^{Tester}$: Testers are rewarded based on the "mutation score" of their generated tests on "golden code." This requires not only a golden reference implementation but also a complete mutation testing framework. This is an extremely powerful and expensive validator, far exceeding simple "pass/fail" signals.

3. The paper's presentation of its reward mechanism is contradictory and misrepresents a critical, hand-tuned component of the method. Sec 5.1 explicitly states: "The reward-mixing coefficient is $\alpha=1$ without further tuning". Equation 4 1 defines $\alpha$ as the weight on the team reward ($r^{team}$). This claim implies the method uses 100% team reward and requires no task-specific reward tuning. This claim is factually incorrect. The appendix, which details the actual reward designs, reveals complex, hand-picked, task-specific mixing coefficients (which are renamed $\lambda$): $\lambda_{math} = 0.70$, $\lambda_{sudoku} = 0.60$, $\lambda_{plan} = 0.50$, $\lambda_{sok} = 0.40$.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focus on developing on-policy RL to Multi-Agent Systems of LLMs. This work introduces AT-GRPO, which uses an "Agent- and Turn-wise" grouped tabular to store agent- and turn-wise state values. This work also developed on-policy training for shared/separated LM parameters, based on VeRL.

Experiments across four domains (game, planning, coding, math) reveal a dramatic bifurcation: near-perfect accuracy on long-horizon planning tasks (96–99.5%), but modest gains on complex reasoning (3–18% avg. improve in code/math).

### Strengths
1. The manuscript is well-written and easy to follow.

2. The authors have invested considerable effort in designing an effective RL training framework for multi-agent systems (MAS), adeptly handling both shared and distinct LLM parameters.

3. Comprehensive experiments were conducted across two model scales (Qwen-3 1.7B and 8B).

### Weaknesses
1. The experiments use an unfair reward engineering for the proposed method: AT-GRPO got 96-99.5% on planning/games while the single agent GRPO baseline only got 0-56%. A major reason is the local reward functions detailed in Appendix A.1, they provide lots of oracle information to the algorithm, e.g. in Plan-Path, the local reward is 1 if and only if the agent's chosen action "lies on at least one shortest path from $s_{k-1}$ to goal." **This is not learning to reason; this is learning to imitate an predefined oracle.** In contrast, no detailed reward function is provided for the single-agent GRPO baseline, suggesting it was unfairly handicapped and may have only used a sparse final reward.

2. The proposed Agent- and Turn-wise Group-normed advantage is a very natural extention of tabular-wise value estimate to MAS, like GiGPO[1] from GRPO, which is not mentioned in this paper. And consider the literature of cooperative MARL, tabular-wise value estimate may not be a reasonable credit assignment, a common practice in this field is to use CTDE methods like MAPPO and MADDPG which introduce a global value function for credit assignment.

3. The author should conduct ablations on the Agent-wise and Turn-wise design of the advantage, e.g. an experiment of MAS + GRPO.

4. This paper also missing some import MARL baselins for LLM, e.g. MAPoRL mentioned in the related work. Additionally this paper should also discussed some related works like MARTI[2] and MARFT[3]

[1] Feng, Lang, et al. "Group-in-group policy optimization for llm agent training." arXiv preprint arXiv:2505.10978 (2025).

[2] https://github.com/TsinghuaC3I/MARTI

[3] Liao, Junwei, et al. "Marft: Multi-agent reinforcement fine-tuning." arXiv preprint arXiv:2504.16129 (2025).

### Questions
1. What is the computational and memory complexity of maintaining the "Agent- and Turn-wise" advantage grouping (Algorithm 1, line 8)? And how does this approach scale as the number of agents ($N$) and the turn horizon ($T$) increase in more complex MAS environments?

2. Tables 1 & 2 show that the "MAS (prompt-only)" baseline significantly outperforms the "Single Agent (prompt-only)" baseline, even without any RL training. Sec 5.1 mentioned that for code and math, the SA baseline is "single-turn termination" while the MAS baseline is allowed $T=4$ turns. Is this performance gap primarily an artifact of this unfair comparison, where the MAS baseline benefits from multiple turn iterative refine while the SA baseline does not?

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
4

### Summary
This paper introduces AT-GRPO, an agent- and turn-wise grouped reinforcement learning method, alongside a novel training system, to enable effective on-policy RL for LLM-based multi-agent systems. Through extensive experiments across gaming, planning, coding, and math tasks, the authors demonstrate that their approach consistently enhances performance, particularly in long-horizon planning where it elevates success rates from 14-47% to over 96%.

### Strengths
- The experimental design is comprehensive, and the results consistently demonstrate the effectiveness of the method. However, the model scales used (1.7B, 8B) are relatively small, leaving uncertainty about how well the approach would scale to much larger models.

- The finding that the choice between a role-sharing policy and role-specialized policies depends on the task characteristics is a very interesting and valuable insight for the field.

### Weaknesses
- The paper is at times hard to read. Key explanations, such as the components of Figure 3, are unclear.

- See Questions.

### Questions
- The description of Figure 3 is unclear. Where is the “middle” section referring to the code debugging task (mentioned around line 214)? Also, what do the gray and blue colors represent in the figure?

- The notation used across sections is inconsistent. For example, Equation 1 uses index  l \in \{1, \dots, K\}, while Algorithm 1 uses c \in \{1, \dots, K\}. This inconsistency hinders readability.

- How is the number of agents \( N \) determined in the MAS setup?

- What is the advantage of tree sampling over parallel sampling?  Additionally, in Algorithm 1, greedily selecting the action with the maximum turn-level reward at each turn t may lead to suboptimal final outcomes. Could this myopic selection affect the overall reward?

- Equation 3 does not include a clipping mechanism to constrain policy distribution changes. Is there a specific reason for omitting this common GRPO stabilization technique?

### Soundness
2

### Presentation
2

### Contribution
2
