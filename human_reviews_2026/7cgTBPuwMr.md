# Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
This paper investigates Reinforcement Learning (RL) approaches to enhance the reasoning capabilities of Large Language Model (LLM) agents in long-horizon, multi-turn scenarios. Such multi-turn agentic tasks can be naturally formalized as turn-level Markov Decision Processes (MDPs). However, most existing methods adopt MDP formulations with trajectory-level rewards, either terminal rewards that provide only a final outcome signal, or delayed rewards that merge intermediate and outcome signals into a single sparse feedback, leading to poor credit assignment. To address this limitation, we reformulate these tasks as MDPs with explicit turn-level rewards and provide theoretical analysis supporting the effectiveness of this design. Building on this formulation, we extend popular RL algorithms, GRPO and PPO, to their respective multi-turn variants, enabling fine-grained credit assignment. We conduct case studies on multi-turn reasoning-augmented search agents, where we carefully design two types of turn-level rewards: verifiable and LLM-as-judge. Our experiments on multi-turn search tasks demonstrate that our proposed formulation, incorporated well-designed turn-level rewards, enables RL algorithms to significantly outperform baseline methods with trajectory-level rewards. Both training and validation reward curves illustrate that our method achieves \textit{greater stability}, \textit{faster convergence}, and \textit{higher accuracy}. Numerical results across diverse question-answering datasets further show that our approach consistently delivers highest answer correctness and 100\% format correctness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
They propose a turn-level reward design strategy to enhance RL algorithms in multi-turn agent tasks. By integrating turn-level rewards, they extend GRPO and PPO to their respective multi-turn variants, enabling fine-grained credit assignment.

### Strengths
- The algorithm is well clarified with a specific case study.
- This paper studies a fundamental problem for multi-turn RL -- the use of turn-level reward.
- The MT versions of PPO and GRPO show better performance compared to their counterparts: PPO and GRPO.

### Weaknesses
- Lack of theoretical support.
- Limited Baselines for Comparison: To provide a more comprehensive evaluation, additional baselines should be included, such as GRPO or PPO augmented with intrinsic rewards. The current comparisons are restricted to open-source LLMs and ablated variants of the algorithm, which may not fully benchmark the approach against state-of-the-art reinforcement learning methods in similar domains.
- Omission of Concurrent Works: The discussion should address relevant concurrent research, such as the work on "Context-lite Multi-turn Reinforcement Learning for LLM Agents," to highlight how the proposed method differentiates itself or builds upon these efforts.

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
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a turn-level reward design framework to improve reinforcement learning for multi-turn LLM agents. The authors extend GRPO and PPO into multi-turn variants (MT-GRPO and MT-PPO) that integrate intermediate rewards to enable finer credit assignment across reasoning steps. They evaluate the approach on search-based QA tasks using both verifiable and LLM-as-judge rewards. Experimental results with Qwen2.5-7B show that MT-PPO achieves more stable training, faster convergence, and better format correctness than PPO and GRPO. The paper highlights turn-level reward design as a promising direction for long-horizon agent training.

### Strengths
1. The paper tackles an important problem: improving multi-turn reasoning in LLM agents through better reward shaping
2. The distinction between single-turn and multi-turn MDP formulations is well presented and conceptually sound.
3. The paper is clearly written and easy to follow, with consistent notation and illustrative examples.

### Weaknesses
1. The main contribution, introducing turn-level rewards into PPO/GRPO, is conceptually straightforward and closely related to prior work on process reward models (PRM) and segment-level credit assignment. The paper overstates its originality by claiming to be the “first systematic study” without adequately discussing/comparing with these prior methods.
2. The experiments are limited to search-based QA tasks, leaving it unclear whether the proposed framework generalizes to other multi-turn or open-ended domains such as code generation, dialogue, or planning.
3. The reported improvement in answer accuracy (approximately +1.5% over PPO) is relatively modest compared to the additional implementation complexity required for designing and tuning intermediate rewards.
4. The motivation for introducing MT-GRPO is weakly justified, and its evaluation is minimal.
5. Novelty concern: Much of the paper’s content overlaps with existing work, and the contributions are largely incremental.

### Questions
1. How does MT-PPO performance compare to prior PRM or step-level RL work?
2. How sensitive are results to the chosen reward weights (retrieval, format, search penalty)?
3. What is the computational overhead of MT-PPO relative to standard PPO (in terms of runtime, tokens, or sample efficiency)?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of training LLM agents for complex, multi-turn tasks using RL, proposing a novel turn-level reward design strategy. This strategy provides fine-grained credit assignment by rewarding the agent at each step of its multi-turn interaction. The authors extend both GRPO and PPO into multi-turn variants and conduct case studies on a reasoning-augmented search agent. Experiments demonstrate that their method achieves greater training stability, faster convergence, and higher accuracy compared to baseline methods that use only outcome-level rewards.

### Strengths
- 1.The work identifies and systematically tackles a fundamental flaw in applying RL to multi-turn LLM agents: the credit assignment problem. By shifting from sparse, end-of-task rewards to dense, turn-level rewards, the method provides the agent with much richer and more immediate feedback, which is crucial for learning complex sequences of actions.

- 2.The paper offers a detailed and practical framework for designing turn-level rewards, which is a significant contribution. It introduces two distinct types of rewards: verifiable rewards and LLM-as-judge rewards. This dual approach ensures both precision and flexibility in guiding the agent's behavior. The authors not only create a multi-turn variant of GRPO but also develop MT-PPO to overcome MT-GRPO's computational limitations.

- 3.Experiments on multiple question-answering datasets consistently show that their approach leads to more stable training, faster convergence, and superior performance in both answer correctness and output format adherence compared to strong baselines.

### Weaknesses
- 1. **High Computational Complexity**: The proposed MT-GRPO method requires exponential trajectory samples, making it infeasible for long-horizon tasks. While MT-PPO reduces this cost via a critic model, it still introduces additional training overhead.

- 2. **Fixed-Turn Constraint Limits Flexibility**: MT-GRPO mandates all rollout groups to have the same number of turns, enforced through system prompts. This rigid structure hinders adaptability to dynamic scenarios where tasks may require variable interaction lengths.

- 3. **Reward Design Relies on Heuristic Priors**: Turn-level rewards are manually tuned without theoretical justification. This risks reward hacking and limits generalizability. The LLM-as-Judge approach also inherits biases from the judge model.

### Questions
- 1. Is there a more reliable and theoretically-grounded method for reward design that enables adaptation to different tasks?
- 2. The experiments focus on structured search tasks with clear turn boundaries. How would the method perform in less structured environments, such as open-ended dialogue or collaborative planning, where turns may involve unpredictable state transitions or partial observability?

### Soundness
3

### Presentation
3

### Contribution
2
