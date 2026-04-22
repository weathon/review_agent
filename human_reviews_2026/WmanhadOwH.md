# Memory-Driven Self-Improvement with Large Language Models

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 6, 2, 4

## Abstract
Large language models (LLMs) have emerged as effective action policies for sequential decision-making (SDM) tasks due to their extensive prior knowledge.  However, this broad yet general knowledge is often insufficient for specific decision-making tasks with limited task-related data, making it challenging to efficiently adapt LLMs to specific SDM tasks. To address this challenge, we propose a memory-driven self-improvement framework that combines LLM general prior knowledge with a compact memory of domain-specific experiences. Memory retains past interactions and associated Q-values, thereby capturing decision-relevant knowledge that facilitates accurate value estimation and informs the LLM prior refinement. The refined LLM prior, in turn, generates higher-reward trajectories that further enrich memory, forming a natural self-improvement framework where memory and LLM prior mutually reinforce each other. Experiments show that our memory-driven approach significantly outperforms both traditional RL and LLM-based baselines, e.g., improving performance by over 40\% on in-distribution tasks and over 75\% when generalized to unseen tasks in ALFWorld.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses the limiation in efficiently adapting LLMs to text-based decision-making tasks, and proposes a method to combine LLM priors with task-specific interactions.

In this method, the LLM prior generates action candidates and memory-driven value estimation guides action selection, establishing a self-imporvement loop between the fine-tuned LLM priors (through LoRA) and the updated value estimation (through Q-table).

Experiments on 2 tasks in Overcooked and 2 tasks in ALFWorld shows the effectiveness of proposed method compared to traditional RL method DQN (and its variants) and fixed LLM priors (and its variants).

### Strengths
1. Memory is a key aspect of LLM for improved performance especially in agent tasks. This paper proposes a memory mechanism that stores Q-values for visited state–action pair, which aims to select more precise actions and refine the LLM parameters.
2. Experiments show effective self-reinforcing effects between updated memory table and refined LLM priors compared to DQN and variants of Mem-EM.

### Weaknesses
1. **Insufficient baselines to compare.**
The compared baselines in the paper only include DQN variants and variants of the proposed method (Mem-EM). 
Many other approaches (e.g., [1][2][3]) incorporating memory into LLM agents are not compared. Thus, current results cannot demonstrate the effectiveness of proposed Q-memory compared to other forms of memory for LLM agents.
2. **Insufficient benchmark experiments raises concern about the generalization of the proposed method across different domains and tasks.**
The paper only chooses 2 tasks from Overcooked and 2 tasks from ALFWorld. Why not consider all the task set in these two environments? Is there any special reason to choose these four tasks in experiments?
3. **Concerns about the practicality of the proposed method.**
Mem-EM needs to store memory table, which is impractical when facing open-ended environments with infinite state/action spaces.
Additionally, Mem-EM needs to compare Q-values of different action candidates for the same state, however, in many text-based environments (e.g, dialogue) we cannot find the same intermediate state within a trajectory.
4. **Issues in both memory and compuation cost.**
Due to the memory table storage and multiple action sampling per state, Mem-EM induces more memory and LLM rollout costs despite it can improve exploration efficiency. 

References:

[1] Xu, Wujiang, et al. "A-mem: Agentic memory for llm agents." arXiv preprint arXiv:2502.12110 (2025).

[2] Packer, Charles, et al. "MemGPT: Towards LLMs as Operating Systems." (2023).

[3] Zhong, Wanjun, et al. "Memorybank: Enhancing large language models with long-term memory." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 17. 2024.

### Questions
1. What is the difference between $LLM_{Mem-EM}$ and $LLM_{Mem-EM}+Q_{Mem-EM_{w/tune}}$ in Table 1? 
Better to provide the mean results and standard error in Table 1 across different seeds.
2. Why do $LLM_{Mem-EM}+Q_{Mem-EM_{w/tune}}$ in Table 1 show better performance when K=5 compared to larger K?
3. It is challenging to estimate value when process rewards are not available according to Eq.2. What is the value of $\gamma$ used in Eq.2? Any ablation studies for this?
4. Is it worth to improve exploration effciency at the cost of more memory and computation costs? Please provide detailed memory and computation costs before and after using Mem-EM.

Typo: citation format in the paper

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a memory-driven self-improvement framework for text-based sequential decision-making that combines LLM general knowledge with a compact memory of domain-specific experiences. The framework consists of two components: (1) memory-driven value estimation using kernel-based Q-value approximation with LLM embeddings, and (2) memory-driven LLM prior refinement via an Expectation-Maximization formulation. The authors claim their approach achieves improvements on both in-distribution and out-of-distribution tasks.

### Strengths
- The paper presents an interesting combination of memory-based Q-learning with LLM priors in the EM framework.
- The paper is well written and easy to follow.

### Weaknesses
Please see my detailed questions and concerns below.

### Questions
- Given that you're using Qwen2.5 as your LLM prior, why not use the same LLM's encoder for state-action embeddings? Wouldn't this create better alignment between the value estimation and policy components?
- How do you handle the cold-start problem? When memory is empty or sparse early in training, how reliable are your Q-value estimates? This could significantly impact early exploration.
- How do you ensure memory diversity? The LRU replacement strategy might keep frequently visited states but miss important rare states.
- When K increases beyond training settings, performance often drops in unseen tasks (Table 1). Doesn't this suggest overfitting to the training distribution?
- Both test environments seem fully observable. How would this extend to POMDPs where memory of past states might be critical?

### Soundness
3

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
The paper proposes a "Memory-Driven Self-Improvement" framework to adapt Large Language Models (LLMs) for specific sequential decision-making (SDM) tasks, addressing the insufficiency of their general prior knowledge. The method combines the LLM's prior with a compact memory that stores domain-specific experiences, specifically past state-action pairs and their estimated Q-values. This framework establishes a mutually reinforcing loop: a memory-driven, non-parametric value estimator retrieves semantically similar experiences to guide efficient exploration, while an Expectation-Maximization (EM) algorithm periodically refines the LLM prior by learning from the high-quality interactions stored in memory. This refined prior then generates higher-reward trajectories, which in turn enriches the memory. Experiments in environments like ALFWorld and Overcooked demonstrate that this self-improvement cycle significantly outperforms traditional reinforcement learning and LLM-based baselines in both sample efficiency and generalization to unseen tasks.

### Strengths
*  **Novel Synergistic Framework:** The paper introduces a novel and principled framework that synergistically combines LLM priors with a memory-augmented Q-learning. This memory-driven approach, which stores and retrieves domain-specific `(s, a, Q)` experiences, allows the agent to effectively integrate the LLM's generalist knowledge with high-value, task-specific trajectories, leading to significant performance gains on complex benchmarks like ALFWorld.

*  **Superior Sample Efficiency:** A key advantage of this framework is its remarkable sample efficiency compared to traditional methods like reinforcement learning fine-tuning (RLFT).

*  **Thorough Empirical Ablation:** The paper's thorough empirical ablation studies demonstrate its robustness and generalization.

### Weaknesses
* The framework's performance appears highly sensitive to parameter tuning. Its multi-component design introduces numerous hyperparameters, suggesting that performance could vary significantly with different settings, potentially impacting robustness.

* The algorithm's effectiveness is critically dependent on the external embedding model. The core memory-driven value estimation is retrieval-based, and the embedding model's performance will significantly impact the algorithm's overall performance.

### Questions
* How does the time-efficiency of the framework at test time compare to baselines like DQN and DQN-Prior?

* The hyperparameter $\tau_1$ governs the stochasticity of the posterior action selection and thus directly manages the exploration-exploitation trade-off of the policy. Could the authors provide results demonstrating the algorithm's performance with different values of $\tau_1$?

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
The paper proposes a memory-driven self-improvement framework for text-based sequential decision making. It has two coupled pieces: (i) Mem-Q, a non-parametric, retrieval-based Q-estimator that stores past  (s,a,Q) in a memory table and estimates Q(s,a) via k-NN with an inverse-distance kernel over LLMembeddings; and (ii) Mem-EM, an EM-style procedure that improve the LLM policy. Experiments on Overcooked (text) and ALFWorld report faster learning than DQN/DQN-Prior and gains for Mem-EM with LoRA fine-tuning.

### Strengths
1. The paper is easy to follow: it combines non-parametric value estimation with control-as-inference type policy learning.

2. Studies over candidate count, memory size and embedding backbones help illuminate when the system works best

### Weaknesses
1. The Mem-Q piece is simply Episodic Control with better embeddings; Mem-EM resembles MPO/EM-style policy updates with a learned prior. The paper cites these lines but does not sharply distinguish what is algorithmically new (beyond swapping in LLM embeddings and an LLM prior). In other words, it seems to me that the paper mainly combines these two ideas in the LLM domain. New insights when LLM coming into the picture is lacking.

2. Off-policy/Q-type-learning convergence is typically unstable. The paper even relies on importance sampling. Furthermore, the importance sampling weight is a heuristic approximation. Finally, does the memory table has sufficient coverage/exploratory enough? This makes people questions the practicality of the algorithms. I believe when we talk about policy optimization for LLMs, we have to be careful and understand what is new/different compared with standard DRL setting.

3. Baselines studies are also limited. For example, no comparisons with search-based planners guided by value, no comparisons with more advanced memory-based RL algorithms.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of adapting Large Language Models (LLMs) for specific sequential decision-making (SDM) tasks, particularly when domain-specific data is limited. The authors note that while LLMs possess broad prior knowledge, this is often insufficient for specialized tasks. To solve this, the paper proposes a **memory-driven self-improvement framework**. The core of this framework is a memory table that stores past interactions (state-action pairs) and their estimated Q-values. This process forms a self-improvement loop: the refined LLM generates better trajectories, which in turn enriches the memory, leading to better value estimates and further refinement. The framework is formalized using an Expectation-Maximization (EM) algorithm. Experiments conducted on ALFWorld and Overcooked environments demonstrate that the proposed method (Mem-EM) significantly outperforms baselines like DQN and DQN-Prior in terms of sample efficiency and final performance, and shows strong generalization to unseen tasks.

### Strengths
- The paper addresses the **important and practical problem** of adapting generalist LLMs for specialized decision-making tasks, which is a key challenge for their real-world application.

- The core idea of using a **memory table to drive both value estimation and LLM prior refinement** is intuitive and well-motivated. The proposed self-improvement loop, where the memory and LLM mutually enhance each other, is a compelling concept.

- The paper is **well-written and easy to follow**. The framework's components and data flow are clearly explained, supported by effective illustrations.

### Weaknesses
The **empirical evaluation, while promising, could be strengthened** in several areas:

- The comparison is primarily against DQN and a DQN-Prior baseline. The exclusion of more advanced or policy-based RL algorithms (e.g., PPO, DAPO), which are common in decision-making, makes it difficult to fully contextualize the performance gains. 

- The evaluation is limited to two benchmarks, ALFWorld and Overcooked. While these are relevant text-based environments, including **more complex decision-making tasks** would provide a more robust validation of the framework's scalability and effectiveness. 

- The experiments rely exclusively on the Qwen model family (Qwen2.5-3B and 7B). Testing the framework with **other foundational LLMs** (e.g., from the Llama families) would be valuable to demonstrate that the approach is model-agnostic.

### Questions
- The paper mentions that the computational overhead is "tolerable" by using LORA and infrequent tuning (e.g., six times in the ALFWorld experiment). Could the authors provide a more concrete analysis of the **computational cost** (e.g., total training time, GPU memory) of the $Mem-EM_{w/tune}$ method compared to the baselines? 

- For the memory-driven Q-estimation, the authors use an **inverse distance kernel** (Eq. 3). What was the rationale for this specific choice? Have other kernel functions or alternative non-parametric estimators been explored, and how do they compare?

### Soundness
3

### Presentation
3

### Contribution
2
