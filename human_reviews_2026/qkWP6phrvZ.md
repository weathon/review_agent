# Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided exclusively upon generating the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate three critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals; (ii) lack of fine-grained credit assignment, where the correctness of intermediate turns is obscured, especially in long-horizon tasks; and (iii) poor sample efficiency, where each rollout yields only a single outcome signal, leading to low data utilization. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer.  Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward signals. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved data efficiency. Our code is available at https://github.com/GuoqingWang1/IGPO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
They propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. Specifically, they use log-likelihood of the ground truth at each turn to define the intrinsic rewards.

### Strengths
- The algorithm is clearly presented. The algorithm design is simple yet effective.
- The empirical evaluation is solid with significance improvements observed over multiple baseline methods.

### Weaknesses
- Limited Novelty: The paper introduces a new intrinsic reward design, but its novelty is somewhat constrained. Further exploration or differentiation from existing designs would enhance its contribution to the field.
- Insufficient Theoretical Support: The paper lacks robust theoretical justification for the proposed intrinsic reward design.
- Lack of Insight on Alternative Designs: The paper would benefit from a discussion on alternative intrinsic reward designs. Exploring how the proposed design could be integrated with other existing designs might provide valuable insights and broaden its applicability.

### Questions
See weaknesses.

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
4

### Summary
This paper presents Information Gain-based Policy Optimization (IGPO), a reinforcement learning framework that enhances multi-turn LLM agents for tasks like web search and multi-hop question answering. IGPO employs dense, turn-level intrinsic rewards derived from the marginal probability increase of generating correct answers at each step. By integrating these per-turn rewards with conventional outcome-based rewards, IGPO addresses reward sparsity and advantage collapse in long-horizon RL, enabling superior credit assignment. Experiments across seven benchmarks demonstrate IGPO's superiority over prompt-based and RL baselines, supported by ablation studies, training analyses, and theoretical foundations.

### Strengths
1. Principled and Well-Motivated Reward Design: IGPO addresses a pertinent weakness in agentic RL for LLMs—the reward sparsity problem—by introducing information-gain signals that provide stepwise, ground-truth-aware supervision. The approach is simple yet effective and is grounded in a clear theoretical motivation (see Appendix A).

2. Thorough Mathematical Formulation: The paper provides clear derivations of the reward formulation (Equation 4–7), discounted advantage calculation, and the overall surrogate objective for IGPO. This mathematical clarity makes the method reproducible and portable to related settings.

3. Comprehensive Empirical Evaluation: The authors conducted careful experiments across in-domain and out-of-domain benchmarks (NQ, TQ, HotpotQA, 2Wiki, MusiQue, Bamboogle, PopQA), and compared IGPO against strong baselines (both RL-based and prompt-based, see Table 1 and Table 2).

4. Reproducibility: Public release of source code enhances transparency and facilitates replication.

### Weaknesses
1. Novelty is Incremental vs. Contemporary Efforts: While IGPO’s design is sound, many core principles—such as dense, turn-level supervision, teacher-forced signals for policy confidence, and combining process and outcome rewards—are parallel to mechanisms introduced or explored in very recent works from 2025 that are not thoroughly contrasted or ablated against. The differences from GiGPO, ReasoningRAG, StepSearch, and especially the missing related work are not sharply drawn.
  - Zeng, S., Wei, Q., Brown, W. (2025): "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment"
  - Wei, Q., Zeng, S., Brown, W. (2025): "LeTS: Learning to Think-and-Search via Process-and-Outcome Reward Hybridization"
  - Tang, X., Xu, W., Wang, Y. (2025): "Eigen-1: Adaptive Multi-Agent Refinement with Monitor-Based RAG for Scientific Reasoning"

2. Reliance on Ground-Truth for Intrinsic Reward: The information gain reward fundamentally requires access to the ground-truth answer for teacher-forcing in every trajectory step (see Section 3.2,). This is not always feasible for open-ended or real-world deployments, limiting applicability. The issue is acknowledged in the limitations, but its practical significance is not fully explored, nor are mitigations proposed.

3. Absence of Fine-Grained Failure Analysis: While the aggregate numbers and training curves in Tables 1–3 and Figures 4–5 are generally positive, there remains a lack of granular breakdown where IGPO underperforms (if any), or qualitative investigation of error modes and tasks/environments where information gain may be less reliable (e.g., ambiguous or multi-answer questions). The failure cases in Figures 6 and 7 are insightful but could be complemented with quantitative measures of failure types.

### Questions
1. Applicability Beyond Ground Truth Rich Environments. IGPO relies on ground-truth answers for teacher-forced reward estimation in every trajectory step. How would the method adapt to settings where ground truths are partial, noisy, or unavailable (e.g., open-ended queries, creative generation, real-world search)? Could unsupervised or self-consistency signals be integrated in lieu of explicit ground truth?

2. Comparison With Omitted Recent Baselines. Please provide direct comparison (either empirical or qualitative) against the most recent 2025 turn-level/process-level RL methods listed above (e.g., Turn-Level Credit Assignment, LeTS, ToolRL). What fundamentally differentiates IGPO in real settings?
  - Zeng, S., Wei, Q., Brown, W. (2025): "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment"
  - Wei, Q., Zeng, S., Brown, W. (2025): "LeTS: Learning to Think-and-Search via Process-and-Outcome Reward Hybridization"
  - Zeng, S., Wei, Q., Brown, W. (2025): "ToolRL: Reward is All Tool Learning Needs"

3. Failure Modes and Robustness. What are the typical failure cases or degenerate behaviors for IGPO—especially when intermediate turns provide little actual information gain, or when there are multiple valid answer paths? Could the dense rewards inadvertently reinforce misleading intermediate confidence?

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
3

### Summary
This paper proposes IGPO, a reinforcement learning framework for training multi-turn LLM agents. IGPO addresses the reward sparsity and advantage collapse issues of outcome-only rewards by introducing a turn-level information gain reward, which quantifies how much each agent turn increases the policy’s probability of generating the correct answer. These dense, ground-truth-aware rewards are integrated with final outcome rewards to form a comprehensive signal for GRPO-style optimization. Experiments on seven search-based QA datasets (NQ, TQ, HotpotQA, 2Wiki, MusiQue, Bamboogle, PopQA) show that IGPO consistently outperforms prior outcome- and step-reward RL baselines, improving both sample efficiency and training stability.

### Strengths
1. IGPO’s reward formulation is Simple yet effective, estimating turn-level information gain without requiring additional annotation or external evaluators.
2. Strong empirical results: Extensive experiments on both in-domain and out-of-domain datasets show consistent improvements over strong baselines such as GiGPO and DeepResearcher.
3. The paper is well-structured and easy to follow.

### Weaknesses
1. The evaluation focuses exclusively on search-based QA tasks. Such settings naturally align with the proposed information-gain reward, as acquiring relevant retrieved evidence directly increases the probability of producing the correct answer. However, in other domains, such as mathematical reasoning, embodied agents, or web agents, this reward may not lead to performance improvements. Consequently, the proposed method appears to be primarily applicable to information-retrieval tasks like search-based QA.

2. Moreover, this reward could reinforce spurious correlations. LLMs are known to exploit shortcuts when solving problems, relying on superficial cues rather than learning the underlying reasoning process. Instead of acquiring new knowledge through comprehensive reasoning, LLMs may overfit to spurious correlations between final answers and intermediate components. In contrast, many recent works on mathematical reasoning explicitly aim to mitigate such spurious correlations. The design of the IGPO method could, in fact, encourage this undesired behavior, potentially having a negative impact on this research area.

3. Finally, the paper computes normalized advantages using: A = r-mean(R)/std(R), where R aggregates all rewards across all steps and all rollouts within a group. This design is theoretically problematic. If the method assumes step-level rewards, the baseline should be computed per step. That is, by sampling multiple trajectories starting from that turn and averaging their returns. Otherwise, the estimator mixes rewards from different time steps with distinct state-action distributions, introducing bias. If, instead, the method assumes trajectory-level rewards, it should not be framed as step-level GRPO. The authors need to clarify or justify this design choice.

### Questions
1. How does IGPO perform on non-search tasks such as mathematical reasoning, code generation, or web-agent? Would the information-gain reward still be meaningful there?
2. Could you provide empirical evidence showing whether IGPO indeed mitigates spurious correlations, e.g., by testing on tasks requiring multi-step reasoning consistency rather than document retrieval?
3. How sensitive is the training to the normalization scheme of turn-level rewards? Have you tried per-turn baselines or PPO?
4. What is the computational cost compared to baseline algorithms, given the need to compute per-turn log probabilities of the ground truth?

### Soundness
2

### Presentation
3

### Contribution
2
