# Uncertainty-Aware Tree Search for Efficient LLM Reasoning

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Tree search is an important class of methods for improving the multi-step reasoning capability of Large Language Models (LLMs) by explicitly exploring the intermediate steps on a search tree guided by a value function. However, the existing tree-search methods often devote equal computational budgets across different reasoning branches regardless of the associated uncertainty, causing significantly high token consumption. In this paper, we first conduct a pilot study to reveal the ubiquitous semantic redundancy of reasoning trajectories starting from an intermediate reasoning step. Such highly certain reasoning steps will ultimately reduce the diversity of the final answers. Further, we theoretically show that under a probabilistic guarantee, the sampling budget required to maintain fixed generation quality grows proportionally with the step uncertainty. Built on top of this, we propose \textbf{Uncertainty-Aware Allocation} (UAA), a plug-and-play framework that allocates search budget adaptively according to the step-wise uncertainty. In particular, UAA detects for highly certain reasoning steps and incorporates (i) uncertainty-aware pruning (UAP) to keep only a high-quality subset of candidate actions, and (ii) uncertainty-aware budgeting (UAB) to shrink the next-step expansion budget. Extensive empirical evaluations demonstrate that UAA can significantly reduce token consumption and wall-clock time without hurting accuracy when applied to Beam Search and MCTS.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The existing tree search methods for multi-step inference in large language models suffer from high computational overhead. The article points out that these methods allocate the same computational budget to all inference branches without considering the uncertainty of the branches themselves. The author found through empirical research that there is a high degree of semantic redundancy in the inference trajectory starting from intermediate steps in tree search inference of large language models, and this redundancy contributes limited to the diversity of final answers. Based on this, the article proposes a plug and play framework for uncertainty aware allocation (UAA) that does not require training, including uncertainty aware pruning (UAP) and uncertainty aware budget (UAB). Experiments show that in Beam Search and MCTS, token consumption and inference time are significantly reduced while maintaining or even improving accuracy.

### Strengths
1、The proposed UAA framework is training-free, requires no auxiliary models or fine-tuning, and can be easily integrated into existing tree-search algorithms.
2、The paper provides a clear information-theoretic motivation (Corollary 1) and supports it with empirical evidence, including a pilot study on trajectory redundancy and extensive experiments across multiple models and benchmarks.
3、UAA consistently reduces token usage and inference time across Beam Search, MCTS, DVTS, and REBASE, without compromising accuracy. The result that highlights its generality and robustness.

### Weaknesses
1、The experiment is limited to mathematical reasoning tasks. The effectiveness of this framework for other types of reasoning, such as logical reasoning, common sense reasoning, or planning, has not been confirmed, which limits its claimed universality.
2、Key hyperparameters such as the uncertainty threshold λ, pruning count k^, and reduced budget B^ are fixed across experiments. Their sensitivity and adaptability across different tasks or model scales are not systematically analyzed, which may affect reproducibility and deployment in new settings.

### Questions
1、How was the intermediate state selected for subsequent redundancy analysis in the pilot experiment of the third section?
2、Have the authors considered evaluating UAA on non mathematical reasoning tasks, such as common sense reasoning ScienceQA and logical reasoning StrategyQA, to better support its universal applicability?
3、Could the authors provide more insight into how the hyperparameters (λ, k^, B^) were chosen? A sensitivity analysis or adaptive strategy for these parameters would strengthen the method's practicality.

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
The authors conduct an empirical study to reveal the ubiquitous redundancies among reasoning trajectories in tree-search methods, and demonstrate that such redundancies would ultimately reduce the diversity of the final answers. To address this, this paper presents Uncertainty-Aware Allocation(UAA), a training-free, plug-and-play module that dynamically allocates budgets for each reasoning step in the search tree. UAA is used for both Beam Search and MCTS. Experiments conducted on the GSM8K, MATH500, and AIME24 benchmarks validate that UAA successfully improves the reasoning efficiency of these tree-search algorithms.

### Strengths
1. The paper motivation is clear, addressing the critical problem of extensive redundancy within the search space of tree-search algorithms for LLM reasoning. The empirical study reveal that this redundancy is prevalent among reasoning trajectories originating from intermediate steps and compellingly demonstrate that it ultimately diminishes the diversity of the final answers. This insight is a significant contribution to the advancement of tree-search methods for LLM reasoning.

2. UAA improves the reasoning efficiency when it integrates with tree-search algorithms like Beam Search and MCTS on challenging benchmarks, including GSM8K, MATH500, and AIME24.

3. The paper provides theoretical proofs to substantiate its claims.

### Weaknesses
1. The experiments is confined to MATH benchmarks (GSM8K, MATH500, AIME24). The method's adaptability and effectiveness across diverse types of datasets remain unverified. This is a notable limitation, particularly as LLM + MCTS methods are increasingly being applied to more realistic and interactive reasoning environments, such as Blocksworld [1] and WebShop [2], where the search dynamics may differ.

2. The empirical study on reasoning trajectory redundancy appears to be limited, as it was conducted using only small-scale LMs and focused exclusively on Beam-Search. And, the paper lacks a corresponding analysis for the MCTS scenario.

[1] Reasoning with language model is planning with world model[J]. At EMNLP.

[2] Webshop: Towards scalable real-world web interaction with grounded language agents[J]. At NeurIPS.

### Questions
1. Could the authors provide experiments applying UAA to Beam Search or MCTS on other LLM reasoning tasks (beyond math benchmarks), preferably on more realistic reasoning tasks, such as web-based tasks?

2. Could you provide additional data to demonstrate that the observed reasoning trajectory redundancy also occurs when employing powerful LLMs, such as GPT-4o, within both Beam Search and MCTS? This evidence would be crucial to confirm that the redundancy is an inherent challenge in the search process itself, rather than a byproduct of using less capable models.

3. How does the introduction of UAA impact the final depth of the search trees? Is there a risk that dynamically allocating budgets might lead to insufficient exploration (i.e., pruning promising paths too early) or premature termination of the search?

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
This paper presents an uncertainty-aware framework denoted as UAA, which is designed to optimize the tree-search reasoning efficiency of Large Language Models (LLMs). Corresponding experiments of the framework are carried out on multiple mathematical reasoning benchmarks.

### Strengths
The study’s focus on addressing uncertainty in LLM-based reasoning holds certain merit, as effective uncertainty management is critical for improving multi-step reasoning efficiency.

### Weaknesses
1. UAA’s premise that "low-uncertainty steps correspond to semantic redundancy with no contribution to answer diversity" is only validated in math reasoning. This assumption may fail in open-domain tasks 
2. Experiments are only conducted on open-source models (Llama3.1-8B-Instruct, Qwen3-32B). For widely used closed-source LLMs (e.g., GPT, Gemini) in industry, the paper only mentions "confidence/PRM variance" as alternative uncertainty proxies but lacks empirical validation, leaving UAA’s practicality unproven.

### Questions
1. Is the correlation between uncertainty and "step value" absolute? UAA defaults to the assumption that "low uncertainty equates to low value (thus prunable)". However, in mathematical reasoning, "low-uncertainty yet critical steps" may exist—for instance, "lemma derivation" in proof-based problems, where the step is uniquely determined but a single error would invalidate the entire reasoning process. Why does UAA rely exclusively on uncertainty for pruning, rather than integrating "step correctness scores" to make a holistic judgment?
2. Can experiments on open-domain reasoning tasks (e.g., question answering, dialogue generation, and planning tasks such as WebShop) be incorporated to verify the generalizability of UAA’s core assumption?
3. Will supplementary experiments be conducted on closed-source LLMs (e.g., GPT, Gemini) to validate the effectiveness of alternative uncertainty proxies (e.g., confidence scores/PRM variance) in black-box scenarios?

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
The paper introduces a plug-and-play framework named Uncertainty-Aware Allocation (UAA), which solves reasoning computational budgets with the associated uncertainty. First, the paper finds that remaining reasoning trajectories from the same step exhibit high semantic similarity in tree-search methods. Second, it proposes the UAA that accelerates tree-based reasoning via dynamically pruning redundant reasoning steps and allocating expansion budget according to step-wise uncertainty. It is training-free and applicable to both black-box and white-box LLM settings. The results show that UAA consistently reduces inference cost without compromising accuracy and generalizes well across different LLMs settings.

### Strengths
1. The paper shows that the ubiquitous semantic redundancy of reasoning trajectories starts from an intermediate reasoning step. Such highly certain reasoning steps will ultimately reduce the diversity of the final answers.
2. The paper proposes Uncertainty-Aware Allocation (UAA), a plug-and-play framework that allocates search budget adaptively according to the step-wise uncertainty with uncertainty-aware pruning (UAP) and uncertainty-aware budgeting (UAB).
3. Empirical evaluations demonstrate that UAA can significantly reduce token consumption and wall-clock time when applied to Beam Search and MCTS.

### Weaknesses
1. The current design of UAA adopts fixed thresholds and budgets for compute allocation, which restricts its adaptability across reasoning depths.
2. The proposed method lacks the evaluation of more realistic tasks in the world. The generalization of the method needs further discussion.

### Questions
1. How effective is the proposed method on larger and stronger LLMs such as Claude 3 or Gemini 2.5 Pro?
2. How does UAB save costs? As shown in Table 2, tokens only show a significant downward trend when both UAP and UAB are used for tree search. Because "UAP focuses the limited budget on high-value trajectories", why do they together achieve the lowest Tokens?
3. Why do you use two types of value functions to guide search? Because in Tables 1 and 2, it means that the PRM is almost better performance than Conf. for the Acc., Tokens, and Time.

### Soundness
3

### Presentation
2

### Contribution
2
