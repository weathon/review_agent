# LearNAT: Learning NL2SQL with AST-guided Task Decomposition  for Large Language Models

- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
Natural Language to SQL (NL2SQL) aims to translate natural language queries into executable SQL statements, offering non-expert users intuitive access to databases. While recent approaches leveraging large-scale private LLMs such as GPT-4 have achieved state-of-the-art results, they face two critical challenges: the lack of openness and reproducibility, and the prohibitive computational cost of test-time scaling. To address these issues, we explore improving the model-level performance of small-scale public LLMs in NL2SQL under resource-constrained settings. Our exploratory experiments reveal the potential of task decomposition for enhancing NL2SQL performance, but also highlight the difficulty of enabling LLMs to decompose queries effectively. Motivated by these findings, we propose LearNAT, a novel framework designed to enhance LLMs’ decomposition capabilities. LearNAT introduces (1) a Decomposition Synthesis Procedure, which leverages AST-guided search with pruning strategies to generate verifiable and efficient decompositions, and (2) Margin-Aware Reinforcement Learning, which provides fine-grained preference optimization for multi-step reasoning beyond standard DPO. Extensive experiments on benchmark datasets demonstrate that LearNAT significantly improves the performance of small-scale LLMs, achieving results comparable to GPT-4 with only a 7B parameter model. These results validate the effectiveness of verifiable decomposition and fine-grained preference learning in advancing NL2SQL towards openness, transparency, and efficiency.
Our code is publicly available at https://github.com/MrBlankness/LearNAT.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents LearNAT, a framework designed to enhance the performance of small, open large language models (LLMs) on natural language to SQL (NL2SQL) tasks. The method combines two ideas:

1. An AST-guided decomposition synthesis process that uses Monte Carlo Tree Search (MCTS) guided by the SQL Abstract Syntax Tree (AST) of gold queries to generate verifiable intermediate subtasks (sub-SQLs).

2. A Margin-Aware Direct Preference Optimization (MDPO) objective that introduces AST-based structural margins between positive and negative steps, providing fine-grained reward signals without a learned reward model.

Experiments on BIRD and Spider benchmarks demonstrate substantial accuracy improvements for open Qwen2.5-coder models (7B/14B/32B) and notable efficiency advantages over GPT-4-based system-level pipelines. The paper provides ablations, cost analyses, and code release, emphasizing openness and reproducibility.

### Strengths
- **Motivated practical problem:** Tackles a highly relevant challenge — enabling small, public models to achieve competitive NL2SQL performance without expensive test-time pipelines.

- **Strong methodological alignment:** AST-guided decomposition is both interpretable and efficient, providing verifiable supervision that directly matches SQL’s structural nature.

- **Novel preference learning variant:** The margin-aware DPO objective elegantly integrates structured information into preference learning without requiring a learned reward model.

- **Empirical results and cost analysis:** Large gains on BIRD and Spider, along with token-cost comparisons, demonstrate both effectiveness and efficiency.

- **Reproducibility:** Clear method description, ablation studies, and commitment to open code release.

### Weaknesses
### Offline Dependence on Gold SQL ASTs
The synthesis process relies on gold ASTs ($AT(Y)$) for search and reward computation, limiting scalability to unlabeled settings.

### Baseline Comparison Fairness
Model-level and system-level results (e.g., GPT-4 pipelines) are mixed without clear labels or cost normalization.

### Limited Reward-Learning Baselines
MDPO is compared only to vanilla DPO.

### Compute and Cost Transparency
Synthesis and fine-tuning costs are underreported.

### Robustness and Variance Reporting
Main results appear from single runs, which limits reliability.

### Scope Limitation to Canonical ASTs
The method depends on well-defined SQL ASTs, which may not exist in less-structured domains.

### Relation to Concurrent Structured-Reasoning Work
The paper should cite Struct-LLM (Stoisser et al., 2025), which also explores structured reasoning over SQL and Cypher using reinforcement learning. Briefly contrast LearNAT’s offline AST-guided preference learning with Struct-LLM’s online RL-based reasoning approach.

### Questions
### Method Clarity & Assumptions

1. **Gold AST availability**  
   You mention that the decomposition synthesis uses the gold SQL AST to guide MCTS.
   - How does this affect scalability to datasets without gold SQLs?
   - Can LearNAT generate training data in a semi-supervised setting, or does it strictly rely on gold supervision?

2. **Verification signal granularity**  
   You mention “verifiable intermediate subtasks.”
   - Are these subtasks verified purely syntactically (AST match) or also semantically (execution match on DB)?
   - How do you handle equivalent but syntactically different SQL forms?

3. **MDPO stability**  
   - Did you observe training instability compared to vanilla DPO due to margin scaling or structural rewards?
   - Are the AST-based margins dynamically computed or fixed?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper propose LearNAT, a framework that enhances LLMs' ability to decompose complex queries through decomposition synthesis procedure and margin-aware reinforcement learning.  Decomposition synthesis procedure uses AST-guided search and pruning to precede efficient and verifiable decomposition on the BIRD-train dataset for the margin-aware reinforcement learning. Margin-aware reinforcement learning modified DPO's loss function by a AST-based reward distinction between samples. The experiment shows that LearNAT enables 7B-parameter models to reach performance close to GPT-4.

### Strengths
1. The paper introduces a novel approach that leverages ASTs for task decomposition, enabling the synthesis of training data for reinforcement learning.
2. This paper further proposes a modification to the DPO framework by incorporating an AST-distance-based reward to better estimate reward margins and enhance performance on BIRD and Spider datasets.

### Weaknesses
1. The authors acknowledge that although LearNAT does not achieve state-of-the-art performance among system-level approaches, it consumes fewer tokens during inference. However, LearNAT should also be compared against model-level approaches of similar model size, such as Reasoning-SQL and OmniSQL, which demonstrate stronger performance on the BIRD leaderboard.

### Questions
None.

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
5

### Summary
This paper introduces LearNAT, a novel framework that starts from task decomposition for NL2SQL. The core innovation lies in leveraging an AST-guided Monte Carlo Tree Search (MCTS) reasoning framework for efficient reasoning and data synthesis, as well as integrating AST-based structural alignment into the optimization objective to enhance the DPO algorithm. These methods collectively boost the baseline model’s performance by more than 10%.

### Strengths
1. Proposes an AST-guided Chain-of-Thought (CoT) task decomposition and verification mechanism, achieving high controllability and impressive success rates in intermediate process validation.
2. Innovatively improves the DPO algorithm by incorporating AST skeleton contrast in the optimization target, enabling fine-grained supervision of multi-step reasoning.

### Weaknesses
1. The writing lacks clarity, particularly regarding the model inference stage: implementation details, methods used, and specific parameters are not sufficiently described. It remains unclear whether Monte Carlo Tree Search (MCTS) or voting methods were employed during the inference process. Furthermore, the rationale behind the specific parameter settings is not discussed, nor is it specified whether hyperparameter analysis was conducted to optimize the inference performance.
2. The baseline selection in this paper is notably insufficient and lacks relevance. Current comparisons fail to directly target key methods such as DPO [1] and MCTS [2,3] with similar model scales, making LearNAT's claimed advantages difficult to substantiate. Without rigorous and fair evaluations against established approaches, the performance improvements may be unconvincing. The authors must provide more targeted and transparent baseline comparisons to truly demonstrate the superiority of LearNAT.
3. Although the abstract and introduction highlight the heavy test-time computational burden of existing methods, the paper does not explicitly quantify the inference efficiency gains brought by AST-pruned MCTS, nor provide detailed time cost statistics. Supplementary experiments in this regard are recommended.


[1] Uncovering the Impact of Chain-of-Thought Reasoning for Direct Preference Optimization: Lessons from Text-to-SQL

[2] SQL-o1: A Self-Reward Heuristic Dynamic Search Method for Text-to-SQL

[3] Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search

### Questions
1. Given the diversity of SQL queries—where different SQL skeletons result in varying AST structures—how does the AST-guided MCTS handle such cases during data synthesis? Are these instances treated as error trajectories?
2. Can the authors provide a more detailed analysis of sample correctness to elucidate the intrinsic incentives of the improved DPO? Specifically, it would be helpful to demonstrate under what kinds of samples LearNAT’s margin-aware DPO exhibits advantages over vanilla DPO, rather than only presenting final aggregate metrics.

### Soundness
3

### Presentation
2

### Contribution
2
