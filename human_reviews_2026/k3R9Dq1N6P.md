# DeepRepoQA: Code Repository Question Answering with Deep Agent Exploration

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Effectively answering developer questions about a software repository is a critical yet under-explored problem in software engineering. While existing repository understanding methods have advanced the field, they predominantly rely on surface-level code retrieval and lack the ability for deep reasoning over multiple files, complex software architectures, and grounding answers in long-range code dependencies. To address these limitations, we propose DeepRepoQA, a repository question answering (QA) approach in realistic code environments. DeepRepoQA builds on the agentic framework where LLM agents find answers through a systematic tree search over structured action spaces. Our key innovations include balanced exploration and exploitation via Monte Carlo Tree Search (MCTS) for multi-hop repository reasoning and LLM feedback that provides learned priors and values to reduce search depth and reduce drift. The system maintains structured memory paths that enable reliable evidence synthesis and traceable reasoning steps. Comprehensive experiments on SWE-QA demonstrate substantial performance gains over strong baselines, validating the effectiveness of systematic MCTS-guided exploration for multi-hop repository reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper propose DeepRepoQA, a agentic framework where LLMs agents find answers through systematic tree search over structured actions spaces on code repositories. The key innovations include Monte Carlo Tree Search (MCTS) for multi-hop repository reasoning. Specifically the framework encompass four major components: (1) Perception Agent which analyze the current state and propose new actions; (2) Panning Agent which is responsible to turn report from Perception Agent to concreate action plans for node expansion; (3) Execution Agent which utilze a series of tools to execute the actions; (4) Evaluation Agent, which assess the outcome of the action and provide feedback values for MCTS learning. Experiment results on SWE-QA demonstrate improved performance when compared with prior academic work and commercial solutions.

### Strengths
(1) The design of the agentic system seems to correspond well with MCTS. The idea of the Preception Agent to gather information beyond current node (include siblings) is interesting. Other key design components correlate well with the MCTS flow of expansion, selection, simulation and back-propagation.

(2) Results seem solid. Good comparision is done with other SOTA methods. Good ablabtion studies and insights on the core components and hyperparameters (max nodes, max iterations, action usage etc.)

### Weaknesses
(1) MCTS is already widely used and popular to boost efficiency in test-time scaling. Although a solid work, the core idea does not seem particularly exciting and novel. Ablation studies find this core component to be important, but it seems as if removing this component still could achieve compelling strong results.

(2) It is without doubt that scaling inference would likely lead to improved performance such as using MCTS. However, the core results in Table 1 does not provide detailed comparisions on the inference cost (such as inference token costs) between the core methods. It would be also interesting to compare between MCTS and just sampling multiple trajectories under the same tool set and inference cost.

### Questions
(1) Can authors compare against the inference cost between different methods in Table 1? 

(2) Some details are not explained: For example what models do the commercial tools use (Tongyi Lingma and Cursor)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DeepRepoQA, an agentic framework that solves repository level QA problems. DeepRepoQA employs MCTS to systematically explore reasoning paths to solve the challenge of multi-hop, cross-file reasoning. The framework involves 4 specialized agents to solve the QA tasks iteratively. Evaluation shows that the approach can substantially outperform SOTA techniques.

### Strengths
1. Novel approach. The paper positions repository-level QA problems as an MCTS planning problem, showing its novelty. Different from prior techniques like ReAct (single-path greedy search) and simple RAG, DeepRepoQA appears to be a more principled way to handle complex, multi-hop reasoning.
2. Strong empirical results. The paper conducts an extensive evaluation, substantially outperforming the baselines. Especially, DeepRepoQA outperforms the baselines by a lot in terms of completeness and reasoning, consistent with the claim that MCTS-guided exploration leads to more comprehensive and reliable answers.
3. Strong ablation analysis. The paper performs solid ablation analysis to show the effectiveness of the designs. The ablation shows that the main design, i.e., the MCTS design, significantly contributes to the performance of DeepRepoQA.

### Weaknesses
1. The mechanism for MCTS value estimation is ambiguous, especially the use of $V_{\theta}$ notation.
2. Token cost analysis is missing, making it hard to evaluate the cost-effectiveness of DeepRepoQA.
3. Overreliance on LLM-as-judge. The framework heavily relies on the LLM-as-judge methods but does not discuss the potential threat introduced by LLM-as-judge.
4. Vague Termination and Answer Synthesis Logic. The process for ending the search is not well-defined.

### Questions
1. The paper's description of the "Simulation" phase (Section 3.5) and the "Evaluation Agent" (Section 3.4) is ambiguous. The agent is supposed to estimate a scalar value $V_{\theta}(s,a)$, where the $\theta$ notation implies a learned model. However, no training process or dataset for this value function is described. Is this agent a fine-tuned model? Or is it, in fact, a zero-shot LLM prompt providing a heuristic score (akin to the LLM-as-a-Judge)?
2. Why token cost analysis is not performed? How expensive is this MCTS-based approach? Since the looks like a test-time scaling for repo-level QA problems, and in stages like perception it needs to summarize and produce a concise situational report from the direct trajectory and sibling attempts.
3. Could LLM-as-judge introduce some bias to the evaluation? The LLM-as-judge is used prevalently in evaluation. Prior works show some threats of using LLM-as-judge for scoring[1] [2][3][4][5].
4. When exactly will the search end? The paper states that termination occurs when the "Finish" action is selected by the MCTS loop. This suggests Finish is just another action (like FindClass or ViewCode ) proposed by the Planning Agent. It is unclear how the agent learns when it has gathered so-called "sufficient evidence" to propose this action, or how the Evaluation Agent scores it to make it a winning choice. An early Finish action would lead to an incomplete answer, while a late Finish would waste resources.

[1] Krumdick, M., Lovering, C., Reddy, V., Ebner, S., & Tanner, C. "No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding." arXiv:2503.05061.
[2] Shi, L., Ma, C., Liang, W., Diao, X., Ma, W., & Vosoughi, S. "Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge." arXiv:2406.07791 (June 2024, rev. December 2024).
[3] The Collective Intelligence Project. "LLM Judges Are Unreliable.".
[4] Szymanski, A., Ziems, N., Eicher-Miller, H. A., Li, T. J., Jiang, M., & Metoyer, R. A. "Limitations of the LLM-as-a-Judge Approach for Evaluating LLM Outputs in Expert Knowledge Tasks." arXiv:2410.20266.
[5] Ye, J., Wang, Y., Huang, Y., Chen, D., Zhang, Q., Moniz, N., Gao, T., Geyer, W., Huang, C., Chen, P., & Chawla, N. V. "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge." The Thirteenth International Conference on Learning Representations (ICLR).

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of answering developer questions about software repositories. It introduces DEEPREPOQA, an agent-based framework that performs multi-hop reasoning over codebases. The approach using Monte Carlo Tree Search guided by LLM feedback to balance exploration and exploitation.

### Strengths
- The paper explores the under-explored problem of RepoQA, with well-designed experiments (benchmark, metrics, and models) that provide useful references for future research in this area.
- The paper is well-structured and easy to follow.

### Weaknesses
- MCTS introduces significant latency, which may make it unsuitable for QA scenarios.
- The paper only compares against SWE-QA-Agent, while other agent-based methods such as OpenHands and SWE-Agent could also be applied to RepoQA. Comparing with these methods would better demonstrate the effectiveness of the proposed approach.

### Questions
- How does DeepRepoQA perform in terms of reasoning efficiency compared to the baselines?
- How did the authors compare with commercial tools like Cursor? Were the queries performed manually on Cursor? It would be better to include the version number used to ensure reproducibility.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents DeepRepoQA, an agentic system for repository level question-answering tasks. It combines four sub-agents (perception, planning, execution, and evaluation) within an MCTS loop to navigate through a repository to collect information to produce a final answer. Empirical evaluations on the SWE-QA benchmark shows DeepRepoQA works well with different models and produces competitive results compared to both open-source and commercial baselines.

### Strengths
1. The paper is well-written and easy to follow. Each sub-agent’s task is clearly divided so that it is simple to understand and evaluate its functionality in the overall pipeline.

2. Experimenting with three different LLMs showcase the generality of the DeepRepoQA system.

### Weaknesses
1. I think the paper would be greatly strengthened with better empirical evaluations. There are two weaknesses related to the SWE-QA benchmark. First, it is constructed from the 12 Python repositories that SWE-bench uses. Recent LLMs have data leakage issues on those repositories. Using more contamination-free datasets such as SWE-bench-Live [1] could help reduce this leakage issue. Second, the final scoring is based on LLM-as-a-judge, which can be biased by response length [2]. DeepRepoQA utilizes MCTS to iteratively collect information to produce a final answer. Compared to direct prompting or RAG-based baselines, it might produce longer answers and the empirical section does not mention controlling final answer length. It would be useful for the authors to comment on this.
Furthermore, it would be useful to have some human correlation data to confirm LLM-as-a-judge does provide sensible evaluation results.

[1] Zhang, Linghao, et al. "SWE-bench Goes Live!." arXiv preprint arXiv:2505.23419 (2025).
[2] Hu, Zhengyu, et al. "Explaining length bias in llm-based preference evaluations." arXiv preprint arXiv:2407.01085 (2024).

2. DeepRepoQA represents an inference-scaling approach to the QA problem. Comparing it to baselines such as direct prompting hides the computational cost. Can the authors comment on how different methods trade off computational cost and accuracy?

### Questions
Please see the weakness section.

Additional question: can DeepRepoQA be applied to repositories in languages other than Python?

### Soundness
2

### Presentation
3

### Contribution
2
