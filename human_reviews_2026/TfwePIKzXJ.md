# Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
The integration of Large Language Models (LLMs) with automated theorem proving has shown immense promise, yet is constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces BFS-Prover-V2, a step-level theorem proving system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of the LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent system that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. BFS-Prover-V2 achieves 95.08% and 41.4% on the miniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents BFS-Prover-V2, a theorem proving system that uses multi-turn off-policy reinforcement learning and multi-agent tree search to scale both training and inference in LLM theorem proving. The model shows sota empirical results on existing benchmarks such as minif2f and proofnet, showing the strength of using BFS in LLM theorem proving.

### Strengths
This work's strongest motivation would lie in combining multi-turn off-policy RL with planner-guided inference for theorem proving. In specific, the planner-enhanced multi-agent search shows a promising way in theorem proving. The methodology is explained in a very clear and thorough way, making it easy to understand. Empirically, BFS-Prover-v2 achieves high performance, which showcases the strength of best first search.

### Weaknesses
1. One of the largest weaknesses would be the computation. Since it requires large-scale multi-turn and retraining cycles, it would be computationally expensive. The authors did not discuss the computation.

2. Result elaborations are insufficient, with limited interpretation of why specific design choices show observed gains.

3. The evaluations are only done on two datasets (minif2f and proofnet); other datasets such as putnambench, deepseek-prover-bench, and others would strengthen the generalisability.

4. The paper omits ablation studies, making it unclear how much each component (adaptive filtering, retraining, planner) contributes individually.

5. The evaluation metrics are not discussed – it's unclear whether the comparisons are measured in pass@n or other metrics.

### Questions
1. Could the author provide a deeper discussion on the results, such as the proof diversity and error cases and other aspects?

2. Computational efficiency would be a large issue since it requires multi-turn. What were the computational resources (GPUs, hours, and others) used for reproducibility?

3. Would the system maintain performance under reduced computational budgets or smaller-scale models?

4. Are the metrics comparable to other works pass@n evaluations?

### Soundness
3

### Presentation
2

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
This paper proposes BFS-Prover-V2, a step-level automated theorem proving system using LLMs. This system addresses two major challenges in LLM-based provers: training-time scaling (performance plateaus) and inference-time scaling (search complexity).

At training time, it introduces a multi-stage expert iteration inspired by AlphaZero. This includes two novel learning techniques to overcome prover performance plateaus: "adaptive tactic filtering" and "periodic retraining."

At inference time, it employs a "planner-enhanced multi-agent architecture." This is a hierarchical approach where a high-level "planner" agent (a general-purpose LLM) decomposes complex theorems into simple subgoals, and a team of specially trained "prover" agents solves them in parallel.

As a result of experiments, BFS-Prover-V2 achieved state-of-the-art (SOTA) performance as a step-level prover on the miniF2F (95.08%) and ProofNet (41.4%) benchmarks.

### Strengths
1.  **Achieving new SOTA in the Step-Level Approach:**
    This research is significant for achieving clear state-of-the-art (SOTA) performance in the "step-level generation" approach, which is distinct from "whole proof generation" (the one-shot generation of an entire proof).
2.  **Uniqueness and Future Potential of the Approach:**
    The step-level approach may possess distinct advantages over whole proof. Specifically, it could play a crucial role in exploring unknown theorems, particularly through interactive collaboration with proof assistants like Lean or by dynamically changing strategies mid-proof. The AI for Math community needs to continue supporting and evaluating this different approach, and this paper is a strong contribution in that direction.
3.  **Robust Training and Inference Architecture:**
    The "adaptive filtering" and "periodic retraining" introduced to overcome performance plateaus during training are robust engineering solutions for practical issues in long-term RL training. Additionally, the hierarchical decomposition by the planner at inference time, combined with "dynamic replanning" when the prover gets stuck, represents a very powerful paradigm for addressing complex problems.

### Weaknesses
1.  **Limitation of the Expert Iteration Design (Absence of the Planner):**
    The proposed expert iteration (training loop) focuses solely on enhancing the prover's (tactic executor) capability, and lacks a mechanism to strengthen the planner (strategy planner).
2.  **Potential Scaling Bottleneck:**
    In the current architecture, the planner's performance (i.e., its problem decomposition ability) relies on the fixed capability of an external, general-purpose model (Gemini 2.5-pro). However powerful the prover becomes, the planner's performance could become the ultimate bottleneck for the entire system, potentially causing performance to plateau.
3.  **Limited Comparative Benchmark Scores:**
    While SOTA for a step-level prover, the absolute score on miniF2F (95.1%) falls short of the leading whole proof approaches, such as Seed-Prover (99.6%) and Delta-Prover (95.9%).

### Questions
A primary weakness of this paper is that the planner is not strengthened within the expert iteration loop. I would like to ask the authors' views on a more integrated expert iteration design that strengthens the planner in parallel with the prover's performance improvements—for example, by training the planner using success/failure feedback from the prover.

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
In this paper, they proposed a step-level theorem proving system of BFS-Prover-V2 by improving both training-time and test-time scaling. In training-time scaling, they introduced a multi-turn scaling-up strategy of repeatedly expert iteration, data synthesis and retrain the model from the checkpoint. In test-time scaling, they introduced both planner and prover agents: planner as a general-purpose reasoning LLM for task decomposition and prover as an LLM-based tactic generator and multi-agentic system for enhancing reasoning capability in the inference time.

In experiments, it is regrettable that the authors compared their model with a limited number of models in the main result of Table 1. In a first look of the Table 1, the proposed BFS-Prover-V2-32B is best among the step-level prover, while some other models like DeltaProver exploit the lemma decomposition strategy and it can be classified as “step-level” models although it is placed in “Whole-proof provers.” Similarly, some results from the latest models are missing from Table 1 such as Baba et al 2025 and Goedel-Prover-V2. Considering this incompleteness of the main result, I am hesitant to adequately assess the effectiveness of the proposed methodology.

### Strengths
- S1: Interesting improvement in training-time scaling and test-time scaling
- S2: Figure 2 is very interesting: it presents the real training-time performance improvement including the model scaling up.

### Weaknesses
- W1: The overall work seems still on-going: if either training-time scaling or test-time scaling is effective, it can be evaluated via multiple ablation studies. In this work, the effectiveness of the Planner is suggested with the BFS-Prover-V2-32B case while it is not presented within the 7B model. Indeed, Table 1 is the unique final result table in this paper and it is difficult to evaluate the effectiveness of this model only from this table.
- W2: The main table 1 is indeed limited and the statement “Our system sets a new state of the art for LLM step-provers” doesn’t sound plausible. DeltaProver is classified as “Whole-proof provers” and some import results are missing.
- W3: The overall writing can be improved. Before I reach the final results at the end of the paper, I see many intermediate results that many readers would like to see after the main result. This may prevent readers from grasping the overall contribution of this paper.

### Questions
- Q1: Do you have some specific reason why you have classified DeltaProver as whole-proof provers?
- Q2: It seems that you have scaled up the model from 7B to 32B at the 13th expert iteration. Do you have some clarification or specific reason for this?

### Soundness
2

### Presentation
2

### Contribution
3
