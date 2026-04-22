# DeepRAG: Thinking to Retrieve Step by Step for Large Language Models

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Large Language Models (LLMs) have shown remarkable reasoning capabilities, while their practical applications are limited by severe factual hallucinations due to limitations in the timeliness, accuracy, and comprehensiveness of their parametric knowledge. Meanwhile, enhancing retrieval-augmented generation (RAG) with reasoning remains challenging due to ineffective task decomposition and redundant retrieval, which can introduce noise and degrade response quality. In this paper, we propose DeepRAG, a framework that models retrieval-augmented reasoning as a Markov Decision Process (MDP), enabling reasonable and adaptive retrieval. By iteratively decomposing queries, DeepRAG dynamically determines whether to retrieve external knowledge or rely on parametric reasoning at each step. Experiments show that DeepRAG improves retrieval efficiency and boosts answer accuracy by 25.41%, demonstrating its effectiveness in enhancing retrieval-augmented reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DeepRAG, a framework that enhances retrieval-augmented generation by modeling reasoning as a Markov Decision Process (MDP). DeepRAG dynamically decides when to retrieve external knowledge and when to rely on internal parametric reasoning through iterative query decomposition. By enabling adaptive and context-aware retrieval, it reduces noise from redundant information and improves both retrieval efficiency and answer accuracy—boosting performance by 25.41%—demonstrating significant advances in retrieval-augmented reasoning for LLMs.

### Strengths
* The paper is well written
* When to conduct retrieval is an important topic
* The experiments are extensive

### Weaknesses
* Compared to Serach R1, the paper mainly differs at using the model's native generation ability to determine when to retrieve, but this has already been well studied and the authors do not provide new solution.

### Questions
* Table 1 shows that Qwen 2.5 32B peforms worse than Llama 3 8B, which is confusing for me, could you explain the reason?
* Online RL methods such as GRPO typically outperform offline approaches, a trend observed in the performance of Qwen. However, in the case of Llama, online and offline methods show comparable results. What accounts for this difference？
* In Section 5.3, line 436 states that most questions require 3–5 decomposition steps. However, the majority of queries in HotpotQA and 2Wiki are designed for 2-hop reasoning. Does this imply the presence of significant redundant retrieval? If so, what causes this inefficiency? And do current sota models, such as GPT-5, still exhibit similar tendencies toward redundant reasoning?

### Soundness
3

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
The paper proposes DeepRAG, a retrieval-augmented reasoning framework that formulates iterative query decomposition and retrieval decisions as a Markov Decision Process. It introduces three core components: Binary Tree Search to exhaustively explore parametric and retrieved paths for each subquery and identify minimal-retrieval correct trajectories; Imitation Learning through supervised fine-tuning on these trajectories with masked loss over retrieved tokens; and a Chain of Calibration that includes both offline and online variants to enhance the model’s awareness of its knowledge boundaries. Evaluated on in-distribution datasets such as HotpotQA and 2WikiMultihopQA, as well as out-of-distribution datasets including PopQA, WebQuestions, and MuSiQue, DeepRAG achieves a 25.41 percent accuracy improvement over previous adaptive retrieval-augmented generation methods while also reducing retrieval calls.

### Strengths
- Strong empirical gains: 25.41% accuracy lift is reported across five datasets, with ablation likely showing each stage’s contribution; out-of-dist PopQA and Freebase-absent WebQuestions stress robustness.
- Uses a fixed-depth priority queue (lowest retrieval count first) and discards unsolvable instances, yielding high-quality imitation data without oracle subqueries.

### Weaknesses
- The method doesn’t seem to offer much innovation. Among the many existing approaches that use reinforcement learning for autonomous multi-turn retrieval, I didn’t find any particularly striking or novel insights.
- All experiments assume a fixed Wikipedia retriever (presumably BM25 or Contriever). What about comparison with some deep research methods, they also use multi-turn retrieval?

### Questions
- For WebQuestions (Freebase-backed), what fraction of ground-truth answers are un-retrievable from Wikipedia, and how does DeepRAG-RLon handle verified absence (e.g., explicit “unknown” vs. hallucination)?
- How does the model behave on single-hop factual queries (e.g., PopQA)? Does calibration reduce retrieval to near-zero, and is parametric accuracy preserved?

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
DeepRAG models retrieval-augmented reasoning as a Markov Decision Process, enabling dynamic per-subquery decisions on whether to retrieve external knowledge or rely on parametric knowledge. The framework uses: (1) Binary Tree Search to explore retrieval strategies, (2) Imitation Learning on minimal-cost trajectories, and (3) Chain of Calibration (offline/online) to teach knowledge boundary recognition.

### Strengths
1. Well-Motivated Problem
Existing RAG systems apply retrieval indiscriminately—either over-retrieving (wasting compute) or under-retrieving (missing information). DeepRAG addresses this through atomic decisions: decompose queries into subqueries, then decide retrieval necessity per subquery rather than per original query.
2. Strong Empirical Results

25.41% accuracy improvement over baselines (Table 1)
Better retrieval efficiency: lower average steps than Multi-Step Retrieval (Table 2)
Good generalization on out-of-distribution datasets (PopQA, WebQuestions)
Higher knowledge boundary alignment (MCC scores in Table 3)

3. Comprehensive Ablations
Validates minimal-cost path selection (Figure 6), atomic query decomposition effectiveness (Figure 4: fewer conjunctions/pronouns per subquery), and necessity of calibration stage (RLoff/RLon > Imi).
4. Practical Design
Masked loss over retrieved documents prevents noise learning; end-to-end trainable without separate classifiers; works across different LLM sizes (Llama-3-8B, Qwen-2.5-32B).

### Weaknesses
1. Overstated Technical Novelty
The "MDP formulation" is superficial packaging:

Transitions are deterministic (not stochastic as typical MDPs)
No actual policy search—just supervised learning on pre-computed trajectories
Binary Tree Search is exhaustive enumeration (2^N paths), not a novel algorithm


2. Narrow Evaluation Scope
Only tested on artificially constructed multi-hop QA benchmarks (HotpotQA, 2WikiMultiHop, MuSiQue). Missing evaluations on:

Single-hop questions (does it over-retrieve?)
Real-world queries: "Draft a rejection email," "Compare iPhone vs Samsung," "When will my order ship?"
Non-QA tasks: summarization, dialogue, creative writing

The query patterns in academic benchmarks don't represent real deployment scenarios.

3. Missing Simple Baselines
The paper requires three complex stages but never compares to a one-stage baseline:

### Questions
1. Algorithm Correctness Issue
Algorithm 1 (page 5) has a critical flaw:
pythonLine 7: if IsEqual(o, y) then return h  # Returns FIRST correct path
This implements greedy search, not optimal search. Counterexample:

Path A: [retrieve, retrieve] cost=2 (found first) 
Path B: [parametric, parametric] cost=0 (explored later) 

The paper claims "minimal retrieval cost" but the algorithm returns whichever correct path is found first in priority queue order—not guaranteed optimal.

2. Questionable "Optimality" Claims
Table 1 shows counterintuitive result:

RLoff (trained on "optimal paths" from Stage I): 41.47
RLon (self-exploration): 41.05 (better!)

If Stage I finds optimal paths, why does self-exploration outperform? Possible explanations:

Algorithm 1's bug means Stage I paths aren't actually optimal
RLon discovers better strategies not in training data
GRPO is simply better than DPO for this task

This undermines confidence in the entire Stage I data synthesis process.

3. Reproducibility Concerns
Many critical details missing:

Binary tree search: max depth? timeout? handling of unsolvable queries?
Training time for tree search (exponential in depth)?
Inference latency breakdown (subquery generation overhead)?
DPO hyperparameter β? GRPO rollout count G?

### Soundness
3

### Presentation
3

### Contribution
3
