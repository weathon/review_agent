# Intra-Request Branch Orchestration for Efficient LLM Reasoning

- Decision: Reject
- Scores: 8, 2, 4, 6

## Abstract
LLMs increasingly rely on inference-time reasoning algorithms such as chain-of-thought and multi-branch reasoning to improve accuracy on complex tasks. These methods, however, significantly increase token usage (cost) and per-request latency.
Prior work has primarily focused on reducing token usage, often at the expense of accuracy, while overlooking other latency factors.

We present DUCHESS, an LLM serving system that reduces computational cost and latency without sacrificing accuracy through intra-request branch orchestration guided by predictions.
Within each request, DUCHESS predicts branch correctness with
a lightweight linear probing model over LLM layer activations. The orchestration policy uses these predictions to decide whether to terminate a branch early, duplicate an existing branch, or continue exploring a branch.
When handling multiple requests, DUCHESS can further reduce latency by prioritizing easier reasoning tasks, when request complexity can be estimated from the prompt.

Experiments on three reasoning benchmarks show that DUCHESS consistently improves the token–accuracy Pareto frontier, reducing token usage by 42–63\% at matched accuracy compared to self-consistency. For request serving with vLLM, DUCHESS reduces mean, median, and tail latencies by 57-81\%, 58-85\%, and 52-84\% with First-Come-First-Served (FCFS) scheduling across three datasets, compared to self-consistency. At higher request rates, scheduling jobs by increasing predicted difficulty reduces latency by 25.1-29.7\% further over FCFS.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces an innovative system called DUCHESS (Intra-request branch orchestration for efficient LLM reasoning). It proposes to make Large Language Models (LLMs) faster and cheaper to run, especially when they are solving complex problems using multi-branch reasoning methods (like Self-Consistency or Chain-of-Thought). While these reasoning algorithms help LLMs achieve higher accuracy on difficult tasks, they typically increase the number of output tokens generated, leading to higher computational costs and longer waiting times (latency) for users. The proposed system DUCHESS tackles this challenge by intelligently managing these reasoning paths, or "branches," to maintain accuracy while drastically reducing cost and latency.
The work is primarily based on the following two mechanisms:
1. Intelligent Intra-Request Branch Orchestration (Stopping, Forking, Continuing).
2. Optional Inter-Request Complexity-Aware Scheduling.

*Trying to make the review bullet points for the clarity and easy to respond.

### Strengths
The paper provides strong evidence that the fundamental approach of DUCHESS is sound and highly effective, leading to significant, quantifiable improvements over existing baselines:
1. Dominance on the Cost–Accuracy Pareto Frontier.
2. Efficacy of Lightweight Prediction and Minimal Overhead.
3. Substantial Latency and Straggler Reduction.
4. Optional Complexity-Aware Scheduling.
5. Novel Intra-Request Branch Orchestration Policy.

### Weaknesses
The paper is transparent in listing several limitations, particularly regarding the aggregation of results and the scope of implementation:
1. Dependencies in Answer Aggregation.
2. Limited Scope of Complexity-Aware Scheduling.
3. Predictor Layer Constraints.
4. Low Complexity Predictor Accuracy.
5. Hyperparameter Tuning Range.
6. Scheduling Overhead.

### Questions
A couple of questions here that the paper clearly covers as limitations-
1. About the Enhanced Answer Aggregation and Termination Strategy: I would wish for request termination strategies that explicitly account for dependencies among collected answers.
2. Broader Applicability and Optimization of Inter-Request Scheduling: 
 a. Extension of complexity-aware scheduling to datasets that lack inherent difficulty labels. 
 b. Improvement in the accuracy of the request complexity predictor.
 c. Profiling of alternative LLM layers.
 d. Mitigation of queueing delays caused by prefilling.
3. More Extensive Predictor Tuning and Exploration.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces DUCHESS, an LLM serving system that aims to reduce computational cost and latency in multi-branch reasoning setups. The system incorporates a linear probing model that leverages the transformer model's activations to predict correctness of a reasoning branch. These predictions are utilized by the system to perform branch level operations including 1) early stopping, 2) additional branching, and 3) continued generation. On datasets like GSM8K, MATH, and MMLU, the work demonstrates token length reductions of 42%-63%, and 57%-81% reduction in mean latency in concurrent request serving scenarios.

### Strengths
- The core intuition and methodology are well-grounded in observations from prior research work about:
    - Informativeness of the middle layers of transformer models.
    - Linear probing techniques for extracting signals from prior layers.
    - However, concerns exist regarding hyperparameter selection and generalizability (especially with regards to model architecture families).
- Overall, the paper is well structured with informative diagrams/tables, along with system workflow explanation.The writing is mostly succinct with minimal filler.
- The work showcases results with deployment considerations in mind. Specifically, the throughput numbers (queries per minute) in figure 4, showcase promising results for production environments. 
- Additional component overhead (branch correctness prediction/answer extraction), in terms of latency, provide a confident argument in favor of the added complexity.

### Weaknesses
- The connection between the two main contributions, i.e., 1) intra-request orchestration and 2) inter-request scheduling, feels disconnected. 
- Explanations about the choice of hyperparameter selection is incomplete at times. For example: It is mentioned in line 211 that probing for branch correctness is performed at intervals of $i=16$ tokens, while in line 215, it is mentioned that a suitable range is $16 \leq i \leq 80$. 
- The originality seems somewhat incremental: 
    - The core idea of early termination based on hidden states has been explored (e.g., Zhang et al., 2025; Afzal et al., 2025). 
    - The probing technique, although claimed to be sufficiently advanced compared to prior works (see lines 127-135), does not seem entirely novel. 
    - The complexity-aware scheduling component feels like an add-on rather than a core contribution, and is only evaluated on one dataset (see figure 5).
- The experiments are performed with just one model, i.e., DeepSeek-R1-Distill-Llama-8B. It is unclear if the methods proposed generalize across different models and if the technique applies to non-reasoning models 
- In lines 204-207, it is said that, based on the validation performance of the model on GSM8K, the layer 14 was chosen for correctness prediction. Selection of the prediction layer based on empirical results on a single dataset/model combination raises concerns about applicability in real-world deployment scenarios. 
- It is unclear whether the methods generalize to tasks involving longer reasoning traces (GPQA-Diamond, for example), as opposed to the presently evaluated datasets (GSM8K, MATH, MMLU), which require relatively lower generation lengths.
- Hyperparameter choices such as the probing interval ($i=16$), or the branch termination threshold($\tau$) would benefit from further analysis and/or advice about selection criteria in deployment scenarios where single-dataset validation data might not suffice.
- In lines 148-150, it is mentioned that the termination of entire request, once a "sufficient" number of answers is collected, prevents stragglers from delaying completion. The claim would benefit from further explanation of the so-called "straggler" branches and how the method prevents them. The main concern here is whether the remaining long-running branches pre-empted by the method, are actually wasteful in practice, or they meaningfully contribute towards alternate reasoning path expansion.
- The work motivates the proposed method from the perspective of improving multi-branch reasoning. With this regard, comparison with existing techniques which already attempt to tackle this problem, such as ESC, RASC, Adaptive Consistency would better support the claims.

The work's significance is primarily in its engineering integration and empirical validation rather than algorithmic breakthrough.

### Questions
- Q1. Regarding linear probing models for branch correctness prediction:
(Refer Appendix A.1, lines 710-711) Why are the hidden layer sizes of the MLP different for MMLU vs. GSM8K/MATH datasets?
- Q2. Regarding Experimental Settings:Why is the max token length per branch is capped at 4,096 (see lines 304-305) ?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces DUCHESS, a serving‑time controller for multi‑branch chain‑of‑thought (CoT) inference. A lightweight MLP probe on intermediate activations predicts per‑branch correctness, enabling three actions at decode time: early‑terminate, selectively fork promising branches (reusing KV cache), or continue. A request terminates when a consensus criterion is met. In multi‑request settings, an optional scheduler prioritizes prompts predicted to be easier. Probes are trained using periodic CoT probes (interval i) with mid‑layers performing best; key results report 42–63% token reductions at iso‑accuracy and large mean/tail latency drops under vLLM, further improved by difficulty‑aware scheduling.

### Strengths
Beyond prior early-stopping work in reasoning (e.g., DeepConf), this paper treats early stop as an end-to-end serving problem with a hardware/system-aware design: a lightweight MLP activation probe that runs co-resident with the model to avoid activation transfer, KV-cache-aware selective branching, and request-level consensus with optional SJF-style difficulty scheduling. This coupling yields consistent iso-accuracy token savings (≈42–63%) and sizable wall-clock gains—including mean/P95 latency and TTFT—on vLLM under load.

Consistent token and latency savings at matched accuracy; helpful layer‑wise probe analysis (mid‑layer peak).

### Weaknesses
- Probe seems trained per dataset/model; multi-task/domain robustness is not stress‑tested.
- Comparisons to incremental/entropy‑aware branching (e.g., ESC/RASC) and similarity‑pruning (Slim‑SC) would better calibrate gains.
- The paper reports per‑component costs, but explicit wall‑clock accounting under large batches/multi‑GPU and periodic probing vs. the achieved savings would help.
- Difficulty‑aware prioritization may starve "hard" requests at load; fairness/quality safeguards are not fully analyzed.

### Questions
- Sensitivity to 𝑖 (probing interval), correctness threshold 𝜏, and the consecutive‑round parameter S (for early termination). Any auto‑tuning procedure?
- Does a probe trained on one backbone/size transfer to others, or is per‑model retraining required?
- Ablate early termination vs. branch‑level pruning: what fraction of latency/tokens each contributes?
- Under difficulty‑aware scheduling, how are starvation or fairness handled when many "hard" jobs arrive?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, the authors study the problem of LLM serving system. In particular, they propose an LLM serving system which they call DUCHESS that orchestrates multi-branch reasoning within a single request using predictions from a lightweight probe over LLM activations. DUCHESS takes one of three branch-level actions: 

- (1) Early termination with CoT probing to elicit a short final answer
- (2) Selective branch-out (forking promising branches while reusing KV cache)
- (3) Continuation (keep generating tokens for branches that are still uncertain, giving them additional steps to potentially reach a correct answer)

Beyond intra-request control, the system optionally adds complexity-aware scheduling across a request pool, which using prefill activations of the first decoded token to predict difficulty, the scheduler serves easier prompts first to reduce mean latency.

Experiments on GSM8K, MMLU, and MATH with DeepSeek-R1-Distill-Llama-8B and vLLM, show effectiveness under both cost–accuracy trade-offs (tokens vs. accuracy) and serving latency.

Admittedly, the area of this research is not my primary research area. I apologise that I can only share some general comments on this paper.

### Strengths
``S1``: The empirical gains on both cost–accuracy and serving latency look good, with clear and reproducible setups. Particularly, 

- ``reducing token usages by 42%-63% at matched
accuracy compared to Self-consistency``, 
- ``reducing mean, P50, and
P95 latencies by 57-81%, 58-85%, and 52-84% on three datasets``, and 
- ``reducing mean latency by up to 29.7% and 34.7%
compared to FCFS``.

``S2``: Selective branch-out with KV reuse is a practical systems optimisation that preserves parallelism and reduces stragglers.

### Weaknesses
I apologise that I’m not very familiar with LLM serving systems. I will carefully refer to the comments of other fellow reviewers, who are more confident than me, for my final judgment. Some general weaknesses that I can notice are as follows.

``W1``: The authors currently focus on math plus general knowledge (GSM8K, MMLU, MATH). A broader commonsense and graduate-level domain generalization is in fact, not tested. The inclusion of SuperGPQA and CommonsenseQA would help strengthen the contributions of the proposed LLM serving system DUCHESS.

``W2``: Request termination and voting don't actually account for dependencies introduced by branch duplication. I have noticed that the authors mention this as a limitation at the end of the paper. This opens the door to weighted voting or de-duplication.

### Questions
Apart from those aforementioned in the weaknesses section:

``Q1``: I wonder would selective branch-out reduce reasoning diversity?

``Q2``: The authors report 16.4% and 7.8% mistaken early termination on MMLU/MATH. It would be great if the authors could elaborate the possibility of a detecting and recovering mechanism based on this finding.

### Soundness
3

### Presentation
2

### Contribution
3
