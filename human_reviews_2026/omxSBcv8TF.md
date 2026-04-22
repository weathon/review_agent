# Teaching Language Model to Act Efficiently

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Tool-integrated reasoning (TIR) augments large language models (LLMs) with the ability to invoke external tools during long-form reasoning, such as search engines and code interpreters, to solve tasks beyond the capabilities of internal reasoning.
While reinforcement learning (RL) has shown promise in training such agents, most of existing approaches typically optimize only for final correctness without considering the efficiency or necessity of external tool use. This often leads to excessive tool calling, leading to increased computational costs and additional latency, and may also shift reliance toward external tools rather than the model’s own reasoning -- a phenomenon referred to as \textit{cognitive offloading}. To this end, we propose Optimized Tool Call-controlled Policy Optimization (OTC-PO), a simple yet effective RL-based framework that encourages models to produce accurate answers with less tool calls. Our method introduces a tool-integrated reward that jointly considers answer correctness and corresponding tool use behavior of model to reach that answer. To validate the effectiveness, we introduce the additional metric of \textit{tool productivity}, defined as the ratio between the number of correct answers and the total number of tool calls across all test cases. This metric reflects how efficiently and effectively tool usage contributes to successful task completion, with higher values indicating more productive external tool calls with the help of internal reasoning. We then instantiate this framework within both Proximal Policy Optimization (PPO) and Group Relative Preference Optimization (GRPO), resulting in OTC-PPO and OTC-GRPO. Experiments with Qwen-2.5 and Qwen-Math across multiple QA benchmarks show that our approach reduces tool calls by up to 68.3\% and improves tool productivity by up to 215.4\%, while maintaining comparable answer accuracy, especially for the larger models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The authors investigate tool-integrated reasoning large language models, and introduce the Optimized Tool Call-controlled Policy Optimization (OTC-PO) method to handle the cognitive offloading issue. The core to OTC-PO is a novel reward function that relates to tool calls. By using the proposed reward function, OTC-PO  not only improves the performances but also reduce the number of tool calls.

### Strengths
- The paper is well written and easy to understand.
- The authors provide sufficient motivation for the proposed reward function.
- Extensive experiments are conducted to validate the effectiveness of **OTC-PO**.

### Weaknesses
- The reviewer would like to know the practical cost of tool invocation. According to **Table 1**, **Search-R1-PPO** calls the tool up to three times, while **OTC-PPO** also requires at least one call. What is the actual time savings achieved in this case? This investigation is essential to assess the real-world significance of the proposed method.
- The reviewer suggests conducting experiments on larger-scale models, if computational resources permit, to further verify the scalability of the approach.
- An important related work on **tool-augmented LLMs** [1] is missing from the discussion in the paper.

[1] Advancing Tool-Augmented Large Language Models: Integrating Insights from Errors in Inference Trees. NeurIPS 2024.

### Questions
See Weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces OTC-PO, an RL framework that trains tool-using LLM agents to optimize both answer correctness and efficiency of tool use. The key idea proposed by the authors is a tool-integrated reward that multiplies a correctness term with a tool-efficiency coefficient that favors trajectories reaching correct answers with fewer tool calls.

Tool-integrated agents increasingly dominate practical workflows. Optimizing only EM produces agents that over-call tools, inflate latency, and offload trivial reasoning to external systems. This paper reframes training as a compute-aware objective and introduces a concrete, RL-compatible shaping that pushes models toward judicious tool use without hand-crafted rules or heavy SFT. The result is a simple, plug-and-play recipe that system builders can directly try in search and code agents.

### Strengths
The multiplicative tool-efficiency factor is easy to add to existing RL pipelines, and the GRPO variant provides a data-driven estimate of the “optimal” call budget per question. This is engineering-friendly and aligns with system constraints. 

The paper shows large TC reductions and strong TP gains on both search and code tool settings, including OOD QA, while preserving accuracy on larger models. This is the right way to argue for deployment. 

The case study and the “reasoning versus acting” analysis are insightful, demonstrating that discouraging gratuitous tools increases internal reasoning, which is exactly the intended effect.

### Weaknesses
EM and TP are informative, but many decisions hinge on end-to-end cost. Please add wall-clock latency, prompt token counts including tool observation length, and number of forward passes so that compute is matched across baselines. Some comparisons could still be explained by budget differences. 

The reward uses alpha and a smooth constant c. Provide ablations showing stability, convergence speed, and EM-TP trade-offs as these vary, ideally on two datasets and two backbones. The text hints at PPO versus GRPO stability but does not quantify failure rates or variance. 

The “minimal calls among correct trajectories” is a moving target. Analyze how often the proxy changes during training, whether it biases toward under-search, and how robust it is to spurious lucky trajectories. A small controlled synthetic task where the true optimum is known would help.

### Questions
(1) Under equalized prompt tokens, number of model calls, and observation lengths, how do OTC-PPO and OTC-GRPO compare to Search-R1 and ToRL on EM and TP, including wall-clock? A clean, matched table would increase confidence that gains are not budget artifacts. 

(2) Show how often the estimated n changes per question during training, and the effect on stability. Include a diagnostic on under-search errors created by too-aggressive penalties. 

(3) Can you demonstrate the same shaping works with larger tool menus or multi-tool plans, for example search, code, and calculator together, where the notion of “optimal calls” may be more complex. A small ablation would help.

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
This paper proposes Optimized Tool Call-controlled Policy Optimization (OTC-PO), a RL framework that trains language models to use external tools more efficiently while maintaining accuracy. It introduces a new reward design and metric, tool productivity, which balances correctness with the number of tool calls. Experiments on multiple benchmarks with Qwen models show that OTC-PO reduces tool calls by up to 68% and improves tool productivity by over 200%, without significant loss in performance.

### Strengths
- OTC-PO can be plugged into existing RL pipelines with minimal modification. Its reward design is lightweight yet conceptually impactful.
- Evaluated on multiple tasks (search and code), datasets, and model scales (3B–7B), showing consistent improvements.

### Weaknesses
- The main novelty lies in the reward design, specifically the cosine-decay modulation to penalize excessive tool use. While effective, this idea feels heuristic and somewhat ad hoc. It would be stronger if compared with or generalized to other decay strategies (e.g., linear, exponential, adaptive).
- The paper lacks comparisons with existing length-control methods such as Length-Control Policy Optimization (LCPO) or budgeted reasoning RL approaches.
-

### Questions
- Could the authors justify why cosine decay is preferable to simpler alternatives (e.g., linear, exponential, or adaptive decay)? Including ablations with different decay types or visualizations of reward curves would help clarify robustness.
- In Section D, why do the authors revise the system prompt of Search-R1 by adding the instruction “You need to make every search call count and gain helpful results”? What motivates this modification, and does it influence the accuracy, training stability, or comparability of baselines?
- In Table 1, OTC-GPRO shows huge improvement over Search-R1-GRPO. Could the authors explain it?

### Soundness
3

### Presentation
2

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
This paper introduces Optimized Tool Call-controlled Policy Optimization (OTC-PO), a simple yet effective RL-based framework that encourages models to produce accurate answers with less tool calls. Tool integrated reasoning is framed as jointly optimizing both correctness and efficiency by modulating the reward signal with a scaling coefficient that reflects tool efficiency. Two instantiations of the framework, OTC-PPO and OTC-GRPO, are extensively evaluated on two tools namely search and code to validate the effectiveness.

### Strengths
- The paper is generally clear and well structured. Figure 1 is a good illustration of the proposed method.
- The proposed method is simple yet effective and is compatible with various RL methods.
- Conducted experiments validate the effectiveness in reducing total tool calls with minimal to no loss of efficacy.

### Weaknesses
- The technical contribution of OTC-PO centers on scaling the correctness reward by a tool-efficiency coefficient that penalizes trajectories with a larger number of tool calls. While the prompt-adaptive estimation of minimal tool usage is a useful heuristic, conceptually it is an incremental extension of standard length penalties and reward shaping techniques for tool-efficiency.

- A follow-up to Search-R1 [1] demonstrates that incorporating a format reward can substantially improve performance, particularly when training from base LLMs. For example, using Qwen2.5-3B-Base, Search-R1 with a format reward reaches **0.428** (NQ) and **0.371** (HotpotQA) under PPO, and **0.429** (NQ) and **0.372** (HotpotQA) under GRPO. With Qwen2.5-7B-Base, the PPO variant attains **0.488** (NQ) and **0.436** (HotpotQA), while the GRPO variant achieves **0.458** (NQ) and **0.412** (HotpotQA); refer to Table 1 in [1]. These results are competitive with, and in some settings exceed, the performance of OTC-PO variants. A more extensive comparison against such methods would better contextualize the contribution.

- The paper reports single-run results and notes that hyper-parameters are reused from Search-R1. However, LLMs are known to exhibit high run-to-run variance, and sampling parameters can significantly influence outcomes. Without repeated trials and confidence intervals, it is difficult to assess the stability and practical significance of the reported improvements.




---

References


[1] Jin et al., An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents

### Questions
1. Could the authors clarify the sampling parameters used during inference for all reported experiments? Such details are important for reproducibility and for understanding potential performance sensitivity to decoding settings.

2. Minor typos. 
Line 106: “Only few of studies” should be revised to “Only a few studies.” Line 157: “generate outputs consists” should be revised to “the generated output, consisting.”

### Soundness
3

### Presentation
3

### Contribution
2
