# A State-Transition Framework for Efficient LLM Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
While Long Chain-of-Thought (CoT) reasoning significantly improves Large Language Models (LLMs) performance on complex reasoning tasks, the substantial computational and memory costs of generating long CoT sequences limit their efficiency and practicality.
Existing studies usually enhance the reasoning efficiency of LLMs by compressing CoT sequences.
However, this approach conflicts with test‑time scaling, limiting the reasoning capacity of LLMs.
In this paper, we propose an efficient reasoning framework that models the reasoning process of LLMs as a state‑transition process.
Specifically, we first apply a linear attention mechanism to estimate the LLM’s reasoning state, which records the historical reasoning information from previous reasoning steps.
Then, based on the query prompt and the reasoning state, the LLM can efficiently perform the current reasoning step and update the state.
With the linear attention, each token in the current reasoning step can directly retrieve relevant historical reasoning information from the reasoning state, without explicitly attending to tokens in previous reasoning steps.
In this way, the computational complexity of attention is reduced from quadratic to linear, significantly improving the reasoning efficiency of LLMs.
In addition, we propose a state-based reasoning strategy to mitigate the over-thinking issue caused by noisy reasoning steps.
Extensive experiments across multiple datasets and model sizes demonstrate that our framework not only improves the reasoning efficiency of LLMs but also enhances their reasoning performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an efficient reasoning framework for Large Language Models (LLMs) that addresses the high computational and memory costs associated with long Chain-of-Thought (CoT) reasoning. Unlike prior work that compresses CoT sequences—potentially limiting reasoning capacity and conflicting with test-time scaling—the authors model the reasoning process as a state-transition system.

The key idea is to maintain a compact reasoning state using a linear attention mechanism, which summarizes historical reasoning information. At each step, the model generates the next reasoning segment based on the current query and this reasoning state, rather than attending to the full CoT history. This allows each token to efficiently access relevant past information without the quadratic complexity of standard attention, reducing computational cost to linear time. Experiments across multiple benchmarks and model sizes show that the proposed method improves both reasoning efficiency and performance compared to standard CoT and other efficient reasoning approaches.

### Strengths
1. The paper proposes a conceptually innovative approach by modeling the LLM reasoning process as a state-transition system, where historical reasoning information is compressed into a compact reasoning state matrix via linear attention. This design effectively decouples reasoning efficiency from CoT length, preserving full reasoning trajectories while avoiding the quadratic attention cost.

2. The proposed state-based reasoning strategy leverages the gradient-descent interpretation of linear attention to compute a global gradient (via momentum) that guides the current reasoning step. This mechanism actively counters the accumulation of noisy or misleading reasoning steps, addressing the over-thinking problem in a principled and trainable manner, which contributes to both improved accuracy and stability in long reasoning chains.

### Weaknesses
1. The major drawback of this paper is the lack of comparison with many relevant baseline methods. These baselines are not included in the experimental comparisons.

[1] Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning

[2] Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging

[3] L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning

[4] Adaptive Group Policy Optimization: Towards Stable Training and Token-Efficient Reasoning

[5] Concise Reasoning via Reinforcement Learning

[6] SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning

[7] Plan and Budget: Effective and Efficient Test-Time Scaling on Large Language Model Reasoning

[8] Not All Tokens Are What You Need In Thinking

[9] Stable Reinforcement Learning for Efficient Reasoning

[10] Just Enough Thinking: Efficient Reasoning with Adaptive Length Penalties Reinforcement Learning

[11] Optimizing Anytime Reasoning via Budget Relative Policy Optimization

[12] Making Small Language Models Efficient Reasoners: Intervention, Supervision, Reinforcement

2. The paper relies on a large amount of data (95K) for fine-tuning, which suggests a clear issue of data inefficiency compared to RL-based methods that incorporate length penalties. The authors should provide a detailed discussion of this limitation in the paper.

3. The performance improvement is marginal. In terms of token efficiency and compression, there is no significant gain. Moreover, the results on AIME24 and AIME25 are based on a single run, which introduces considerable randomness and undermines the reliability of the evaluation.

4. The experimental results appear to be sensitive to hyperparameters, yet the paper does not include a joint analysis or visualization (e.g., a heatmap or contour plot) of the effects of key hyperparameters such as $\alpha$ and $\beta$. Such an analysis would strengthen the validity and reproducibility of the findings.

### Questions
During the long chain-of-thought reasoning process, for certain questions, the model sometimes exhibits repetitive generation, leading to hitting the maximum length without producing a final answer. These cases significantly increase inference length while still failing to solve the problem. I would like to know to what extent the performance gain of the proposed method stems from mitigating such repetitive generation behavior.

### Soundness
2

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
This paper aims to alleviate the computational and memory burden of long-chain CoT reasoning: the authors explicitly model LLM reasoning as “state–transition,” using linear attention to maintain a cross-step “reasoning state” so that each token in the current step can directly retrieve historical essentials from this state instead of re-attending to all tokens from previous steps; meanwhile, the softmax-attention branch only attends to the prompt and the current-step prefix, reducing attention complexity from quadratic to linear, together with a “state-based reasoning strategy” to mitigate noisy steps and overthinking. The abstract claims that experiments across datasets and model scales show not only substantial efficiency gains but also improvements in reasoning performance.

### Strengths
1. The problem is well-targeted and the motivation is clear. The paper tackles the latency and memory blow-up of long CoT reasoning: by restricting the SA branch to “prompt + current step” and introducing an LA branch to maintain a “historical reasoning state matrix,” it reduces attention complexity from quadratic to linear and the KV cache from linear to near-constant. The exposition is clear and technically coherent.

2. The method is novel and modular in practice. The proposed Mixed Attention Module (MAM) replaces standard attention with parallel SA (local, current-step) and LA (global, cross-step state) branches; tokens in the current step fetch historical essentials directly from the “state blackboard,” without re-reading all past tokens. The idea is natural, and implementation is compatible with existing Transformer interfaces.

3. Experiments are broad and the gains are significant. (1) Accuracy: Outperforms a range of efficient-reasoning and KV-compression baselines across multiple benchmarks. (2) Efficiency: Advantages become clear when CoT > 4K; at 32K, inference is accelerated by over 40%, and memory usage remains approximately constant with length, in line with the theoretical claims.

### Weaknesses
The methodological description is not sufficiently clear. I recommend adding a schematic of the attention matrices to reduce the reader’s cognitive load. In addition, I recommend including pseudocode or a diagram in the main text for both the training and inference procedures of the MAM method.

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework that models the reasoning process of LLMs as a state-transition process, framing it as a sequence of state evolution. Specifically, the paper designs a Mixed Attention Module (MAM), which incorporates the LLM's original attention module alongside a linear attention module, to address the computational and memory efficiency issues associated with CoT reasoning. The proposed framework is evaluated on test sets spanning mathematical, scientific, and code reasoning tasks. The results demonstrate that this framework significantly enhances the efficiency of LLM reasoning.

### Strengths
1. In terms of research motivation, the focus of this paper is highly significant. While CoT enhances LLM performance on complex reasoning tasks, it also incurs substantial computational and memory costs. Current academic approaches to this efficiency problem often employ prompting, supervised fine-tuning (SFT), or reinforcement learning (RL) to compress CoT, which can lead to the loss of critical information. This paper innovatively addresses this issue, aiming to improve LLM reasoning efficiency while minimizing information loss.

2. In the design of the framework, to prevent performance degradation that might arise from directly replacing the LLM's original attention module, the MAM retains it. This design facilitates a division of labor between different attention mechanisms, allowing them to work collaboratively.

3. Regarding the experimental results, the model's performance is comparable to other baseline models with shorter CoT lengths. However, its performance significantly surpasses other baselines when the CoT length exceeds 4K. When the CoT length reaches 32K, the model's reasoning speed is even over 40% faster than other baseline models. These experimental results strongly validate the effectiveness of the proposed MAM framework.

4. Regarding the research outlook, the MAM proposed in this paper significantly enhances the reasoning efficiency of LLMs in CoT. Looking forward, as the complexity of tasks assigned to LLMs increases, requiring longer CoT to maintain high performance for complex reasoning, the consumption of computational and memory resources becomes particularly critical. Therefore, the focus of research on this paper is innovative and holds substantial practical significance.

### Weaknesses
1. The framework relies on segmenting long CoT sequences. The paper does not elaborate on the extent to which this segmentation method is applicable to different types of reasoning tasks or whether it can generalize effectively. Furthermore, it states that all reasoning steps in the training set are clustered, but the specific method used for this clustering is not described. It is also unclear to what extent the different thinking patterns effectively correspond to distinct reasoning types. Concerns remain about whether scenarios analogous to "over-fitting" or "under-fitting" of thinking patterns could occur, rendering them ineffective for different CoT and reasoning tasks, thereby questioning the framework's robustness.

2. The experiments are primarily concentrated in the mathematical domain, making the experimental scope relatively narrow. Only one dataset was used for testing in the scientific and code reasoning domains. Moreover, since the training data is sourced entirely from the mathematics-focused OpenR1-Math-220K dataset, the framework might perform well in mathematics, but its generalization capability to other domains remains questionable.

3. The explanation of the "state-transition process" lacks depth, making it difficult for the reader to gain a clear and comprehensive understanding of its specific working principles.

4. The experimental design lacks sufficient justification for the chosen parameter values. The rationale behind the specific parameter settings is not adequately explained.

### Questions
See the Weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper integrates full attention with linear attention to construct an efficient state-transition-based inference framework. By applying full attention to the current state and linear attention to historical states, the proposed framework effectively reduces the attention burden in long CoT scenarios.

### Strengths
1. The idea of employing a hybrid attention mechanism to achieve efficient reasoning is innovative.

2. Calibrating the current state based on the global state is convincing, and the experimental results demonstrate strong performance.

### Weaknesses
1. The method relies on step-level segmentation, which may limit its applicability to more general tasks.

2. The paper lacks certain implementation details, such as the diversity of thinking patterns and the specific configurations used in LoRA training.

### Questions
1. Could you provide an example to illustrate how diverse thinking pattern samples were constructed?

2. What are the specific LoRA configurations and the size of the state space used in the LA component of the proposed framework? These factors could significantly influence the training cost of the method.

### Soundness
3

### Presentation
2

### Contribution
3
