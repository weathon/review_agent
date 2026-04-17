# ACON: Optimizing Context Compression for Long-horizon LLM Agents

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations.
This expansion raises memory costs and reduces token efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications.
We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations.
ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly.
Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module.
Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Agent Context Optimization (ACON), a framework designed to address the challenge of ever-growing context length for LLM agents operating in long-horizon tasks. The core contribution is a novel, gradient-free method for optimizing context compression guidelines. This is achieved by analyzing pairs of trajectories—one with full context that succeeds and one with compressed context that fails—using a powerful LLM to identify the causes of failure and iteratively refine the compression prompt. The authors also propose distilling the optimized compressor into a smaller, more efficient model. Experiments conducted on three benchmarks (AppWorld, OfficeBench, and Multi-objective QA) demonstrate that ACON can significantly reduce peak token usage (26-54%) while largely preserving task performance, and in some cases, even enhancing the capabilities of smaller agent models.

### Strengths
Originality and Significance: The paper tackles a highly significant and practical problem for the advancement of LLM agents: context management. The proposed method for optimizing compression guidelines is novel and clever. Using the task outcome (success vs. failure) as a supervisory signal in a gradient-free, natural language optimization loop is an original approach that is broadly applicable, even to closed-source API-based models.


Clarity: The paper is exceptionally well-written and clearly structured. The problem, the proposed solution, and the experimental results are all explained with high clarity. The figures, particularly Figure 1 and 3, are effective at illustrating the core trade-offs and the optimization mechanism.

Empirical Rigor: The experimental evaluation is comprehensive, covering three distinct and challenging long-horizon benchmarks. The results are strong, showing that ACON not only maintains performance close to the "no compression" upper bound but also significantly outperforms other compression baselines, which often suffer from severe performance degradation.

### Weaknesses
Practical Cost vs. Token Efficiency: The primary weakness lies in the trade-off between peak token reduction and overall computational/API cost. The paper itself acknowledges this limitation in Section 4.5. While ACON successfully reduces the maximum context length (peak tokens), the process of history compression (which involves frequent calls to the compressor LLM) can break the KV-caching mechanism of the agent LLM. This forces re-computation and can lead to a higher total number of tokens processed and thus higher API costs, as shown in Figure 7. This is a significant practical drawback that might limit the adoption of the history compression part of the framework where cost, not just memory, is the main concern.

Cost and Scalability of the Optimization Process Itself: The paper details the effectiveness of the ACON framework but does not sufficiently discuss the "meta-cost" of the guideline optimization process. This process requires running multiple full trajectories (both with and without compression) and then using a powerful "optimizer" LLM for analysis. For new, complex domains, this optimization phase could be prohibitively expensive and time-consuming. The scalability of this approach to a wide variety of new tasks without incurring substantial upfront costs is unclear.

Generalizability of Optimized Guidelines: The experiments show that guidelines optimized on a specific benchmark's training set work well on its test set. However, the generalizability of these highly specialized guidelines across different domains remains an open question. For instance, would a guideline optimized for AppWorld's application-based tasks be effective for a radically different domain like code generation or scientific literature review? The paper could be strengthened by including an experiment that tests this cross-domain transferability.

### Questions
Regarding the critical issue of computational overhead: Could the authors provide a more detailed analysis of the trade-off between peak token reduction and total API cost, especially for history compression? For what types of tasks or interaction patterns does the cost-saving from reduced context outweigh the extra cost of the compressor calls and KV-cache invalidation?

Could the authors elaborate on the cost and complexity of the guideline optimization phase? What is the estimated computational cost (e.g., in terms of number of LLM calls or GPU hours) required to generate an optimized guideline for a new benchmark of similar complexity to AppWorld?

The quality of the optimized guideline seems to depend heavily on the capability of the "optimizer" LLM (O3 model in this case). How sensitive is the final guideline quality to the choice of this optimizer? If a less capable model (e.g., gpt-4.1-mini) were used as the optimizer, would the process still yield significant improvements, or does ACON fundamentally rely on having access to a state-of-the-art reasoning model for optimization?

### Soundness
3

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
Large language models serve as agents in dynamic environments where they accumulate extensive interaction histories, leading to increased computational costs and inefficiencies in long-horizon tasks. The motivation arises from the need to compress these growing contexts effectively, as prior methods primarily address single-step or domain-specific scenarios and fail to preserve essential multi-step signals. Challenges involve retaining diverse information such as states, causal relations, preconditions, and decision cues across heterogeneous tools without losing critical details. ACON introduces a unified framework that optimizes compression guidelines through natural language failure analysis and distills them into smaller models to achieve efficient, informative condensations.

### Strengths
1. ACON reduces peak tokens by 26-54% while preserving or enhancing task performance. This efficiency stems from targeted compression that eliminates redundancies without sacrificing key information. Agents can thus handle longer horizons more cost-effectively.

2. The guideline optimization leverages contrastive feedback from successful and failed trajectories. This process refines prompts in natural language space to better capture task-specific needs. As a result, compression becomes more adaptive and effective across diverse environments.

3. Experiments demonstrate consistent gains on AppWorld, OfficeBench, and Multi-objective QA benchmarks. These validations cover varied domains like productivity and question answering. The broad applicability underscores the framework's robustness.

4. ACON improves smaller agents' performance by 20-46% by mitigating long-context distractions. Concise summaries focus reasoning on essential details. This equalization empowers less capable models to tackle complex tasks.

5. The method operates gradient-free, making it suitable for API-based LLMs. No parameter updates are required during optimization. This flexibility supports integration with proprietary systems.

### Weaknesses
1. The optimization phase demands collecting feedback from multiple trajectories. This requires significant upfront computation for contrastive pairs. Deployment in time-sensitive scenarios becomes challenging.

2. Benchmarks are simulated and may not capture real-world variability. Unforeseen environmental changes could degrade performance. Broader testing in live settings is essential.

3. Distillation incurs a minor performance drop despite high retention. Critical applications risk failures from lost nuances. Enhanced techniques to minimize this gap are necessary.

4. Thresholds for invoking compression need per-benchmark tuning. Suboptimal values lead to either excessive calls or insufficient reduction. This hyperparameter dependency complicates usage.

5. Comparisons omit some recent agent-specific compression methods. Relative advantages remain unclear without these baselines. Expanding evaluations could better position ACON.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This authors proposed a framework called ACON designed to reduce the computational cost of LLM agents for long-horizon tasks. The authors identify the growing context length due to accumulated histories of actions and observations as a key obstacle to efficiency. ACON tackles this by introducing a compression guideline optimization that learns how to summarize and retain essential information across the steps in long horizon tasks through a contrastive, min–max formulation. The authors also experimented with distillation of the learned compressor into smaller models using LoRA fine-tuning. Experiments on three benchmarks of AppWorld, OfficeBench, and multi-objective QA show improvements in task success rates and moderate reductions in peak input tokens. However, while ACON achieves better reasoning stability, the actual efficiency gains in terms of total token usage and runtime cost are less convincing.

### Strengths
* Formulation of context compression as learning problem: ACON elegantly formulates context compression as a contrastive optimization problem. By pairing successful trajectories with failed ones after compression, it directly trains the model to preserve information that determines success. This min–max objective formalizes what to keep and what to drop in a principled way, moving beyond rule-based or heuristic memory truncation.
* Clear and rigorous methodological description: The paper provides a detailed explanation of the compression guideline optimization process. The use of LLM-as-a-judge evaluation for multiple candidate guidelines, iterative feedback generation, and adaptive prompt selection is described with strong clarity. This makes the method reproducible and highlights the thoughtfulness of the design.
* Exhaustive evaluations with multiple benchmarks: The authors conducted experiments on three distinct benchmarks under varying conditions and provided detailed analyses to understand various aspects of the proposed framework. Evaluations on three distinct long-horizon agentic benchmarks demonstrate consistent improvements in accuracy and moderate token reductions. The inclusion of both full-scale and distilled compressors supports the framework’s flexibility and practical deployment value.
* Strong contribution to reasoning stability: Even though ACON’s original goal was efficiency, its most significant contribution appears in reasoning stabilization. Compressed and structured contexts improve coherence in long-horizon planning, reducing redundant exploration and logical drift in LLM agents.

### Weaknesses
* Limited resolution of the claimed efficiency problem: Although the paper motivates ACON as a solution to computational inefficiency caused by long contexts, experiments show that overall token usage and runtime cost did not decrease significantly. In fact, repeated compressor invocations increased API calls, and the authors acknowledge that execution latency rose. The framework thus enhances task performance but not genuine efficiency
* High optimization cost in the guideline learning phase: The compression guideline optimization is extremely expensive, involving iterative LLM calls across the full D_cont dataset. With 20–25 candidate prompts per iteration and multiple iterations, the process may require hundreds of thousands of API calls and many hours of training time. The paper admits this cost but omits quantitative measurements, treating it as an offline overhead. This weakens the practicality of ACON for large-scale or domain-adaptive deployment. The authors argue that guideline optimization is performed once per domain and reused across tasks. However, in realistic multi-domain settings, new environments would require data collection and retraining that reintroduces the same heavy computational overhead. This limits ACON’s scalability and adaptability for general-purpose agent systems.
* Distillation effect Is limited: The distillation step offers only marginal performance gains. The paper itself notes that even GPT-4.1-mini without distillation performs comparably to distilled small models. Hence, the true utility of distillation lies in cost reduction rather than learning transfer, making its contribution modest.
* Guideline optimization depends heavily on heuristic search: Although the optimization is presented as learning-driven, it fundamentally relies on prompt-based heuristic exploration with LLM-as-a-judge feedback. This process lacks theoretical guarantees of convergence or optimality and may depend heavily on model biases and dataset idiosyncrasies.

### Questions
Please refer to the Weaknesses section to address the raised issues.

### Soundness
1

### Presentation
3

### Contribution
2
