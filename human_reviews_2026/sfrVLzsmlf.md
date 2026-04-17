# AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Long-horizon interactions between users and LLM-based assistants necessitate effective memory management, yet current approaches face challenges in training and evaluation of memory. Existing memory benchmarks rely on static, off-policy data as context, limiting evaluation reliability and scalability. To address these gaps, we introduce AMemGym, an interactive environment enabling on-policy evaluation and optimization for memory-driven personalization.
AMemGym employs structured data sampling to predefine user profiles, state-dependent questions, and state evolution trajectories, enabling cost-effective generation of high-quality, evaluation-aligned interactions. LLM-simulated users expose latent states through role-play while maintaining structured state consistency.
Comprehensive metrics based on structured data guide both assessment and optimization of assistants.
Extensive experiments reveal performance gaps in existing memory systems (e.g., RAG, long-context LLMs, and agentic memory) and corresponding reasons. AMemGym not only enables effective selection among competing approaches but also can potentially drive the self-evolution of memory management strategies.
By bridging structured state evolution with free-form interactions, our framework provides a scalable, diagnostically rich environment for advancing memory capabilities in conversational agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces AMemGym, an interactive framework for evaluating and optimizing memory management in long-horizon conversations with LLM-based assistants. It addresses limitations of static, off-policy benchmarks by enabling on-policy evaluation through simulated users and structured data. AMemGym supports memory personalization, role-play-based latent state revelation, and structured state evolution. Experimental results highlight its capability to identify performance gaps and foster the self-evolution of memory systems, making it a scalable and diagnostic tool for conversational assistant development

### Strengths
1. The paper introduces a novel on-policy evaluation framework, AMemGym, which effectively addresses limitations in existing memory evaluation systems.

2. The integration of simulated users and structured state evolution ensures reliable and scalable assessments of memory capabilities.

3. Experimental evidence supports the framework’s ability to foster memory self-evolution, offering practical insights for conversational assistant optimization.

### Weaknesses
1. The statistical information about the data is insufficiently described. For example, what categories of dialogues are included in the dataset? How many dialogues are there in total? What is the distribution of token lengths across these dialogues? These details are missing throughout the paper and should be explicitly addressed.

2. The "Memory Implementation" section on page 6 is somewhat confusing. How exactly is the AWE method implemented? Why is parameter tuning only applied to the AWE method? Does this imply that the RAG and AWI methods do not have relevant parameters to adjust?

3. It is unclear why the paper does not adopt common generation evaluation metrics, such as GPT4Judge or BLEU, for performance assessment. Instead, it uses a self-constructed memory score for the final evaluation. The rationale behind this choice should be explained in more detail, especially given the availability of well-established evaluation metrics.

4. The structure of the appendix appears overly simplistic. It merely lists prompts without clearly explaining at which stage of the process each prompt was used. This lack of context makes the appendix difficult to interpret and detracts from its utility.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces **AMEMGYM**, an **interactive (on-policy)** benchmarking and optimization environment for long-horizon conversational memory. A **structured state-evolution blueprint** anchors free-form, LLM-driven role-play and enables **diagnostic** scoring with attribution to **write / read / utilization** stages. Experiments compare native LLMs, RAG, and two agentic-write variants, reveal sizable **on- vs off-policy** ranking shifts, and demonstrate feedback-driven **self-evolution** of memory policy.

### Strengths
- **Clear motivation**: Off-policy evaluations can induce reuse bias; AMEMGYM offers an on-policy, diagnostically rich setup.  
- **Methodological novelty**: Persona/state trajectories, exposure utterances, QA variants (with reflection) enable constrained interaction and automated scoring; normalized memory score and stage-wise failure analysis are useful.  
- **Thorough evaluation**: Quantifies on- vs off-policy discrepancies; characterizes long-horizon degradation of native LLMs; provides granular failure attributions and analyses of frequency, short-term buffers, and top-k.  
- **Meta-evaluation**: Human ratings on exposure clarity and dialogue consistency support data quality.

### Weaknesses
- **External validity of simulated users**: Add a small human-in-the-loop comparison and a systematic study of user-LLM choice.  
- **Broader baselines**: Include structured memory graphs/event stores, hierarchical compression, and explicit state trackers.  
- **Leakage control**: Provide anti-leak prompt design and automatic leakage checks.  
- **Metric reporting**: Add variance/CI and difficulty-conditioned analyses for the normalized memory score.  
- **Scope of self-evolution**: Jointly evolve retrieval and utilization (e.g., top-k plus utilization prompting), and report stability/convergence.

### Questions
1. **Trace reusability**: Must interactions be regenerated for each system, or can exposure prompts and user policy be reused for fair comparison?  
2. **Noise control**: How is the ratio of non-informative chatter quantified and manipulated as \(N_i\) grows?  
3. **Upper bound (SUB)**: Does the UB solver access all ground-truth states plus a strong reasoner? Would an expert-solver UB better isolate utilization bottlenecks?  
4. **Privacy**: Guidance for minimizing PII in real-user deployments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The work introduces an interactive environment designed to benchmark the memory capabilities of llm assistants. The authors argue that existing benchmarks are flawed because they rely on static, off-policy data. This means the assistant is evaluated on a fixed conversation history it did not create, which fails to capture how an assistant's own responses influence the dialogue and can lead to unreliable evaluations. They first generate the state of the user using structured outputs and then the simulated user uses their attributes in a natural way during the conversation. The framework also introduces diagnostic metrics that decompose memory failures into three stages: write, read, and utilization. They demonstrate that this "gym" can be used for agent self-evolution, where an agent uses the environment's feedback to autonomously improve its own memory-writing policy

### Strengths
1. The papers central critique of off-policy evaluation is compelling and well-articulated. The authors provide concrete evidence in table 2 that evaluation rankings of memory systems change when moving from an off-policy to an on-policy setup, proving that the distinction is not just theoretical but has practical consequences.
2. The introduction of write, read, and utilization failure metrics which gives better insights into failure modes than the usual accuracy metric.
3. The self-evolution experiment shows that agents can improve its memory policy by learning from the environment's feedback.

### Weaknesses
1. The work only evaluates memory for selecting the correct answer using multiple choice questions, but doesn't test the generation capabilities. 
2. The memory is tested using structured key-value pairs and doesn't test the episodic memory or memory where the assistant has to reason over multiple facts.

### Questions
1. In section 5, "Complete Feedback" is described as including the questions, agent's answers, and ground-truth answers, which are summarized into <feedback.summary>. Could you provide an example of how this summary is formatted? Is it a natural language paragraph, structured JSON, or another format?
2. How do you ensure diversity in the structured generation? 

- Some weird formatting: L373-375

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces AMEMGYM, a new interactive, on-policy benchmark designed to  evaluate memory management in long-horizon conversational assistants. AMEMGYM differs from existing static datasets and provides a fully automated environment where LLM-simulated users engage in structured evolving conversations that reveal latent user states through role-play. Through experimental comparisons, they found that on-policy evaluation changed the rankings and scores substantially. They also show that even the latest LLMs struggle to maintain long-term conversational memory, confirming context-length degradation. Agentic-Write External systems performed best which highlights that selective, structured writing yields higher information utilization accuracy.

### Strengths
* The paper is grounded on a novel motivation: existing benchmarks mostly focus on off-policy memory evaluation which might have a gap between realistic systems.
* The introduction of write/read/utilization decomposition is a meaningful contribution.
* The paper also conducted extensive comparisons that shed light on long-context degradation and memory design trade-offs.

### Weaknesses
* While the paper positions on-policy evaluation as a core contribution, the empirical and conceptual justification for its advantage over off-policy settings remains underdeveloped. Although Table 2 and Figure 5 show some rank changes between on- and off-policy settings, the paper does not provide insights on why those differences matter. It remains unclear what specific behavioral aspects of “interactive memory” are uniquely captured. In fact, one could argue that well-curated off-policy datasets offer simpler, cheaper, and more reproducible alternatives, and the paper does not convincingly rule out this possibility.
* The claimed novelty of introducing an on-policy memory evaluation environment is somewhat weakened by the existence of several recent long-horizon, on-policy frameworks: AgentGym, DeepResearch, SWE-Agent. These already feature real-time decision-making, persistent contexts, and memory management. The paper should sufficiently clarify what AMEMGYM contributes beyond these environments.
* The entire benchmark relies on LLM-simulated users rather than real human interaction data. While this enables scale and control, it raises a question: whether simulated dialogues genuinely capture the noise, inconsistency, and ambiguity of real users. Even though the authors conduct a “meta-evaluation”, it mostly checks internal coherence rather than human-likeness or behavioral realism.
* Although the paper evaluates several general architectures (e.g. RAG, agentic write, and long-context LLMs), it does not include direct experiments on established open-source memory frameworks such as Mem0 (Chhikara et al., 2025) or A-Mem (Xu et al., 2025), despite citing both as prior work. While the authors’ custom agentic write setups are reasonable for controlled analysis, the paper should better justify why this self-defined configuration was chosen instead of real implementations that are already widely used.

### Questions
* Could you provide the cost analysis?
* Could you also provide results on popular open-source models?

### Soundness
2

### Presentation
3

### Contribution
2
