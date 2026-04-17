# Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Large language models face challenges in long-context question answering, where key evidence of a query may be dispersed across millions of tokens.
Existing works equip large language models with a memory corpus that is dynamically updated during a single-pass document scan, also known as the "memorize while reading" methods.
While this approach scales efficiently, it suffers from irreversible forward-only processing, information loss through overwriting, and sparse reinforcement learning signals.
To tackle these challenges, we present ReMemR1, a memory-augmented agent with callback-enhanced memory that allows selective retrieval from the entire memory history and allows non-linear reasoning and revisiting of early evidence.
To further strengthen training, we propose Reinforcement Learning with Multi-Level Rewards (RLMLR), which combines final-answer rewards with dense, step-level signals that guide effective memory use.
Together, these contributions mitigate information degradation, improve supervision, and support multi-hop memory utilizing.
Experiments on long-document QA show significant gains over existing memory-based approaches, which validates ReMemR1 as an effective solution for long-context reasoning agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents ReMemR1, a memory-augmented large language model (LLM) agent for long-context question answering (QA). The key innovation is a callback-enhanced memory mechanism that enables the model to revisit and retrieve earlier memory states instead of relying solely on a forward-only “memorize while reading” paradigm. To address the sparse reward issue commonly faced in reinforcement learning (RL)-based memory agents, the authors propose Reinforcement Learning with Multi-Level Rewards (RLMLR), which combines trajectory-level (final-answer) and step-level (intermediate) rewards. Experiments on two datasets demonstrate consistent improvements over strong baselines such as MemAgent and Qwen2.5, particularly under long-context and distant-evidence settings.

### Strengths
- The paper is clearly written, well-organized, and easy to follow, which facilitates understanding of the technical contributions.
    
- The proposed method achieves consistent gains on both in-distribution and out-of-distribution datasets, with notable advantages in challenging long-context scenarios.
    
- The idea of non-linear reasoning via callback-based memory is conceptually interesting and potentially useful for improving long-context reasoning.

### Weaknesses
- While the callback mechanism is intuitively appealing, it can be viewed as an incremental extension of existing memory-retrieval or recurrent-state models. The distinction from prior works (e.g., MemAgent) is not sufficiently emphasized.
    
- The paper provides limited formal or analytical insight into *why* the proposed retrieval and multi-level reward structure lead to improved reasoning or generalization.
    
- The authors argue that existing memory-based agents suffer from irreversible forward-only processing and information loss because they cannot revisit previous states. However, this argument is not entirely convincing. When the policy updates its memory at each step, it already considers both the previous memory and the current input (query or document chunk), which inherently allows recursive integration of prior information. It is unclear why an additional “memory recall” mechanism is strictly necessary, rather than improving the existing update function to retain or reweight key information.
    
- The proposed design retrieves relevant content from all previous memory states, which implies that the model must maintain access to the entire memory history. This could be computationally expensive and may not scale to extremely long contexts. The paper should clarify whether ReMemR1 explicitly stores all intermediate memories or uses a compressed or differentiable representation to manage memory growth.
    
- Important implementation aspects, such as retrieval efficiency, computational overhead, and inference latency, are under-discussed. It remains unclear how the proposed approach scales in both training and inference compared to other memory-based methods.

- Additional baselines, especially recent long-context or memory-augmented LLMs, would make the evaluation more comprehensive and convincing.

### Questions
1. How does ReMemR1 perform when the memory corpus exceeds the retrieval limit or available storage capacity? Or how does other methods enlarge their memory storage?

2. Does the system employ any pruning or compression strategy to prevent unbounded memory growth?

### Soundness
3

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
4

### Summary
The paper presents a novel approach for long-text QA comprised of two components. The first component is a memory-augmented agent that stores multiple historical memories while processing the long text rather than rewriting a single memory state. It allows the agent to revisit past memories via query-based retrieval, enabling non-linear reasoning paths.  The second component is a modification of GRPO with a detailed reward function that combines traditional outcome rewards and step-level state rewards to improve the relevance of the generated reasoning sequences. The proposed method is evaluated on two MHQA datasets and shows improvement over vanilla model and a MemAgent baseline.

### Strengths
1. The paper provides a clear motivation for addressing the limitations of "memorize while reading" paradigm. 
2. The experimental results demonstrate consistent improvements across multiple model sizes and varying context lengths. 
3. The proposed RLMLR approach is clearly described.

### Weaknesses
1. Poor figure presentation. Throughout the paper, figures are consistently misaligned with their textual references, which significantly disrupts the reading flow and makes it difficult to follow the arguments.
2. The memory efficiency of the proposed method compared to the "memorize while reading" baseline is not addressed. The proposed approach requires storing multiple historical memory states, and it is not discussed how such large memory storage affects inference time and cost.
3. The statement about per-step reward innovation should be clarified. Per-step rewards are a common practice in RL. 
4. The paper does not provide the callback mechanism failures analysis.

### Questions
1. How does the memory overhead scale with task complexity and document length compared to the "memorize while reading" paradigm?
2. What are the typical failure modes of the proposed retrieval mechanism?

### Soundness
3

### Presentation
2

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
This paper introduces ReMemR1, an LLM training paradigm designed to overcome memory-related challenges in long-context QA. Existing long-context methods process documents in a single forward pass, while memory agents may result in irreversible information loss due to the linear processing of memories. In contrast, ReMemR1 addresses this by using callback-enhanced memory, in which the agent generates a callback query each step to selectively retrieve and revisit information from its previous memories and thereby enabling non-linear reasoning. To train this agent effectively, the authors propose RLMLR and combines sparse outcome-based rewards with dense, step-level rewards based on the "information gain" in memory updates and callback retrievals. Experiments show that the proposed ReMemR1 can outperforms existing memory-based approaches and long-context alternatives.

### Strengths
1. The proposed method allows memory agents to selectively retrieve information from its previous memories. This counters the potential information loss in standard memory agent methods and enables non-linear reasoning paths.

2. The propsoed RLMLR introduces a RL-learning framework with multi-level reward design. This provides more dense and step-level rewards for each states in the reasoning trajectory, which offers better guidance than RL methods that solely rely on the final outcome.

3. The authors perform extensive experiments to show that ReMemR1 can outperfor memory agents and lon-context LLMs in open-domain QA benchmarks. They also highlight the performance gains on out-of-distribution datasets.

### Weaknesses
1. Limited datasets / baselines. For open-domain QA benchmarks, the authors should also consider one-hop datasets like triviaQA, natural questions and further multi-hop questions like bamboogle, musique. Further long-context baselines like IterDRAG and search agents like Search-R1 should also be considered as baselines.

2. The method requires storing the entire history of all previous memory states over many retrieved document chunks. For each step, the LLM then perofrms a retrieval search across this history and generate new states / memory or final answer. This adds additional compute, memory overhead and latency compared to standard single-pass methods or search agents.

3. The authors adopt a very simple retrieval function based on word overlapping, other alternative methods like embedding-based retrieval is not discussed. Furthermore, the proposed method is designed to return one single past memory with the maximum word overlapping. This could be a limitation preventing the agent from synthesizing information across multiple documents / reasoning steps.

### Questions
1. On the RL training level, is each state treated as one training instance in RLMLR?

2. What is the precise format of the memory? Can you provide more qualitative examples of the memories and the generated queries?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of question-answering over long contexts, in which large language models (LLMs) typically utilize a memory corpus to summarize a single pass over the context in order to answer the question. This approach, which uses only forward processing, is storage-efficient but often suboptimal, as the importance of some parts of the context may only be apparent after scanning the initial context. This paper proposes to mitigate this challenge by appending a query and retrieval function to the agent memory for each document chunk. This query can retrieve information from prior chunks, allowing them to be “revisited.” The paper further introduces chunk-level rewards that measure the information gain in memory updates to encourage retention of new information relevant to the ground truth, using reinforcement learning to learn the memories and retrievals for each document chunk. Experiments on multiple short and long context datasets show that the proposed ReMemR1 technique outperforms MemAgent as well as existing standard and long-context LLMs.

### Strengths
+ The experiment results show a clear gain in performance relative to the baselines for both in-distribution and out-of-distribution test datasets, for most document context lengths. The accuracy of ReMemR1 is also fairly consistent as the number of context documents increases.

+ The proposed idea of using learned queries to revisit past context information, while simple to conceptualize, is intuitively likely to work. It also appears to use about the same amount of memory as the usual single-pass approaches (though I’d appreciate if the authors could confirm this explicitly).

### Weaknesses
--It seems like the presence of learned queries in ReMemR1 would increase the LLM inference time, as the model would need to query past context while making an inference. How much more time would this take in practice (and if I am incorrect and the inference time is comparable to that of single-pass memory agents, can the authors explain why?)

-- The experiments pad the contexts of both HotPotQA and 2WikiMultiHopQA with random documents. This would seem to give an unrealistic picture of true long-context documents, which would very likely have different patterns from random. Indeed, it seems that the random padding would simply lead to the model learning not to query these random documents.

### Questions
Please see also Weaknesses above.

1) What is the tradeoff introduced by varying alpha? I understand that a smaller value of alpha corresponds to more emphasis on step-based rewards, but conceptually, since the evaluation target is the final accuracy, the optimal value of alpha should be alpha = 1, which emphasizes only the final reward. Thus, the paper should better justify the need for step-based rewards. Given enough training iterations, will alpha=1 eventually converge to a better (i.e., higher accuracy) solution?

2) Why does ReMemR1 use the relatively simple word similarity as a reward in the step-based rewards? Would a more sophisticated information gain measure (e.g., one that evaluates the semantics) work better?

3) The experiment section also includes an ablation study of sorts that compares ReMemR1 to that of MemAgent with a rule-based callback, showing that ReMemR1’s learned callbacks outperform a rule-based one. It is not clear, however, whether this is a true ablation study on ReMemR1; is MemAgent equivalent to ReMemR1 without revisitable memory? It’s also not clear why MemAgent + rule-based callback often performs worse than MemAgent itself in Table 3.

4) In Figure 5 of Appendix B, it’s not clear why the paper states that performance rapidly improves around epoch 20. There seems instead to be a smoothly decreasing increase in performance across the training duration.

### Soundness
3

### Presentation
3

### Contribution
3
