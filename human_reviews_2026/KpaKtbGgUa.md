# MemoryField: Exploiting Gravitational Field for Long-term Memory Management

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Despite the rapid progress of large language models (LLMs) enabling agents to perform complex decision-making and interaction, their limited long-term memory capacity hinders effective retention and organization of historical interactions. This often leads to instability and semantic fragmentation in multi-turn dialogues and long-range reasoning tasks. Existing memory mechanisms struggle with structural reorganization, dynamic semantic retrieval, and the modeling of cognitive phenomena such as memory consolidation and forgetting. To address these challenges, we propose MemoryField, a novel dynamic spatial cognitive memory architecture driven by an attention-based gravitational field model. MemoryField represents memory items as nodes in a high-dimensional semantic space, where semantic attraction, repulsion, attention-driven forces, and decay mechanisms enable self-organized evolution and adaptive restructuring. By integrating node dynamics with fusion and forgetting processes, our approach ensures semantic coherence and cognitive stability. We validate the effectiveness of our approach on multi-turn dialogue and multi-type reasoning tasks. In dialogue tasks, MemoryField outperforms baseline models, achieving improvements of up to 4.9 points in Mauve and 3.3 points in ROUGE-L. In long-context reasoning tasks, the F1 score is improved by up to 14.7 points on adversarial and temporal reasoning benchmarks. These results demonstrate that the proposed method offers significant advantages in memory modeling and can serve as a general solution for long-term memory management in LLM agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MemoryField, an innovative memory architecture for large language models (LLMs) based on an attention-driven gravitational field model. MemoryField addresses challenges in long-term memory management, including structural reorganization, semantic retrieval, and cognitive phenomena like memory consolidation and forgetting. It models memory as nodes in a high-dimensional semantic space, enabling adaptive restructuring through forces like attraction, repulsion, and decay. Extensive experiments demonstrate superior performance in dialogue coherence, reasoning stability, and real-world benchmarks compared to existing methods.

### Strengths
1. The MemoryField framework effectively integrates semantic dynamics, memory consolidation, and forgetting mechanisms, providing a scalable and novel approach to long-term memory management.

2. Experimental results highlight significant improvements in dialogue coherence and reasoning stability over state-of-the-art baselines across diverse benchmarks.

3. The paper presents a well-grounded theoretical foundation, with clear descriptions of the gravitational field model and its impact on memory reorganization.

### Weaknesses
1. As a study addressing the challenge of long-term memory in large language models, it is concerning that the authors did not evaluate their approach on well-established memory benchmarks such as LongMemEval [1] or LoCoMo [2], raising doubts about the model's memory capabilities.

2. Is the memory forgetting module necessary? For humans, forgetting is essential due to limited brain capacity. However, in scenarios where storage space is sufficient, forgetting may become redundant or even detrimental, especially if critical information is forgotten, potentially impacting performance negatively. Additionally, even if certain information is not queried immediately, there is no guarantee it will not be required in future contexts.

3. The proposed method involves multiple components, including attraction, repulsion, forgetting, and fusion modules. However, the paper lacks the necessary ablation studies to demonstrate the effectiveness of these individual components.

4. The title of Table 1 appears to be inconsistent with the table's content. Furthermore, it is recommended to include evaluation metrics similar to GPT4Judge in the results for a more comprehensive assessment.

[1] Wu, Di, et al. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." The Thirteenth International Conference on Learning Representations.

[2] Maharana, Adyasha, et al. "Evaluating Very Long-Term Conversational Memory of LLM Agents." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

### Questions
See weakness.

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
4

### Summary
This paper proposes MemoryField, a dynamic spatial cognitive memory architecture. This architecture is driven by an attention-based gravitational field model. This model allows the memory structure to self-organize and adaptively restructure. The system also explicitly integrates node fusion, which serves as a form of conceptual abstraction to reduce redundancy, and a forgetting mechanism to prune long-term, low-activity memory nodes, ensuring semantic coherence and cognitive stability. Extensive experiments were conducted across diverse benchmarks, including multi-turn dialogue long-context reasoning, and real-world agent tasks. The results shows that MemoryField consistently outperforms existing memory mechanisms.

### Strengths
1. This paper introduces a gravitational-field concept from physics into the memory module of LLMs, enabling self-evolution and natural forgetting. This is a interesting idea.

2. This framework provides a unified and intuitive simulation of advanced cognitive functions. Specifically, the mechanism of “Fusion” aligns with memory consolidation, whereas “Activity Decay” and “Source Repulsion” reflect the dynamics of natural forgetting.

3. The experimental design is comprehensive, covering three major scenarios—dialogue, reasoning, and agent-based tasks. It compares multiple baselines and further validates the framework’s generalization ability across several state-of-the-art models.

### Weaknesses
1. There are too many new hyperparameters. It is difficult to reproduce the results. Moreover, it raises concerns about the robustness and generalization ability of the proposed method, as the performance might be highly sensitive to specific hyperparameter settings.

2. The framework appears to be quite complex, yet the authors do not provide any analysis or discussion of its computational cost. A detailed examination of the time and space complexity would help readers better understand the practicality of the method. In addition, the authors should report the inference latency and compare it quantitatively with other baseline methods to demonstrate the efficiency and scalability of their approach.

### Questions
1. Which embedding model is used to obtain the semantic content vector?
2. How is the spatial position vector obtained? 
3. Why can’t the semantic vector replace the position vector?

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
The paper proposes MemoryField, a long-term memory module that models stored memories as particles in a high-dimensional “gravitational field,” with four forces (attraction/repulsion/attention pull/peripheral pushback), plus fusion and forgetting. It reports gains on dialogue quality (e.g., +4.9 MAUVE, +3.3 ROUGE-L) and reasoning F1, and claims cross-model generalization (abstract & Sec. 4).

### Strengths
1. The introduction of a field-based memory representation is quite novel and creative, providing a new physical metaphor for modeling interactions among memories.

2. The paper conducts extensive experiments across multiple datasets and tasks (dialogue, reasoning, and real-world benchmarks), demonstrating consistent improvements over strong baselines.

3. The approach shows broad applicability across different model backbones, suggesting good generalization potential.

### Weaknesses
- The paper could be strengthened by connecting to Trace Theory — where each perceived sentence or event is mapped into a sequence of nodes (a trace) rather than a single node in the field. Such a design would better capture the temporal and structural continuity of cognition, and make the framework fundamentally different from standard RAG systems. The current formulation, which stores each paragraph or text chunk as an independent node, remains conceptually close to conventional retrieval-based architectures.

- If the goal is to show superiority in text organization and retrieval, the comparisons should include RAG baselines such as BM25, Dense Passage Indexing (DPI)[2], HippoRAG[1], or HippoRAG-v2[3]. At present, the baselines are mostly non-RAG methods (e.g., ReAct), making it unclear whether the proposed method truly outperforms simpler yet competitive retrieval pipelines.

- Despite being named MemoryField, the system primarily operates as a structured RAG rather than a true memory system. It lacks properties associated with long-term memory, such as global understanding, skill accumulation, or adaptive reuse of prior knowledge (as discussed in MemoryAgentBench[4]). In its current form, the work feels closer to an “advanced RAG” rather than a genuine “memory” architecture.

[1] HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.  
[2] Dense Passage Retrieval for Open-Domain Question Answering.  
[3] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models.   
[4] Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions.

### Questions
When merging multiple nodes, how do you merge the texts in these nodes?

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
3

### Summary
The authors propose MemoryField, a dynamic architecture for long-term memory in LLM agents. Each “memory node” is treated as a particle in a high-dimensional semantic space with a content vector $𝐶_𝑖$, position $𝑃_𝑖$, velocity $𝑉_𝑖$, and activation $𝐴_𝑖$. Nodes are subject to four forces: inter-node attraction/repulsion and attraction to/repulsion from the origin, with periodic merge and forgetting rules. The system answers a query, updates links, and relaxes the configuration until the energy converges. Experiments are reported for dialogue (MSC, CC), long-context settings (five task types), and “real-world tasks” (AlfWorld, ScienceWorld, HotPotQA, FEVER).

### Strengths
* The idea of dynamic, physics-inspired memory control (four forces plus merge/forgetting) is fresh and potentially useful for mitigating long-context noise. The formalization of the forces and update rules is sufficiently clear.

* An appealing intuition: store the answer and its context as a new memory node; reinforce frequently used knowledge and weaken the “periphery.”

* A broad set of setups (dialogue, reasoning categories, and environment/interaction tasks) is intended to test the approach’s generality. The work also features clear, informative visualizations that aid understanding.

### Weaknesses
1. In Section 4.1 (“Experimental Setup”), the ‘Datasets’ paragraph lists categories (single-hop, multi-hop, temporal, open-domain, adversarial) and mentions only illustrative datasets in Appendix A.5 (e.g., NQ, MuSiQue, HotPotQA, 2WikiHop), but there is no clear mapping between these datasets and the listed categories. Table 2 summarizes results by the five reasoning categories rather than listing the exact datasets, and for the Single-hop Reasoning, Temporal Reasoning, and Adversarial Reasoning categories no concrete benchmarks are specified. For HotPotQA/FEVER, the SR (success rate) metric is reported, but SR is not clearly defined (is it EM? EM@1? the fraction of successful agent episodes?), and the evaluation protocol is not described (e.g., whether HotPotQA distractor passages were used, how hops were constructed, number of steps, etc.).

2. What exactly is stored in a memory node, and how memory is initialized and grows, is described too generally.
In particular, it is unclear what precise textual payload constitutes the semantic content vector $𝐶_𝑖$. Is it the raw answer, the question+answer pair, retrieved passages, an extractive span set, a summary, or a prompt-formatted bundle (with instructions/system text)? The paper states that retrieved nodes plus the current question are fed to the LLM, after which “the answer and its context” are saved as a new node, but there is no benchmark-specific recipe: no concrete templates or examples, no accumulation depth, allowable top-k sizes, update frequency, merge triggers, or forgetting thresholds in the evaluation setup.

3. $𝐶_𝑖$ is defined as a “semantic content vector,” but the specific embedding model is not specified, nor is it clear how it was chosen or fine-tuned, or whether it is the same across tasks and LLMs. For $𝑃_𝑖$, the dimensionality 𝑛 is not fixed in the method. In the training “example log,” a position matrix with shape (1,128) (and later (2,128), (3,128)) appears, which suggests that the implementation likely uses $n=128$, but this is not stated in the main text. The ablation visualizations (Figure 4) appear two-dimensional, yet the projection or reduction method (t-SNE, UMAP, PCA, or force-directed) is not described.

4. Ablations are shown only visually, without metrics. Section 4.3 presents visual configurations with “forces turned off,” but there is no table quantifying the impact of each force on key metrics (Mauve, ROUGE-L, F1, SR). This weakens the evidence for the necessity of all components.

5. Unclear/inconsistent descriptions and captions in the tables.
Table 1 is titled “F1… across context lengths,” but the table itself reports BLEU-4/ROUGE-L/Mauve/BERTScore for MSC/CC, with no context lengths and no F1. This is a clear mismatch between caption and content. Table 3 uses “automatic scoring, higher is better,” but the metric is neither named nor defined (no scale, source, or validity).  In Table 2, there is an “Overall” column, but it is not explained how it is computed; judging by the numbers, it is not the simple average of the five categories (and this needs to be clarified).

### Questions
1. In your Table 4, you include the Reflexion baseline. Please clarify whether you reproduced these results under your own protocol, or adopted them from the source. If taken from the source, provide an exact citation and justify protocol comparability, since in Shinn et al. the metrics are computed over trials and are notably sensitive to agent configuration. In the original Reflexion paper, Figs. 3–4 show stronger curves on AlfWorld/HotPotQA and explicitly test the ReAct + Reflexion configuration. Please explain why the ReAct + Reflexion variant is not reported in your Table 4.
2. Please refer to Weaknesses §1–§3

### Soundness
1

### Presentation
2

### Contribution
3
