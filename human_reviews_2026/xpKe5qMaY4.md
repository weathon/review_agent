# GPS: Graph-guided Proactive Information Seeking in Large Language Models

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
Equipping Large Language Models (LLMs) with the ability to proactively ask clarifying questions is essential to mitigate ambiguity when faced with underspecified user queries in retrieval-augmented generation (RAG) systems. However, existing methods often neglect the rule-based reasoning structures embedded in the retrieved knowledge that are central to ambiguity, making it challenging to learn an effective and efficient question-asking strategy. To address these issues, we introduce \textbf{GPS}, a two-stage framework for enhancing proactive information seeking abilities of LLMs in RAG systems. In the reasoning stage, we propose a Directed Acyclic Graph (DAG) reasoning structure with theoretical guarantees of logical completeness, which facilitates capturing all conditional logic in the retrieved knowledge and supports effective clarification. In the clarification stage, we design a traversal-based algorithm that dynamically prunes the DAG based on user responses, enabling efficient clarification. To further enhance DAG construction, we first propose a conditional paths guided data synthesis method to address data scarcity challenge, then we apply a clarification-oriented reinforcement learning method with a hybrid reward that jointly considers effectiveness and efficiency to optimize the LLM. Experiments on three benchmarks demonstrate that \textbf{GPS} outperforms baseline methods in both success rate and clarification efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The GPS (Graph Guided Proactive Information Seeking) framework proposed in this paper addresses the issue of active clarification in RAG systems when LLM processes queries with insufficient information. It innovatively introduces a conditional inference structure based on directed acyclic graphs (DAGs) and combines conditional path guided data synthesis with clarification oriented reinforcement learning, effectively balancing the effectiveness and interaction efficiency of clarification.

### Strengths
1. The design of DAG conditional reasoning structure combines logic and efficiency, breaking through the limitations of traditional prompting methods that rely on LLM spontaneous reasoning, and providing structured logical support for active clarification.

2. The conditional path guided data synthesis method not only optimizes the logical correctness of DAG, but also suppresses redundant interactions and structural redundancy, achieving multi-objective collaborative optimization.

### Weaknesses
1. The specific implementation logic for extracting DAG is not clear, such as "how to parse conditional variables and logical relationships from unstructured documents" and "when there are fuzzy rules in the document.

2. Suggest adding parameter sensitivity experiments to clarify the optimal parameter selection strategy; Fully derive the calculation process of structural quality rewards and verify its rationality through examples.

3. In the relevant work section, there was no in-depth comparison of the essential differences between GPS and existing models in "structured inference targets" In addition, many SOTA methods have already studied knowledge graphs and their logic. What is the difference between this article and them?

### Questions
1. Why was the method in this article not compared with the GraphRAG series models?

2. The second stage of this method uses models around 7B, which is too small compared to the deepseek used in the first stage. Should we use LLMs of the same scale? Existing mainstream methods also tend to use larger models

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces GPS, a two-stage framework that enables large language models in retrieval-augmented generation systems to proactively clarify underspecified user queries by modeling conditional logic in retrieved documents as a directed acyclic graph. In the reasoning stage, a Reasoner LLM constructs a logically complete graph that captures the AND and OR relationships among condition variables and possible answers. In the clarification stage, a Clarifier LLM dynamically traverses the graph, selecting questions from a candidate set of nodes based on their expected remaining depth, and prunes inconsistent paths according to user responses to achieve efficient clarification with only a few turns.

To train the Reasoner, the authors propose conditional path-guided synthesis that augments ConditionalQA by generating underspecified queries with multiple path assignments and filtering them through a Verifier LLM to ensure consistency. They further apply reinforcement learning with a hybrid reward combining accuracy, efficiency, and structural quality to enhance reasoning precision and interaction efficiency.

### Strengths
- Proposition 1 rigorously proves logical completeness via DNF encoding, with every root-to-leaf path corresponding to a conjunction. 
- Conditional path-guided generation from documents produces underspecified queries with explicit missing conditions and corresponding reasoning paths, each consisting of variable–answer pairs. These queries are filtered based on necessity—retained only if a Verifier model can answer correctly when given the full conditions but fails when any are masked. This process yields high-quality augmentations of ConditionalQA’s limited underspecified samples, expanding the dataset significantly without requiring human annotation.

### Weaknesses
- Filtering retains samples only when the Verifier LLM predicts the correct answer under full conditions but fails under partial ones. However, this approach discards nuanced cases involving partial ambiguity resolution and inherits the biases of the Verifier model, such as inconsistencies observed in DeepSeek-R1. 
- There is no inter-annotator agreement or human validation for the roughly 75.5% of samples discarded during ConditionalQA augmentation.

### Questions
- How often does the Reasoner generate invalid directed acyclic graphs—for instance, containing cycles, incomplete edge coverage, or mismatched condition variable domains—and what fallback mechanism is applied during traversal if graph parsing fails or if the candidate set becomes empty prematurely?
- Regarding the efficiency reward, what specific value of α was used, and how does the model’s performance degrade when user responses deviate from the simulator assumptions, such as providing evasive answers or multi-valued inputs outside the defined variable domains?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles ambiguity in RAG by teaching LLMs to proactively ask better clarifying questions. It proposes GPS, a two-stage approach: first, represent the retrieved knowledge with a Directed Acyclic Graph so the model can reason over conditional rules in a logically complete way; second, traverse and prune that graph interactively based on user answers to keep clarification efficient. To make this practical, the authors generate training data that reflect conditional paths and further fine-tune with a clarification-oriented RL objective balancing effectiveness and efficiency.

### Strengths
1. Rule-structured reasoning: Modeling conditional rules as a graph is a clean, principled way to surface ambiguity and drive targeted clarification, rather than ad-hoc question asking.

2. Theory with practical bite: The logical completeness guarantee plus average-case O(r) clarification complexity gives both soundness and efficiency.

3. End-to-end system design: A coherent pipeline—conditional-path data synthesis, clarification-oriented RL with a hybrid (accuracy/efficiency) reward, and dynamic traversal—aligns training with the actual interaction objective.

### Weaknesses
1. Baseline fairness (Clarify-DPO): The original Clarify-DPO does not has the RAG part. It’s unclear whether Clarify-DPO had access to retrieved documents (true RAG) or only engaged in Q&A without retrieval. If the latter, the comparison is unfair; if the former, the paper should specify how evidence was integrated to ensure parity.

2. Training data parity: GPS is trained on ConditionalQA (same policy domain as ShARC). Were baselines also trained on exactly the same splits and sources?

3. Efficiency evidence gap: The claim of higher efficiency isn’t supported by the experiments. There is no comparison in the aspect of efficiency with the baselines. 

4. Domain generalization: Evaluations center on rule-heavy policy/regulation datasets; it’s unclear how GPS transfers to domains where rules are fuzzier (open-ended QA, multi-hop encyclopedic tasks) or where conditional structures are incomplete/noisy.

### Questions
If the retrieved document is incomplete or underspecified, can the model leverage its own parametric knowledge to supplement missing conditions during DAG construction and clarification?

### Soundness
3

### Presentation
2

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
The paper presents GPS, a two-stage framework for improving proactive clarification in retrieval-augmented generation (RAG) systems when user queries are ambiguous or underspecified. The method explicitly models conditional reasoning dependencies using a Directed Acyclic Graph (DAG) that captures logical structures across retrieved documents. The first stage constructs the DAG with theoretical guarantees of logical completeness, while the second stage performs dynamic traversal to prune inconsistent reasoning paths based on user feedback. To mitigate data scarcity, the authors propose a conditional path-guided data synthesis strategy and optimize DAG extraction using clarification-oriented reinforcement learning with hybrid rewards balancing accuracy and efficiency. Experimental results are conducted on Synthetic, ConditionalQA, and ShARC benchmarks.

### Strengths
+ GPS introduces a novel approach by integrating graph-based reasoning with proactive clarification in retrieval-augmented generation systems, which significantly advances the state of the art in handling underspecified queries.
+ The framework is supported by theoretical foundations, including formal guarantees of logical completeness for the constructed DAG, ensuring systematic and reliable reasoning.
+ Empirical results demonstrate that GPS outperforms baseline methods on Synthetic and conditional QA dataset on Success rate, and achieve comparable results on OOD dataset ShARC, showing generalization capabilities.

### Weaknesses
- The performance across the three benchmarks does not show clear and consistent superiority on all evaluation metrics, suggesting room for further improvement in robustness and overall gain.
- The methodological exposition could be clearer. the paper would benefit from more detailed algorithmic descriptions and richer examples to illustrate how conditional relationships are captured and resolved.
- The examples provided involve relatively simple condition–conclusion links; it remains unclear whether the DAG-based reasoning can handle more complex logical hierarchies, such as when a conclusion becomes a condition in a nested structure.
- Interestingly, on in-domain data, GPS exhibits only marginal gains and performs worse than Clarify-DPO in clarifier prediction accuracy. This suggests that the reasoner may struggle to interpret graph-based representations effectively when determining whether a query requires clarification. It would be valuable to explore enhancing the reasoner’s capability through joint training on synthesized DAG–QA pairs, allowing it to better align graph structures with clarification decisions.

### Questions
Since the improvement in Success Rate is only marginal, it would strengthen the paper to include qualitative comparison examples that clearly demonstrate the advantages of GPS in constructing higher-quality and more logically complete DAGs compared to existing methods. Such examples are essential to highlight why graph construction is both necessary and beneficial, beyond quantitative gains—showing how the proposed approach leads to more accurate condition–conclusion reasoning, better clarification paths, and enhanced interpretability relative to baselines.

### Soundness
2

### Presentation
1

### Contribution
3
