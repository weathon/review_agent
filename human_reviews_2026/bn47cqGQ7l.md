# GTool: Graph Enhanced Tool Planning with Large Language Model

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Tool planning with large language models (LLMs), referring to selecting, organizing, and preparing the tools necessary to complete a user request, bridges the gap between natural language understanding and task execution. However, current works treat different tools as isolated components and fail to leverage the inherent dependencies of tools, leading to invalid planning results. Since tool dependencies are often incomplete, it becomes challenging for LLMs to accurately identify the appropriate tools required by a user request, especially when confronted with a large toolset. To solve this challenge, we propose GTool, which is the first work aiming to enhance the tool planning ability of LLMs under incomplete dependencies. GTool constructs a request-specific tool graph to select tools efficiently and generate the \<graph token\> which provides sufficient dependency information understandable by LLMs. Moreover, a missing dependency prediction task is designed to improve the reliability of GTool with incomplete dependencies. Without trimming LLMs, GTool can be seamlessly integrated with various LLM backbones without extensive retraining. Extensive experiments show that GTool achieves more than 29.6% performance improvements compared with the state-of-the-art (SOTA) baselines with a light-weight (7B) LLM backbone.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents GTool, a novel graph-enhanced framework for tool planning with large language models (LLMs). Existing tool-using systems (e.g., HuggingGPT, ToolNet) typically treat tools as independent modules and fail to model the latent dependency structure among them, leading to suboptimal planning. GTool addresses this limitation by explicitly encoding tool dependencies as a graph and integrating graph neural networks (GNNs) with frozen LLMs to perform structured reasoning.

### Strengths
1.This paper presents a very interesting perspective. Unlike traditional approaches such as ToolNet, it does not mechanically learn and plan tool invocation patterns. Instead, it systematically performs graph neural network–assisted learning based on historical tool dependencies and invocation paradigms. The theoretical foundation is strong, and the relationships between dependencies and corresponding tool-handling strategies are clearly and directly defined.

2.GTool introduces a novel direction for tool invocation schemes. While traditional methods focus more on handling the current state, this work emphasizes the dependency relationships among tools. It provides detailed analysis on potential issues in planning and decomposing a problem, especially in the formalization of multi-dependency iterative invocations. Traditional tool-calling frameworks tend to be linear, but this approach can address deeper, multi-layered problems more effectively.

3.Another strength of this paper lies in its impressive experimental results. Considering the complexity of tool invocation, the authors conducted experiments across multiple datasets and demonstrated the superiority of their method in terms of invocation time and performance. In particular, the proposed lightweight GNN architecture showed strong optimization capability compared to traditional long-CoT–based tool learning approaches.

### Weaknesses
1. My biggest concern is that the authors did not provide any examples or analysis of incorrect tool calls—they only examined incomplete graphs. Compared to the results on incomplete graphs, I would like to see whether the framework has the capability to adjust tool invocations after extraction through GTool. Put more simply, if we find that a certain dependency extracted by GTool fails to satisfy the basic task requirements, how does the framework adapt or adjust? Or does this paper simply not cover such adaptive capability? If the proposed method is to be applied to broader tool-use scenarios, then the ability to identify and adjust tools is even more important than fast inference.

2. Following the previous point, we know that methods such as Tool-Planner and ToolNet mainly focus on handling problematic tool calls—their performance improvements often come from repairing or re-planning calls when errors occur. If GTool merely produces a plan without further revising its correctness, then I think the reported performance gains are somewhat unfair. After all, in traditional approaches like Toolformer or DFSDT, if every tool call were correct, their performance metrics would essentially match GTool’s.

3. Could you further explain your final loss function? In what specific setting or stage is this loss jointly trained?

4. In large-scale tool scenarios, why does GTool perform better in your experiments? Do you think this is related to the graph’s inherent understanding of tool relations? Given that with 16k tools the contextual information could become excessive—leading to higher hallucination rates—I wonder whether the graph’s strong dependency modeling might also increase the overall error rate.

### Questions
I remain open to further discussion from the authors. If they can address these issues more thoroughly or provide additional analysis or experiments, I would be willing to raise my original score.

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
This paper proposes GTool, a graph-enhanced framework for tool planning with large language models. It constructs a request-specific tool graph to model dependencies among tools, encodes this structure with a graph neural network, and injects the resulting <graph token> into the LLM to guide tool selection and ordering. A missing-dependency prediction task allows GTool to handle incomplete graphs, and only the lightweight GNN is trained while the LLM remains frozen. Experiments show that GTool improves planning accuracy by over 29%, maintains robustness under missing dependencies, and greatly reduces inference cost.

### Strengths
1. The paper proposes a genuinely novel and well-motivated idea — representing tool dependencies explicitly as a graph and integrating it with LLMs through a GNN encoder. This design feels natural yet surprisingly underexplored in prior work, and the authors manage to make it both elegant and technically grounded. I particularly appreciate how the paper bridges symbolic structure (graph reasoning) and LLM-based semantic planning in a coherent way.
2. One of the most impressive aspects is the robustness under incomplete or noisy dependency graphs. The “Missing Dependency Prediction” component is not only clever but also practical — it addresses a real pain point in existing methods that assume full dependency knowledge. The fact that GTool still performs well when 90% of edges are missing is very compelling.
3. I like that the model does not fine-tune the LLM itself — only the GNN encoder is trained. This is a huge advantage in terms of scalability and reproducibility. The reported reduction of 95% in token length and 10× faster inference time makes the method appealing not just conceptually but also computationally.
4. The experiments are thorough and consistent. The improvement (around +30% F1 on average) is significant and holds across multiple datasets and backbones (LLaMA, Vicuna, Qwen). The ablation studies are also well-designed, showing the clear contribution of each component.
5. The paper is well-structured and easy to follow. Figures and equations are clean, and the intuition behind each design choice (e.g., why a <graph token> is needed) is clearly articulated. The work feels mature — not just a proof of concept, but something that could realistically be integrated into production-level systems.

### Weaknesses
1. The paper provides basic ablations (e.g., removing <graph token> or MDPL), but it would be interesting to see more analysis of what the GNN actually learns — for example, visualizing attention weights or node embeddings to illustrate that it truly captures tool relationships rather than acting as a generic feature compressor.

2. While the writing is overall clear, the training description could be elaborated — e.g., how the <graph token> is injected into the LLM embedding space in practice, and whether this alignment requires any additional projection layer.

3. The construction of the initial tool dependency graph relies on observed invocation trajectories. In domains where such history is sparse or noisy, the method might struggle to bootstrap meaningful structure. It would strengthen the paper if the authors could discuss how GTool performs in zero-history or cold-start scenarios.

### Questions
1. If the size of the tool graph becomes very large, can the model still maintain good inference performance when reasoning over multiple tools?

2. If there are multiple similar tools, how does the paper address the resulting conflicts?

3. Moreover, in the case of erroneous tool invocations, is there a proposed mechanism to resolve such issues and overcome the bottlenecks in the overall tool-calling process?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the practical challenge of tool planning for LLMs when tool dependencies are incomplete. The proposed framework, GTool, represents tools and their relationships as a graph, which is dynamically contextualized for each user request. Its core novelty lies in a two-part approach: a GNN encoder is trained with an auxiliary Missing Dependency Prediction task, enabling it to robustly infer latent tool relationships from incomplete data, and this learned structural knowledge is efficiently injected into a pre-trained LLM via a single, compact graph token. Extensive experiments demonstrate that GTool significantly outperforms baseline methods in accuracy, robustness to missing dependencies, and computational efficiency without requiring LLM fine-tuning.

### Strengths
1. The paper introduces an efficient method for the real-world problem of incomplete tool dependencies. It effectively uses a Graph Neural Network to learn the underlying tool structure and then injects this complex information into any frozen LLM using just a single, compact token.
2. The approach is validated by extensive experiments, showing it consistently outperforms existing methods. Critically, it does not require fine-tuning the LLM, which makes it highly practical, efficient, and easy to apply across different models.

### Weaknesses
1. The method's performance depends on having good historical data to build the initial tool graph. The paper doesn't explore how well it works in 'cold-start' scenarios where such data is scarce or unavailable.
2. The training process, which involves predicting missing links in the graph, could become a bottleneck for extremely large-scale toolsets. The paper does not fully address how the method scales when the number of tools becomes massive.
3. The use of a single 'graph token' to represent the entire tool structure is a black box. The paper offers no insight into what specific dependency information is captured by this token or how the LLM interprets it, making the reasoning process difficult to understand.

### Questions
Refer to Weaknesses.

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
This paper investigates the problem of tool planning with LLMs, which involves selecting and organizing the appropriate tools to fulfill a user’s request. The authors propose GTool, a framework designed to enhance LLMs’ tool planning capabilities under incomplete dependency settings. GTool constructs a request-specific tool graph that captures the relationships among tools and generates a corresponding graph token to represent dependency information in a format interpretable by LLMs. In addition, the authors introduce a missing dependency prediction task to improve the robustness of the model when faced with incomplete or partially observed tool dependencies. The method can be integrated with various LLM backbones without additional fine-tuning or retraining.

### Strengths
1. The paper addresses a practical and important problem in tool planning — how to model and utilize dependencies among tools. The issue of accurate dependency localization is indeed a real challenge in real-world tool-using scenarios.

2. The experiments are extensive and cover multiple evaluation settings, demonstrating the authors' efforts to comprehensively assess the proposed approach.

### Weaknesses
1. The construction of the tool graph appears problematic. The authors assume that the use of 𝑣𝑖(𝑗+1), depends on 𝑣𝑖𝑗, which cannot be guaranteed. The order of tools in a trajectory does not necessarily imply dependency between them. This assumption weakens the validity of the proposed graph-based representation.

2. The method section is difficult to follow due to inconsistent and overloaded notations. In addition, the mixed use of symbols such as 
𝜏𝑖, 𝑡𝑖1, 𝑡𝑖∈𝜏, ti∈τ, and 𝑡𝑛∈𝑇 leads to unnecessary confusion and makes the methodology harder to understand.

3. The claim that multiple LLM interactions further improve the performance of GTool may not hold. While the multiple-interaction setup indeed improves tool planning, it remains unclear whether GTool truly mitigates the inherent limitations of multi-interaction settings or merely benefits from them.

4. The generalization ability of GTool is questionable. Given that the dataset seems to involve a large number of tools, it is unclear whether GTool genuinely learns transferable tool dependencies—as claimed—or merely memorizes the tool usage patterns within the dataset. If the latter is true, static tool dependency graphs could achieve comparable results without additional model training or inference.

5. Two critical ablation studies are missing:
(a) Using static tool dependencies to verify that GTool performs reasoning rather than memorization.
(b) Using an LLM to directly predict tool dependencies to demonstrate the additional effectiveness of MDPL.
These analyses are essential to support the paper’s claims.

### Questions
1. The statement “leading to invalid planning results” (Line 015) is overly strong and lacks empirical support. Could the authors provide specific evidence or examples?

2. The use of annotations and notations throughout the paper is inconsistent and confusing (e.g., 𝜏𝑖, 𝑡𝑖, 𝑡𝑖∈𝜏). Please clarify the notation and maintain consistency across sections.

3. Could the author provide some tool dependency graphs with the same tool list but different user queries?

### Soundness
2

### Presentation
2

### Contribution
3
