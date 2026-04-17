# Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference

- Decision: Accept (Oral)
- Scores: 6, 8, 6, 2

## Abstract
Large language models (LLMs) are increasingly leveraged for text-rich graph machine learning tasks, with node classification standing out due to its high-impact application domains such as fraud detection and recommendation systems. 
Yet, despite a surge of interest, the field lacks a principled understanding of the capabilities of LLMs in processing graph data.
In this work, we conduct a large-scale, controlled evaluation across the key axes of variability: the LLM-graph interaction mode, comparing prompting, tool-use, and code generation; dataset domains, spanning citation, web-link, e-commerce, and social networks; homophilic vs. heterophilic regimes; short- vs. long-text features; LLM sizes and reasoning capabilities. We further analyze dependencies by independently truncating features, deleting edges, and removing labels to quantify reliance on input types.
Our findings provide actionable guidance for both research and practice. (1) Code generation mode achieves the strongest overall performance, with especially large gains on long-text or high-degree graphs where prompting quickly exceeds the token budget. (2) All interaction strategies remain effective on heterophilic graphs, challenging the assumption that LLM-based methods collapse under low homophily. (3) Code generation mode is able to flexibly shift its reliance to the most informative input type, whether that be structure, features, or labels.
Together, these results establish a clear picture of the strengths and limitations of current LLM–graph interaction modes and point to design principles for future methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper systematically evaluates how large language models perform on node classification across multiple settings, including interaction mode (prompting, tool-use, and code generation), dataset type, graph structure, and model scale. The authors find that Graph-as-Code, which leverages LLMs’ code generation ability, achieves the strongest and most adaptive performance—especially on long-text or high-degree graphs. Through controlled ablation studies, the paper demonstrates that Graph-as-Code dynamically adjusts its reliance on structural, feature, and label information, offering an efficient and flexible approach to LLM-based graph reasoning.

### Strengths
S1. The paper conducts a large-scale controlled evaluation of LLMs for text-rich graph tasks (e.g., node classification) across six key variability axes, filling the field’s gap in principled understanding of LLM-graph interactions.

S2. It proves all LLM-graph interaction modes work on heterophilic graphs (challenging prior assumptions) and identifies Graph-as-Code as optimal, especially for long-text/high-degree graphs.

### Weaknesses
W1. The paper lacks comparisons with GNN baselines (e.g., GCN, GAT), making it hard to judge LLMs’ competitiveness against GNN methods.

W2. The evaluation focuses mainly on accuracy, with limited discussion of computational efficiency, latency, or interpretability trade-offs, which are critical for practical use.

W3. The paper lacks case studies or examples illustrating Graph-as-Code’s internal reasoning or generated code behavior, which would help clarify its mechanisms beyond aggregate metrics.

### Questions
Q1. Can you include comparisons with GNN baselines such as GCN, GraphSAGE, or GAT?

Q2. Can you report computational efficiency metrics such as runtime, token usage, or latency for each LLM–graph interaction mode?

Q3. Can you provide case studies or examples showing how Graph-as-Code performs reasoning or generates code during inference?

Q4. When edges are deleted, how does Graph-as-Code access feature and label information of other nodes—does it query arbitrary nodes rather than neighbor nodes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the effectiveness of LLMs for solving the node classification task through extensive controlled experiments. The authors consider various factors, including the mode of interaction between LLMs and graphs, graph domain and structural/feature characteristics, as well as LLM model size and reasoning capabilities. The experiments reveal several valuable findings, such as the strong and robust performance of the "Graph-as-Code" interaction mode across different graphs, even under perturbed features or structures. These insights provide a useful foundation for advancing research in both the graph learning and LLM communities.

### Strengths
1. **This paper broadens the discussion on how LLMs interact with graphs** by bringing attention to the tool-use manner (GraphTool) and code generation (Graph-as-Code). These interaction paradigms have largely been overlooked in existing benchmarks for LLMs applied to the node classification task.

2. **The authors evaluate a wide range of variables**, including interaction modes between LLMs and graphs, graph characteristics (domain, homophily, text-feature informativeness, etc.), and LLM capabilities (model size, reasoning ability, and series). Additionally, they investigate how different graph manipulations (e.g., truncated features, deleted edges, removed labels) interplay with these variables, providing a comprehensive understanding of performance under diverse conditions.

3. **The experiments are extensive and bring many actionable insights**. Among the interaction modes, "Graph-as-Code" achieves strong overall performance and demonstrates robust generalization across perturbed graphs. This makes it a highly favorable method. Furthermore, even on heterophilic graphs, LLMs leverage their reasoning capabilities to deliver reliable and accurate predictions. These findings are promising to benefit the community by encouraging further exploration of the Graph-as-Code approach, both for developing new methodologies and establishing new benchmarks.

4. The paper is **well-written**, with a logical flow that is easy to follow. The authors effectively use vivid figures and tables to present their findings, making the results accessible and facilitating understanding.

### Weaknesses
1. The paper could discuss the computational costs associated with the different interaction modes, e.g., token consumption. This would help readers better understand the performance-efficiency trade-offs of different methods.

2. The evaluation focuses primarily on scenarios with abundant supervision, e.g., up to 60% of training data on most datasets. However, it would be valuable to compare the few-shot or zero-shot performance of LLMs, as these capabilities are a key strength of LLMs and could reveal additional insights.

3. The considered tasks are limited to node classification. Expanding the scope to include other graph computational problems, e.g., shortest path, could further demonstrate the utility of the Graph-as-Code methodology, particularly in its ability to directly generate Python code for problem-solving.

### Questions
Please refer to Weaknesses section

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
The paper presents a systematic, large-scale empirical study on the performance of LLMs in graph inference tasks—particularly node classification. The authors design three LLM–graph interaction paradigms: Prompting, GraphTool, and Graph-as-Code. They conduct controlled experiments across multiple dimensions—including dataset domains, homophilic vs. heterophilic graph structures, text length of node features, and LLM model scales—and further employ ablation analyses to reveal how different methods rely on graph structure, node features, and labels.

### Strengths
- **Systematicity and Experimental Breadth**: This work is the first to factorize LLM-based graph reasoning capabilities across six key axes: interaction mode, dataset domain, homophily level, feature text length, model scale, and reasoning capability. The experimental design is rigorous and covers the major variables present in real-world scenarios.  

- **Proposal and Validation of Graph-as-Code**: The paper introduces *Graph-as-Code* as a novel LLM–graph interaction paradigm that leverages LLMs’ code generation abilities (e.g., via pandas operations) to dynamically query graph data. It significantly outperforms traditional prompting—especially on long-text or high-degree graphs. (**This is an excellent contribution that finally provides clear research direction for the emerging field of LLMs for graphs.**)

- **Challenging Prevailing Assumptions**: The paper refutes the commonly held belief that “LLMs fail on heterophilic graphs.” Experimental results show that all interaction modes perform well on heterophilic graphs, demonstrating that LLMs can effectively leverage node features rather than relying solely on neighborhood label consistency.

### Weaknesses
- All experiments are limited to **node classification**; it remains unclear whether the conclusions generalize to other graph tasks such as graph classification, link prediction, or subgraph matching.  
- The **baselines are simplistic**, including only Label Propagation, Random, and Majority voting—without comparison to state-of-the-art GNNs or hybrid LLM+GNN models. *(Note: this is understandable given the paper’s focus on empirical analysis rather than SOTA performance.)*  
- Experiments are **dominated by closed-source models**, which limits reproducibility.

### Questions
> *I originally intended to give this paper a score of 7, but ICLR doesn’t provide that option. I hope the authors can further strengthen the completeness and rigor of their experiments and deepen the discussion around Graph-as-Code—especially its scalability, efficiency, and extensibility to other graph tasks.*

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper benchmarks these modes on node classification tasks across homophilic vs. heterophilic graphs, short- vs. long-text datasets, and different model scales (from LLaMA to GPT-5). The authors report that Graph-as-Code consistently outperforms prompting and tool-calling due to better token efficiency and compositional reasoning.

### Strengths
- Timely and high-impact topic.
- Reproducibility transparency.

### Weaknesses
- Conceptual overlap between GraphTool and Graph-as-Code.
- Weak theoretical framing.

### Questions
1. The Graph-as-Code paradigm is described abstractly. If code generation and execution are treated as a form of tool invocation, then Graph-as-Code is effectively a special case of GraphTool with compositional or batched tool calls. Both rely on external execution environments to retrieve graph information. 
2. In Prompting, neighborhood hops are truncated due to token limits. In GraphTool/Graph-as-Code, there is no equivalent limit, and the LLM can query indefinitely. This makes the modes incomparable in resource exposure: Graph-as-Code effectively sees more information than Prompting. How many tokens are used per query on average?
3. The work attributes improvement to “graph reasoning,” but Graph-as-Code might simply exploit code-LLM priors (e.g., structural induction, algorithmic recall). There is no control experiment using the same LLM on shuffled or random adjacency matrices to test if improvements are truly structure-dependent.
4. In GraphTool, how many reasoning steps are allowed? 
5. Could authors present an example of how Graph-as-Code behaves in practice?

### Soundness
3

### Presentation
2

### Contribution
2
