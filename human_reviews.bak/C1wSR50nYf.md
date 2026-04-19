# Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 6

## Abstract
In recent years, graph prompting has emerged as a promising research direction, enabling the learning of additional tokens or subgraphs appended to original graphs without requiring retraining of pre-trained graph models across various applications. This novel paradigm, shifting from the traditional "pre-training and fine-tuning" to "pre-training and prompting," has shown significant empirical success in simulating graph data operations, with applications ranging from recommendation systems to biological networks and graph transferring. However, despite its potential, the theoretical underpinnings of graph prompting remain underexplored, raising critical questions about its fundamental effectiveness. The lack of rigorous theoretical proof of why and how much it works is more like a "dark cloud" over the graph prompting area for deeper research. To fill this gap, this paper introduces a theoretical framework that rigorously analyzes graph prompting from a data operation perspective. Our contributions are threefold: **First**, we provide a formal guarantee theorem, demonstrating graph prompts’ capacity to approximate graph transformation operators, effectively linking upstream and downstream tasks. **Second**, we derive upper bounds on the error of these data operations for a single graph and extend this discussion to batches of graphs, which are common in graph model training. **Third**, we analyze the distribution of data operation errors, extending our theoretical findings from linear graph models (e.g., GCN) to non-linear graph models (e.g., GAT). Extensive experiments support our theoretical results and confirm the practical implications of these guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Graph prompting is a popular method to adapt pre-trained graph models for downstream tasks. While many graph prompting methods work well on real-world graph datasets, theoretical analyses on graph prompting methods are still not well explored. This study provides comprehensive analysis of popular graph prompting methods. The authors conduct extensive experiments on both synthetic and real-world datasets to validate the theoretical findings in this study.

### Strengths
1. The studied problem of theoretical analyses on graph prompts is a significant topic.
2. The authors provide experiments on both synthetic and real-world datasets.

### Weaknesses
1. Some arguments lack supporting evidence. The authors claim that “The rest graph prompt designs can usually be treated as their special cases or natural extensions.” in line 105. However, drawing this conclusion is not straightforward. For example, GPPT [1] designs graph prompts as structure tokens and task tokens, and GraphPrompt [2] designs graph prompts as prompt-based Readout operations. The authors may discuss how to treat these graph prompt methods as special cases or natural extensions of GPF or All-in-one.
2. Incorrect formulation of All-in-One. According to Section 2, All-in-One obtains the prompt graph by connecting $k$ learnable prompt tokens with $N$ original graph nodes. However, All-in-one in Lemma 1 adds prompt tokens to node features, which conflicts with Section 2. Since most theoretical analysis in this study is related to All-in-one, the authors should at least provide correct formulation of All-in-One as the basis of this paper.
3. The description of All-in-One-Plus in Section 4.1 is missing. The authors should provide the citation of this method and introduce it in the paper.
4. Notations are inconsistent in the paper, which confuses readers a lot. For example, the authors use two different notations $l$ and $i$ to represent the layer index of GNNs in the paper, while $l$ is also used to denote a column vector. The authors should use consistent notations to avoid ambiguity.
5. Duplicated sentences should be avoided. For example, the paragraph *Model Settings* appears twice in Section 5.1 and Appendix C.

[1] Sun, Mingchen, et al. "Gppt: Graph pre-training and prompt tuning to generalize graph neural networks." Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022.  
[2] Liu, Zemin, et al. "Graphprompt: Unifying pre-training and downstream tasks for graph neural networks." Proceedings of the ACM Web Conference. 2023.

### Questions
1. The authors take GCNs as linear graph models and GATs as nonlinear graph models based on different aggregation mechanisms. Could the authors provide some previous studies that use such categories? According to my knowledge, nonlinear graph models are typically GNNs with nonlinear activation functions between GNN layers.
2. What is meaning of $SG_\omega$ in Theorem 4?
3. What does $\mu$ mean in Theorem 5?
4. Could the authors provide more details about All-in-One-Plus?
5. Why do the authors not include the results of GPF-plus and All-in-one-plus in Section 5?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This study provides a solid theoretical analysis of graph prompts. The theoretical findings include the capabilities of graph prompts on GCN models with non-linear layers, the error bound of the data operations by graph prompts for both a single graph and batch of graphs, and the error distributions of the data operations by graph prompts. This work also provides empirical studies to confirm these theoretical findings.

### Strengths
The theoretical analysis in this study addresses the gap in establishing a theoretical basis for the capabilities of graph prompts with non-linear pretrained models and training on batches of graphs. The theoretical findings demonstrate the capabilities of graph prompts across several typical GNN models, providing detailed error bounds and error distributions. Notably, the authors show that the scale of prompts does not increase linearly with the size of the graph dataset, a positive result that supports the scalability of graph prompts.

### Weaknesses
The writing could be further improved to assist readers who may not have sufficient background knowledge about graph prompts. It also lacks some explanations, which might lead to misunderstandings among readers. For instance:

1. The description of the function $C$ is vague and could be clarified using a specific task, such as binary classification. Additionally, $C$ should be denoted as a function of a certain downstream task.
2. The terms  $\Phi ,  \mu , and  \lambda$  in Equation (4) are not explained and should be clarified.
3. The precondition in Corollary 1 is not specified and should be stated explicitly.

### Questions
1. Why is the error related to the prompt design in Theorem 5? Intuitively, the term $||C(G)||$ appears to be related to the graph itself and the downstream task, suggesting it may not depend with the prompt design.
2. Does the number of non-linear layers affect the error bound in Theorem 5?
3. How is $C(G)$ computed in Figure 1?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a theoretical framework for understanding graph prompting, a method of incorporating additional tokens or subgraphs without requiring retraining of pre-trained graph models over various downstream tasks. The authors introduce a comprehensive theoretical framework that establishes formal guarantees on the effectiveness of graph prompts in approximating various graph transformation operations. They derive upper bounds on the errors introduced by these prompts for individual graph, and further extend the findings across different GNN architectures, including both linear and non-linear models. The empirical study supports the theoretical findings and showcases the practical benefits.

### Strengths
The paper introduces a comprehensive theoretical framework for graph prompting, significantly advancing the understanding of how and why graph prompts work. Specifically,
- There are some key concepts such as "bridge sets" and "ϵ-extended bridge sets," which are pivotal for understanding how graph prompts function. By establishing the existence of a bridge graph for any given input graph, the authors provide a strong theoretical foundation that justifies the use of graph prompts.
- The paper provides a thorough and systematic analysis of the errors introduced by graph prompts, establishing upper bounds that offer valuable insights into their performance. By deriving specific error bounds, the authors contribute significantly to the theoretical understanding of graph prompting. 

The paper is well-structured, featuring clearly defined notations and formulas that significantly contribute to its readability and comprehension

### Weaknesses
The paper offers rigorous theoretical analysis related to data operation perspective, upper bound study, etc. In the meantime, it may lack sufficient contextualization regarding how these error bounds apply in real-world scenarios. A more practical interpretation of the results could help bridge the gap between theory and application. 

In Section 4, the theorems are based on the assumption of full-rank weight matrices. It would be helpful to investigate how well the assumption holds in practical applications

The experimental section of the paper lacks a comprehensive integration of real-world dataset evaluations within the main text, which limits the visibility and perceived relevance of the findings. By omitting a detailed discussion of these real-world results in the primary analysis, the paper misses an opportunity to contextualize its findings and demonstrate the effectiveness of graph prompting in practical applications. Including these insights in the main text would strengthen the paper’s overall impact

### Questions
In the experimental section, the authors utilize GCN and GAT as the primary models. Given that graph transformers have emerged as SOTA models especially in the context of prompting, could you elaborate on your decision to focus on GCN and GAT? How do you believe your findings might differ if applied to graph transformers?

Based on the theoretical study in the paper, what are the next steps you envision for further research in graph prompting? How can practical application benefit from the theoretical framework? It would be good to have the paper linked to real-world applications

### Soundness
3

### Presentation
3

### Contribution
3
