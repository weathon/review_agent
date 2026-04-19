# GraphText: Graph Learning in Text Space

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 8, 3, 3

## Abstract
Large Language Models (LLMs) have gained the ability to assimilate human knowledge and facilitate natural language interactions with both humans and other LLMs. However, despite their impressive achievements, LLMs have not made significant advancements in the realm of graph machine learning. This limitation arises because graphs encapsulate distinct relational data, making it challenging to transform them into natural language that LLMs understand. In this paper, we bridge this gap with a novel framework, GraphText, that translates graphs into natural language. GraphText derives a graph-syntax tree for each graph that encapsulates both the node attributes and inter-node relationships. Traversal of the tree yields a graph text sequence, which is then processed by an LLM to treat graph tasks as text generation tasks. Notably, GraphText offers multiple advantages. It introduces training-free graph reasoning: even without training on graph data, GraphText with ChatGPT can achieve on par with, or even surpassing, the performance of supervised-trained graph neural networks through in-context learning (ICL). Furthermore, GraphText paves the way for interactive graph reasoning, allowing both humans and LLMs to communicate with the model seamlessly using natural language. These capabilities underscore the vast, yet-to-be-explored potential of LLMs in the domain of graph machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces an innovative method for translating graph-structured data into a natural language that Large Language Models  (LLMs) can understand. The authors demonstrate that their proposed approach facilitates training-free graph reasoning and enables interactive graph-based reasoning.

### Strengths
- **Innovative Graph-Syntax Tree Design**: One of the notable strengths of this paper is the introduction of the graph-syntax tree. Particularly impressive is the discretization of continuous node features. This novel approach is significant as it strikes a balance between providing informative features to LLMs while avoiding the issue of massive input tokens. Besides, this creative design facilitates feature propagation and thus significantly enhances LLMs' graph reasoning capabilities.

- **Comprehensive Experimental Evaluation:** The authors demonstrated the effectiveness and flexibility of the proposed method by testing it in different scenarios. These include training-free settings with closed-source LLMs, fine-tuning with open-source LLMs, interactive graph reasoning, etc. This comprehensive evaluation underscores the practical utility and versatility of the proposed approach, making it a valuable contribution to the field.

### Weaknesses
- **Heavy Parameter Tuning**: The paper's heavy reliance on dataset-specific parameter tuning, as demonstrated in Table 6, raises concerns about the generalizability and effectiveness of the proposed method. Conducting ablation studies on the selection of these hyperparameters would be beneficial to determine whether the performance boost is primarily a result of heavy parameter search or the inherent design of the method itself. This clarification would help assess the true impact of the proposed method.


- **Reliance on Domain Knowledge for Interactive Graph Reasoning**: While the paper successfully highlights the adaptability of LLMs to human feedback for interactive graph reasoning, it is important to recognize that this approach heavily depends on the quality and relevance of the human feedback. Not all tasks can be addressed using a single inductive bias, such as the PPR label. This necessitates tailoring human feedback for specific graph tasks.

### Questions
- Q1: Could you provide information on the number of prompts/training samples used in the experiment presented in Table 1? 

- Q2: When transforming the continuous features into a discrete space using K-means, it would be valuable to know the specific value of K that was used in your experiments. Additionally, could you explain the heuristic or rationale behind selecting this particular value for K?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors introduce GRAPHTEXT, a novel framework designed to bridge the gap between Large Language Models (LLMs) and graph machine learning. While LLMs have excelled in natural language understanding and reasoning, they have faced challenges in applying their capabilities to graph-structured data. GRAPHTEXT addresses this issue by translating graphs into natural language, enabling LLMs to perform graph reasoning tasks. GRAPHTEXT presents a promising approach to extend the capabilities of LLMs into the realm of graph machine learning, offering potential benefits for various applications that involve graph-structured data.

### Strengths
1. The paper presents an elegant solution to a fundamental problem: How to derive a language for relational data. The proposed tree-based solution provides a principled way to bridge relational data and one-dimensional sequential language. This innovation has the potential to catalyze significant future research.

2. GraphText equips LLMs with the ability to reason over graphs using natural language, thereby enabling interactive graph reasoning. The distinct aspects of interpretability and interactiveness of GraphText differentiate it from traditional GNNs.

3. Another outstanding feature of GraphText is its training-free reasoning ability, which not only reduces the computational overhead but also delivers impressive performance, even surpassing some supervised GNNs. Such capabilities indicate great potential for real-world applications.

### Weaknesses
1. A comparative analysis with existing baselines is required. It would be especially beneficial to compare GraphText against other methods like GraphML and GML, which explore a similar problem

2. How to construct discrete text from continuous features is not comprehensively studied. According to the hyper-parameters, the best settings are mostly based on label propagation.

3. The algorithm is overall good. Nonetheless, there is a lack of time complexity analysis. I think it should be added then.

### Questions
See in weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper addresses an important problem of bridging the gap between LLM and GNN. A GRAPHTEXT method is introduced to construct a graph-syntax tree by incorporating node features and structural information in a graph. The graph-syntax tree is then converted to a graph text sequence, which is then processed by an LLM to treat graph-related tasks as text generation tasks. However, the experiments seem insufficient in some aspects.

### Strengths
Paper Strength:

1. Exploring the gap between LLM and GNN is highly meaningful, as it effectively utilizes the potential of LLMs.

2. The proposed graph-syntax tree is novel and conceptually sound.

### Weaknesses
Paper Weakness:

1. The article lacks a comparison with large text-attributed graph datasets such as OGB-Arxiv, which is a commonly-used dataset extensively discussed in numerous papers regarding text-attributed graphs [1-4]. 

2. Regarding the experiments, I have several questions. (1) Since Wisconsin, Texas, and Cornell datasets are relatively small datasets, it would be beneficial if the authors could provide variance analysis of the results. The results on these datasets may exhibit considerable variance. (2) Moreover, Wisconsin, Texas, and Cornell datasets exhibit high heterophily. Therefore, when evaluating performance on these datasets, it is crucial to include comparisons with MLP. Based on the results from prior studies [5], it appears that GraphText may not show a significant advantage over MLP. (3) There appears to be an inconsistency in the accuracy of GraphText on Cora between Table 3 and Table 1. Could you provide an explanation for this inconsistency? (4) Providing the results under different hyper-parameters regarding the selection of text attributes and relations in Appendix A.3 would be valuable. (5) The comparison with directly utilizing LLMs to handle text attributes while ignoring the graph structure is essential.

3. It would be helpful if the authors can provide a time complexity analysis for constructing the graph-syntax tree and run-time requirements. It seems that it would be costly to construct a graph-syntax tree on large-scale graphs.

4. Can the proposed framework help other graph-related tasks, like link prediction, community detection?

[1] Chen, Zhikai, et al. Exploring the potential of large language models (llms) in learning on graphs.

[2] Duan, Keyu, et al. Simteg: A frustratingly simple approach improves textual graph learning.

[3] Zhao, Jianan, et al. Learning on large-scale text-attributed graphs via variational inference.

[4] He et al. Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning.

[5] Zhu, Jiong, et al. Beyond homophily in graph neural networks: Current limitations and effective designs.

### Questions
See weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method GRAPHTEXT to solve graph tasks by LLMs. The method constructs a syntax tree to describe necessary information about the graph. Then the syntax tree is traversed to the prompt. LLMs can use the prompt to get information about the graph. The paper also proposes to use discretization methods like clustering to transform continuous feature into discrete space. Experiment results show that the method can achieve good performance in some datasets.

### Strengths
1. Bridging the gap between graphs and LLMs is an interesting and important problem.

### Weaknesses
1. The performance on Cora/CiteSeer is very low, According to [1], GPT3.5 can achieve 67% accuracy on Cora with target text features, but ChatGPT performance shown in Table is much worse than it (label+feat, original). I guess it is because of the prompt used. I suggest to use a better prompt for the baseline. Besides, comparing 67% with the highest 68.3% performance in Cora, the proposed method does not provide good benefit to the task.
2. Some datasets used (Texas, Wisconsin, Cornell) are heterophily graphs where many GNNs cannot outperform MLP. You should compare with MLP and heterophily graph methods as baselines.
3. Some essential parts are missing in the paper. See questions.
4. in Section 4.2, the observation 2 is confusing. Why in-context learning being good indicate that GPT-4 outperforms ChatGPT?

[1] Chen, Z., Mao, H., Li, H., Jin, W., Wen, H., Wei, X., ... & Tang, J. (2023). Exploring the potential of large language models (llms) in learning on graphs. arXiv preprint arXiv:2307.03393.

### Questions
1. How to generate the pseudo labels in Figure 2a?  
1. Section 4.2 uses human feedback to improve LLM prediction, can you provide the details about human feedback?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
