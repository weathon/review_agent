# Understanding Graph Self-Supervised Pre-training under Distribution Shifts: A Scaling Law Perspective

- Decision: Reject
- Scores: 4, 6, 4, 4, 2, 4

## Abstract
Scaling laws have played a fundamental role in the development of foundation models for NLP and vision, but their applicability to large-scale pretrained graph-based models remains unclear—particularly under distribution shifts intrinsic to graph data. In this work, we systematically investigate how model capacity and data scale affect downstream performance in graph pre-training under distribution shifts. To disentangle how distribution shifts impact the scaling, we construct synthetic benchmarks based on contextual stochastic block models, with precise control over both structural and feature-level shifts across the pre-training and testing graphs. Our initial experiments on GCN, a standard Graph Neural Network (GNN) baseline, reveal a striking asymmetry: increasing model capacity consistently improves performance, while increasing data size often degrades it, even under mild shift. We show that this degradation is not inevitable; properly configuring the pretraining model with deeper, wider, and transformer-based architectures enables favorable data scaling, even when distribution shifts. As data scales, graph transformer models achieve up to +9\% gains over GCN, which holds for both synthetic and real-world graph domain adaptation tasks. To explain this phenomenon, we develop a theoretical framework based on Fisher separability and Wasserstein domain divergence, which formally characterizes how distribution shifts affect representation transferability. Our results highlight architecture- and shift-aware strategies as the key to unlock scalable graph-based model pre-training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a nice innovation and introduces innovative ideas about scaling laws for graph-based pre-training under distribution changes. However, I think this paper lacks an overall framework diagram and a clearer visual description of the novel contributions, which would further improve the completeness and clarity of the paper. In addition, the results section is relatively thin, and a visual presentation of the method will enhance the persuasive power of the method and also help the reader better understand the experimental setup. Finally, the comparison with existing SOTA methods is not explicitly emphasized in this paper, making it difficult to measure the relative advantages of the proposed method. Including. Overall, this work is promising, but would benefit from improved presentations and stronger empirical analysis.

### Strengths
The originality of this paper is good, the quality is acceptable, but the explanation of the problem is not clear enough. Graph neural network is a good research direction, and it is very meaningful to explore the influence of model capacity and data scale on the downstream performance of graph pre-training under the condition of distribution shift.

### Weaknesses
In my opinion, the biggest problem of this paper is that the summary of the problem solved is not concise enough, and there is no prominent point in the description of the method. If the overall framework diagram can be added, this paper will be more complete. At the same time, the comparison with the existing baseline or similar methods will make the innovation points of this paper more convincing.

### Questions
1.It would be helpful to include a diagram illustrating the key innovations or core concepts of the paper.
2.The overall framework could be further streamlined for clarity.
3.Try to add more visualized results to enhance the presentation of your findings.

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
3

### Summary
The scaling law for graphs is an important issue. This work focuses on this point and demonstrates that graph transformers are the architectures capable of scaling. Additionally, the authors construct a framework to characterize the relationship between distribution shift and representations.

### Strengths
Originality: The authors’ work is original and goes beyond a simple combination of existing ideas (i.e., not merely “A + B”). They identify and address an important problem with their own thoughtful perspective and answer. 

Quality: The quality of the work is substantial. The experiments, hypotheses, and interpretations are well-aligned and mutually supportive. 

Clarity: The manuscript is clearly written and free of confusing or ambiguous passages. 

Significance: The question of scaling laws for graphs is an important and timely topic in the field.

### Weaknesses
The authors should provide a more detailed description of the datasets in the main experiments, rather than using an oversimplified notation like `USA → EU`. Although they mention that more details are available in the appendix, I was unable to locate them.

The experiments appear to involve only the same underlying data with different representations—i.e., graph data of the same type but with varying node attributes. In my view, this level of analysis are far from scaling law on graph.

### Questions
1. Do you  have experimental results that examine shifts across fundamentally different types of graph data? E.g., a graph-level data shift to a node-level data?

2.This paer employed three GNN variants and one graph transformer variant. Have you considered other graph transformer architectures as well?


I understand that conducting extensive additional experiments during the rebuttal period is impractical; therefore, the authors could instead clarify their claims through illustrative examples or simple pilot experiments.

Overall, the problem this paper address is critical. However, the limited experimental validation raises concerns.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a systematic investigation of scaling laws in graph self-supervised pre-training, with a particular focus on the challenges posed by distribution shifts. The authors construct a synthetic benchmark based on the Contextual Stochastic Block Model (CSBM), which enables precise and independent control over both structural and feature-level distribution changes between pre-training and evaluation graphs. Through extensive experiments, the paper uncovers a striking asymmetry: while increasing model capacity consistently improves transferability under distribution shift, merely increasing the amount of pre-training data can degrade performance, especially when the source and target graph distributions differ. The authors further demonstrate that expressive architectures—particularly deep and wide Graph Transformer models—can overcome this negative scaling effect and benefit from larger datasets. On the theoretical side, the paper develops a unified framework grounded in Fisher separability and Wasserstein domain divergence, providing a principled explanation of how distribution shifts affect representation transferability and why higher model capacity is crucial for positive data scaling.

### Strengths
1.The paper pioneers the study of scaling laws under distribution shifts in graph self-supervised learning — a gap in the current literature where most scaling analyses focus on NLP and vision.
2.The paper's use of CSBM for synthetic data generation is a major advantage. It provides the authors with fine-grained control to isolate feature-level and structural distribution shifts, which is crucial for a clean causal analysis. This makes the findings substantially more reliable than those derived from observations on a handful of real-world graphs.
3.The paper demonstrates an asymmetry between model size and data size — making the model larger steadily improves performance, but when a distribution shift occurs, adding more data can actually hurt. This finding challenges the common belief that “more data always leads to better results.”

### Weaknesses
1.The authors have developed a strong and formal theoretical analysis to support their claims. A potential area for improvement could be in bridging the formal mathematics with more accessible, intuitive commentary. For instance, walking the reader through the intuition behind why certain assumptions are made, or what a particular theorem implies in practice, could make this dense but valuable section more digestible for a wider audience.
2.The paper strongly advocates for the Graph Transformer as the key architecture for solving data scaling challenges. However, it completely overlooks the high computational and memory complexity of these models relative to traditional GNNs, thereby ignoring the critical trade-off between performance and efficiency.
3.The conclusion that Graph Transformers outperform GCNs under distribution shift is not particularly surprising, given that their powerful global attention mechanism is inherently better suited for complex scenarios. While the paper's contribution lies in connecting this to data scaling, it fails to clearly distinguish its novel perspective from the common understanding that Transformers are simply more powerful models.

### Questions
1.The real-world experiments in Section 3.5 simulate data scaling by using fractions of a single source graph. This tests scaling within one distribution, whereas Graph Foundation Models (GFMs) are typically pre-trained on multiple, heterogeneous domains. How do the paper's findings on data scaling generalize to this more realistic multi-domain GFM setting, where adding new domains introduces more diverse distribution shifts?
2.Your striking finding about negative data scaling is primarily demonstrated on a synthetic CSBM benchmark. Is it possible that this phenomenon is an artifact of the specific data generation process, where all source graphs are drawn from the same distribution? Have you observed similar strong negative trends on a more diverse collection of real-world pre-training datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work systematically explores scaling laws in graph self-supervised pre-training under distribution shifts, addressing the ambiguity of their applicability in graph-based pretrained models. It constructs synthetic benchmarks (via CSBM) to control structural and feature-level shifts, and conducts experiments on GNNs (e.g., GCN) and graph transformers. Key findings include: increasing model capacity consistently boosts performance, while data scaling often degrades it, though deep/wide or transformer-based architectures enable favorable data scaling. A theoretical framework based on Fisher separability and Wasserstein domain divergence explains these phenomena.

### Strengths
1. Provides systematic analysis of graph pre-training scaling laws under distribution shifts.
2. Combines empirical results with theoretical insights, offering both practical guidance and principled explanations.
3. Uses controllable synthetic benchmarks to disentangle structural and feature shifts.

### Weaknesses
1. In line 66, the authors mention "whether the absence of scaling is due to intrinsic limitations of graph models or the insufficiency of available data". In fact, there is also a lack of data. The experiments using data generated by CSBM proposed in the paper seem difficult to reflect the limitations of graph models in real-world scenarios. Although I notice the use of real datasets (USA, Europe, Brazil), these datasets are very small in scale, and the paper lacks cases with large-scale data.
2. Could the authors explore the impact of different pretraining tasks on transfer performance? Besides model architecture and parameters, the choice of graph pretraining tasks may also have a significant impact. However, the paper only seems to adopt the GraphMAE task. This could help us investigate which tasks are suitable for studying scaling laws.
3. Is the selection of structural shifts (Shift 3–5) rather simplistic? They seem to only reflect differences in data augmentation. Could CSBM be used to control the degree of structural shifts? In line 153, Shift 1 and Shift 2 appear to be indistinguishable, as both reduce inter-class separation. What is the significance of setting these two separate shifts?
4. The paper focuses primarily on node classification, which limits insights into other graph tasks (e.g., link prediction, graph classification).

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the scaling behavior of graph self-supervised pre-training under distribution shifts. The authors design controlled synthetic benchmarks using contextual stochastic block models and examine how model capacity and data scale influence transferability. They find that while increasing model capacity consistently improves performance, enlarging the pre-training dataset can hurt accuracy under distribution shifts. Through experiments on GCN, GraphSAGE, GAT, and graph transformers, the study shows that transformers exhibit more stable and positive scaling effects. The authors further propose a theoretical framework based on Fisher separability and Wasserstein divergence to explain why model capacity helps while data scaling can fail.

### Strengths
The paper addresses an important and timely problem which is understanding scaling laws for graph foundation models, especially under distribution shifts that are inherent to real-world graph data. The motivation is clear, the methodology is well-structured, and the experiments are described clearly. The proposed synthetic benchmark provides controlled conditions for studying the phenomenon, and empirical results show clear improvements of graph transformers over GCN-based baselines.

### Weaknesses
The paper a little lacks novelty to me. In detail, the experimental setup mainly repackages known components such as CSBM data, GraphMAE pretraining. The related work section is incomplete and does not sufficiently compare to concurrent studies on graph scaling or graph foundation models, graph learning under distribution shifts. The empirical evaluation includes only limited baselines, and important state-of-the-art SSL methods and graph distribution shifts methods are not fully considered. The theoretical part is more descriptive, and the connection between the derived inequalities and observed empirical trends is not convincingly supported.

### Questions
Can the authors clarify how the proposed theoretical framework improves over existing analyses of domain shift in GNNs? Also, since the synthetic benchmarks are highly controlled, how confident can we be that the same scaling behavior generalizes to complex real-world graph distributions beyond the small domain adaptation examples presented?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work presents the systematic study of scaling laws in graph self-supervised pretraining under distribution shifts. The key finding is that increasing model capacity consistently enhances transferability, whereas the benefits of data scaling depend on model expressiveness and architecture.

### Strengths
1. The investigation on scaling law for graph pre-training is interesting.
2. The experimental setting is well designed.
3. The results are interesting and well justified.

### Weaknesses
1. The evaluation is limited to only four graph self-supervised learning (SSL) methods, all of which are not the most recent, ranging from 2016 to 2022. This makes the study less representative of the current state of graph SSL. Additionally, evaluating only four methods is insufficient to draw strong conclusions. To strengthen their claims, the authors are expected to include more recent and diverse graph SSL methods in the evaluation. Authors may consider incorporating some of the following works [1-7] for comprehensive evaluation.

[1] Bo, Deyu, et al. "Graph contrastive learning with stable and scalable spectral encoding." Advances in Neural Information Processing Systems 36 (2023): 45516-45532.

[2] Hou, Zhenyu, et al. "Graphmae2: A decoding-enhanced masked self-supervised graph learner." Proceedings of the ACM web conference 2023. 2023.

[3] Li, Wen-Zhi, et al. "Homogcl: Rethinking homophily in graph contrastive learning." Proceedings of the 29th ACM SIGKDD conference on knowledge discovery and data mining. 2023.

[4] In, Yeonjun, Kanghoon Yoon, and Chanyoung Park. "Similarity preserving adversarial graph contrastive learning." Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023.

[5] Xiao, Teng, et al. "Simple and asymmetric graph contrastive learning without augmentations." Advances in neural information processing systems 36 (2023): 16129-16152.

[6] Wan, Guancheng, et al. "S3GCL: Spectral, swift, spatial graph contrastive learning." Forty-first International Conference on Machine Learning. 2024.

[7] Ji, Cheng, et al. "Regcl: Rethinking message passing in graph contrastive learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 8. 2024.

### Questions
1. in the feature construction, can authors provide more details on how to obtain the soft labels?

2. Regarding feature perturbation, do the authors apply normalization to the node features before feeding them into the GNN during both the pre-training and linear probing stages?

### Soundness
2

### Presentation
3

### Contribution
3
