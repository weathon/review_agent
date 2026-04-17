# S$^2$-ETC: Semantic-Structure Aware Encrypted Traffic Classification via Hyper-Bipartite Graph

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
In the era of pervasive encryption, encrypted traffic classification serves as a fundamental technique, underpinning diverse applications including intrusion detection and network management. 
It is commonly approached with deep learning methods that rely on semantic feature extraction or on traffic interaction graphs; however, these approaches suffer from three major limitations: semantic signal obfuscation under strong encryption, which suppresses distinguishable single-flow semantics and undermines accuracy and robustness; inter-flow over-squashing, which constrains the expressivity of interaction graphs and degrades classification performance; and the absence of intra–inter fusion combined with limited scalability, which prevents effective reconciliation of semantic and structural cues and hinders deployment on massive traffic graphs.
To address these challenges, we propose S$^2$-ETR (Semantic-Structure Encrypted Traffic Representation), a novel framework that integrates traffic semantics with communication topology graph. The framework includes a Hyper-Bipartite Graph (HBG), which takes two branches to fuse topology and semantic features. The topology branch models structural relations with an IP–flow bipartite graph, decoupling flows from communication entities to mitigate overfitting. The semantic branch employs a lightweight adapter to capture flow semantics, enhancing cross-domain robustness; meanwhile, it constructs semantic hyperedges via implicit hypergraph learning, propagating global semantic representations without extra information. Finally, a conditional probability–based hierarchical classification strategy is introduced to augment scalability on massive traffic graphs. Furthermore, through a mathematical proof, we demonstrate that HBG reduces long-range dependencies and over-squashing, leading to better efficacy and generalization compared to traditional topology graphs. Experimental results show that S$^2$-ETR consistently achieves state-of-the-art performance across 5 datasets of varying scales, outperforming 15 baselines by 2.4\%–17.1\% on encrypted application classification datasets, and surpassing the best baseline by 9.2\% on the more complex and challenging IoT dataset.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes S²-ETR, a novel framework for encrypted traffic classification that integrates semantic and topological features via a Hyper-Bipartite Graph (HBG). The method consists of an adapter, an HBG module, and a hierarchical classifier, and is evaluated on four datasets, showing superior performance over 11 baselines.

### Strengths
- Comprehensive experiments across multiple datasets, scales, and scenarios;

- Strong performance in efficiency and scalability, especially in large-scale hierarchical classification.

### Weaknesses
- [Mandatory] Many statements in the Introduction require additional references or other forms of support, such as the over-squashing problem in topology-based GNNs.

- [Mandatory] Anonymization (e.g., IP/port replacement) is a strong prior and may introduce bias.

- [Mandatory] Some definitions referenced in Section 3 are supposed to be provided in Section 2, but this part of the definitions is missing in Section 2.

- [Mandatory] The theoretical analysis is based on MPNN, but its connection to the attention-based IHL used in the model is not clearly established;

- [Mandatory] Key implementation aspects of IHL (e.g., initialization, convergence criteria) are omitted;

- [Mandatory] The fusion weight $\alpha$ is heuristic and lacks theoretical or empirical justification;

### Questions
Please refer to Weaknesses. Btw, I have some optional questions:

- [Optional] How are hyperedges initialized in IHL, and did you encounter non-convergence during iterative updates?

- [Optional] Were the GNN baselines ensured to have graph construction comparable to the topological branch of HBG?

- [Optional] Could anonymization discard discriminative semantic patterns (e.g., port-based features), thus limiting performance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a framework for encrypted traffic classification designed to address issues of overfitting and generalization. The proposed method is centered on a model, the S^2-ETR, which implements a Hyper-Bipartite Graph (HBG) structure.

The framework utilizes a dual-branch architecture to process both semantic and topological information from network traffic. For the semantic branch, the system preprocesses PCAP data by extracting the first six packets of each flow, consisting of an 80-byte header and a 240-byte payload. Within this branch, IP addresses and port numbers are anonymized. The resulting data is then passed to a configurable encoder, referred to as an adapter (e.g., 1D/2D CNN, ResNet, Linear), to generate a content-based representation. For the topological branch, the original, non-anonymized IP information is used to construct an IP-Flow bipartite graph. The model learns embeddings for the IP nodes within this graph to represent structural relationships.

The paper's theoretical analysis posits that the HBG structure is intended to mitigate the over-squashing problem common in graph neural networks. It is argued that this is achieved by replacing long, multi-hop information paths with constant-depth aggregation steps, thereby reducing long-range dependencies. The framework was evaluated on three datasets (CICIoT2023, ISCXVPN2016, and USTC-TFC2016) using the four different encoder configurations.

### Strengths
1. The main new idea in this work is the "Hyper-Bipartite Graph" (HBG), which is a framework with two parts. The first part (the topological branch) models the problem in a smart way by representing how IP addresses and data flows are connected. The second part (the semantic branch) uses an attention mechanism to cleverly learn more complex relationships.
2. The authors present and provide a mathematical proof explaining why their proposed HBG architecture is superior to traditional IP topology graphs.
3. The method was evaluated through comprehensive comparative experiments against eleven deep learning-based, pre-training-based, and graph neural network-based models across three distinct datasets. Some of the dataset choices are intriguing.

### Weaknesses
1. This paper is heavily packed with technical details and expresses many concepts and designs through mathematical formulas. However, most of these equations do not actually aid understanding; instead, they significantly reduce the readability of the paper. I suggest reducing the use of formulas in the main text. Rigorous definitions and proofs can be moved to the appendix, while the main body should use simpler and more intuitive representations such as figures.
2. The authors claim that S^2-ETR addresses the limitations of deep learning methods, pretraining-based methods, and topology-based GNNs. However, the paper only discusses improvements over topology-based GNNs. The authors should elaborate more on why a topology-based approach was adopted and what theoretical motivations underlie this choice, in order to strengthen the overall motivation.
3. A short overview at the beginning of Section 3—summarizing Figure 1’s workflow and core idea—would greatly improve readability and prepare readers for the detailed designs later on.
4. Accuracy and efficiency are claimed as major contributions, possibly explaining why the authors did not employ large models. However, as far as I know, the most efficient methods remain traditional machine-learning approaches such as FlowPrint and AppScanner. Even when using raw packet bytes as input, traditional models like XGBoost can still achieve high accuracy. It is therefore suggested that the authors include comparisons with such traditional baselines.
5. In the experiments, it would be useful to include comparisons with NetMamba, which likewise emphasizes efficiency. In addition, the reported accuracies on ISCX-VPN2016 are noticeably lower than those in the original papers (e.g., YATC and TrafficFormer). This may reflect differences in pretraining data or other factors; please report the exact pretraining datasets used and explain the lower accuracy. Finally, because the performance gap on USTC-TFC2016 is relatively small across methods, adding more datasets would help substantiate the generality of the conclusions.

### Questions
1. Add more discussion on the motivation for adopting a topology-based approach.
2. Restructure Sections 2 and 3 to present the architecture and key designs more intuitively, avoiding excessive reliance on formulas.
3. Include additional datasets and comparative experiments, such as with traditional machine-learning methods (FlowPrint, AppScanner, raw packet–based XGBoost), and NetMamba.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel encrypted traffic classification framework named S2-ETR, designed to address core challenges in existing encrypted traffic identification methods. To this end, the authors developed a dual-branch architecture integrating both semantic modeling and topological modeling, while introducing an implicit hypergraph learning mechanism to adaptively model higher-order relationships between nodes.

### Strengths
1.The paper presents a clearly defined and practically significant problem motivation, accurately identifies key bottlenecks in current encrypted traffic classification, and proposes a targeted architectural design. The research background and motivations are well-justified and persuasive.
2.The dual-branch architecture integrating semantic and topological branches, along with the introduction of IHL, constitutes the core innovation of this work. This design effectively mitigates the over-squashing issue inherent in traditional GNN models while enhancing the complementary relationship between semantic and topological information.

### Weaknesses
1.The statically computed α value fundamentally contradicts the claimed "adaptive learning" capability. Furthermore, while contemporary graph neural networks commonly employ dynamic weighting strategies such as gating mechanisms, this study fails to substantiate the comparative advantages of the fixed-weight approach.
2.The experimental design and argumentation completeness of this paper raise several concerns. Firstly, the comparative experiments on the most challenging large-scale dataset CIC-AndMal2017 are insufficient due to the absence of comparisons with stronger pre-training baseline models like YaTC, which undermines the persuasiveness of the claimed performance superiority. Secondly, the ablation study casts doubt on the effectiveness of the semantic branch, particularly on the USTC-TFC2016 dataset where its removal yields better performance than the full model. Finally, the timeliness of the experimental datasets is inadequate, with most versions being outdated (e.g., from 2016/2017), making it difficult to validate the model's adaptability to contemporary encrypted traffic.

### Questions
1.It is recommended to explore at least one dynamic weighting strategy as a comparative baseline to demonstrate the trade-offs against the static method.
2.On large-scale datasets such as CIC-AndMal2017, it is essential to include comparisons with widely recognized strong baseline models (e.g., YaTC). This is a critical prerequisite for validating whether the method's performance achieves state-of-the-art levels.
3.Regarding the anomalous observation in the ablation study where "removing the semantic branch resulted in improved performance," further tests should be conducted on additional datasets to determine whether this is an isolated case or a general phenomenon.
4.To address concerns about the model's capability to handle modern encrypted traffic, it is strongly recommended to incorporate validation experiments on recently released datasets.

### Soundness
3

### Presentation
3

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
This paper propose S2-ETR , a framework that integrates traffic semantics with communication topology graph for encrypted traffic classification (ETC), effectively alleviating the over-squashing and long-range dependence problems in traditional graph neural networks. Experimental results show that S2-ETR achieves SOTA performance.

### Strengths
+ It provides mathematical proof that the HBG structure is better than traditional IP topology in terms of information dissemination efficiency, which enhances the persuasion of the paper.
+ It was compared with 11 baselines on multiple public datasets (such as CIC-IoT2023, ISCX-VPN2016, USTC-TFC2016), covering different topologies and data sizes.

### Weaknesses
+ The expression of HBG and IHL  is very dense, and some formulas and illustrations are not intuitive enough. Some terms (such as "implicit hypergraph" and "semantic semantic hyperedges") lack clear definitions.
+ The datasets used are old (year 2016) and do not represent the realistic characteristics of real-world encrypted traffic. Please consider evaluating it on the latest dataset, such as CipherSpectrum[S&P 2025].
+ The proposed method alleviates the over-squashing problem, but lacks discussion and comparison with other over-squashing solutions.

### Questions
+ What does semantics mean in encrypted traffic, Why HBG can capture semantics, and how to prove that semantics are indeed captured.
+ Over-squashing is a classic topic in graph neural networks, but the paper lacks discussion and comparison of common solutions. Please clearly state the contribution and innovation of the method in solving over-squashing problems.
+ Why select 500 flows per class? Is this can reflect the distribution of the real world?
+ Why consider CNN, Linear and ResNet? Without considering other network structures, such as transformer.
+ The best recall for the ISCX-VPN2016 dataset in Table 3 should be S2-ETR (ResNet) rather than S2-ETR (2D-CNN).
+ Why S2-ETR (ResNet) performs worse than S2-ETR (2D-CNN) on CIC-IoT2023.
+ Implementation details are inconsistent with settings disclosed in the code, such as epoch, batch size for each datasets.

### Soundness
2

### Presentation
2

### Contribution
2
