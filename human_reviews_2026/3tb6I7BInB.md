# Debugging Tabular Log as Dynamic Graphs

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 8, 2, 2

## Abstract
Tabular log abstracts objects and events in the real-world system and reports their updates to reflect the change of the system, where one can detect real-world inconsistencies efficiently by debugging corresponding log entries. However, recent advances in processing text-enriched tabular log data overly depend on large language models (LLMs) and other heavy-load models, thus suffering from limited flexibility and scalability. This paper proposes a new framework, GraphLogDebugger, to debug tabular log based on dynamic graphs. By constructing heterogeneous nodes for objects and events and connecting node-wise edges, the framework recovers the system behind the tabular log as an evolving dynamic graph. With the help of our dynamic graph modeling, a simple dynamic Graph Neural Network (GNN) is representative enough to outperform LLMs in debugging tabular log, which is validated by experimental results on real-world log datasets of computer systems and academic papers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes a method for transforming tabular logs into dynamic graphs to perform anomaly detection using a temporal graph architecture.
The motivation is to move beyond text-based or static-graph log models toward a 
representation that captures temporal and relational structure without relying on
 large language models.

The idea is potentially interesting. However, the paper currently suffers from missing clarity and reproducibility details, which make it impossible to fully assess the true contribution.

### Strengths
1. Lighter than full LLM pipelines:
The method avoids the computational overhead of large language models, offering 
a practical and efficient alternative for log understanding.

2. Related-work coverage broad (even if slightly disorganized):
The authors review a wide range of related work, showing awareness of the field 
and positioning their approach within both log-analysis and dynamic-graph research.

### Weaknesses
1. Clarity and Formalism
* Many definitions (e.g., of event nodes, object nodes, and anomalies) are vague 
or semi-formal.
* The notation and formal setup are fragmented, critical parts are only in the appendix. This
interrupts reading flow.
* Key concepts (such as the distinction between “features” and “objects”) remain unclear to me.

2. Model Definition

Many aspects of the model are unclear and need clarification:
* Mapping from logs to graph structure unclear: I cannot fully understand how log entries are converted into nodes and edges.
The text suggests that each log line corresponds to an event node, but I do not understand 
whether this node connects to multiple “object” nodes (e.g., sender, receiver, message) or 
whether each object is instantiated once per entry.
A clearer formal definition and an illustrative example would be very helpful.

* Confusing color and node-type semantics in figures:
The graphical illustration (e.g., pink, yellow, red nodes) is confusing.
The “data receiver” appears in red in the log table but pink in the graph; 
yellow and pink nodes seem to denote different entity types but are not labeled.
The lack of a legend or consistent mapping confuses me in how the graph 
encodes events and objects. It is also confusing, because the table in the 
figure has many repeated entries but with different timestamps. This would lead to 7 
distinct snapshots with three nodes each, no?

* Notation scattered and incomplete:
Important variables are introduced late or only in the appendix.
This disrupts the narrative flow and hinders understanding.

* Online vs. offline setting introduced too late:
The paper suddenly states on page 4 that it operates in an “online” setting,
 which makes earlier descriptions unclear.

* Unclear definition of anomalies:
The paper informally states that a snapshot can be labeled as anomalous but never 
defines what constitutes an anomaly in graph terms.
Concrete examples of anomalies would help readers understand what constitutes an 
“anomalous” object or structure.

* Unclear concepts of “object” and “feature” nodes:
The text claims that features and objects are “mutually convertible” without 
further explanation. What are features, why are they needed in contrast to objects, and are they even used?

* Temporal granularity of graph snapshots:
What is the granularity? For example, if two log entries happen one second apart, 
does this lead to two separate graph snapshots?

3. Experimental Design and Reproducibility

* It seems like there is only a train and a test set, no validation set, making it unclear how hyperparameters were determined.

* No information on hyperparameters or tuning is provided.

* No code or implementation details shared, thus no reproducibility.

* Suggestion: Algorithms 1 and 2 describe standard training routines that could be moved to the appendix 
to make room for more substantial model details.

4. Dataset and Evaluation

* The datasets appear to contain no real anomalies, only artificially injected ones, 
thus, if I am not mistaken, reducing the task to link prediction rather than anomaly detection.

* The authors should clearly state what constitutes an anomaly in each dataset 
and discuss whether the injected anomalies reflect realistic log patterns.

* There is no discussion of a validation split.

* The authors limit logs to 20,000 entries to compare with RAG baselines. 
While understandable, the paper claims scalability: For this reason, the authors 
should demonstrate that the approach also works for longer logs.

5. Comparison and Baselines

* The paper does not compare against existing graph-based log anomaly detection methods, 
despite citing them.

* There is no ablation or analysis of the proposed approach, leaving unclear which 
design choices affect performance.

Minor Comments

M1. Related work structure:
While comprehensive, the related work section is oddly organized: it first discusses 
tabular and LLM-based methods, then jumps to general dynamic graphs (unrelated to logs),
 and then back to log anomaly detection. 
 The section would read better if restructured thematically.

M2. LLM inconsistency:
The paper claims to avoid LLMs but still uses a pretrained transformer (e.g., MiniLM).
This is still a language model, thus the claim of being “non-LLM” is somewhat misleading.

M3. Not rigorous definitions:
Many key statements (e.g., definition of anomalies, formal model description) lack precision 
or mathematical grounding.

M4. Section 4.3:
“For details, our GNN θ takes the dynamic graph snapshot Gn and the incoming subgraph 
gn  as inputs and predicts the likelihood of all links in gn.”
This is not clear. How can parts of the graph 
Gn be “already known” in the online setting?

### Questions
1. How exactly are “objects,” “features,” and “events” defined and interconnected in the graph?

2. Does each log entry correspond to a single new event node, or can multiple entries share the same event? Are logs in one subgraph connected by sharing the same objects?

3. What is the temporal granularity of snapshots (e.g., one per second, one per event)?

4. How realistic are the generated anomalies in your experiment? Could you please give examples of why your test setup is realistic for an anomaly detection scenario? How is your task different from standard link prediction?

5. How were hyperparameters tuned in the absence of a validation set?

6. Can you provide or release code for reproducibility?

7. Why are no dynamic-graph baselines included in the comparison?

8. Could the model scale to longer logs? If yes, could you include experiments?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes GraphLogDebugger, a novel framework designed to debug tabular logs by transforming them into dynamic heterogeneous graphs. The central idea is to model tabular logs as evolving systems, where objects and events are represented as nodes in a graph, and the edges represent their relationships. The framework uses a dynamic GNN to perform link prediction for anomaly detection. Experiments demonstrate that GraphLogDebugger is not only more efficient than traditional LLMs but also provides better scalability for real-time anomaly detection in high-throughput environments.

### Strengths
1. The integration of dynamic heterogeneous graphs to represent tabular logs and  redefine the object-event connections as link prediction on the evolving graph are unique.

2. One of the key strengths is its efficiency. Unlike LLM-based models that suffer from high computational overhead, GraphLogDebugger delivers high throughput and low GFLOPs, making it suitable for real-time applications in large-scale systems. The approach demonstrates significant improvements in processing speed compared to RAG and LLM-based methods, which is crucial for handling streaming log data.

3. The authors validate their framework on a variety of datasets from different domains, including Arxiv, HDFS, Analyst, and Landslide. GraphLogDebugger consistently outperforms existing baselines in terms of both detection effectiveness and efficiency. This diverse testing strengthens the claims of generalizability and robustness.

### Weaknesses
1. In some cases, GraphLogDebugger performed worse than RAG, like your Case 3 in the case study. The authors have mentioned that there may be limitations in scenarios with low connectivity between objects and events, where the dynamic graph might not provide enough information for accurate anomaly detection. It would be better if the authors could provide further analysis of these edge cases and discuss ways to improve the model in scenarios where the object-event graph is sparse or the event history is limited.

2. The paper uses GAT as the backbone for the dynamic GNN. While this is a reasonable choice, it would be better if provide enough justification for choosing GAT over other types of GNNs like GCNs or GraphSAGE, especially in the context of dynamic graphs.

### Questions
1. The framework uses GAT for anomaly detection. Could you justify why GAT is more suitable than other GNN architectures like GCN, GraphSAGE, in this specific context of dynamic log graphs? What specific advantages does GAT offer over these other architectures?

2. In Case 3, GraphLogDebugger fails to predict an anomaly correctly. Could you discuss in more detail how the model performs in sparse graph settings? Are there any strategies you could employ to improve its performance in such scenarios?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents GraphLogDebugger a framework for online anomaly detection in tabular logs using dynamic graph neural networks (GNNs).  Each log entry is represented as a small heterogeneous subgraph connecting object nodes and an event node; new log entries are incrementally merged into a dynamic graph over time. The anomaly detection task is formulated as a link prediction problem, where abnormal connections correspond to low predicted likelihoods under the learned GNN. Empirical validation on four datasets (ArXiv, Analyst, Landslide, and HDFS) shows that the proposed model outperforms both MLP and retrieval-augmented generation (RAG) baselines in accuracy and computational efficiency.

### Strengths
- The idea of framing tabular log debugging as online link prediction on a dynamic, heterogeneous graph is interesting. This formulation could inform future work on efficient, online, graph-based anomaly detection.
- Reporting GFLOPs and throughput alongside accuracy is valuable, highlighting the model’s efficiency compared to RAG baselines.
- The paper explains the framework clearly, mapping each anomaly type to a specific graph operation. Figures and algorithms help the reader understand the pipeline and the overall architecture.

### Weaknesses
Limited novelty:
- The idea of representing log data as graphs has been explored in prior works (e.g., [LogKG](https://ieeexplore.ieee.org/document/10179162/), [GLAD](https://arxiv.org/pdf/2309.05953), [GuARD](https://arxiv.org/pdf/2412.03930v2)). A comparison with these methods is therefore necessary to clarify the method’s novelty and relative performance.

Empirical design lacks clarity and rigor:
- The experimental protocol lacks a validation set, making it unclear whether hyperparameters are inadvertently tuned on the test set. This raises concerns about the validity of reported performance.
- The hyperparameter search space for both the proposed model and the baselines is not reported, which reduces the transparency and reproducibility of the experimental setup.
- Several experimental choices lack justification, such as the decision to train GNN models for only ten epochs, which may be insufficient for convergence.
- The rationale for selecting GAT as a backbone is unclear, particularly given the existence of more suitable temporal GNN architectures (e.g., [DCRNN](https://arxiv.org/abs/1707.01926), [GCLSTM](https://arxiv.org/abs/1812.04206)) that are explicitly designed for dynamic data or heterogeneous GNNs (e.g., [HGT](https://arxiv.org/abs/2003.01332), [HTGNN](https://arxiv.org/abs/2110.13889)) that can handle (temporal) heterogeous graphs.
- The paper does not specify which performance metric is optimized during training, making the optimization objective ambiguous.
- It is unclear whether the RAG-based baselines were appropriately fine-tuned for the target tasks; if not, the resulting comparison may not accurately reflect the relative capabilities of the methods.

  
Limited experimental validation:
- In each iteration, GraphLogDebugger operates on a graph whose number of nodes increases with the number of log entries, since at least one event node is added per step. The authors should include a memory consumption analysis compared to baseline methods to assess scalability.
- Additional quantitative analyses, such as confusion matrices, would help illustrate the model’s effectiveness in distinguishing normal and anomalous events.
- The baseline coverage is somewhat limited. Indeed, only a simple MLP and LLM-based RAG methods are considered. The paper does not compare with: (i) established online log anomaly detectors such as [DeepLog](https://doi.org/10.1145/3133956.3134015), [LogAnomaly](https://nkcs.iops.ai/wp-content/uploads/2019/06/paper-IJCAI19-LogAnomaly.pdf), or [AutoLog](https://doi.org/10.1016/j.eswa.2021.116263), which are standard references in this area; (ii) temporal and/or heterogenous GNNs (see above); (iii) previous works on tabular log anomaly detection (see above), or (iv) classical [GNNs for Anomaly Detection](https://arxiv.org/abs/2409.09957). 
- The experimental evaluation is limited to 4 tasks. Authors should consider including more datasets, such as from [LogHub](https://arxiv.org/pdf/2008.06448), to provide a more comprehensive assessment of the method.

Minors:
- Line 259: *is → are*  
- Section 3 (Preliminaries): The formalism introduced in this section is never used in later parts of the paper, while definitions related to dynamic graphs are completely missing.

### Questions
See weaknesses

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
4

### Summary
This paper proposes GraphLogDebugger, a framework that models tabular logs as heterogeneous dynamic graphs (with object and event nodes) and performs online anomaly detection via link prediction using a lightweight GNN. The method avoids reliance on large language models (LLMs) and demonstrates good performance across four diverse datasets, outperforming both simple MLP baselines and RAG-based LLM approaches in both accuracy and efficiency.

### Strengths
- The formulation of tabular log debugging as a dynamic heterogeneous graph link prediction task is well-motivated.
- The framework is efficient, scalable, and avoids the computational burden of LLMs.
- Experiments are conducted across several domains (academic, system, finance, geology), and the ablation studies support design choices.

### Weaknesses
- Novelty is limited: the core idea — modeling logs as dynamic graphs for anomaly detection via link prediction — was already introduced in TempoLog (arXiv:2501.12166, Jan 2025). TempoLog focuses on discrete event logs (parsed into templates) and constructs a homogeneous temporal graph of template dependencies, and this work targets structured tabular logs with explicit object – event semantics and builds a heterogeneous graph, the high-level paradigm (dynamic graph + link prediction for log anomaly detection) is similar. The authors did not cite or discuss TempoLog. This omission significantly undermines the claimed novelty.

Qi et al., Beyond Window-Based Detection: A Graph-Centric Framework for Discrete Log Anomaly Detection, https://arxiv.org/abs/2501.12166,  Jan 2025.

- Baseline comparison is limited: This paper positions itself as a novel, general-purpose framework for tabular log debugging by contrasting primarily against a simple MLP (Multi-Layer Perceptron) and heavyweight RAG+LLM (Retrieval-Augmented Generation + Large Language Model) pipelines. However, it omits comparisons with a range of recent, highly relevant graph-based log anomaly detection methods — most notably TempoLog, GLAD-PAW, GLAD, LogGD, and OCDiGCN (see below the references). These recent methods all belong to the family of graph-based log analysis, sharing the common approach of modeling logs as dynamic or semantic-rich graphs. This significant omission could undermine the validity of the evaluation, making it difficult to determine the effectiveness of the proposed approach. 

Y. Wan, Y. Liu, D. Wang, and Y. Wen, “Glad-paw: Graph-based log anomaly detection by position aware weighted graph attention network,” in Pacific-Asia conference on knowledge discovery and data mining, pp. 66–77, Springer, 2021.

Y. Xie, H. Zhang, and M. A. Babar, “Loggd: Detecting anomalies from system logs with graph neural networks,” in 2022 IEEE 22nd International conference on software quality, reliability and security (QRS), pp. 299–310, IEEE, 2022.

Z. Li, J. Shi, and M. Van Leeuwen, “Graph neural networks based log anomaly detection and explanation,” in Proceedings of the 2024 IEEE/ACM 46th International Conference on Software Engineering: Companion Proceedings, pp. 306–307, 2024.

Y. Li, Y. Liu, H. Wang, Z. Chen, W. Cheng, Y. Chen, W. Yu, H. Chen, and C. Liu, “Glad: Content-aware dynamic graphs for log anomaly detection,” in 2023 IEEE International Conference on Knowledge Graph (ICKG), pp. 9–18, IEEE, 2023.

- Concern regarding synthetic anomaly injection: The proposed method is evaluated using anomalies generated by randomly swapping existing objects or events within the log (Section 5.1). While this setup enables controlled experiments, it may not reflect realistic failure modes in real-world tabular logs. Such "swapped" entries often remain syntactically and semantically valid. In contrast, real systems typically encounter many anomaly scenarios such as semantic contradictions, temporal violations, out-of-distribution values, or structural inconsistencies — none of which can be captured by simple entity swapping. This synthetic anomaly model significantly weakens the validity of the evaluation and raises doubts about the method’s practical utility.

⦁ Dataset: The evaluation uses HDFS as the system log dataset. This dataset is relativly easier. The authors may consider other commonly-used system log datasets such as BGL, Thunderbird, and Spirit.

### Questions
- How is the proposed approach compared with recent graph-based log anomaly detection baselines? 
- Do the experiments reflect real-world anomaly scenarios ?

### Soundness
2

### Presentation
3

### Contribution
2
