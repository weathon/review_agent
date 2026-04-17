# Full-Graph vs. Mini-Batch Training: Comprehensive Analysis from a Batch Size and Fan-Out Size Perspective

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Full-graph and mini-batch Graph Neural Network (GNN) training approaches have distinct system design demands, making it crucial to choose the appropriate approach to develop. A core challenge in comparing these two GNN training approaches lies in characterizing their model performance (i.e., convergence and generalization) and computational efficiency. While a batch size has been an effective lens in analyzing such behaviors in deep neural networks (DNNs), GNNs extends this lens by introducing a fan-out size, as full-graph training can be viewed as mini-batch training with the largest possible batch size and fan-out size. However, the impact of the batch and fan-out size for GNNs remains insufficiently explored. To this end, this paper systematically compares full-graph vs. mini-batch training of GNNs through empirical and theoretical analyses from the view of the batch size and fan-out size. Our key contributions include: 1) We provide a novel generalization analysis using the Wasserstein distance to study the impact of the graph structure, especially the fan-out size. 2) We uncover the non-isotropic effects of the batch size and the fan-out size in GNN convergence and generalization, providing practical guidance for tuning these hyperparameters under resource constraints. Finally, full-graph training does not always yield better model performance or computational efficiency than well-tuned smaller mini-batch settings. The implementation can be found in the github link: https://github.com/LIUMENGFAN-gif/GNN_fullgraph_minibatch_training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates mini-batch training for GNNs. It provides theoretical analysis on both convergence and generalization within the mini-batch training framework, and conducts experiments to validate the proposed theory.

### Strengths
S1: Comprehensive theoretical analysis.

S2: The choice of datasets and models are reasonable.

S3: Results make sense.

### Weaknesses
W1: The results are somewhat plain. The first notable conclusion is that when trained with the MSE loss, larger batch size leads to slower convergence. But empirically, this is validated using node classification datasets. The second claim is that "mini-batch could leads to better generalization than full-graph", but in Fig. 4, the influence of batch size seems marginal. Reducing fan-out size is useful, but to the best of my knowledge, community is already aware of that, due to those DropEdge papers.

W2: Presentation could be further improved. Lots of statements in the experiments section refer to the Appendix N. If the information in the appendix is important enough, it should be presented in the main paper. Defer them to the appendix but continually mention them will not save space. For example, line 363, line 415, figures in the appendix is mentioned before figures in the main paper.

### Questions
Q1: May I ask what is the target loss in iteration-to-loss, the target accuracy in iteration-to-accuracy and time-to-accuracy? I try to look them in appendix N, but there is only description, not the actual value. If we change these value, will the conclusion change?

Q2: What is the setting and hyperparameters of Fig. 4(a)?

Q3: In the mini-batch training, different batch sizes means different amount of samples the model saw after one iteration. Is it fair to compare the number of iteration in such context? 

Q4: The most relevant work is [1]. Could you briefly summarize the most important new insight of your paper in comparison to [1] (except using iteration-to-accuracy to avoid hardware influence)?

[1] Bajaj, Saurabh, et al. "Graph Neural Network Training Systems: A Performance Comparison of Full-Graph and Mini-Batch." Proceedings of the VLDB Endowment 18.4 (2024): 1196-1209.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper aims to provide a systematic and comprehensive study on comparing the full-graph vs. mini-batch GNN training, focusing on the impact of batch size and fan-out size. Theoretical analysis shows that increasing batch size slows convergence under MSE but accelerates it under CE, while increasing fan-out size consistently improves convergence. The provided theories show batch size has a greater impact on GNN optimization dynamics, while the fan-out size more strongly affects GNN generalization. Empirical results validate these findings, using a hardware-agnostic iteration-to-accuracy metric. Experiments show that moderate fan-out size balances speed and efficiency, and overly large batch size and fan-out size degrades accuracy. Overall, well-tuned mini-batch training can match or surpass full-graph training in both performance and computational efficiency.

### Strengths
1. Both main observations—(Obs.1) convergence is more sensitive to batch size, and (Obs.2) generalization is more sensitive to fan-out size - are consistently supported by theoretical derivations and empirical results across multiple datasets and GNN models.
2. The paper addresses a practically important topic - understanding the differences between full-graph and mini-batch GNN training， which is beneficial to large-scale GNN training.
3. Code is provided for reproducibility.

### Weaknesses
1. The study only covers node-wise sampling; other major mini-batch paradigms (e.g., graph-level and layer-wise sampling) are omitted, which limits the comprehensiveness of the conclusions.
2. The main conclusion—that full-graph training does not always outperform well-tuned mini-batch training—has already been reported in prior empirical works (e.g., Bajaj et al., 2024). The novelty thus lies mainly in providing a formal theoretical framing rather than discovering a new phenomenon.

### Questions
1. Scope of Mini-Batch Paradigms:
The paper focuses on node-wise neighbor sampling when discussing mini-batch training. However, mini-batch GNNs also include subgraph-level (e.g., GraphSAINT, Cluster-GCN) and layer-wise (e.g., LADIES) sampling approaches, which may exhibit different trade-offs between variance, memory, and structural bias. Could the authors clarify whether their theoretical framework—especially the convergence and Wasserstein-based generalization analysis—can extend to graph-level or layer-wise mini-batching? Have the authors considered running at least small-scale comparisons (e.g., Cluster-GCN vs. neighbor sampling) to test whether the same sensitivity patterns still hold under these paradigms?


2. Conceptual Contribution and Practical Value:
The concluding message—“full-graph training does not always outperform well-tuned mini-batch training”—has been noted in prior empirical studies (e.g., Bajaj et al., 2024). What remains unclear is the new insight or practical benefit beyond formalizing known intuitions. The two key observations (Obs.1: convergence more sensitive to batch size; Obs.2: generalization more sensitive to fan-out size) are interesting, but they seem to reaffirm what many practitioners already do during hyper-parameter tuning. Could the authors elaborate on what new guidance or algorithmic implications their framework provides? How does this work go beyond offering “interpretive understanding” and translate into actionable or theoretical advancement for future GNN training design?

### Soundness
3

### Presentation
3

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
This paper presents a comparative study between full-batch and mini-batch training in GNNs, in terms of model convergence, generalization and computatioal efficiency. Based on their theoretical analysis and by using hardware-agnostic measures, the authors aim to guide practitioners on selecting the two key parameters of mini-batch training: batch size and fan-out size (i.e. the number of neighbors to be sampled for each node in the batch).

### Strengths
1. This work sheds light on factors such as selecting between mini and full batch training and tuning subsequent hyperparamers that are often overlooked and decided without much thought but that impact model performance in multiple ways.

2. Insights stem from theoretical analysis and are backed by comprehensive experiments. 

3. The paper is well-organized.

### Weaknesses
1. The figures are a little difficult to read and follow. For e.g. in Fig 1, it would be better to place the legend as a single horizontal line above or below the subplots instead of repeating it in each subplot and covering the information that needs to be displayed. Also, it would make the comparison easier to read if the bar grouping is done based on dataset rather than training setting, for example blue represents mini-batch and orange represents full batch, and the each groups of bars represents a dataset. The later heatmaps are also a little overwhelming with lots of numbers thrown in. Perhaps only the numbers that reveal crucial information can be retained and the general trends can be left to be depicted by the heatmap color. This would make it easier for the reader to pick up key insights more easily than going through a lot of data. 
 
2. The number of layers of the GNNs in the experiments is restricted to 3, hardly enough to be considered 'deep', so oversmoothing does not seem to be a plausible explanation on lines 368 and 369. 

3. Regarding 'observed low variance' of experiment results repeated thrice, while I trust the authors on this claim, it would still be better to provide some sort of evidence for it, in table format perhaps, in the appendix atleast as standard scientific practice.

### Questions
1. Is there likely any role of the homophily level of the input graph in determining what batch and fanout sizes would be better for training a GNN for the task?

2. While the authors provide results for three GNN architectures of different natures in their experiment, could they also explicitly comment on whether the same trend was observed across them or was their any variation? For e.g. is a particular architecture (or type thereof) more suited to a training paradigm?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a systematic empirical and theoretical analysis comparing full-graph and mini-batch training for GNNs. The authors propose using the dual lenses of batch size and fan-out size as a unified framework for this comparison, noting that full-graph training is a special case with maximal values for both parameters. The work aims to address a significant gap in the literature by providing a holistic understanding of how these hyperparameters impact optimization dynamics convergence and generalization, moving beyond isolated or purely hardware-dependent comparisons. The central, counter-intuitive claim supported by the study is that full-graph training does not always outperform well-tuned mini-batch training in terms of performance or efficiency.

### Strengths
1. The paper's core strength is its unified framework for analyzing full-graph and mini-batch training through batch and fan-out sizes.

2. The topic is of high importance to the GNN community. As graph datasets grow larger, the question of which training paradigm to choose and how to configure it is critical. This paper provides a foundational study that can guide future research in optimized training algorithms and systems

3. This paper is easy to understand.

### Weaknesses
1. While the theoretical analysis is a key contribution, the paper would be strengthened by a more explicit discussion of the assumptions made in the theorems (e.g., Theorem 1 & 2 on optimization dynamics). The claim of "better aligning with practice" by considering irregular graphs and non-linear activations is commendable, but the specific limitations and the gap between the theoretical model and real-world GNN training dynamics should be clarified to assess the generality of the conclusions.

2. The empirical evaluation is currently focused on transductive node classification. While this is a standard and important task, the findings' applicability to other fundamental GNN tasks like inductive learning, link prediction, or graph classification remains unverified. Different tasks may exhibit different sensitivities to batch and fan-out sizes, and demonstrating the generality of the findings across tasks would significantly increase the impact.

### Questions
Do the same conclusions hold for other types of tasks?

### Soundness
3

### Presentation
3

### Contribution
3
