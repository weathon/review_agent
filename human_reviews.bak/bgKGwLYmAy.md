# DGTAT: DECOUPLED GRAPH TRIPLE ATTENTION NETWORKS

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5

## Abstract
The Message Passing Neural Network (MPNN) is a foundational paradigm in graph learning algorithms, demonstrating remarkable efficacy in its early implementations. Recent research has focused on using Transformer on graph data or combining Transformer with MPNNs to address issues like over-squeezing and over-smoothing while capturing long-range dependencies. However, Graph Transformers (GT) often perform poorly on small datasets. More seriously, much position and structure information encoded by GT-based methods is coupled with node attribute information, affecting node attribute encoding while propagating structure and position information, implicitly impacting on expressiveness.  In this paper, we analyze the factors influencing the performance of graph learning models. Subsequently, we introduce a novel model, named DECOUPLED GRAPH TRIPLE ATTENTION NETWORKS (DGTAT). Based on the MPNN+VN paradigm and a sampling strategy, DGTAT effectively decouples local and global interactions, separates learnable positional, attribute, and structural encodings, and computes triple attention. This design allows DGTAT to capture long-range dependencies akin to Transformers while preserving the inductive bias of the graph topology. As a result, it exhibits robust performance across graphs of varying sizes, excelling on both large and small datasets. DGTAT achieves state-of-the-art empirical performance across a variety of node classification tasks, and through ablation experiments, we elucidate the importance of each decoupled design factor within the model. Compared to GT-based models, our model offers enhanced interpretability and flexibility.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper analyzes the factors influencing the performance of graph learning models. The authors proposes a model named DGTAT, which is based on MPNN and a sampling strategy. The proposed method decouples local and global interactions, separates learnable positional, attribute, and structural encodings, and computes triple attention. This design allows DGTAT to capture long-range dependencies akin to Transformers while preserving the inductive bias of the graph topology.

### Strengths
1. The idea of decoupling local and global interactions, as well as separating learnable positional, attribute, and structural encodings, is interesting;

2. The proposed method combines several techniques, i.e. laplacian, random walk, positional encoding, GNN, which is solid;

### Weaknesses
1. The major concern is the experimental result of the proposed method. According to Table 3 and Table 4, the proposed method does not show significant improvement over baseline methods. Most of the numbers are quite close to the previous SOTA results. It is unclear to see from the results that the proposed techniques in this paper works.

2. There is no experimental result showing that how the proposed components in this paper performs. It is necessary to do ablation study to see how the performance changes with/without a particular module.

3. The presentation of this paper could be improved. The authors introduces a lot of notations, and sometimes it is hard to find the actual meaning of a particular notation. Figure 1 and figure 2 is also confusing and hard to understand.

### Questions
see weaknesses above

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes to decouple (position, structure, attribute) and (global interaction local interaction) in graph transformers with a novel model DGTAT. DGTAT is consistently effective on both Homophilic and heterophilic graphs at different scales.

### Strengths
S1 The proposed method is consistently effective on both Homophilic and heterophilic graphs at different scales.

S2 The proposed method avoids over-smoothing and has better expressivity.

### Weaknesses
W1 The contribution of the paper is marginal. The performance improvement is most likely due to the incorporation of existing techniques like MPNN+VN, LapPE, JaccardPE, SE. 

W2 Why we need decoupling in graphs is not well discussed. There are only vague claims on page 14 stating that "compared to the GT with SE/PE, through the decoupling of PE, SE and AE, DGTAT can distinguish some graphs with more sensitive position, structure, and attribute information that coupled PS/SE cannot learn" (this is an assumption, not an explanation/analysis). Figure 5 also cannot support this claim well.

W3. Only node classification experiments are performed. In contrast, standard GTs are evaluated by graph classification.

### Questions
Explain W2 please.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a novel graph neural network architecture called DGTAT that decouples the propagation of positional, structural, and attribute information to improve model expressiveness and interpretability. The key ideas are: 1) Use dedicated encodings to represent positional and structural information separately from attributes; 2) Compute triple attention based on positional, structural, and attribute information; 3) Sample relevant nodes based on positional/structural attention for capturing long-range dependencies. Experiments show SOTA results on node classification tasks.

### Strengths
- Clearly motivates the need for decoupling different types of information in GNNs, both theoretically and empirically.
- Proposes a clean design framework to achieve decoupling of positional, structural, and attribute information.
- Achieves SOTA results across multiple datasets, especially on heterophilic graphs.
- Ablation study illustrates the contribution of each decoupled component.

### Weaknesses
- The sampling strategy based on positional/structural attention is heuristic and may not optimally capture long-range dependencies.
- Increased model complexity due to decoupled computations and triple attention.
- Lacks analysis of computational efficiency compared to baselines.

### Questions
- How is the sampling distribution optimized during training? Is there a learnable component?
- What is the empirical complexity of DGTAT compared to GT and MPNN baselines?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
