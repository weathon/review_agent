# Bridging Indexing Structure and Graph Learning: Expressive and Scalable Graph Neural Network via Core-Fringe

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 5

## Abstract
Existing message passing-based and transformer-based graph neural networks (GNNs) can not satisfy requirements for learning representative graph embeddings due to restricted receptive fields, redundant message passing, and reliance on fixed aggregations. These methods face scalability and expressivity limitations from intractable exponential growth or quadratic complexity, restricting interaction ranges and information coverage across large graphs. Motivated by the analysis of long-range graph structures, we introduce a novel Graph Neural Network called Core-Fringe Graph Neural Network (CFGNN). Our Core-Fringe structure, drawing inspiration from the graph indexing technique known as Hub Labeling, offers a straightforward and effective approach for learning scalable graph representations while ensuring comprehensive coverage of information. CFGNN leverages this structure to enable selective propagation of relevant embeddings through a carefully designed message function. Theoretical analysis is presented to show the expressivity and scalability of the proposed method. Empirically, CFGNN exceeds standard GNNs on tasks including classification and regression, especially for large, long-range graphs where scalability and coverage matter. Ablation studies further confirm the benefits of our core-fringe based graph neural network, including improved expressivity and scalability.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors question whether a GNN model with both scalability and expressivity in learning graph information exists, and propose to use a new message-passing paradigm based on hub labeling. The proposed paradigm only maintains the message-passing between core nodes and their corresponding fringe, which reduces the amount of message-passing operation in regular GNN, while cover all information in the graph. Experimental results show that the method is comparable to existing approach.

### Strengths
The motivation and intuition behind the hud labeling based message-passing is well explained. Overall, the paper is easy to follow with informative illustrations highlighting the proposed methods against existing approaches. 

This approach significantly reduces message-passing operations required for distant nodes to read information from each other.

The approach seems well-attuned to large graphs, and experimental results show that the approach is promising compared to existing ones that potentially require more computations.

### Weaknesses
- The message passing is only between the core and fringe, meaning that none of the local structural information is kept. However, these structures are very important for some applications (including molecular graphs which are the main targets of the work).

- While the performance is on par with the existing approaches, CFGNN is not the clear winner. This is fine, as I believe the key contribution is that CFGNN requires much less computation resources. However, no experimental results verify the claim. (Except transformer model OOM on some datasets that CFGNN can process)

- The hub labeling process is not coupled with the learning process, meaning that the core/fringe assignment can be irrelevant to the actual learning task, which potentially cause the model to ignore information.

- While using the hub core nodes to aggregate information seems natural, it could be detrimental. For example, supernodes exist in the graph. Then, at the end of the message passing, all nodes will have very similar embedding as their messages are all from supernodes. In such cases, node embeddings will be substantially over-smoothed. In contrast, a regular GNN, because every node keeps local connectivity and has message from non-supernodes, is less affected.

### Questions
- How do the actual core nodes look like? Are they supernodes? If they possess similar properties across different datasets, then is it valid to use the same tool to choose core nodes for different data?

- Why do transformer models cause OOM? If you shrink the batch size, you should still be able to train the model.

Also, weaknesses mentioned above.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors proposes a new way of aggregating messages in graph neural networks, called CFGNN.

CFGNN divides nodes into core and fringe groups based on hub labeling.

CFGNN first propogates node embeddings from the fringe groups into the core nodes, then propogates node embeddings from the code nodes to the corresponding fringe groups.

This technique scales the receptive fields of the GNN such that the receptive field is not linear to the number of layers. CFGNN is more computationally efficient than transformers, as it applies sparse message passing along core and fringe nodes. The authors evaluate their approach against 7 datasets and achieve superior performance.

### Strengths
How to extend GNNs receptive field and scalability are both important questions in the field.

The authors propose a model which is scalable and theoretically has large receptive field and beats baselines.

The authors carefully detail their motivations throughout the paper.

### Weaknesses
I have several issues with the work:

1) Novelty
- The paper essentially takes the Graph U-Net[1] approach of pooling and unpooling node embeddings from a core subset of node embeddings, yet the authors do not mention this work.
- Unlike Graph U-Net[1], the core subset of node embedding are determined by hub labeling rather than being learned. I believe this would hurt CFGNN's performance given sufficient data to learn the hub labeling.
- There have been several related works addressing receptive fields of GNNs, which are not compared against: Graph U-Net[1], Deeper GNNs[2], Multihop GNNs[3].
- As a suggestion, I implore the authors to look more into how indexing can improve the scalability of their approach over said existing techniques. However, I believe more explanation and experiments are needed to distinguish this work from others in this aspect.

2) Experiments
- There are GNN baselines [1,2,3] that address the receptive fields issue, which the authors do not compare against.
- The authors do not compare against scalable deep GNN techniques such as sampling[4,5] or distillation[6].
- Given CFGNN tradeoffs the expressive power of graph transformers for increased scalability, the paper lacks scalability studies between CFGNN and Graphnormer, which would clarify when one should be used over the other.
- I am unsure what the * mean?
- There is a lack of ablation study (ex. alternatives to hub labeling for choice in core nodes, how many hub nodes?, etc.)

3) Writing
- I am unsure of the hub labeling notation: what is the difference between c and v in "L(c) and L(v)" in the section "HL-based core-fringe structure"?
- Sections 2.1 and 2.2 could be shortened. Much of the discussion in these sections are well-known (definition of receptive field, linear scalability of receptive field and layer count, and neighborhood definitions of transformers). I appreciate the author's description of these concepts, but believe this section can be greatly shortened to leave more space for the author's own proposals.

References:

[1] Graph U-Nets (ICML2019)

[2] Towards Deeper Graph Neural Networks (KDD2020)

[3] Multi-hop Attention Graph Neural Network (IJCAI2021)

[4] Inductive Representation Learning on Large Graphs (NeurIPS2017)

[5] Training Graph Neural Networks with 1000 Layers (ICML2021)

[6] Graph-less Neural Networks: Teaching Old MLPs New Tricks Via Distillation (ICLR2022)

### Questions
Please see the weaknesses section. I am mainly concerned with:
1) What is the relationship between CFGNN and Graph U-Net? Are there any empiracal comparisons between the two?
2) What is the relationship between CFGNN and scalability techniques applied on regular GNNs/transformers?
3) What is the scalability relationship between CFGNN and transformers? Are there any empiracal comparisons?

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The overall research problem is interesting and the proposed method is easy to follow and understand. However,  some necessary technical details are missing. The experiment seems weak. The empirical and theoretical analysis of scalability is not adequate. Please refer to the following sections for more details.

### Strengths
- The research problem is interesting and pragmatic.

- The related work discussion is comprehensive.

- The experimental setting is extensive.

### Weaknesses
- The experiments seem weak, the experimental results are not competitive, and the selected baselines are not state-of-the-art [1].

- Motivated by the scalability, but the selected datasets are not large-scale, and the time complexity analysis is not formal.

- Section 2.1 is not informative, and its relation with the following proposed method is not clear.

- It is better to add necessary contexts for proposed equations.

- It is better to add references for the selected benchmarks and baselines.

[1] Hamed Shirzad, Ameya Velingker, Balaji Venkatachalam, Danica J. Sutherland, Ali Kemal Sinop. Exphormer: Sparse Transformers for Graphs. ICML 2023

### Questions
How core nodes are selected in the proposed method?

In Eq. 3, how the matrix $\mathbb{A}_{N}$ is constructed?

In Eq. 4, how the matrix $\mathbb{A}_{L}$ is constructed?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
