# Understanding the Role of Spectral Signal in Unsupervised Graph Domain Adaptation

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6, 3

## Abstract
Unsupervised graph domain adaptation (GDA) addresses the challenge of transferring knowledge from labeled source graphs to unlabeled target graphs. However, existing methods primarily implement spatial message-passing operators, which are limited by the neglect of the unique roles of spectral signals in unsupervised GDA. In this paper, we initially investigate an experimental study and find that the low-frequency topology signals signify the shared cross-domain features, while the high-frequency information indicates domain-specific knowledge. However, how to effectively leverage the above findings persists as a perplexing conundrum. To tackle the above issue, we propose an effective framework named Synergy Low-High Frequency Cross-Domain Network (SnLH) for unsupervised GDA. Specifically, we decouple the low- and high-frequency components in the original graph, extracting global structures and local details to capture richer semantic information and enhance the graph-level semantics. For the low-frequency components, we design an optimization objective to maximize the mutual information among low-frequency features, promoting the model to learn more generalized low-frequency information. To further mitigate domain discrepancy, we introduce high-frequency information cross-domain contrastive learning to impose constraints on the domains. By effectively leveraging both low and high-frequency information, the learned features turn out to be both discriminative and domain-invariant, thereby attaining effective cross-domain knowledge transfer. Extensive experiments demonstrate the superiority and effectiveness of the proposed framework across various state-of-the-art unsupervised GDA baselines.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work separates graph data into low- and high-frequency components and applies specialized processing techniques: maximizing mutual information for low-frequency consistency across domains and using contrastive learning for high-frequency components.

### Strengths
Separating low- and high-frequency signals for UGDA introduces an innovative approach to better capture cross-domain information.

### Weaknesses
1. The idea of separating low- and high-frequency information is not novel, like [1][2]. Although these work faces different tasks, the core idea of guiding the model in learning the low-frequency and high-frequency information separately is the same.
2. It lacks new and related baselines, like [3].
3. Writing can be improved. For example, the first paragraph in the Introduction is too long. You should talk about graph data and graph domain adaptation in two different paragraphs. Besides, some words are too long, like 
4. The use of mutual information and contrastive learning with frequency-based filters may add significant complexity, making the method harder to implement. Scalability on very large graphs with complex structures remains uncertain. You should provide computational complexity analysis or runtime comparisons on larger graph datasets. 

[1] Bo D, Wang X, Shi C, et al. Beyond low-frequency information in graph convolutional networks[C]//Proceedings of the AAAI conference on artificial intelligence. 2021, 35(5): 3950-3957.
[2] Chen J, Lei R, Wei Z. PolyGCL: GRAPH CONTRASTIVE LEARNING via Learnable Spectral Polynomial Filters[C]//The Twelfth International Conference on Learning Representations. 2024.
[3] Luo J, Gu Y, Luo X, et al. GALA: Graph Diffusion-based Alignment with Jigsaw for Source-free Domain Adaptation[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2024 (01): 1-14.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper presents a framework for unsupervised graph domain adaptation (GDA) through the introduction of the Synergy Low-High Frequency Cross-Domain Network (SnLH). It identifies gaps in existing methodologies, including utilizing spatial message-passing operators while neglecting the potential of spectral signals. The authors conduct an experimental study revealing that low-frequency topology signals correlate with shared cross-domain features, while high-frequency signals denote domain-specific knowledge. SnLH disentangles these frequency components, optimizing low-frequency features to maximize mutual information and employing high-frequency contrastive learning to address domain discrepancies.

### Strengths
1. The model performs well in most databases.
2. SnLH provides a spectral signal view in solving graph-level domain adaption problems.
3. This work notably highlights the spectral signal information discrepancy in graph-level DA.

### Weaknesses
1. Novalty is limited. The paper claims they first explore the influence of frequency domain information and effectively leverage this knowledge to mitigate domain discrepancies. However, [1] also highlights its issue in the 2023 of GDA. 
2. Lack of theoretical analysis. This work mentions mutual information many times when using this method. I doubt the effectiveness of this approach in practical terms. I doubt whether its impact on GDA is significant unless they can prove that the performance improvement is due to the introduction of the mutual information method rather than other domain alignment methods.
3. Lack of innovative methods. Low-high-frequency signal and low-frequency interclass consistency are basically existing losses, and improvement is incremental.
4. Graph-level DA impact is limited. Most existing GDA methods focus on node-level tasks. Recent graph-level work needs to clarify the importance of solving graph classification tasks due to the lack of work on that.





[1] Pang, Jinhui, et al. "Sa-gda: Spectral augmentation for graph domain adaptation." Proceedings of the 31st ACM international conference on multimedia. 2023.

### Questions
Same as Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the problem of Unsupervised graph domain adaptation and proposes a new method named Synergy Low-High Frequency Cross-Domain Network (SnLH) for unsupervised GDA. It decouples the low- and high-frequency components in the original graph, extracting global structures and local details to capture richer semantic information and enhance the graph-level semantics. Extensive experiments demonstrate the superiority and effectiveness of the method across various state-of-the-art unsupervised GDA baselines.

### Strengths
- The studied problem is interesting and important.
- The paper is well-organized and clearly written.
- The idea of incorporating graph spectral signals into GDA is quite interesting and effective.

### Weaknesses
- Why A2GNN is introduced in the baseline? Is this method for node classification? It seems to be a wrong citation as well. 
- The paper lacks some recent SOTA baselines such as "Multi-View Teacher with Curriculum Data Fusion for Robust Unsupervised Domain Adaptation". 
- How about the influence of different GNN encoders?
- I suggest that the authors include some comparisons of computation time.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper addresses unsupervised graph domain adaptation (UGDA) by proposing the Synergy Low-High Frequency Cross-Domain Network (SnLH), which leverages low- and high-frequency spectral signals to handle cross-domain data transfer. Through disentangling and optimizing low- and high-frequency information, SnLH aims to enhance generalization across domains without target labels. Experimental results indicate that SnLH achieves competitive or superior performance compared to state-of-the-art UGDA methods on multiple datasets.

### Strengths
1. The approach provides a unique take on UGDA by distinguishing between low- and high-frequency spectral components, addressing previously overlooked aspects of spectral signal impact in GDA.
2. SnLH exhibits strong empirical performance, surpassing several baselines across diverse datasets, demonstrating its robustness and versatility.
3. The authors implement cross-domain mutual information maximization for low-frequency signals and contrastive learning for high-frequency signals, showcasing a well-structured approach to utilizing spectral information.
4. "Experimental studies reveal that low-frequency topology signals represent shared cross-domain features, while high-frequency information reflects domain-specific knowledge" is an interesting and intuitively reasonable finding.

### Weaknesses
1. Equations 9 and 10 represent a KL-divergence loss, not mutual information, and therefore are not equivalent to mutual information maximization, as claimed by the authors.

2. The authors claim that maximizing mutual information ensures the model learns global domain invariance on low-frequency features. However, this claim is unsubstantiated, and a more robust demonstration is needed to support this point.

3. Clarification is required on how  $P_s$ and $P_t$ are expressed or estimated within the model.

4. The motivation for applying contrastive learning to high-frequency features is insufficiently developed. A demonstration is necessary to justify why minimizing relative distances is appropriate for graph domain adaptation.

5. The proposed method appears inconsistent with the authors' motivations. Initially, the authors argue that low-frequency features capture domain-shared information, while high-frequency features are domain-specific. However, both the contrastive learning on high-frequency features and the KL minimization on low-frequency features aim to align feature distributions to achieve domain invariance. This approach does not align with the authors' original intent to treat high-frequency and low-frequency features differently due to different intrinsic properties.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2
