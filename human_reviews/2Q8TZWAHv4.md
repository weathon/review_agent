# GOAt: Explaining Graph Neural Networks via Graph Output Attribution

- Decision: Accept (poster)
- Scores: 6, 8, 6, 5

## Abstract
Understanding the decision-making process of Graph Neural Networks (GNNs) is crucial to their interpretability. Most existing methods for explaining GNNs typically rely on training auxiliary models, resulting in the explanations remain black-boxed. This paper introduces Graph Output Attribution (GOAt), a novel method to attribute graph outputs to input graph features, creating GNN explanations that are faithful, discriminative, as well as stable across similar samples. By expanding the GNN as a sum of scalar products involving node features, edge features and activation patterns, we propose an efficient analytical method to compute contribution of each node or edge feature to each scalar product and aggregate the contributions from all scalar products in the expansion form to derive the importance of each node and edge. Through extensive experiments on synthetic and real-world data, we show that our method not only outperforms various state-of-the-art GNN explainers in terms of the commonly used fidelity metric, but also exhibits stronger discriminability, and stability by a remarkable margin.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors introduce  a new method for interpreting GNNs. Specifically, this method provides transparent explanations by attributing graph outputs to input features. It employs an analytical approach that expands the GNN into a sum of scalar products, which allows for efficient calculation of feature contributions. Empirical results on three well-known datasets show the effectiveness of the proposed method.

### Strengths
1.Unlike previous methods that either rely on back-propagation with gradients or train complex black-box models, this method uses an analytical method based on the sum-product structure of the GNN’s forward pass.
2. The proposed method demonstrates promising performance on synthetic and real-world data. These experiments demonstrate that GOAt outperforms existing state-of-the-art methods in several key metrics, such as fidelity, discriminability, and stability.
3. The paper seems to be well-organized in its presentation.

### Weaknesses
1. It isunclear what limitations of previous methods the author's method can overcome, and the reasons for overcoming these limitations. It would be good to clarify these two points.
2. Please explain why it makes sense to define each edge as an Equal Contribution.
3. The latest methods mentioned in the Related Work: GLGExplainer and GCFExplainer, do not appear in the experiments as baselines.
4 .The experimental datasets could be more diverse, such as including the BBBP, MNIST, and other datasets, to make the model more convincing.
5. The reference to 'Equation(??)' in the paragraph below Equation 5 is unclear.

### Questions
Please refer to Weaknesses 1-5.

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces GOAt, a novel method for GNNs by representing the output with product terms. Specifically, the authors propose an efficient analytical method to compute the contribution of each node or edge feature to each scalar product and aggregate the contributions from all scalar products in the expansion form to derive the importance of each node and edge. GOTt is shown to be applicable to different GNN architectures, including GCN, GraphSAGE, and GIN. It outperforms various SOTA GNN explanation baselines in terms of fidelity, discriminability, and stability as demonstrated through experiments on both synthetic and real-world data. The authors also provide qualitative case studies to show that GOAt can identify the most important nodes and edges in the input graph that contribute to the output prediction and that it can also detect the presence of adversarial attacks and the robustness of the GNN model.

### Strengths
1. Novelty: Explaining GNNs via output attribution is a novel approach to my knowledge.

2. Effectiveness: GOAt outperforms SOTA baselines in terms of fidelity, discriminability, and stability, and case studies show GOAt can generate qualitatively meaning explanations.

3. GOAt is shown to work for different GNN architectures and for both graph classification and node classification.

4. Code is provided for reproducibility.

### Weaknesses
1. Given the GOAt method is relatively complex, a time complexity analysis or empirical speed evaluation helps assess its significance.

2. For the newly introduced metrics, i.e., discriminability and stability, a more detailed discussion of how they are computed can help improve clarity.

### Questions
1. For the discriminability metric, which the class labels were used? The ground truth labels or the GNN predicted labels? In these two cases, the discriminability has different meanings. My current understanding is using the ground truth labels, but this means when GNN makes an incorrect prediction, the embedding is also included in the discriminability computation.

2. I am a bit confused about how the coverage for the stability measure is computed. Can the authors explain more? Also, although coverage for BA-2Motifs is high for GOAt, it is very low on Mutagenicity and NCI1 for all methods. Any discussions about this observation?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a new GNN explanation method by expanding the GNN as a sum of scalar products. The proposed method efficiently computes the contribution of each node and feature. From extensive experiments, the paper shows significant improvements over baselines and also has better stability across samples.

### Strengths
1. The GNN explanation problem is important for decision-making.
2. The proposed method is sound and could explain both edges and features.
3. The proposed method significantly outperforms baselines.

### Weaknesses
1. There is no computation complexity analysis for the proposed method.
2. It is better to have a figure to illustrate the key idea of the proposed method.

### Questions
Please refer to the weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on a problem of explaining graph neural networks. Specifically, the author proposed an explaining method that lies in the category of decomposed approaches. The proposed method aims to compute the contribution of each node/edge for the final predictions. Experiments are conducted to compare with some perturbation-based explanation methods.

### Strengths
1. This paper focuses on an important problem of explaining graph neural networks. 
2. The proposed method is reasonable and technically sound.
3. The results compared with the representative perturbation-based explainers are promising.

### Weaknesses
1. There are already a group of decomposition methods for explaining graph neural networks [1, 2]. They share very similar idea with the proposed method. However, they are not discussed in the paper. It is necessary to clarify the contributions of this work compared with these related works. 
2. In the experimental section, the decomposition methods are not compared with the proposed framework. It is suggested to add decomposition-based explainers as basslines.
3. It is claimed that the proposed method can be faithful and also stable across similar samples. I suggest to discuss and compare with a recent work [3] that also works in both directions of GNN explainer.

[1] Pope, Phillip E., et al. "Explainability methods for graph convolutional neural networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
[2] Schnake, Thomas, et al. "Higher-order explanations of graph neural networks via relevant walks." IEEE transactions on pattern analysis and machine intelligence 44.11 (2021): 7581-7596.
[3] Zhao, Tianxiang, et al. "Towards Faithful and Consistent Explanations for Graph Neural Networks." Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining. 2023.

### Questions
Please refer to the weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
