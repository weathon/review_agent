# Escaping Low-Rank Traps: Interpretable Visual Concept Learning via Implicit Vector Quantization

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 2, 6

## Abstract
Concept Bottleneck Models (CBMs) achieve interpretability by interposing a human-understandable concept layer between perception and label prediction. We first identify that the condition of \textit{many-to-many} mapping is necessary for robust CBMs, a prerequisite that has been largely overlooked in previous approaches. While several recent methods have attempted to establish this relationship, we observe that they suffer from the fundamental issue of \textit{representation collapse}, where visual patch features degenerate into a low-rank subspace during training, severely degrading the quality of learned concept activation vectors, thus hindering both model interpretability and downstream performance. To address these issues, we propose Implicit Vector Quantization (IVQ), a lightweight regularizer that maintains high-rank, diverse representations throughout training. Rather than imposing a hard bottleneck via direct quantization, IVQ learns a codebook prior that anchors semantic information in visual features, allowing it to act as a proxy objective. To further exploit these high-rank concept-aware features, we propose Magnet Attention, which dynamically aggregates patch-level features into visual concept prototypes, explicitly modeling the many-to-many vision–concept correspondence. Extensive experimental results show that our approach effectively prevents representational collapse and achieves state-of-the-art performance on diverse benchmarks. Our experiments further probe the low-rank phenomenon in representational collapse, finding that IVQ mitigates the information bottleneck and yields cross-modal representations with clearer, more interpretable consistency. Code is available at \url{https://github.com/Daryl-GSJ/IVQ-CBM}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper talk about Concept Bottleneck Models, which try to make AI more explainable by using human concepts. But they find a problem, the features collapse and lose information, making model not good. They propose a new method called IVQ and Magnet Attention to fix this, and they say it works better in many experiments. They also provide some interesting explanable visualzations.

### Strengths
Intuitive solution -- implicit vector quantization for binding human text concepts.
Clean model architecture design with consistent performance boosts.
Explanable results -- the visualization of the concepts learnt aligns well with human knowledge.

### Weaknesses
W1 Typo
---
Line 178. 
`Building upon obtrained high-rank concept-aware features`.
Should be obtained.


W2 Unclear Presentation
---
Figure 2, top left.
It draws `color`, `texture` and `shape` as text encoder input, so it implies #codes should be equal to #color + #texture + #shape.
However, Line 215 says that `M equals the number of textual concepts K`. Which one is true?

Besides, what is the form of text inputs? Do all the baselines have access to this information? How are the correspondences between texts and visual concepts made, so as to calculate Equation (8)? The using of BCE suggests that the number of text items is equal to the number of visual concepts; but the ablation studies used different numbers of visual concepts -- How could BCE be calculated?


W3 Unclear presentation
---
In Equation (5), the normalization is conducted along dimension $K$, i.e., codes in the codebook. However, in Equation (6), dimension $L$, i.e., patch embeddings, are eliminated (weighted summed). Routinely, if you want to eliminate some dimension in a matrix multiplication, you should normalize along that dimension. Do the authors have any insights for not following the routine?

```python
Z = ...  # (L,D)
Q = ...  # (K,D)
# (L,1,D) (1,K,D) -> (L,K)
dist = norm(Z[:,None,:] - Q[None,:,:], dim=2)
A = softmax(-dist, dim=1)  # TODO XXX why not `dim=0` or `L` ???
# (L,K)
M = einsum("LK,LD->KD", A, Z)  # TODO XXX `L` is eliminated
```


W4 Incomplete Analysis
---
Figure 7a.
Visualizing the code indexes of explicit vector quantization is missing.


W5 Incomplete Review
---
Section Related Works:
Vector-Quantization-related works should be reviewed here, given the VQ contributes a lot in the proposed method.
Possible literatures include:
- Neural Discrete Representation Learning.
- Generating Diverse High-Fidelity Images with VQ-VAE-2.
- Restructuring Vector Quantization with the Rotation Trick.
- Addressing Representation Collapse in Vector Quantized Models with One Linear Layer.
- Vector-Quantized Vision Foundation Models for Object-Centric Learning.
- Scaling the Codebook Size of VQ-GAN to 100,000 with a Utilization Rate of 99%.
- MGVQ: Could VQ-VAE Beat VAE? A Generalizable Tokenizer with Multi-group Quantization.
- ...

---

That said, I am open to revising my ratings, if the author could address my concerns.

### Questions
Please refer to the former section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses \textit{representational collapse} in Concept Bottleneck Models (CBMs), where visual patch features degenerate into a low-rank subspace, consequently degrading concept quality. The proposed solution introduces Implicit Vector Quantization (VQ) to discretize visual features into a high-dimensional concept space. This acts as a regularizer to prevent the "low-rank trap" and is shown to improve both concept quality and overall task classification performance. Nevertheless, I am not an expert in Concept-based learning, so the following comments and questions could be based on potentially inaccurate understandings or biased perspectives, and I will raise the rating if the author could give a sound explanation.

### Strengths
1. The paper identifies a critical and pervasive challenge in CBMs (representational collapse) and provides a clear mechanism for its cause. Implicit VQ is a novel and well-justified technique for regularizing the feature space and enforcing feature diversity, directly tackling the collapse problem. 

2. The method claims to simultaneously enhance model interpretability (concept quality) and maintain or improve predictive accuracy.

### Weaknesses
The implicit VQ regularization might introduce computational overhead compared to standard CBM training. The concept learning process relies on pre-computed text concept embeddings, which may limit the discovery of new, fine-grained, or complex concepts not covered by the initial textual set (Algorithm 1, steps 2, 13).

### Questions
How sensitive is the performance (concept quality and task accuracy) to the choice of the commitment cost hyperparameter $\beta$ in the VQ loss, and how does the size of the implicit codebook affect the trade-off between preventing collapse and maintaining expressive feature representation? Could the author give more empirical or theoretical evidence?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel method to utilize patch level visual representation to capture fine-grained relationships with concepts for training Concept Bottleneck Model (CBM).

### Strengths
- The authors propose a novel idea of utilizing patch level representation from CLIP for modeling fine-grained and detailed relationships between image regions and concepts.  
- The authors demonstrate representation collapse in CBM training with patch-level details and propose novel implicit vector quantization to preserve feature diversity.

### Weaknesses
- The claim of evaluating proposed methods on diverse benchmarks is misleading. While the authors demonstrate results on 8 diverse medical datasets, they should also evaluate the proposed CBM on standard benchmark datasets including CIFAR, CUB, ImageNet, and Places [1,2,3,4]. Such comparisons are crucial for assessing the scalability and generalization of the proposed approach across general classification tasks with increasing dataset sizes.
- The paper is missing qualitative comparison visualizing the concepts learned by the method and with other baselines.
- Recent studies [3, 4] have shown that CBM training can leak information through the bottleneck layer, allowing models to achieve high accuracy without actually learning meaningful concepts. To address this issue, [4] introduced the A-NEC metric to enable fair comparisons between CBMs based on their concept learning ability and IVQ-CBM should be evaluated on this metric.
- Notations and writing in section 3.2 such as $Z_{v}, Z_{q}$ are unclear.

[1] Yang, Yue, et al. "Language in a bottle: Language model guided concept bottlenecks for interpretable image classification." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.

[2] Oikarinen, Tuomas, et al. "Label-free concept bottleneck models." arXiv preprint arXiv:2304.06129 (2023). 

[3] Yan, An, et al. "Learning concise and descriptive attributes for visual recognition." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[4] Srivastava, Divyansh, Ge Yan, and Lily Weng. "Vlg-cbm: Training concept bottleneck models with vision-language guidance." Advances in Neural Information Processing Systems 37 (2024): 79057-79094.

### Questions
My primary concerns are lack of results on standard datasets and metrics and missing qualitative results. Please see weaknesses for more details.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper identifies representational collapse as a key obstacle in CBMs. It proposes Implicit Vector Quantization (IVQ) to preserve feature diversity without hard quantization and Magnet Attention to explicitly model many-to-many patch–concept relations. Experiments on eight benchmarks show that IVQ-CBM prevents collapse, improves interpretability, and achieves state-of-the-art performance.

### Strengths
1. It proposes a novel view of representational collapse as low-rank degeneration and introduces IVQ, a soft regularizer that avoids hard quantization.
2. The proposed method addresses a fundamental CBM limitation. It improves both interpretability and performance with broad applicability to multimodal learning.
3. This paper is well-organized and clearly written. The motivation and method is easy to follow.

### Weaknesses
1. The interpretability assessment mainly relies on qualitative visualization and concept-level alignment metrics. Incorporating human evaluation or concept intervention tests would better validate the interpretability and causal faithfulness of the learned concepts.
2. A theoretical analysis and a deeper connection between IVQ’s optimization dynamics and rank preservation would clarify its mechanism.

### Questions
Do the IVQ-CBM generalizes to broader multimodal reasoning or relational concept tasks? Are there any experiments to validate it?

### Soundness
3

### Presentation
3

### Contribution
3
