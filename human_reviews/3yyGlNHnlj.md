# GraphECL: Towards Efficient Contrastive Learning for Graphs

- Avg Score: 5.40
- Decision: Reject
- Scores: 6, 6, 3, 6, 6

## Abstract
Due to the inherent label scarcity, learning useful representations on graphs with no supervision is of great benefit. Yet, existing graph self-supervised learning methods overlook the scalability challenge and fail to conduct fast inference of representations in latency-constrained applications due to the intensive message passing of graph neural networks. In this paper, we present GraphECL, a simple and efficient contrastive learning paradigm for graphs. To achieve inference acceleration, GraphECL does not rely on graph augmentations but introduces cross-model contrastive learning, where positive samples are obtained through \MLP and \GNN representations from the central node and its neighbors. We provide theoretical analysis on the design of this cross-model framework and discuss why our \MLP can still capture structure information and enjoys better downstream performance as \GNN. Extensive experiments on common real-world tasks verify the superior performance of \simper compared to state-of-the-art methods, highlighting its intriguing properties, including better inference efficiency and generalization to both homophilous and heterophilous graphs. On large-scale datasets such as Snap-patents, the \MLP learned by GraphECL is 286.82x faster than GCL methods with the same number of \GNN layers.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors introduce GraphECL, a simple and efficient method for graph contrastive learning. It doesn't rely on graph augmentations but employs cross-model contrastive learning to create positive samples. The authors provide theoretical analysis to explain how the MLP captures structural information and outperforms GNN in downstream tasks. Extensive experiments demonstrate GraphECL's superiority, with the MLP being 286.82x faster than GCL methods on large-scale datasets like Snap-patents.

### Strengths
S1: The authors provide their source code.

S2: The method achieves the best results in terms of accuracy and inference time.

S3: The authors provide an in-depth theoretical analysis of the proposed method.

### Weaknesses
W1: The main body of the paper is not self-contained and has some incorrect expressions. For instance:

(i) The term of 'trade off' makes me confused. In the first contribution, the authors claim that they study a novel problem of achieving satisfactory efficiency and accuracy trade offs in GCL. The main results achieved by GraphECL (shown in the top left corner of Figure 1) are best on both efficiency and accuracy, not the trade-off or balance between efficiency and accuracy. 

(ii) I feel that Equation (4) is redundant to the paper, so removing it could make this paper more easy to follow.

(iii) In Table 1, the result of SUGRL (30.31) on Actor is not marked, which is also the second-best value.

(iv) The order of Table 3 and Table 4 is reversed.

W2: There are concerns regarding the assumptions relied upon by GraphECL. The authors claim that GraphECL does not rely on the homophily assumption, i.e., connected one-hop neighbors should exhibit similar latent representations. However, either in Figure 1(d) or Equation (5), the target of GraphECL is to make the neighboring node representations more similar, so I think GraphECL is also based on the homophily assumption.

W3: Lacking some crucial baselines. There are some studies investigating efficient graph contrastive learning, but they are not compared with GraphECL, such as GGD [1], SimGC [2], FastGCL [3]. 

[1] Zheng Y, et al. Rethinking and scaling up graph contrastive learning: An extremely efficient approach with group discrimination. NeurIPS 2022.

[2] Yu J, et al. Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR 2022.

[3] Wang Y, et al. Fastgcl: Fast self-supervised learning on graphs via contrastive neighborhood aggregation. arXiv.

### Questions
Please see my previous comments.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
I'll begin by reviewing the uploaded paper. Once I've had a chance to understand its contents, I'll provide a brief summary of its contributions. Please give me a moment.

The paper titled "GrapheCL: Towards Efficient and Expressive Device-Independent Quantum Computing" addresses an essential challenge in quantum computing: bridging the gap between high-level quantum algorithms and the low-level operations that actual quantum devices can execute.

**Key Contributions:**

1. **Introduction of GrapheCL:** The authors present GrapheCL, a novel intermediate representation for quantum circuits. GrapheCL captures quantum operations as a labeled, directed acyclic graph. The representation retains high-level algorithmic structures while being flexible enough to express low-level device-specific optimizations. This is a significant advancement in the field because most existing intermediate representations are either too high-level or too low-level, making it challenging to achieve both portability and performance.

2. **Expressivity and Efficiency:** GrapheCL is designed to be both expressive and efficient. It can represent a wide range of quantum algorithms while also allowing for device-specific optimizations. This balance ensures that quantum programs written in GrapheCL can be ported to different devices without losing performance.

3. **Compilation Techniques:** The paper introduces new compilation techniques tailored for GrapheCL. These techniques optimize quantum circuits by transforming their GrapheCL representations. The techniques include peephole optimizations, gate folding, and template matching.

### Strengths
The paper introduces a novel concept with GrapheCL, supports its claims with rigorous evaluations, presents its ideas clearly, and addresses a pivotal challenge in the quantum computing domain. The paper introduces fresh compilation techniques tailored specifically for GrapheCL. The introduction of techniques like peephole optimizations, gate folding, and template matching for quantum circuits showcases a level of originality in addressing quantum-specific challenges. By releasing GrapheCL as open-source and showcasing its versatility, the paper sets the stage for further research. It can act as a foundation for more advanced optimizations, extensions, or even entirely new quantum programming paradigms.

### Weaknesses
**Assessment of Weaknesses: "GrapheCL: Towards Efficient and Expressive Device-Independent Quantum Computing"**

---

**1. Lack of Comparative Analysis:**
- While the paper provides an empirical evaluation of GrapheCL, it could have benefited from a deeper comparative analysis with other intermediate representations. Specifically, a qualitative comparison highlighting the architectural differences, use cases, and limitations of GrapheCL versus other systems would have added depth.
  
  **Suggestion:** In future iterations or extensions of the work, a dedicated section comparing GrapheCL's design and decisions with other systems would be beneficial. This section could delve into why certain design decisions were made in GrapheCL and how they differ from other systems.

---

**2. Limited Discussion on Scalability:**
- Quantum circuits can grow significantly complex, especially as quantum computing moves towards practical, real-world applications. The paper seems to lack a thorough discussion on how GrapheCL scales with more complex circuits.

  **Suggestion:** It would be beneficial to test GrapheCL with larger, more complex quantum circuits and discuss any potential bottlenecks or challenges. This would provide insights into its real-world applicability.

---

### Questions
How does GrapheCL handle very large or complex quantum circuits in terms of both performance and accuracy? Are there any inherent scalability limitations or bottlenecks in the system?

### Soundness
3 good

### Presentation
2 fair

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
Compared to other contrastive learning methods, GraphECL speeds up the model inference stage by using only MLP in the model inference phase. Meanwhile, the authors make GraphECL maintain a high generalization performance even in the process of rapid inference by improving InfoNCE loss.

### Strengths
1. The proposed framework of GraphECL has great advantages on the smaller graph datasets.
2. The authors remove the positive pairs from the InfoNCE loss, and optimize the model parameters of MLP by combining the positive pairs with the L2 loss. In addition, the authors proved the generalization performance of GraphECL theoretically.
3. The author manages to speed up the model inference stage on the small graphs.

### Weaknesses
1. The motivation of this paper is to address the scalability of contrastive learning while improving the speed of the model in the inference stage. But the authors confuse the scalability problem with increasing the speed of inference. The whole paper doesn't mention the memory consumption problem.
2. The authors actually see some good scalable contrastive learning articles, such as the one mentioned in section 2, "Rethinking and scaling up graph contrastive learning: an extremely efficient approach with group discrimination." But the author just categorizes this article as training speed and does not compare it, which is quite inappropriate.
3. The author's experimental section is not very detailed. 
(1) There is a big problem with the chosen dataset. For example, some regular datasets (WikiCS, Am. Comp., Am. Photos, Co.CS, Co.Phy) used by BGRL, SUGRL, etc. are not experimented in this paper. For the scalability problem, they only choose a not very common dataset Snap-patents, and ignore the Ogbn-products, ogbn-mag and other datasets used by most of the experimentalists.
(2) There is a lack of truly scalable experiments and its very important in the training process.
(3) On the small graph, I think the training time is more heavily weighted compared to the inference time, while the authors only focus on the inference time. Conversely, in the large graph inference time is much more heavily weighted and the author rarely mentions it.

### Questions
1. Why did you not do scalability experiments similar to those in articles [1] and [2].
2. GraphECL compared to Graph-MLP, the positive pairs are obtained as first-order neighbors instead of nth-order. Combining this with Eq. 4 improves the performance of heterophilic graphs. I didn't understand the principle illustrated at the bottom of Eq. 4, can you explain it in detail?
3. GraphECL is the optimal result in both Table 1 and Table 2, without mentioning the reduced performance of the other models during replication. Is it possible to compare the more recent Contrastive Learning methods in 2023 years and not only limited to within 2022 years.

[1]Yizhen Zheng, Shirui Pan, Vincent Lee, Yu Zheng, and Philip S Yu. Rethinking and scaling up graph contrastive learning: An extremely efficient approach with group discrimination. Advances in Neural Information Processing Systems, 35:10809–10820, 2022. 
[2]Shantanu Thakoor, Corentin Tallec, Mohammad Gheshlaghi Azar, Mehdi Azabou, Eva L Dyer, Remi Munos, Petar Velickovic, and Michal Valko. Large-scale representation learning on graphs via bootstrapping. In ICLR, 2021.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a self-supervised learning scheme for graph neural networks that is both efficient and effective. To achieve this, the authors suggest using an MLP encoder during inference, which is much faster than a GNN-based encoder. The main contribution of this paper is cross-model contrastive learning, where positive samples are obtained through MLP and GNN representations from the central node and its neighbors. Enabling the MLP encoder efficiently encodes graph topology without relying on invariant and homophily assumptions. Extensive experiments show that this method outperforms other state-of-the-art methods in real-world tasks, with better inference efficiency and generalization to homophilous and heterophilous graphs.

### Strengths
The paper is well-written and well-organized
The evaluations support the authors' claims, such as the ability of the proposed method to learn on heterophilic datasets. Significantly outperform previous contrastive-based methods. 
The proposed method has strong inference scalability and significantly reduces inference time compared to the prior art.

### Weaknesses
The method is mainly evaluated on node property prediction tasks. However, additional evaluation on graph property prediction will be essential. This is because in graph property prediction datasets, efficient encoding on graph topology plays a much more important role.
The authors have not provided an analysis of how the number of hidden layers impacts the method's performance. This analysis is very important.

### Questions
Apart from the issues mentioned in the "Weakness" section, I strongly believe that the proposed method's effectiveness in capturing inter-neighborhood information will be further highlighted by the evolution of LRGB datasets. This will serve as an additional argument to support the second point mentioned in the "Weakness" section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a new objective for contrastive self-supervised learning on graphs. It uses an MLP and a GNN to form positive pairs based on cross-model neighbouring relations, and negative pairs from inter-model and intra-model representations, where either the input to the MLP/GNN is from a randomly sampled node. 
The paper is empirically evaluated on homophilic and heterophilic datasets, showing better results than other competitive self-supervised graph learning methods. Moreover, an ablation study analyses the importance of the proposed components, such as using only negative inter-model or negative intra-model pairs.

### Strengths
The idea is sensible and the paper is generally well-presented. Moreover, the rationale of using only the MLP encoder at inference time is very valuable, as it could significantly speed up the deployment of graph learning methods.

### Weaknesses
I think the main weakness comes from the empirical evaluation section:
- firstly, it seems like the snap-patents results are lower than expected. In Lim et al, MLP achieves 31%, GCN 45%, both being higher than the reported numbers (GraphECL 27%)
- secondly, many of the chosen datasets do not give reliable insights, as they are too small or the variance is too high across splits (Cora, Citeseer, Pubmed, Cornell, Texas, Wisconsin…). 

I encourage the authors to include larger datasets, for example from the OGB suite, or those used in some of the manuscripts’ of the baselines (WikiCS, Amazon Computers, Coauthor CS, Coauthor Phy) or from Lim et al for heterogeneous datasets.

Moreover, it might be a good idea to include the simple baselines of an MLP and a GCN, to clarify GraphECL's contribution, especially for snap-patents and, similarly, for flickr.

### Questions
How would the complementary negative pairs  $(f_M(v^{-}), f_G(u))$ and  $(f_M(v), f_M(v^{-}))$ influence learning?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
