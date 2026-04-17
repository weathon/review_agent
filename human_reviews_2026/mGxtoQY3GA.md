# HarmonyGNNs: Harmonizing Heterophily and Homophily in GNNs via Self-Supervised Node Encoding

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Graph Neural Networks (GNNs) have made significant advances in representation learning on various types of graph-structured data. However, GNNs struggle to simultaneously model heterophily and homophily, a challenge that is amplified under self-supervised learning (SSL) where no labels are available to guide the training process. This paper presents HarmonyGNNs, an end-to-end graph SSL framework designed to harmonize heterophily and homophily through two complementary innovative perspectives: (i) Representation Harmonization via Joint Structural Node Encoding. Nodes are embedded into a unified latent space that retains both node specificity and graph structural awareness for harmonizing heterophily and homophily. Node specificity is learned via linear and non-linear node feature projections. Graph structural awareness is learned via a proposed Weighted Graph Convolutional Network (WGCN). A self-attention module enables the model learning-to-adapt to varying levels of patterns. (ii) Objective Harmonization via Predictive Architecture with Node-Difficulty–Aware Masking. A teacher network processes the full graph. A student network receives a partially masked graph. The student is trained end-to-end, while the teacher is an exponential moving average of the student. The proxy task is to train the student to predict the teacher’s embeddings for all nodes (masked and unmasked). To keep the objective informative across the graph, two masking strategies that guide selection toward currently hard nodes while retaining exploration are proposed. Theoretical underpinnings of HarmonyGNNs are also analyzed in detail. Comprehensive evaluations on benchmarks demonstrate that HarmonyGNNs achieves state-of-the-art performance on heterophilic graphs (e.g., +7.1% on Texas, +9.6% on Roman-Empire over the prior art) while matching SOTA on homophilic graphs, and delivering strong computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces HOD-GNN, an architecture that enhances GNN expressivity by computing derivatives of a base MPNN with respect to input node features and using these derivatives as additional structural information to input to the second GNN.  The paper proves that k-HOD-GNN can approximate k-OSAN subgraph GNNs, which is strictly more expressive than MPNNs with random walk structural encodings, and achieves favorable computational complexity for sparse graphs with shallow base MPNNs by exploiting an efficient message-passing-like derivative computation algorithm. Empirically, HOD-GNN consistently ranks in the top two tiers across seven benchmarks.

### Strengths
1. The paper presents a fresh perspective on enhancing GNN expressivity through derivatives. The intuition is clearly explained through the connection to marking-based GNNs and Taylor expansion. I think the presented approach is conceptually elegant. 

2. The theoretical analysis provides a complete expressivity characterization by proving k-HOD-GNN approximates k-OSAN subgraph GNNs while offering constructive separation examples that precisely identify when the method is efficient.

3. The three-component pipeline from base MPNN, intermediate derivative computation, and the downstream GNN is clear and easy to follow.

### Weaknesses
1. The claimed "inductive bias toward derivative-aware representations" is mentioned but never explained.

2. The triangle counting example in the Motivation section is thin as a driver of the whole method. This motivating toy showing first-order derivatives recover triangle counts uses a very special linear stack, i.e., identity activations, and does not show whether derivatives remain informative for typical non-linear, normalized, or regularized MPNN on real tasks. Thus, it’s illustrative but not compelling evidence that derivatives are broadly the right signal. Also, in the theoretical analysis, the separation beyond k-WL depends on analytic activations that  iss fine for softplus/tanh, but not ReLU.

3. Flattening $D^{out}$ could be large in practice. 

4. For $U^{node}$, a 2-IGN is powerful but can be heavy. The paper claims scalability via sparsity, but constants can bite (derivative channels $\times$ feature dims $\times$ orders). 

5. I didn’t see normalization for $D^{out}$ or $D^{(T)}$. Some stabilization may be necessary because  derivatives can vary wildly with feature scaling.

6. I’m confused about the rationale for computing two derivative tensors that target different functions. Specifically, $D^{(T)}$ computes derivatives of the final node embeddings, while $D^{out}$ computes that based on the graph-level output. Why is it necessary to encode both, rather than using only node-level derivatives with an invariant head, or using only graph-level derivatives if the target is graph classification?

### Questions
See Weaknesses

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes H3GNNs, a self-supervised framework aimed at unifying the learning of homophilic and heterophilic graphs within a single model. It introduces a teacher–student predictive architecture with dynamic node-difficulty-based masking to create informative self-supervision signals, and a joint structural node encoding module that fuses linear, nonlinear, and structural features through a Weighted GCN and Transformer-based hierarchical fusion. Experiments on multiple benchmark datasets show that H3GNNs outperforms prior self-supervised GNN models.

### Strengths
1.The paper shows reasonable originality by combining teacher–student predictive learning, dynamic masking, and weighted structural encoding to address both homophily and heterophily in self-supervised GNNs.
2.The technical quality is solid, with extensive experiments and consistent gains across benchmarks, though some design choices and analyses lack depth. 
3.The paper is clearly written overall.

### Weaknesses
1.The roles of WGCN, MLP, linear projection, and transformer fusion are not clearly separated or analyzed; it is unclear how each module contributes to handling homophily and heterophily individually.
2.The dynamic masking update relies on previous loss values, yet the paper does not specify how frequently these scores are recomputed or how the warm-up and exploitation phases are scheduled.
3.The efficiency comparison table does not include information about the hardware platform, GPU type, or framework version, so the claimed improvements in training time and memory cannot be verified.
4.The paper lacks systematic sensitivity analysis for critical hyperparameters like the number of WGCN layers.
5.On homophilic datasets, the reported improvements are minor and may fall within variance, yet no statistical tests or confidence intervals are presented to confirm significance.
6.The theoretical analysis assumes strong convexity and smoothness for deep GNNs, which are unrealistic; hence the derived convergence results have limited practical meaning.

### Questions
1.Could the authors provide a clearer explanation or visualization of how the WGCN, MLP, linear projection, and transformer fusion interact? It is not obvious which module primarily contributes to handling heterophily versus homophily.
2.Why are exactly four “tokens” used in the joint encoding, and why is the fusion order fixed? Have the authors tried using different token numbers or fusion sequences, and how sensitive is performance to these choices?
3.How are the learnable edge weights in WGCN regularized or constrained to prevent degenerate solutions (e.g., trivial scaling or sparsity collapse)? Are they shared across layers or trained independently?
4.The dynamic masking variants show small numerical differences. Can the authors provide more analysis (e.g., convergence behavior, difficulty distribution, or qualitative examples) to demonstrate why dynamic masking is preferable to random masking?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a unified SSL framework addressing the challenge of learning on graphs with mixed structural patterns. It achieves this through Representation Harmonization (joint structural node encoding with WGCN and self-attention) and Objective Harmonization (a predictive teacher–student architecture with dynamic masking). The method effectively balances homophilic and heterophilic signals, showing strong performance across benchmarks. Overall, H3GNNs offers a significant advancement in adaptive, structure-aware graph self-supervised learning.

### Strengths
The strength of this paper lies in its innovative unification of homophily and heterophily modeling within a self-supervised graph learning framework. The authors identify a key limitation in existing GNN and SSL methods—the inability to handle mixed structural patterns—and propose a comprehensive solution (H3GNNs) that achieves both representation harmonization and objective harmonization. It provides both stability and adaptability in learning from complex graph structures. Moreover, the paper is well-written, with extensive experiments showing state-of-the-art performance on heterophilic and mixed graphs, demonstrating strong generalization and interpretability.

### Weaknesses
The paper lacks experiments on large-scale graphs and comparisons with more recent or relevant algorithms, limiting its scalability claims. Efficiency analysis is narrow, involving few baselines without clear justification for their selection as representative methods. Broader comparisons and rationale would strengthen the empirical evaluation and conclusions.

### Questions
See weakness

### Soundness
4

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
This paper proposes a unified SSL framework for both homophilic and heterophilic graph. The framework is built on a teacher-student predictive architecture, where the student network achieves representation harmonization via joint structural node encoding. The experimental results show the effectiveness of the proposed framework.

### Strengths
1.	The idea of : teacher–student predictive architecture is interesting, which eliminate the need for complex negative sampling.

2.	The author provides theoretical analysis of the proposed method, although the analysis is really complicated and hard to understand.

3.	The proposed method outperforms existing SSL methods on heterophilic graph datasets.

### Weaknesses
1. Besides the teacher-student predictive architecture, the overall novelty is limited and the design of the student network is somewhat engineering (e.g., Learning Multi-Head Self-Attention and Fusing and Selecting Tokens Hierarchically as SSL Node Encoding). The motivation of these strategies is unclear.

2. The overall framework involves huge memory and computation overhead. It is suggested to analyze the memory complexity and computation complexity in detail. The authors perform computation and memory comparisons in Section 4.3. It would be better to perform similar experiments on more datasets (i.e., homophilic graphs and large-scale graphs).

3. Many details are missing. For example, in Section 3.1, the authors claimed that the teacher network is not trained. It is not clear how to get the parameters of the teacher network. Moreover, the architecture of the teacher network is also not introduced.

### Questions
see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
