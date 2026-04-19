# Improving Soft Unification with Knowledge Graph Embedding Methods

- Decision: Reject
- Scores: 5, 6, 6, 6

## Abstract
Neural Theorem Provers (NTPs) present a promising framework for neuro-symbolic reasoning, combining end-to-end differentiability with the interpretability of symbolic logic programming. However, optimizing NTPs remains a significant challenge due to their complex objective landscape and gradient sparcity. On the other hand, Knowledge Graph Embedding (KGE) methods offer smooth optimization with well-defined learning objectives but often lack interpretability. In this work, we propose several strategies to integrate the strengths of NTPs and KGEs. By incorporating KGE objectives into the NTP framework, we demonstrate substantial improvements in both accuracy and computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes combining NTP with knowledge embedding methods to remedy NTP training difficulties. Specifically, the authors verified four combinations, and all of them show good performance across the symbolic reasoning datasets (Kinship, Nations, and UMLS). In some tasks, their methods give SOTA results. They also conducted some ablations experiments for configurations of KGEs.

### Strengths
1. This paper tries to solve a real problem of NTP: sparse proofs lead to training difficulties. The authors propose to use KGE to remedy this problem. By combining KGE and NTP, the embeddings become easier to train, because KGE is computationally dense and efficient.
2. The structure of the paper is clear, with four combinations to investigate: it proposes using KGE in various modules of NTP, including step, unification, and loss computation. The experimental results show the effectiveness of them.
3.  The experiments are extensive.

### Weaknesses
1. It claims that "We provide the first study for integrating KGEs into NTPs", however, it is not very appropriate. Although some integration methods proposed in this paper modify the NTP modules, the method that combines losses between NTP and KGE was already proposed in the first NTP paper [1]. In Sec. 4.2, it uses neural link prediction (ComplexE) as an auxiliary loss. 
2. I believe that the equation in Line 108 is incorrect: the second term should be (1 - NTPxxx).
3. The proposed methods are trivial to some extent. Actually, except for the CTP1, the others seem to be irrelevant in easing training difficulties. Why does changing the unification equation improve training? 
4. There are more systemic methods to convert the sparse computation into dense ones: [2] converts the grounding into dense matrix computation, [3] uses an Einsum operation for fast parallel groundings. They seem to be more promising in this direction.

[1]. End-to-End Differentiable Proving

[2]. Logic Tensor Network

[3]. LogicMP: A Neuro-symbolic Approach for Encoding First-order Logic Constraints

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes integrating Knowledge Graph Embedding (KGE) methods with Neural Theorem Provers (NTPs) to enhance neuro-symbolic reasoning systems, addressing the optimization challenges of NTPs. It presents four specific integration strategies (CTP1 to CTP4) and evaluates their performance across various datasets. The results demonstrate substantial improvements over baselines, especially in terms of Mean Reciprocal Rank (MRR) and inference speed, with CTP2 standing out as the most effective integration approach. Ablation studies and detailed analysis add depth to the findings, although limitations in scaling and efficiency are noted.

### Strengths
The integration of KGE methods with NTPs is well-motivated, bridging the optimization smoothness of KGE with the interpretability of symbolic logic inherent in NTPs. The systematic exploration of different integration strategies adds scientific rigor.

The paper evaluates the proposed methods across multiple benchmarks, showing a clear improvement over baselines. The use of detailed ablation studies, t-SNE visualizations, and analyses of embedding space provides strong empirical support for the proposed approach. Also, the authors applied 4 integration strategies (CTP1 to CTP4) and evaluates their performance across various datasets. The integration strategies are diverse and comprehensive.

The paper is well-organized and well-written. All 4 strategies are described with enough details.

### Weaknesses
While the proposed integration strategies are explained, some technical details, such as the choice of parameters for λ and specific KGEs for each variant, could be clarified further to assist replication and general understanding.

I am not familiar with the FB122 dataset. I know FB15K-237 is a commonly used KG reasoning database. The authors may want to explain why not using FB15K-237 but use FB122 instead.

Adding related work at the end is okay in general. But in this paper, the related work is entangled with the ablation study figures and tables, which makes the end of this paper hard to read. Please add related work after introduction.

### Questions
The authors may want to explain why not using FB15K-237 but use FB122 instead.

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
4

### Summary
This work proposes to integrate a knowledge-graph embedding (KGE) loss function into the optimization of a neural theorem prover (NTP) model, a type of neurosymbolic search architecture that is particularly hard to optimize due to subgradient sparsity in its backward pass. The goal of this integration is to alleviate the optimization challenges faced by the standard NTP objective, which contains many operations with sparse gradients and is plagued by local minima. The authors explore several alternative formulations of this integration, 
all based on the Conditional Theorem Prover (CTP; Minervini et al., 2020):
- CTP$_1$: a straightforward linear mixture of the original NTP loss and the KGE loss
- CTP$_2$: adding the KGE scoring function to the NTP stepwise unification scores
- CTP$_3$: changing the goal-reformulation module of the CTP to use the KGE's tuple completion geometry, viable for path-based KGE methods that support calculating the tail entity embedding in a triple directly from the head entity and relation embeddings
- CTP$_4$: replacing the final unification computation in a CTP proof path (normally a very large kernel evaluation) with a KGE lookup.

The authors evaluate their approaches on several link prediction tasks against three prior NTP variants, various KGE methods, Minerva, and NeuralLP. On the Kinship, Nations, UMLS, and FB122 (Test-II and Test-ALL splits), one of CTP$_1$ or CTP$_2$ performs best. On the FB122 Test-I split and WN18RR, KGE baselines outperform the proposed systems.

The authors additionally include analysis of the learned embedding spaces and training dynamics, including the observation that on large datasets like WN18RR, the embedding spaces learned by their hybrid methods do not appear to recover the structure seen in the spaces learned by the baseline KGE methods.

### Strengths
- I feel that the authors successfully demonstrate that hybridizing the two approaches (KGE and NTP) can yield complementary benefits.
- The analysis and ablations are very thorough: I found that they improved my understanding of the differences between the proposed systems and fairly characterized the remaining shortcomings of the best-performing options.
- The paper does a good job of describing the NTP background clearly. While I'm familiar with the method, it doesn't seem like it would be hard to follow for someone who was less so.

### Weaknesses
- While the proposed methods are successful on certain datasets, they fail to match baselines in certain cases (as the authors note). The paper presents clear diagnostic evidence of this issue in section 4.3 and some speculation on why this occurs, but the fact remains that the proposed methods are clearly situational and not straightforwardly scalable to large/complex domains.
- The specifics of CTP$_2$, CTP$_3$ and CTP$_4$ could use more elaboration; it's unclear how the KGE negative samples are included in these cases.

### Questions
Q1: In CTP$_2$, CTP$_3$, and CTP$_4$, how are contrastive negatives included? Is the original KGE objective still optimized (it seems like it's not)?

### Soundness
3

### Presentation
2

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
This paper handles the link prediction by combining Neural Theorem Proving (NTP) and Knowledge Graph Embeddings (KGE). Specifically, the KGE is injected into the CTP framework (a successor of NTP) in four ways. Empirical studies on large-scale datasets justified the empirical performances of those four ways of integration. Notably, CTP 1 and 2 perform way better than but significant slower than CTP 3 and 4.

### Strengths
- The empirical study is systematic and details the effectiveness of four variants and the impact of some crucial hyperparameters, including the weight and number of negative samples.
- the visualization also helped people understand the role of KGEs, which provides better global embedding structures.

### Weaknesses
- One of the key weaknesses is that all findings developed in this paper are solely for neural theorem provers. This makes the paper less impactful, given the existence of other technical solutions, such as GNN-based link predictors like NBFNet, rule miners like LERP, and a large language model for knowledge harvesting.
- In light of the weakness above, it would be much better if the authors could also analyze the proof produced when conducting link predictions as the key feature of neural link predictors.

### Questions
- In terms of performance, what is the performance and speed of using KGE to make link predictions?

### Soundness
3

### Presentation
3

### Contribution
2
