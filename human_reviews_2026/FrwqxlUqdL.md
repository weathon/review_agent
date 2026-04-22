# Enhancing Graph Transformers with Spectral Guidance in Attention

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Existing Graph Transformers often overlook the limitations of self-attention mechanism without inductive bias. The pure self-attention tends to aggregate features from unrelated nodes and misalign attention with graph structures, leading to suboptimal modeling of relational dependencies. Moreover, operating solely in the spatial domain, self-attention underutilizes graph spectral components that correspond to more detailed and comprehensive relational patterns. To address the above issues, we propose the Spectral-Guided Attention Graph Transformer (SGA-Former), which introduces rich structural priors from the graph spectral domain to guide attention learning. Specifically, we design two Spectral Relation Metrics as attention bias, which capture complementary low and high-frequency structural patterns. To leverage these priors, we develop the Spectral-Guided Attention Enhancer (SGA-Enhancer), which filters redundant attention scores and emphasizes important node relationships based on the spectral metrics. Incorporating SGA-Enhancer, SGA-Former builds dual-branch Spectral Attention Layers that jointly utilize both spectral views, enabling more balanced and structure-aware attention learning. Extensive experiments show that SGA-Former consistently achieves superior performance across a wide range of graph learning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a spectral-guided Graph Transformer model, SGA-Former, which introduces spectral domain information as an inductive bias. By incorporating Laplacian-powered high- and low-frequency spectral relation matrices into the attention mechanism, it achieves structure-aware attention distributions that efficiently capture both global and local graph structural features. The paper theoretically demonstrates that its expressive power surpasses Graph Transformers with shortest-path biases. Extensive experiments validate its significant performance improvements across multiple benchmark tasks.

### Strengths
1. The paper presents a novel Graph Transformer architecture that leverages spectral domain information as an inductive bias, effectively addressing limitations in existing Graph Transformers, demonstrating clear innovation.

2. It provides an in-depth analysis of the relationship between spectral information and graph structure, designs two spectral relation metrics to guide attention learning, and theoretically validates the effectiveness of the proposed method.

3. Extensive experiments are conducted on multiple graph learning tasks, comparing with various baseline models, and the results convincingly demonstrate SGA-Former’s superior performance and generalization capability.

### Weaknesses
1. Although the authors claim that the spectral relation matrices can be computed via simple matrix operations on the existing graph structure, the per-layer/per-graph time complexity, memory consumption, and actual runtime on large-scale graphs are not provided.

2. The definitions of (M_{\text{low}}) and (M_{\text{high}}) involve multiple powers of (L_{\text{sym}}) and linear combinations of (A) and (D). While avoiding eigen-decomposition, in practice these matrices may incur high storage and transmission costs for batched graphs, and the generality across different graph sizes is unclear.

3. The SA-Pruner adopts a hard top-α strategy, which prevents direct gradient propagation and may affect the stability of end-to-end training.

### Questions
1. Please provide an analysis of the time and space complexity of computing the spectral relation matrices per layer and per graph, and report runtime performance on large-scale graphs.

2. Explain the storage, transmission, and sparsity optimization strategies for spectral matrices in batched graph processing, and validate the method’s scalability across different graph sizes.

3. Discuss the impact of the non-differentiable pruning strategy in SA-Pruner on end-to-end training, and consider providing differentiable alternatives or stability verification.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper develops a new graph Transformer called SGA-Former. The authors argue that existing attention-based GNNs are built on spatial domain which naturally ignores the graph spectral information. To this end, the authors develop a new attention module based on low- and high- frequency filters to preserve various frequency information between nodes. Experimental results on various datasets show the effectiveness of the proposed method on graph data mining tasks.

### Strengths
1.This paper is well-organized and easy to follow.

2.The authors provide the theoretical analysis of the proposed method.

3.The proposed SGA-Former provides new insights for GTs.

### Weaknesses
1.The research gap is somewhat overclaimed.

2.The complexity analysis is missing

3.Mainstream baselines are missing.

4.The proposed method seems to be sensitive to the hyper-parameters.

### Questions
1.The authors claim the limitation in existing GTs which lack objectivity. There are also several works, such as Specformer and GrokFormer, which are built on spectral information-guided attention modules.

2.Matrix eigendecomposition and node sampling are time-consuming operations. Given the marginal performance gains according to the experimental results, I suggest the authors provide the corresponding complexity analysis of the proposed method.

3.Moreover, the efficiency study is also required to determining the training cost of SGA-Former as well as baselines.

4.Some recent GTs are suggested to be added as baselines.

5.According to Table 8, I have noticed that the proposed method seems to be sensitive to some hyper-paramerters, such as “hidden dim”, “k” and “PE dim”. Based on the results of performance comparison, it is questionable whether the proposed method can bring meaningful performance gain in graph mining tasks.

6.In addition, according to Table 7, the average number of nodes within a graph is quite small. Does this situation imply that SGA-Former can only handle small graphs and hard to be conducted on large-sacle graphs with thousands or millions of nodes?

### Soundness
2

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
This paper introduces low-frequency and high-frequency matrices to inject additional inductive biases into the attention mechanism of Graph Transformers, with theoretical proofs demonstrating the model's strong expressive power. Extensive comparisons against numerous baselines on standard graph datasets highlight the model's superior empirical performance.

### Strengths
1. The paper is clearly written and easy to follow, with well-chosen figures that enable readers to quickly grasp the proposed method.

2. The theoretical analysis is rigorous, providing solid proofs of the model's strong expressive power.

3. The experiments are comprehensive, demonstrating the method's effectiveness through comparisons with a wide range of baselines on multiple standard graph datasets. Ablation studies further validate the necessity of each component.

### Weaknesses
1. Minor errors:
 - Page 1, line 085: "with both both" contains a redundant "both".
 - Page 1, line 090: "demenstrate" should be "demonstrate".
 - Table 2, Peptides-func: the second-best model is GRIT, not MSA-GT.


2. The hyperparameter α is only analyzed on the MNIST and CIFAR datasets. It would be insightful to evaluate its impact on long-range benchmark datasets to better understand how it influences long-range dependencies.

3. It would be beneficial to include visualizations comparing the learned attention coefficients $ A_{\text{enhancer}} $ with those from the baseline Graph Transformer to highlight their differences.

### Questions
See above

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
4

### Summary
The paper proposes SGA-Former, a Graph Transformer that injects spectral inductive bias into attention via two polynomial spectral relation metrics (M_low and M_high) and a two-stage Spectral-Guided Attention Enhancer (prune + scale). The design yields a dual-branch attention layer combining low- and high-frequency structure. Experiments across Benchmarking-GNNs, LRGB peptides, and ZINC-full show strong results, with ablations on pruning/scaling, α keeping rate, and polynomial order k.

### Strengths
1. Simple, principled mechanism to inject spectral bias directly into attention via prune-and-scale—easy to implement on top of existing GTs.
2. No eigendecomposition; clean polynomial construction with clear spatial interpretation.
3. Consistent empirical gains across diverse tasks; strong ZINC-full/LRGB performance. 
4. Useful expressivity analysis relative to SPD biases.

### Weaknesses
1. Theoretical-practical gap: Proposition 1 assumes access to each term $(\tilde{A})^t$ and $k>\text{diam}(G)$, while practice uses the summed metric with moderate k. Please reconcile and provide conditions under which the practical SGA-Former inherits the stated advantage.
2. Efficiency: Despite pruning, runtime stays essentially quadratic unless attention computation itself is sparsified. Compare wall-time/VRAM vs. Exphormer/GPS at similar accuracy; report FLOPs/throughput.
3. Scope of datasets: Add OGB molecular + heterophily/social graphs and very large graphs to validate claims on long-range and boundary modeling beyond peptides and ZINC.

### Questions
1. Complexity: What is the end-to-end time/memory delta vs. GRIT and Exphormer on LRGB (same batch sizes)?

### Soundness
3

### Presentation
3

### Contribution
3
