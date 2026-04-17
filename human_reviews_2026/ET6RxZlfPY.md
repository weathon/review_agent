# Community-Aware Hard Subgraph Mining for Out-of-Distribution Generalization

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Graph Neural Networks (GNNs) are widely used for node classification tasks but often struggle to generalize when training and test nodes follow different distributions, limiting their real-world applicability. Recent approaches based on invariant learning attempt to address this issue but rely on impractical predefined environment labels or low-quality synthetic environments, and their strict invariance assumptions often fail under complex community-level variations. 
In this work, we propose  \textbf{CHASM} (\emph{\textbf{C}ommunity-Aware \textbf{Ha}rd \textbf{S}ubgraph \textbf{M}ining}), a novel framework for OOD generalization on graphs that explicitly leverages latent community heterogeneity. CHASM adversarially mines the hardest subgraphs via a learnable mask model, imposes community-aware regularization to enforce structural coherence, and applies adaptive subgraph augmentation to enhance robustness. A stability-driven learner is then optimized against these hardest cases, yielding a principled and effective solution to community-level shifts. 
Extensive experiments under covariate and concept shifts demonstrate that CHASM consistently outperforms state-of-the-art baselines, while theoretical analysis provides further justification of its robustness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CHASM, a framework for improving GNN generalization under distribution shifts by mining "hard subgraphs" from latent community structures. CHASM uses adversarial learning to identify challenging subgraphs, applies community-aware regularization, and uses adaptive augmentation. Experiments on benchmark datasets show improvements over existing methods under both covariate and concept shifts.

### Strengths
1. The method doesn't require manual environment labels, which is a real pain point in applying invariant learning to graphs. 

2. The paper includes extensive ablations, scalability tests across different backbones (GraphSAGE, PolyNormer, GAT, etc.), and evaluations on multiple datasets. The visualizations of mined subgraphs are helpful.

3. CHASM shows consistent improvements, often +5% or more over strong baselines like CIA and FLOOD on the GOOD benchmark.

### Weaknesses
1. The paper introduces "community heterogeneity" and "hard subgraph mining" as if they're established concepts, which lacks a formal definition. What exactly makes a subgraph "hard"? How is "community heterogeneity" different from homophily/heterophily? The paper needs to define these formally.

2. There's a gap in the motivation. The introduction criticizes invariant learning for requiring environment labels and using synthetic augmentations, then jumps to "leverage community heterogeneity" as the solution. But why are communities the right answer to these problems? The connection isn't justified theoretically or empirically. It seems to me that communities are more crucial for graph-level tasks, rather than node-level tasks. 

3. The paper says community-level variations "naturally give rise to hard subgraphs," but this isn't explained. Hard in what sense: high loss? Low confidence? And why does community structure naturally create hardness? This claim lacks theoretical support or empirical evidence.

### Questions
1. Can you provide a formal definition of "community heterogeneity" and explain how it differs from existing concepts like homophily or subgraph spurious correlations?

2. Why is community structure particularly suited for node-level OOD? Can you provide theoretical or empirical justification for this design choice?

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
The paper proposes CHASM (Community-Aware Hard Subgraph Mining) for node-level OOD generalization. It adversarially mines a “hardest” community/subgraph via a learnable mask maximizing KL divergence from the global predictor, regularized for structural coherence, performs adaptive subgraph augmentation (feature masking + edge dropout) with strengths tied to discrepancies between the mined subgraph and the full graph, and trains a stability-driven learner with a first-order KL approximation. Experiments on GOOD splits of WebKB, CBAS, Twitch, Cora, and Arxiv claim consistent SOTA improvements under covariate and concept shifts, with ablations and limited backbone studies.

### Strengths
1. The proposed method is intuitive. Treating latent communities as implicit environments is natural for graphs, and mining worst-case subgraphs aligns well with distributionally robust optimization (DRO) and hard-case training. The min–max formulation with community-aware KL regularization and structural coherence (Bernoulli–Poisson-inspired) is also well-motivated.

2. The proposed method demonstrates strong empirical performance, and a comprehensive ablation study is provided to support the effectiveness of each component.

3. The literature review is thorough and well-organized, showing a solid understanding of related work.

### Weaknesses
1. It would be helpful to include a complexity analysis, especially for the bi-level optimization, which appears computationally expensive. Clarifying its runtime and scalability would strengthen the paper.

2. The proposed method relies on strong assumptions (for example, the “community expansion” assumption in Assumption 4.1). It would be helpful to either provide more theoretical justification or include intuitive real-world examples that illustrate when this assumption holds.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets node-level OOD generalization on graphs where latent communities induce distribution shifts between train and test. It proposes CHASM, which mines worst structurally coherent subgraphs, applies adaptive feature/structure augmentations to those hard subgraphs, and trains a community-stability learner that penalizes the KL divergence between the hardest-community conditional label distribution and the overall training distribution, approximated by a first-order Taylor expansion. The overall objective couples ERM with a community-aware regularizer over a set of distributions centered on the training distribution. Experiments using GOOD benchmark show gains over baselines. The paper also visualizes mined hard subgraphs and argues that CHASM is backbone-agnostic.

### Strengths
1. The paper formalizes latent community heterogeneity and why iid learnability may fail, justifying community-aware training. 

2. Using GOOD splits to separate covariate vs concept shift improves comparability to recent literature. Experiment shows mined subgraphs align with community structure under both shift types, supporting the intended behavior. 

3. Backbone-agnostic proposal is useful. The paper states CHASM works with diverse GNNs, which, if evidenced, matters for practice.

### Weaknesses
1. Community mining lacks comparative baselines. To justify the mined hardest subgraph module, compare against strong hard-example mining/adversarial structure-editing methods on graphs (e.g., EERM’s adversarial context explorers) to disentangle gains from mining vs. the rest of the pipeline. Comparing to EERM as a mining component variant (not just as a full baseline) would clarify contribution. Because EERM explicitly creates virtual environments with adversarial graph edits, it stands for the class of adversarial environment synthesis methods. 

2. Augmentation strategy needs stronger ablations. Showing results with and without adaptive strength, feature-only, structure-only, and uniform augmentations, plus graph augmenters like Graph Mixup can help evidence that adaptive subgraph-targeted augmentation is the key driver, not generic regularization. 

3. The KL penalty uses a first-order Taylor surrogate, which can use more justification. The sensitivity to the step size, numerical stability, and comparison to directly optimized f-divergence surrogates or DRO objectives are needed to validate the approximation is not the bottleneck. 

4. The paper states CHASM is backbone-agnostic, thus including a table with main-stream backbones like GCN/GAT/GIN/GraphSAGE and one recent Graph Transformer to substantiate scalability across architectures can support the contributions better.

### Questions
What are training-time overheads? Any batching or parallel mining tricks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CHASM, a method for improving OOD generalization in graph neural networks by leveraging latent community heterogeneity rather than predefined environments or heuristic augmentations. The method adversarially mines the hardest subgraphs using a learnable mask model that ensures structural coherence. CHASM then applies community-aware regularization and adaptive subgraph augmentation. A stability-driven learner minimizes loss under these hardest subgraphs, with theoretical analysis proving a bound on generalization error under community shifts. Experiments on multiple OOD benchmarks are provided where CHASM achieves better performance over baselines such as IRM, VREx, FLOOD, and CIA.

### Strengths
1. The min-ERM plus sup over community distributions with a KL penalty (approximated by first-order expansion) is grounded and optimizable by SGD. 

2. Augmentation strength that reacts to hardness is a reasonable mechanism to expose the model to realistic stress tests. 

3. Baseline panel includes ERM, IRM, VREx, DRO, DANN/DeepCoral, and graph-specific OOD methods (EERM, SRGNN, FLOOD, CIT, CaNet, CIA).

### Weaknesses
1. Some graph OOD baselines are missing, such as SizeShiftReg, which are relevant to CHASM’s focus on community/structure shifts. 

2. The method asserts mined subgraphs are structurally coherent. It would be beneficial to including metrics (e.g., conductance/modularity) and show how coherence relates to OOD gains to move beyond qualitative plots. 

3. Worst-community extraction plus adaptive augmentation suggests non-trivial cost. Computational overhead and scalability should be discussed by providing asymptotic analysis, wall-clock vs. ERM, and memory usage on the large dataset, and discussing parallelization. 

4. The optimization doesn’t tightly define/estimate C in the main text. The paper should make explicit whether C is an empirical set over mined subgraphs, a divergence-ball, or both. This matters for understanding guarantees.

5. Is mining solved exactly or approximately? What objective/constraints enforce structural coherence and how do authors avoid degenerate (e.g., tiny/bridge) subgraphs?

### Questions
Please see weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
