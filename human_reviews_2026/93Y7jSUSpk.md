# Out-of-Distribution Graph Models Merging

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
This paper studies a novel problem of out-of-distribution graph models merging, which aims to construct a generalized model from multiple graph models pre-trained on different domains with distribution discrepancy. This problem is challenging because of the difficulty in learning domain-invariant knowledge implicitly in model parameters and consolidating expertise from potentially heterogeneous GNN backbones. In this work, we propose a graph generation strategy that instantiates the mixture distribution of multiple domains. Then, we merge and fine-tune the pre-trained graph models via a MoE module and a masking mechanism for generalized adaptation. Our framework is architecture-agnostic and can operate without any source/target domain data. Both theoretical analysis and experimental results demonstrate the effectiveness of our approach in addressing the model generalization problem.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the problem termed Out-of-Distribution Graph Models Merging, aiming to build a generalized graph model by merging multiple pre-trained GNNs trained on different domains, without requiring access to source data. The authors propose a two-stage framework: (1) a label-conditional graph generation phase that inverts each pre-trained GNN to synthesize pseudo-graphs representing its domain knowledge, and (2) a model merging phase that employs a fine-tuned Mixture-of-Experts with learnable masks and sparse gates to integrate the knowledge of heterogeneous GNNs. Theoretical analyses are provided based on a mixture distribution assumption, and experiments o graph classification benchmarks show improvements over ensemble and model soup baselines.

### Strengths
1. The idea of merging multiple pre-trained graph models without source data is interesting and practically relevant in the context of model reusability and privacy-aware learning.

2. The combination of graph generation and MoE-based merging is conceptually coherent and could inspire further exploration of source-free graph model integration.

3. The paper is well-organized and provides technical clarity, including equations, regularizers, and ablation settings.

### Weaknesses
1. The proposed task largely overlaps with graph-free knowledge distillation [1] and model soup [2], making the originality less substantial. The contribution seems more like a synthesis of existing ideas than a fundamentally new paradigm.

2. The mixture distribution assumption $G_{T} = \sum_{i} \alpha_{i} G_{i}$ is overly strong and lacks empirical justification, especially in discrete graph spaces. Theoretical analysis is abstract and disconnected from the implemented MoE mechanism.

3. All experiments are conducted on small, low-diversity datasets. No results are shown on large-scale or node-level tasks (e.g., OGB benchmarks). Hence, the claimed “OOD generalization” may not hold in realistic settings.

4. The “domain” split based solely on edge-to-node ratio may not meaningfully reflect true distribution shifts. The evaluation therefore risks testing within-distribution variations rather than genuine OOD generalization.

5. There is no investigation into how experts or masks behave, how gating decisions distribute across domains, or what knowledge is actually merged.

6. Competing baselines are mainly adapted from non-graph domains; missing comparisons to recent source-free domain generalization or prompt-based GNN adaptation methods weakens the empirical argument.

[1]  Graph-free knowledge distillation for graph neural networks.

[2] Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.

### Questions
1. How sensitive is the method to the number and diversity of pre-trained GNNs? Would OGMM still perform well when experts are highly redundant or of poor quality?

2. How stable is the label-conditional graph generation? Are there cases of mode collapse or low diversity?

3. How does OGMM scale with the number of experts (in terms of computation and performance)?

4. Could the authors provide a qualitative or quantitative analysis of the learned gating and mask patterns to better interpret the merging mechanism?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a novel problem in the domain of GNNs: Out-of-Distribution Graph Models Merging (OGMM). The authors propose a framework that merges pre-trained GNNs from multiple domains with distribution shifts to create a unified, generalized model. This approach overcomes the challenge of needing to retrain models from scratch by leveraging pre-trained models to preserve specialized domain knowledge. The two-stage process first generates label-conditional graphs using each model and then fine-tunes and merges them using a MoE module. Experimental results show that OGMM outperforms previous methods on multiple datasets, establishing a new state-of-the-art in graph model generalization.

### Strengths
1. The paper introduces a novel challenge of merging out-of-distribution graph models without needing to retrain from scratch, which is both practical and impactful for real-world applications where data is scarce.
2. OGMM consistently outperforms previous fusion methods and demonstrates robustness on large-scale datasets like REDDIT-B and NCI1, showing that it can handle diverse graph domains with different GNN architectures.
3. The authors provide a solid theoretical framework for the problem and support it with comprehensive experiments that demonstrate the framework's effectiveness across multiple domains and datasets.

### Weaknesses
1. The two-stage process for merging out-of-distribution models involves multiple steps, including fine-tuning and the use of the Mixture-of-Experts (MoE) module. While effective, the overall time complexity of this process could be quite high, especially as the number of pre-trained models and the size of the graphs increase.
2. The experiments primarily focus on datasets such as REDDIT-B and NCI1, which are not necessarily representative of the most commonly encountered graph types in real-world applications. It would strengthen the paper to include additional tests on more widely used or complex datasets, such as social network graphs or biological networks, to better understand the model's applicability across various domains.

### Questions
1. See weaknesses.
2. How are these pre-trained GNNs selected? Are they arbitrarily chosen graph models? If not, how is their optimality determined?
3. Negative transfer is a common issue in generalization domains. I would like to know how OGMM avoids this problem, and whether there is any theoretical or empirical evidence supporting this.
4. Some graph-MoE works should be compared and discussed in the experiments.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies a new problem called out-of-distribution graph models merging, which aims to merge multiple pre-trained GNNs from different domains with distribution shifts into a single generalized model. The proposed OGMM is a two-stage framework, leveraging graph generation and a fine-tuned MoE module to enable generalization under graph OOD scenaros. The paper provides theoretical bounds on generalization error, and extensive experiments on multiple datasets demonstrate substantial performance gains over strong baselines.

### Strengths
1) The integration of graph generation and MoE-based model fusion is conceptually coherent, enabling domain knowledge transfer at both data and model levels.

2) The mixture distribution assumption and accompanying error bound provide a formal justification for the merging process.

3) The framework is applicable to heterogeneous GNNs, enhancing its generality and practical relevance.

### Weaknesses
1) The motivation for merging multiple pre-trained GNNs is not sufficiently justified. It remains unclear why model-level merging is preferable to retraining on aggregated data or simply using the best domain-specific model. No empirical or application-level evidence is given to show that scenarios requiring model-level merging without data access are common or practically constrained.
2) Methodological novelty appears incremental, as the proposed approach largely builds upon existing techniques in graph distillation and mixture-of-experts regularization without a distinct new principle or paradigm.
3) While the paper qualitatively illustrates synthetic graph realism, it does not quantitatively assess their fidelity to the underlying data manifold, leaving the actual extent of domain knowledge recovery uncertain.
4) Notation suffers from inconsistencies such as {G_i}_{i\in M}, and the overall writing is occasionally unclear, making it difficult to follow the theoretical formulation.

### Questions
1) How robust is OGMM to the quality of pretrained GNNs? For example, if one model performs poorly or overfits its domain, does the merging process degrade significantly?
2) Why focus masks primarily on classification heads? Have you ablated masking other layers such as message-passing modules and observed differences in generalization?
3) How do you handle varying graph sizes or node/edge feature dimensions across domains during generation and merging? Is there preprocessing involved?
4) Could OGMM be extended to incremental merging, where new pretrained models arrive sequentially?
5) Would adding a contrastive or alignment loss between expert embeddings further stabilize the merging process?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel problem of out-of-distribution graph model merging, aiming to merge models pre-trained on graphs from different domains into a model that generalizes under distribution shifts. A two-stage approach is proposed: first, a graph generator is trained to generate synthetic graph data, which is then used in the second stage to fine-tune the MoE module. Experiments demonstrate that the proposed method effectively generalizes to data with distribution shifts.

### Strengths
- The proposed problem of Out-of-Distribution Graph Models Merging is novel.
- The proposed method, OGMM, has a solid theoretical foundation. Generating synthetic data to extract domain knowledge makes sense.
- The experiments demonstrate the effectiveness of OGMM and the contributions of each component.

### Weaknesses
- The scenarios considered are limited. More results on other node classification graph datasets could be provided. Additionally, the paper only focuses the OOD scenario within a single graph, without considering cross-dataset or cross-domain scenarios.
- The proposed OGMM relies on a mixture distribution assumption, which is not likely to hold in more complex scenarios.
- The proposed OGMM seems to rely on many hyperparameters, some of which significantly impact model performance according to the analysis in the paper. This could limit its applicability in practical settings.
- Although OGMM is theoretically computationally efficient, some experimental results could be provided.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
