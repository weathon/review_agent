# An Effective and Efficient Generation Framework for Condensing the Graph Repository

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Graph repositories with multiple graphs are increasingly prevalent in various applications. As the amount of data increases, training neural networks on graph repositories becomes increasingly burdensome. However, existing condensation methods focus more on reducing the size of a single graph, they fail to address the challenges of efficiently and effectively compressing multiple data graphs. In this work, we propose a novel end-to-end graph repository condensation framework (GRCOND) that effectively condenses a large-scale graph repository with multiple graphs, while preserving task-relevant structural and feature information. Unlike traditional methods, our approach pretrains a dataset-specific GNN model to create and optimize synthetic graphs so that we can capture both intra-graph structures and inter-graph relationships, enabling a more holistic representation of the repository. Through experiments, our proposed approach consistently delivers higher accuracy and feature retention with different compression ratios, which highlights the potential of our framework to accelerate GNN training and expand the applicability of graph-based machine learning in resource-constrained environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GRCOND for condensation of large graph repositories. The method learns a compact set of latent codes that are decoded into synthetic graphs and features. A frozen pretrained encoder provides representational priors, and two dedicated decoders reconstruct topology and attributes. The synthetic set is optimized with class-wise gradient matching so that training on the condensed data yields update trajectories that closely follow those induced by the full corpus. The study evaluates multiple benchmarks across multiple compression ratios, showing consistent improvements over sampling and prior condensation methods, together with notable reductions in training cost.

### Strengths
1. The studied problem is important for large-scale graph research and applications, where full training cycles are costly.

2. The document-to-latent pipeline is clear from pretraining through initialization to trajectory matching and decoding.

3. Cost and speed evidence is provided, which strengthens the real-world usefulness of the framework.

### Weaknesses
1. Missing important and strong recent baselines. The paper should align budgets and hyperparameter tuning across all methods, add at least one recent distribution matching approach adapted to graphs, and document model selection using a shared validation protocol.

2. Reported averages lack a consistent statement on seeds and confidence intervals. Please provide three to five seeds, report mean and percentile confidence intervals, and run standard significance tests on headline gains.

3. All tasks concern graph classification on molecules and proteins. Results on social or heterogeneous graphs would help test transfer. A small out-of-domain repository would improve external validity.

4. Several symbols are introduced without complete definitions, including $\alpha$, $\beta$, $\psi$, $f$, and $\sigma$. The structure of the distance term Dis($\cdot$) and the batching scheme for class-wise gradient matching are not fully specified. The paper should present the complete loss composition, normalization choices, and batch formation rules in one place.

### Questions
1. Are the encoder and both decoders trained strictly on training graphs?

2. What exact distance is used in Dis($\cdot$) and how is it normalized across layers?

3. What is the effect of removing class-wise matching in favor of a global objective?

4. How sensitive is GRCOND to the chosen frozen encoder family?

5. Can you show results on a repository with more classes and significant class imbalance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses the computational overhead in GNN training on large graph repositories, which is not met by existing single-graph condensation methods. It proposes GRCOND, a novel end-to-end framework designed to effectively condense multiple graphs to a small synthetic set while preserving structural and feature information. GRCOND uses a pre-trained GNN decoder to optimize synthetic graphs in a continuous latent space to bypass the discreteness of graphs. Empirical results demonstrate that GRCOND substantially reduces training costs while maintaining high classification accuracy across various real-world benchmarks.

### Strengths
GRCOND shows a novel application of dataset condensation to multiple graphs. The paper solves the challenge of graph data's discrete structure by using a pretrained decoder, making the condensation technique effective for multiple graph samples. A thorough ablation study and comprehensive evaluation that checks the method from diverse perspectives.

### Weaknesses
- The paper suffers from poor clarity and inconsistent presentation, making it very difficult to read and understand. It is not self-contained; mathematical notations (e.g., $L, t, Dis$) are used without formal definition, some notations are duplicated (e.g., $D$ for set vs decoder), inconsistent (e.g., $D_o$ vs $D_O$), confused (e.g., $J$ vs $\mathcal{J}$), or abused (e.g., minus operation on synthetic graph $S$ generated by the decoder. How can we define minus on graphs?) The connection between the main text and the figures is weak (e.g., CE loss in Figure 1 is absent from the main text), and the pretraining loss is not clearly specified. There should be a full review (not only examples I described) and a cleanup of all notations, descriptions, and illustrations for better readability
- A significant weakness is the lack of a public code release or placeholder. Accepting a paper without code is not acceptable.
- The paper needs to provide a stronger justification and a detailed ablation study for using the pretrained decoder, which is the largest contribution. The paper must clarify precisely how the variants in Table 4 were implemented: whether the trainable autoencoder variant was optimized during the condensation process, and whether the other two variants ("We directly perform gradient descent on the node features and structural features of the graph to optimize graph data. The other variant used the untrained model to test") were emplyed using frozen parameters. A more detailed ablation focusing purely on the state of the decoder (Frozen Pretrained vs. Frozen Untrained vs. Trainable during Condensation) is required.
- There is a lack of benchmark domains (chemical and biological domains). There is doubt that it works on other domains like ogbg-code2 (also an example).

### Questions
.

### Soundness
3

### Presentation
1

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
This paper addresses the scarcity and low efficiency of existing graph-level graph condensation methods. The proposed graph-level compression model that determines the parameters of its encoder and decoder through a pre-training phase. And to optimize the learning process, the model employs a gradient matching technique between the original and synthetic graphs to adaptively adjust the sampling strategy for the original dataset. Experiment results demonstrate the superior efficiency of the method. The main contributions of this work are the proposal of an optimization strategy based on gradient matching and end-to-end condensation framework for graph repositories.

### Strengths
1. Pre-training is utilized to freeze key parameters within the model, which reduces the training workload and enhances the overall compression efficiency.

2. The authors use the concept of "graph repository" to interpret an intrinsic, yet previously overlooked, means that graph data originating from the same dataset possess inherent correlations. This approach concurrently establishes the distributional relationships among the condensed graphs.

3. The method demonstrates excellent efficiency.

### Weaknesses
1. The number of baseline models included in the state-of-the-art (SOTA) comparison, generalization performance tests, and computational cost analysis is insufficient, and the majority of them are outdated. It is highly recommended to include more recent graph-level models to better substantiate the claimed superiority of the proposed method in terms of both performance and efficiency. *e.g., KDD2024- Wang Y, Yan X, Jin S, et al. Self-supervised learning for graph dataset condensation.*


2. The abstract and introduction do not specify the concrete problems in prior algorithms that the proposed graph-level condensation method aims to solve. Instead, its contribution is broadly summarized as an efficient solution for multi-graph dataset condenstaion.

### Questions
Q1： The paper only mentions random sampling. Have other sampling strategies been considered? Would including them in the ablation study potentially yield better results?

Q2：Could the generalization performance be validated on a broader range of datasets beyond just PROTEINS, similar to the experimental setup in Table 4? Furthermore, for improved readability, it is suggested to bold or highlight the best-performing results in both tables.

Q3：How exactly does the matching loss optimize the sampling process, which is currently based on random sampling?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes an end-to-end Graph Repository Condensation (GRCOND) framework that effectively condenses a large-scale graph repository with multiple graphs while preserving task-relevant structural and feature information. Unlike traditional methods focusing on a single graph, the approach pretrains a dataset-specific GNN model to create and optimize synthetic graphs, capturing both intra-graph structures and inter-graph relationships for a more holistic representation. Experiments show that GRCOND achieves higher accuracy and retains features across different compression ratios, highlighting its potential to accelerate GNN training and enhance applicability in resource-constrained environments.

### Strengths
1. The paper introduces an end-to-end framework that effectively condenses large-scale multi-graph repositories while preserving both structural and feature information.
2. It leverages a pretrained, dataset-specific GNN to generate and optimize synthetic graphs, capturing intra- and inter-graph relationships comprehensively.
3. Experiments demonstrate strong performance across compression ratios, showing high accuracy and scalability for resource-constrained GNN training.

### Weaknesses
1. The paper lacks experiments on the training time of the graph generation model and the total time combining both generation and condensation processes.
2. The motivation for condensation is unclear—although efficiency is mentioned, if the overall training time is long, direct training on downstream tasks might be more practical.
3. While the paper claims that one condensed graph can support multiple downstream tasks, it does not provide experiments to validate this claim.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
