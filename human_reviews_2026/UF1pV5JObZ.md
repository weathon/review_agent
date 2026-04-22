# Unlocking Universal Graph Knowledge in the View Space

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Generalizing a pretrained model to unseen datasets without retraining is an essential step toward a foundation model. However, achieving such cross-dataset, fully inductive inference is difficult in graph-structured data where feature spaces vary widely in both dimensionality and semantics. Any transformation in the feature space can easily violate the inductive applicability to unseen datasets, strictly limiting the design space of a graph model. In this work, we introduce the *view space*, a novel representational axis in which arbitrary graphs can be naturally encoded in a unified manner. We then propose Graph View Transformation (GVT), a node- and feature-permutation-equivariant mapping in the view space. GVT serves as the building block for Recurrent GVT, a fully inductive model for node representation learning. Pretrained on OGBN-Arxiv and evaluated on 27 node-classification benchmarks, Recurrent GVT outperforms GraphAny, the prior fully inductive graph model, by +8.93% and surpasses 12 individually tuned GNNs by at least +3.30%. These results establish the view space as a principled and effective ground for fully inductive node representation learning. Code and datasets are available at https://anonymous.4open.science/r/RGVT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a new way to train inductive gnn model by projecting the original features into a view space. By learning a shared projecter, the learned representation can be projected to desired task dimension through weight sharing. A recurrent variant is further adopted to deal with tasks requiring different hops.

### Strengths
1. the problem is interesting
2. the presentation is generally good 
3. the performance boost over graphany is clear, which demonstrates that universal representation is much better than directly aggregating the closed-form solution

### Weaknesses
1. I don't agree with the claim in the abstract. Graphany is obviously not the only universal graph model. First, you can't say it's universal. graph is not just about node classification. In their papers, they also don't claim they are graph foundation models (they just say fully-inductive classification). Second, if graphany is universal, they i don't see why others like OneForAll are not universal. 
2. In the introduction part, the motivation makes me feel like the authors think the feature heterogeneity is the main problem of building a graph foundation model. I strongly disagree with this viewpoint. FIrst, it's obvious that structure heterogeneity is much harder. For example, from a geometric perspective, the 1-order (entity-level) and 2-order (link-level) tasks are not compatible. Moreover, the homophily-heterophily problem is very challenging to solve. Second, in practice, the feature heterogeneity may be not a "real" problem. There are many relational foundation mdoels like kumorfm and tabular foundation models that can work on heterogeneous feature types. You just need a type-aware encoder, which works generally well. 
3. In the introduction part, authors point out that the way OneForAll takes results in clear performance loss. I also need to point out that the way GraphAny leads to even more performance loss by the inductive transformation. 
4. Section 3 is in general identical to Graphany. 
5. One obvious drawback of recurrent one is the expressivenss problem, You have to weight share the parameters across layers, and non-identical weight learning for heterophilous graph is very important (for example, in EvenNet) 
6. the experimental dataset are selected with relatively weak features. Vanilla GNN may be much better with better features like LLM-encoded ones. 
7. THis model can only work for node classification tasks, and can't do in-context learning, zero-shot learning, generation. I would say it's far away from a foundation model.

### Questions
What's the main design component that makes GVT so much better than graphany? Better explain this in experiment sections.

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
This paper tackles cross-dataset generalization for graph learning by introducing a “view space” that can encode graphs of any size and feature specification, enabling parameter sharing across incompatible node feature spaces. The authors propose Graph View Transformation (GVT), a learned mapping that projects any node-feature matrix through this shared space, and build Recurrent GVT (RGVT) on top as a foundation model for universally shareable graph knowledge. Trained on OGBN-Arxiv and tested on 27 node classification benchmarks, RGVT outperforms GraphAny and beats a dozen tuned GNN baselines. The results suggest that learning in the view space provides a principled and effective route to general node classifier, with code and data released for reproducibility.

### Strengths
1. The heterogeneous feature space is an important problem for GFM. The proposed view-space formulation is a compact and novel solution.
2. Theoreticall study on properties and expressivity results help justify the design.
3. Extensive experiments show consistent improvements with statistical significance
4. Writing is concise, figures are readable, and the narrative is easy to follow.

### Weaknesses
1. The statement given by the title is too strong. The proposed method provides a general model for node classification. Edge level and graph level tasks are not explored, and coverage of graphs without node features or with edge features is not addressed.
2. The choices for view finders is unclear. The paper does not clearly state how many views are used, how each view is constructed, or how sensitive performance is to these choices.
3. There is no complexity analysis. It would be better to provide a comparison of the the training and inference cost with GraphAny and other baselines.

### Questions
1. Is there a theoretical or empirical guideline for selecting the number of views?
2. Can the framework extend to graphs without node features, to graphs with edge features, and to other downstream tasks such as link prediction and graph classification? If so, what changes are required in the view transformation or the predictor?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles a critical problem for graph foundation models: datasets often have incompatible features. The authors solve this by proposing the view space, a novel representation axis that is independent of node features. They then introduce Graph View Transformation (GVT), a theoretically universal function that learns representations within this new space. GVT serves as the core building block for their RGVT foundation model. Impressively, experiments show the pre-trained, frozen RGVT model surpassed 12 specialized GNNs that were individually tuned for their specific tasks.

### Strengths
1. Novelty: The paper introduces the "view space," a new representation paradigm for graphs. This is a significant conceptual leap, as it bypasses the central problem of feature heterogeneity (incompatible dimensions and semantics) rather than trying to fix it with alignment or projection, which is what most prior work does.
2. Theoretical Grounding: The authors provide a solid theoretical foundation for their method. They mathematically prove that the proposed Graph View Transformation (GVT) achieves "dual permutation equivariance" (for both nodes and features), which formally guarantees its universality across graphs with arbitrary feature specifications.
3. Strong Results: The experiments provide compelling evidence for the "foundation model" claim. A single, pre-trained, and frozen RGVT model, with only a lightweight predictor, was able to outperform 12 different, specialized GNN models that were fully and individually tuned for each of the 27 downstream tasks. This demonstrates a remarkable level of generalizable knowledge transfer.

### Weaknesses
1. Overclaim. This manuscript claims that the proposed model is the first graph foundation model that is unlimited by the feature space barrier, enabling universal knowledge transfer. However, recent advances in text-free multi-domain graph pre-training generally do not struggle with the feature space heterogeneity, supporting the knowledge transfer across different graphs. Also, SAMGPT (WWW25) can be viewed as an initial success of GFM in this direction.
2. Miss of related work. The authors state that `` GraphAny, the only universal graph model to date". However, there has been a series of GFM attempts very recently.
3. Weak evaluations. RGVT is validated in the node classification task. However, as a general-purpose foundation model, important tasks, such as link prediction, graph classification, and node clustering, are touched on in the experiment.

### Questions
1. How does the model scale for graphs with no features?
2. GVT processes each feature channel independently. How does the model learn interactions between different features if they never mix?
3. By treating all features identically (to ensure feature equivariance), doesn't the model lose the ability to apply feature-specific logic, such as processing categorical and numerical features differently?
4. How sensitive is performance to the specific set of view finders?
5. How does the model's computational cost scale with a large number of features? A runtime benchmark against standard GNNs (which project the feature dimension to a small dimension) is missing.

### Soundness
2

### Presentation
3

### Contribution
2
