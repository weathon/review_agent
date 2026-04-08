## Human Reviewer 1

### Summary
This paper introduces a new trajectory representation approach that models trajectory semantics through three hierarchical layers at different granularities: point, segment, and the whole trajectory. Specifically, it follows similar ideas to JEPA and T-JEPA by training a trajectory encoder for each layer and applying different random masks to encourage the model to predict the masked portions. Experiments on several benchmarks for trajectory similarity search tasks demonstrate the generalization and effectiveness of the proposed method.

### Strengths
S1. The paper is easy to read, and the argument that existing methods lack hierarchical representations is both interesting and important.
S2. The methodology, although based on techniques used in previous studies, is sound and well-justified.

### Weaknesses
W1. Although it is interesting to consider different levels of trajectory semantics, the authors do not provide sufficient experimental evidence to show the effectiveness of their approach. For example, it is unclear which representation layers (S1, S2, S3, or their combination) are used for the trajectory similarity search experiments. These details appear to be missing. If the proposed framework indeed works as claimed, the different representations should capture distinct semantic meanings of trajectories. Therefore, different tasks tailored to each learned representation should be included. For example, trajectory clustering, which groups trajectories based on high-level semantic similarities, should be compared with existing deep-learning-based clustering methods. Additionally, behavioral pattern detection would be an ideal task to demonstrate the usefulness of intermediate representations. Currently, the experiments are limited to trajectory similarity tasks (self-similarity and similarity search via fine-tuning), which are insufficient to demonstrate the advantages of hierarchical representations.

W2. The use of self-similarity does not very meaningful in this context. Since HiT-JEPA is trained with random masking of trajectory points, strong self-similarity results are somewhat expected. The paper focuses heavily on similarity matching as the sole measure of embedding quality, which is inadequate. Moreover, for the Porto dataset, where performance is worse than on others, the authors attribute this to higher sampling frequency. While this explanation is plausible, the paper should more explicitly discuss this limitation and clarify the scenarios where the proposed method may or may not be effective.

W3. Can the learned encoder be applied to trajectories from different geographic regions without retraining from scratch? Discussing this aspect would strengthen the claims about generalization and practical applicability.

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces HiT-JEPA, a hierarchical self-supervised framework for learning urban trajectory embeddings suitable for similarity computation. HiT-JEPA employs a three-level joint embedding predictive architecture (JEPA) to progressively capture point-level, segment-level, and global trajectory abstractions, integrating top-down and multi-scale semantic information. The approach is evaluated extensively against state-of-the-art baselines on multiple urban trajectory datasets, including zero-shot, and downstream tasks. Results claim remarkably improvements in mean ranking performance, generalization, and ablation integrity on various trajectory similarity and retrieval tasks.

### Strengths
S1. HiT-JEPA proposes an explicit three-layer architecture that learns and aligns representations at point, segment, and trajectory levels.

S2. The method implements a novel attention propagation strategy between abstraction levels. And it upsamples higher-level attention maps to guide feature extraction at lower levels.

S3. HiT-JEPA demonstrates exceptional zero-shot performance across heterogeneous datasets. Experimental design is rigorous and covers critical scenarios like self-similarity, ablation and visualization.

### Weaknesses
W1. This manuscript has presented limited novelty comparing to prior works. The JEPA and multi-scale structure build heavily on established elements in self-supervised learning, attention propagation, and urban trajectory modeling. The direct design leap over T-JEPA, is incremental rather than a decisive paradigm shift, especially given already cited work such as HIBERT and HiCLRE in NLP and CV. So, this work simply applies these models to the trajectory dataset without making more targeted designs based on the characteristics of trajectory data.

W2. The manuscript claims high-level attention "guides local feature extraction" but does not clarify what specific semantic information high-level attention captures. The authors should provide specific explanations regarding the "semantic information," such as the clustering characteristics of trajectories or the motion features of moving objects. They may even need to include a case study to illustrate this point. Otherwise, the high-level attention mechanism remains a black box, merely effective by coincidence in the current dataset.

W3. HiT-JEPA uses Uber H3 grids to map GPS coordinates instead of traditional rectangular grids but failing to explain how H3’s hexagonal symmetry improves trajectory spatial relation modeling. The effectiveness of these two grid-based methods should also be compared experimentally.

### Questions
Q1. Provide more description about the trajectory-specific model designs, explaining how these designs enhance the effectiveness of trajectory similarity computation, and discuss the differences when applied in the NLP and CV domains.

Q2. Add attention-weight visualizations and controlled experiments to clarify and show how it quantifies the guidance. Or add a case study to show the effectiveness of a high-level attention mechanism.

Q3. Conduct an ablation study comparing H3 grids with rectangular grids on selected datasets.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
5

---

## Human Reviewer 3

### Summary
The paper presents HiT-JEPA, a hierarchical self-supervised framework for trajectory representation that explicitly models point-level, segment-level, and trajectory-level semantics with cross-level interactions. A JEPA (Joint Embedding Predictive Architecture) objective aligns representations in embedding space, complemented by VICReg-style regularization to prevent collapse. Experiments across multiple cities/modalities (including AIS) show strong retrieval performance and promising zero-shot generalization.

### Strengths
- Address single-scale bias by introducing an explicit multi-scale hierarchy with information flow across levels.
- Propose a JEPA-style training reduces reliance on heavy data augmentation; ablations indicate the benefit of hierarchical interactions.
- Extensive experiments across datasets and zero-shot settings supports generalization claims.

### Weaknesses
- Only two contrastive learning baselines are compared in the experiments. There are many trajectory representation learning works should be compared. 
- The interpretations of the results of HiT-JEPA are quite confusing. It would be better to explain why the two showcases can illustrate the interpretation of the proposed method.
- For different trajectory abstraction method, the results would change. From equation (3)-(5), the sampling rate are 50% and 25% of orginal trajectories which may not be the best setting.  How  the abstraction method influence the results should be disscussed. 
- The efficiency of JEPA should be considered.

### Questions
See the weakness above.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper proposes HiT-JEPA, a framework for learning multi-scale urban trajectory representations across different semantic abstraction levels. This framework constructs a three-layer hierarchical structure, which generates local point-level, intermediate segment-level patterns, and high-level global route structure. HiT-JEPA also includes a joint embedding predictive architecture to incoporate multiple levels of abstractions and approximate the target trajectory. Experiments on real-world urban GPS trajectory datasets show that HiT-JEPA exhibits stable performance in self-similarity search, zero-shot scenarios, and downstream fine-tuning tasks.

### Strengths
1. HiT-JEPA constructs a three-layer hierarchical structure within a single model using three pooling layers. It is the first architecture to explicitly unify both fine-grained and abstract trajectory patterns in one framework.

2. Experiments on multiple real-world datasets and few-shot learning tasks demonstrate its generalizability.

### Weaknesses
1. The paper lacks sufficient novelty and mainly appears to be an implementation of industry approaches.
2. The experimental section lacks sufficient and diverse baselines, for example transformer-based sequence forcasting methonds.
3. The paper does not provide detailed interpretations or examples to demonstrate how the learned three-layer structure captures different levels of semantics. It remains unclear whether the extracted multi-scale trajectory representations truly correspond to point-level or higher-level semantics.
4. Multi-layer Transformer struction can also capture multi-granular information: lower layers typically encode specific input semantics, intermediate layers abstract higher-level concepts, and final layers contribute to generation or prediction. It would be helpful to explain why the authors chose a multi-layer convolutional structure instead of multi-layer Transformer. Would the convolutional yield any advantage in performance or interpretability?

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
2