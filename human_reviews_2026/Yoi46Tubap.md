# OpenGLA: Topology and Task Adaptive Foundation Model for Power System Graph-Language Answering

- Decision: Reject
- Scores: 6, 2, 8

## Abstract
Foundation models have shown impressive cross-modal generation and problem-solving abilities, yet directly applying them to power systems remains challenging due to strict requirements on accuracy, efficiency, and physical interpretability. We propose OpenGLA, a distinct graph–language foundation model for power systems. OpenGLA encodes grid states with a topology-adaptive GCN and an adaptive nodal feature encoder, projects them into the language embedding space, and fuses with textual instructions via a Mixture-of-Transformers module. A lightweight transformer detokenizer is designed to enable precise floating-point outputs. Experiments demonstrate that OpenGLA generalizes across diverse tasks and grid topologies while achieving superior accuracy, establishing a scalable foundation model architecture in critical infrastructure.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces OpenGLA, a novel foundation model specifically designed for Power System Graph-Language Answering (GLA) tasks. The authors argue that applying existing Large Language Models (LLMs) directly to power systems is challenging due to the inherent graph-structured nature of the data and the strict requirements for high-precision, floating-point outputs.

To address this gap, OpenGLA introduces an innovative multi-modal architecture with the following core contributions:

1. Graph Encoder: It utilizes a Graph Convolutional Network (GCN) to effectively encode grid data with varying topologies, achieving adaptability to different power system structures.
2. Mixture-of-Transformers (MoT): A sparse model designed to efficiently fuse graph embeddings with natural language instructions. It uses separate parameters for each modality, which effectively mitigates gradient conflicts in multi-task learning.
3. Specialized Output Module: To handle the need for precise floating-point outputs, the model incorporates a lightweight transformer "detokenizer" that generates numerical vectors instead of discrete text tokens.
4. Two-Stage Training Strategy: Inspired by vision-language model training, the strategy first aligns the feature space of the graph encoder with the pre-trained LLM, followed by end-to-end fine-tuning.

The authors conduct comprehensive experiments across four critical power system tasks: Optimal Power Flow (OPF), Fault Detection, State Estimation, and Local Marginal Price (LMP) prediction. These experiments span multiple grid scales, from the IEEE 14-bus system to the realistic Texas 2000-bus system. The results demonstrate that OpenGLA significantly outperforms existing task-specific models and fine-tuned LLMs in nearly all tested scenarios, showcasing its superior performance, generalization, and scalability.

### Strengths
1. Clear Problem Formulation with High Application Value: The paper is well-motivated, precisely targeting the challenges of applying foundation models to the critical domain of power systems. The proposed solution has significant real-world relevance and substantial potential for industrial application.
2. Innovative and Well-Justified Model Architecture: The design of OpenGLA is both innovative and thoughtfully crafted. Each component addresses a specific and significant challenge: the GCN for topological adaptability, the adaptive nodal encoder for task-input flexibility, the Mixture-of-Transformers (MoT) for efficient multi-modal fusion, and the Transformer Detokenizer for handling precise numerical outputs. Together, they form a cohesive and powerful system.
3. Extensive and Compelling Experimental Validation: One of the most outstanding merits of this paper is its rigorous and comprehensive experimental validation. The authors have tested their model across a diverse set of tasks and on power grids of vastly different scales (from standard IEEE cases to the large-scale Texas 2000-bus system). The comparisons against multiple strong baselines provide irrefutable evidence of the model's superiority. The successful evaluation on the Texas grid, in particular, demonstrates the model's potential to scale to practical, large-scale applications.
4. Demonstrates Excellent Scalability: The paper showcases a key characteristic of a true foundation model. Through the scaling law analysis (Figure 8), it shows that the model's performance improves predictably with increases in parameter and data scale. This provides strong evidence for its future potential and aligns the work with the principles of large-scale model development.

### Weaknesses
1. Limited Evaluation of Topological Generalization: The most significant limitation is that the model's generalization capabilities are not tested in the strictest sense. According to the experimental setup, the model is trained and evaluated on the same grid topology (e.g., trained on IEEE-118 data, tested on IEEE-118 data). For a model positioned as a "foundation model," a more rigorous test would be cross-topology generalization—for instance, training on a set of topologies (e.g., 14, 39, and 57-bus systems) and evaluating its zero-shot performance on an entirely unseen topology (e.g., the 118-bus system). The current evaluation primarily demonstrates generalization to new operational scenarios within a known structure, rather than to new structures themselves.
2. Performance Trade-offs Against Specialized Models: As observed in Figure 4, while OpenGLA's overall performance is exceptionally strong, task-specific SOTA models like PatchTST can achieve comparable or even slightly better performance on their niche tasks (i.e., time-series-based fault detection and localization). This is not a flaw but rather highlights a classic trade-off between a generalist and a specialist. The paper could benefit from a more direct discussion of this trade-off, acknowledging that while OpenGLA’s strength is its unified and versatile nature, highly optimized specialist models may still hold an edge in their specific domains.
3. Absence of Hyperparameter Sensitivity Analysis: The appendix provides the final set of hyperparameters used for training, but the paper lacks a sensitivity analysis. A study on how performance is affected by key hyperparameters (e.g., learning rate, number of GCN layers, or embedding dimensions) would significantly strengthen the paper's claims of robustness. Without it, it is difficult to assess the model's stability and how easily the results can be reproduced.
4. Unfulfilled Promise of Physical Interpretability: The ABSTRACT rightfully highlights "physical interpretability" as a critical requirement for power system applications, setting it up as a key motivation for the proposed model. However, the paper fails to deliver on this promise. The experimental section focuses entirely on performance metrics like accuracy and error rates, without providing any analysis to verify that the model's reasoning aligns with the physical laws of power systems. There is no qualitative analysis, such as visualizing conducting case studies, to show why the model makes its predictions. This omission creates a significant gap in the paper's narrative, leaving the claims of suitability for critical infrastructure partially unsubstantiated.
5. Inadequate Benchmarking Against Non-Learning Baselines: For operational tasks like Optimal Power Flow (OPF), a key motivation for adopting a deep learning model is the potential for substantial inference speedup over traditional numerical solvers. However, the paper's efficiency evaluation is confined to comparisons with other learning-based models. It critically omits any performance benchmark against standard, industry-accepted numerical solvers (e.g., the interior-point or simplex methods available in PyPower/MATPOWER). This is a significant flaw, as it fails to demonstrate the practical utility of the proposed method. Without this essential comparison, it is impossible to assess whether OpenGLA provides a meaningful speed advantage that would justify its use over conventional, highly reliable methods. Consequently, the paper's claims regarding computational efficiency and practical viability are not sufficiently substantiated.
6. Limited Scope of Ablation Studies: The paper provides a valuable ablation study demonstrating the effectiveness of the Mixture-of-Transformers (MoT) module in mitigating gradient conflicts. However, the analysis does not extend to other novel components of the OpenGLA architecture. For instance, the adaptive nodal feature encoder (Figure 2B) employs a specific 1D-convolutional design to handle task-varying inputs. It would be insightful to see a comparison against a simpler alternative, such as a standard Multi-Layer Perceptron (MLP), to quantify the actual benefits of this specific design choice. Without a more comprehensive ablation, it is difficult to ascertain whether the complexity of each proposed component is fully justified, or if a simpler design could have yielded comparable performance.

### Questions
1. On Cross-Topology Generalization: Regarding the model's adaptability, the current experiments demonstrate strong generalization to new scenarios within previously seen topologies. To truly substantiate the claims of OpenGLA as a "foundation model," have the authors considered conducting a zero-shot cross-topology generalization test? For example, what is the performance if the model is trained on the IEEE 14, 39, and 57-bus systems and then evaluated directly on the unseen IEEE 118-bus system? Any insight into this more challenging scenario would be highly valuable.
2. On the Necessity of the Mixture-of-Transformers (MoT): The paper convincingly shows that MoT alleviates gradient conflicts compared to a dense model. However, a simpler baseline for multi-modal fusion, such as the one used in LLaVA (projecting visual tokens and concatenating them with language tokens), was not included in this specific comparison. Could the authors comment on why the more complex MoT architecture was chosen over this simpler, widely-used concatenation approach, and what specific advantages it provides in the context of graph-language fusion?
3. On Physical Interpretability: The ABSTRACT rightly emphasizes physical interpretability as a key requirement for power system models, yet the experimental validation focuses on quantitative metrics. Could the authors provide a qualitative case study to shed light on the model's decision-making process? For instance, in a fault localization task, is it possible to visualize the attention weights of the Mixture-of-Transformers to verify that the model correctly focuses on the grid areas physically close to the fault? Such an analysis would be crucial for building trust and ensuring the model's reasoning is physically grounded.

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
4

### Summary
This paper proposes OpenGLA, a graph-language foundation model for power system operations that combines a topology-adaptive GCN encoder with language instructions to handle diverse tasks (OPF, fault detection, state estimation, LMP prediction). The model uses an adaptive Conv-1D nodal feature encoder to handle task-varying inputs, projects graph embeddings into language space, fuses modalities via Mixture-of-Transformers (MoT), and employs a lightweight transformer detokenizer for floating-point outputs. The model is trained on synthetic data covering multiple IEEE test cases and evaluated across different tasks and grid topologies, demonstrating superior performance compared to task-specific baselines.

### Strengths
Domain-Motivated Design: The paper addresses a real need in power systems by designing topology-adaptive components (adaptive Conv-1D encoder + GCN) and a transformer detokenizer for floating-point outputs, which are well-motivated by domain requirements rather than just applying off-the-shelf models.

Evaluation: The experimental evaluation spans multiple tasks (OPF, fault detection, state estimation, LMP prediction) and grid scales (IEEE 14-300 buses, Texas2000),  ablation studies show the efficacy of the MoT design in reducing gradient conflicts.

Task-Specific Performance: OpenGLA achieves relative good performance compared to task-specific baselines across most task-grid pairs, with detailed metrics and statistical reporting.

### Weaknesses
Overstated Foundation Model Claims: The paper claims to present a "foundation model" but provides no evidence of the key properties that define foundation models: (1) no large-scale pretraining on diverse data, (2) no zero-shot or few-shot generalization to new tasks/topologies, (3) training is done separately for each task-grid pair with only 32k samples each, (4) all evaluation is within-distribution on synthetic data. This is essentially a multi-task model for power systems, not a foundation model in the established sense.

Insufficient Literature: There is no related work section. It seems authors have limited understanding of the broader graph learning literature. In the preliminary, the paper positions GCN (2016) as the core GNN method while ignoring the evolution of graph neural networks, spectral methods, heterophilic GNNs, and recent graph foundation models (e.g., GraphGPT, G-Retriever, InstructGLM mentioned in the first paper's review). The comparison is limited to outdated baselines (GAT, GIN from 2017-2018) rather than recent graph transformers or graph-language models.

Scalability: Despite targeting "critical infrastructure," all experiments use synthetic data from simulators (PyPower, PowerWorld). The largest system (Texas2000) shows poor performance on fault localization (Fig. 5), and there's no evaluation on real SCADA/PMU data from actual grids. For a system claiming to support operational decision-making, this is a critical gap affecting credibility and practical applicability.

Limited Architectural Novelty: The core architecture is largely assembled from existing components: GCN (Kipf & Welling 2016), standard projector (from LLaVA), MoT (Liang et al. 2024), and LLaMA backbone. The adaptive Conv-1D encoder is a straightforward application of adaptive pooling. The main contribution is domain-specific engineering rather than fundamental methodological innovation.
Weak Generalization Evidence: The paper provides no evaluation of generalization to: (1) unseen grid topologies, (2) new tasks not in the training set, (3) transfer learning scenarios, or (4) out-of-distribution conditions. Figure 6 shows task embeddings are well-separated after training, but this only demonstrates that the model has learned to distinguish tasks it was trained on—not that it can generalize. The conclusion acknowledges "it remains to explore whether the model performs in a new task or new topology," which should have been addressed in the experiments.

Dataset Scale and Diversity Concerns: Training with only 32k samples per task-grid pair (1.05M total) is modest for a claimed "foundation model." The power-law scaling analysis (Fig. 8) is interesting but the curves haven't saturated, suggesting the model is data-hungry. More critically, all data comes from simulations with synthetic faults and scenarios—the model hasn't been validated on the heterogeneous, noisy, real-world operational data that exists in actual power systems.

Incomplete Ablation Studies: Key design choices lack thorough ablation: (1) Why Conv-1D specifically for the adaptive encoder versus other architectures? (2) How sensitive is performance to the number of GCN layers? (3) What's the contribution of the two-stage training versus end-to-end training? (4) How does performance degrade when certain instructions or graph components are ablated? The only substantive ablation is MoT vs. dense model.

Presentation Issues: The paper has several clarity problems: (1) Figure 2 is overly complex and hard to parse, (2) the relationship between "topology-independent" and "topology-dependent" data (Fig. 9-10) is confusing, (3) critical implementation details are relegated to appendix, (4) the evaluation metrics section (A.6) with its complex reward functions should be in the main paper to understand what "optimality gap" means, (5) inconsistent terminology (e.g., "graph-language answering" vs. "graph-language foundation model").

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes a multi-modal graph–language foundation model for power systems. It integrates a graph encoder, a mixture-of-transformers backbone, and a transformer-based detokenizer to generate both continuous floating-point outputs and language responses. This framework is highly relevant for power-system applications, which are inherently graph-structured and involve diverse downstream tasks. Overall, I consider this work an important and timely contribution to the community.

### Strengths
1. The idea of designing a graph–language foundation model is very suitable for power-system tasks, which have varying graph topologies and require multi-task capability across heterogeneous downstream tasks.

2. The model is well designed to accommodate both language-response tasks and exact-value prediction tasks, reflecting a good understanding of the domain requirements.

3. The experimental study and ablation analyses are comprehensive and demonstrate solid empirical validation.

### Weaknesses
1. The training data include only six fixed topologies (IEEE 14/39/57/118/300 and Texas2000). Thus, the claimed “topology adaptivity” refers to performance consistency across these seen grids, not to unseen networks. Incorporating N-1 topology perturbations, as explored in CANOS, would significantly strengthen the evidence for robustness to topological changes.

2. The detokenizer converts hidden states into continuous values through autoregressive generation, which may slow down inference. Moreover, Figure 11 reports inference time without ensuring a fixed optimality gap threshold across methods, which makes the comparison less fair. It would be more informative to report the time required to reach the same gap.

3. Since the model learns OPF solutions via supervised regression, it does not guarantee physical feasibility. Discussing potential post-processing or feasibility-repair mechanisms would clarify the model’s practical applicability.

### Questions
1. Can you evaluate the inference time at a fixed optimality gap?
2. Can you extend the framework with N-1 contingency or unseen-topology training data to verify robustness to topological perturbations?
3. How does the model handle feasibility violations in predicted OPF or LMP results?
4. Would a parallel decoding or any other methods improve inference time?

### Soundness
3

### Presentation
3

### Contribution
4
