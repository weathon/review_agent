# MoLE-GNN: Parameter-Efficient Fine-Tuning of Graph Neural Networks with Mixture-of-Experts

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Graph Neural Networks (GNNs) are gaining popularity for modeling non-Euclidean data due to their ability to capture local and global structure using message-passing techniques. In real-world scenarios, such as graph classification task, the size of graphs within the same dataset can vary significantly. This warrants an investigation into \emph{depth-sensitivity} of graphs, leading to selection of optimal number of GNN layers according to the size of the graph. Traditional GNNs suffer from a static choice of number of layers for the graphs as it leads to underfitting in the large graphs and overfitting in the smaller ones. Although recent Mixture-of-Experts (MoE) GNN models solve this problem by adaptively selecting depth-sensitive expert networks, they have high computational and memory overhead. To overcome these challenges, we introduce a new hybrid model named MoLE-GNN that combines parameter-efficient adapter modules with GNN experts, supporting dynamic expert assignment with minimal fine-tuning. It drastically minimizes trainable parameters (tunes only 5.1\% of the total parameters) and improves generalization, particularly in low-resource environments. Our extensive experiments across inductive, transductive, and link prediction tasks demonstrate that MoLE-GNN consistently outperforms both full fine-tuning and state-of-the-art PEFT baselines, offering a scalable and effective approach for fine-tuning GNNs on diverse graph topologies. Moreover, MoLE-GNN surpasses existing MoE-based GNNs on inductive and link prediction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MoLE-GNN, a hybrid framework that integrates parameter-efficient fine-tuning (PEFT) with Mixture-of-Experts (MoE) for Graph Neural Networks (GNNs). The motivation stems from the observation that traditional GNNs suffer from depth-sensitivity — fixed-layer architectures lead to underfitting in large graphs and overfitting in small ones. While MoE-based GNNs alleviate this by dynamically routing inputs to specialized experts, they remain computationally heavy. MoLE-GNN overcomes this by embedding lightweight adapter modules within frozen expert GNNs and using a structure-aware gating network for dynamic expert selection. The model tunes only ~5% of parameters, achieving better efficiency and generalization. Extensive experiments on graph classification, node classification, and link prediction tasks show that MoLE-GNN consistently surpasses both full fine-tuning and state-of-the-art PEFT/MoE baselines.

### Strengths
**Innovative Hybrid Design**: 
The combination of MoE and PEFT for GNNs is novel. By freezing backbone experts and adding adapters, the model strikes a compelling balance between adaptability and parameter efficiency.

**Comprehensive Evaluation**: 
The authors evaluate MoLE-GNN across multiple domains (molecular, social, and citation graphs) and tasks (inductive, transductive, link prediction), demonstrating broad applicability and consistent improvements.

**Strong Empirical Results**: 
The model outperforms full fine-tuning and state-of-the-art baselines such as AdapterGNN, GCNconv-Adapter, and Link-MoE by large margins while training only ~5% of parameters, highlighting strong efficiency–performance trade-offs.

**Solid Theoretical Analysis**: 
The inclusion of Lipschitz-bound analysis and graph-dependence proofs adds mathematical rigor, validating the model’s stability and topology-awareness.

### Weaknesses
**Limited Exploration of Pre-training Variability**: 
The experts’ pre-training settings (datasets, architectures, and objectives) are not deeply analyzed. It remains unclear how much diversity among experts contributes to the overall gain.

**Complexity and Interpretability**: 
The dynamic routing and adapter interactions make the system architecturally complex. A clearer ablation isolating the contribution of gating, adapter position, and scaling parameters could improve interpretability.

**Sparse Discussion of Computational Overheads**: 
Although the paper claims parameter efficiency, runtime and inference latency comparisons against MoE baselines are limited. Detailed wall-clock analyses would strengthen the argument for real-world efficiency.

**Limited Theoretical Depth on Routing Stability**: 
While adapters are theoretically analyzed, the gating network’s stability and potential overfitting risks are treated as out-of-scope, leaving a gap in understanding the robustness of expert selection.

### Questions
**Expert Diversity**:
How crucial is heterogeneity among experts (e.g., depth, receptive fields) to MoLE-GNN’s success? Would similar improvements hold if experts share identical pre-training data and architectures?

**Scalability and Deployment**:
Can the model’s structure-aware gating generalize to very large graphs or streaming settings without retraining the router?

**Adapter Placement Strategy**:
How sensitive is the performance to adapter placement (before vs. after message passing)? Could the same principle extend to Transformer-style attention layers in graph transformers?

**Future Integration with Graph Transformers**:
The authors mention future work on incorporating pre-trained graph transformers. How would MoLE-GNN handle attention-based structural representations differently from message-passing GNNs?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel idea of integrating Mixture-of-Experts (MoE) into Parameter-Efficient Fine-Tuning (PEFT) for graph neural networks (GNNs). The concept is interesting and potentially useful for improving efficiency and adaptability in GNN models. However, the manuscript in its current form requires substantial revision. The motivation, contextual framing, and writing style feel outdated and do not align well with the expectations for a 2025-era GNN paper. While the technical core is promising, the paper lacks the necessary refinement and depth to justify acceptance at this stage.

### Strengths
S1. The paper’s logic is generally coherent, and the motivation is clearly articulated.

S2. scaling and fine-tuning GNNs efficiently is relevant and meaningful.

S3. Experimental design appears reasonable, and the results suggest balanced performance across datasets and model sizes.

### Weaknesses
**Concerns**

C1. The introduction frames the work around limitations such as “traditional GNNs suffer from a static choice of number of layers,” and issues related to graph-size heterogeneity. However, these challenges have been extensively explored since 2019, with numerous advances in dynamic depth, heterogeneous GNNs, and adaptive graph architectures. The authors should update the motivation and literature review to better reflect the current state of the field. A 2025 GNN paper better position itself relative to modern paradigms such as graph foundation models, auto-GNN architectures, and scalable graph pretraining frameworks.

C2. The paper uses phrases like “we conduct experiments to validate our findings,” implying discovery of known phenomena—such as the performance gap between shallow and deep GNNs. Yet these issues (e.g., over-smoothing, over-squashing, heterophily) have been well studied for years. The authors are expected to reframe their contribution not as rediscovering these effects but as proposing a new perspective—perhaps showing how the MoE-PEFT mechanism mitigates them efficiently.

C3. From the results, the proposed method seems to achieve a reasonable balance across generalization, model size, and training cost. However, this balance is under-emphasized. The authors can explicitly contrast their approach with (1) conventional GNNs, (2) automated or neural architecture search–based GNNs, (3) graph foundation models, and (4) existing MoE-style GNNs.
A summary table comparing key properties—such as fine-tuning cost, parameter efficiency, scalability, and generalization—would make the contribution clearer and more compelling.

C4. The paper claims that “integrating PEFT into GNNs is challenging since most existing PEFT methods were originally developed for sequence models.” However, this overlooks recent progress. For instance, ICDE 2024 introduced S2PGNN a plug-and-play PEFT framework that selectively freezes and adapts parameters for different graph tasks. The current claim is therefore partially inaccurate and should be revised to reflect existing graph-specific PEFT research.

C5. Besides, S2PGNN has demonstrated better performance on several overlapping graph classification tasks compared to what is reported here. To strengthen the contribution, the authors should conduct or at least discuss a comparison with S2PGNN and highlight where MolE-GNN differs or excels—for example, in its cross-task generalization or balance between efficiency and performance.

### Questions
Better emphasize the contributions of this submission regarding to C3.

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
This study introduces an MoE-based Peft framework for GNN training and inference. One major claim is that this approach addresses the depth-sensitivity issue in traditional finetuning strategies. Extensive experiments have been conducted to demonstrate its effectiveness.

### Strengths
(1) The proposed method is in general straightforward (although I think more explanations about Figure 2b are necessary).

(2) Extensive experiment results on multiple benchmarks demonstrate the effectiveness of the proposed method.

(3) The idea of combining PEFT with MoE GNN is original.

### Weaknesses
Major:
(1) I think overall the paper is well written. However, it would be better if there can be more discussion and results about how the proposed method improves efficiency in the main text as efficiency is the main focus of this method.

(2) Moat selected datasets are not very large which may raise questions about whether the method can still perform well on larger graphs.

(3) More experiments regarding the MLP hyperparameters would provide more insights. I wonder for different graphs/tasks, do we need to design different sizes of MLP to increase the expressiveness of the GNN layer.

Minor:

(1) The table presentation should be improved. It could be a bit challenging to distinguish colors in tables and Figure 2. For Figure 2, I could see from the legend that there is green and pink. But I really could not tell the green or pink from the illustration of those parallel components.

### Questions
(1) Can the authors clarify Figure 2b more? What are the two smaller groups of K experts' illustration for? 

(2) How does the method perform if GNN components are also finetuned (in addition to the MLP layers)?

(3) Would the impact of the proposed method vary across different GNN backbones? For example, GIN already possesses good expressiveness. Will the proposed method further improve it? If so, how many more parameters do we need?

### Soundness
3

### Presentation
1

### Contribution
3
