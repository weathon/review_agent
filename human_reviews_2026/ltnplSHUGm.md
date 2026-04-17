# TEPO: A Transferable EDA Prediction Optimization Method Based on Learngene Characterization

- Decision: Reject
- Scores: 4, 4, 0, 4

## Abstract
This paper introduces TEPO, a novel multi-task learning framework to optimize Electronic Design Automation (EDA) in integrated circuit (IC) design by addressing increasing complexity and the limitations of traditional independent design task approaches. TEPO systematically decomposes design knowledge into gene knowledge and class knowledge, which are referred to as Learngenes. This framework employs a dual-pathway architecture with an adaptive gating mechanism, allowing for fine-grained control over knowledge activation and enhancing computational efficiency and interpretability. In the data input section,  the VIT-GNN fusion processor, which integrates Vision Transformer (ViT) features from layout images with Graph Neural Network (GNN) features from circuit topology, spatially aligning them onto a unified 256x256 grid to preserve both global visual patterns and local structural relationships. Our approach tackles four critical challenges in EDA: knowledge fragmentation, feature integration, transferability and data scarcity. The methodology involves pre-training an upstream model to extract Learngene, which is then used to initialize a downstream 12-layer Transformer model for various prediction tasks. Experiments are conducted on CircuitNet-N28, a dataset providing multi-modal features for Congestion, DRC violations, and IR-drop prediction tasks, as well as a new thermal prediction task. The transferability of learning genes not only performs well in existing categories but also shows a faster convergence speed in new task categories. The data required for its training is also less, it saves more computing costs while achieving the same performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes TEPO, a transferable framework for EDA prediction. TEPO factorizes each layer’s weights into task-agnostic “gene knowledge” and task-specific “class knowledge” (via an SVD-style decomposition) and combines them through dual-path gating during transfer. On the feature side, layout images (ViT) and netlist/topology features (GNN) are spatially aligned and fed to a Transformer. Experiments cover standard backend targets (congestion/DRC/IR-drop) and include an additional Thermal task aimed at testing cross-category transfer. The manuscript reports faster convergence and lower final errors than Xavier initialization and several EDA baselines.

### Strengths
- Clear modular design. The weight factorization plus gating is conceptually coherent; the multi-modal alignment (layout + netlist) is a sensible engineering pipeline.
- Transfer signal. Results indicate faster convergence and improved endpoints over vanilla initialization across multiple tasks, suggesting value as a pretraining/initialization strategy.
- Practical motivation. Addressing multi-task transfer in data-constrained EDA settings is relevant to downstream design flows.

### Weaknesses
- Missing efficiency and reproducibility details. Training time, inference runtime/latency, computing resources (hardware, memory), and model size/compute (parameters/FLOPs) are not reported, limiting assessment of the claimed efficiency and hindering reproduction.
- Limited statistical reporting. The manuscript does not provide evaluations over multiple random seeds or dispersion measures (e.g., mean ± std, confidence intervals) and does not report significance testing, weakening the strength of the empirical claims.
- Thermal label transparency. The Thermal ground truth appears to be taken directly from the referenced dataset (so transparency would largely inherit from the dataset), yet the paper does not explain how those labels are defined or verified within the dataset context; a short clarification would help readers assess fidelity and reproducibility.
- Task coverage gap (timing). Timing prediction (e.g., slack/WNS/TNS) is a key objective in EDA because it directly guides placement optimization and impacts final performance and PPA quality. Recent timing-prediction frameworks such as E2ESlack [1] and PreRoutGNN [2] explicitly address pre-routing timing prediction using global pretraining and local delay learning. Moreover, cross-stage optimization work such as LaMPlace [3] highlights that improving timing correlation during placement leads to better downstream metrics, underscoring the importance of evaluating timing transfer.

### Questions
1) Gene/Class ranks and gating. How are the per-layer ranks for “gene” vs. “class” chosen (fixed ratios or data-driven), at what granularity is gating applied (per-layer, per-head, or per-channel), and what sensitivity is observed on convergence, final accuracy, and compute/latency?  
2) Thermal ground truth. If Thermal labels are inherited from the dataset, how are those labels defined (solver, boundary conditions, grid/resolution, dataset QA) and how is leakage between training and test designs or stages avoided?  
3) Timing prediction. Since timing is one of the most critical tasks in EDA and directly guides placement for better downstream performance, why is timing not included among the evaluated tasks, and how would TEPO be expected to transfer to timing (e.g., pre-routing slack/WNS/TNS) compared with baselines such as E2ESlack [1], PreRoutGNN [2]?

[1] Bodhe, S., Zhang, Z., Hamidizadeh, A., Kai, S., Zhang, Y., & Yuan, M. (2025). E2ESlack: An End-to-End Graph-Based Framework for Pre-Routing Slack Prediction. arXiv preprint arXiv:2501.07564.  

[2] Zhong, R., Ye, J., Tang, Z., Kai, S., Yuan, M., Hao, J., & Yan, J. (2024, March). Preroutgnn for timing prediction with order preserving partition: Global circuit pre-training, local delay learning and attentional cell modeling. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 38, No. 15, pp. 17087-17095).

[3] Geng, Z., Wang, J., Liu, Z., Xu, S., Tang, Z., Kai, S., ... & Wu, F. LaMPlace: Learning to optimize cross-stage metrics in macro placement. In The Thirteenth International Conference on Learning Representations.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes TEPO, a transferable multi-task framework for EDA prediction. TEPO fuses ViT features from layout images with GNN features from circuit topology by spatially aligning both on a 256×256 grid, and  decomposes model weights into “gene” (shared) vs “class” (task-specific) components with adaptive gating to route knowledge per task. The system is pre-trained on congestion, DRC, and IR-drop and then transferred to downstream tasks, including a new thermal prediction task. Experiments on CircuitNet-N28 claim faster convergence, better data efficiency, and improved accuracy over random (Xavier) initialization and several SOTA baselines.

### Strengths
1. There is Clear motivation for early, transferable prediction across EDA tasks.
2. The article has few writing errors and the charts are clear in meaning.

### Weaknesses
1. TEPO underperforms RouteNet on IR-drop in Table 3 (0.027 vs 0.014), so “surpasses existing EDA models” needs qualification by task.
2. CircuitNet-N28 provides a large amount of data on different designs. Only 100 designs were used for training and 20 designs for testing, which raises concerns about the effectiveness and scalability.
3. The selected SOTA is not the current or recent single-task SOTA.
4. This article lacks ablation experiments to prove the effectiveness of each part of the design.
5. This multi-task learning method should have a large number of gradient conflicts among different tasks, but no solution to the gradient conflicts has been seen. So, I am skeptical that the experimental results are better than those of models specifically designed for a single task.

### Questions
1.CircuitNet-N28, this dataset has no thermal ground truths. How did you obtain the labels for the experiment? Any validation vs. physics-based solvers?
2.What is the exact feature difference loss between ViT patches and GNN nodes? Any alternatives tested (e.g., cross-attention alignment)?
3.Table 2 lacks 30-sample results, is not matched with the description before,
4. Please provide the mathematical form of σ_gene, σ_class, and g_task, and an ablation bewteen gating and simple additive mixing.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This work proposed a transferable EDA prediction optimization method by Learngene, which is a weights initialization method by leveraging singular value decomposition (SVD) on the pre-trained weights. By this initialization method, this work performs better than random (Xavier) initialization. Besides, this work used VIT and GNN for multi-modal fusion to improve quality.

### Strengths
1. This work first leveraged Learngene in the EDA prediction task.
2. The performance of this work is better than random initialization.

### Weaknesses
1. The experiments of this work are too weak to verify the transferability of their method. As the title of this paper suggests, this work targets the transferable learning problem in the EDA domain. However, the authors only compare their method with random (Xavier) initialization.
1. The paper lacks a detailed explanation of data distribution, especially since the authors only select 120 samples (100 for training, 20 for testing) from a large-scale CircuitNet-N28 dataset (over 10K samples). Meanwhile, I think experimenting on the whole CircuitNet dataset will be more convincing.
1. The experiment settings are wrong for the thermal prediction task. As shown in Tables 1/2/3, I can not understand why the value of temperature is used to evaluate the model performance. In my opinion, the authors should use MSE, which is used for Congestion, DRC, and IR-drop predictions. Meanwhile, the authors didn't mention how to generate the thermal data in the experimental section.

### Questions
1. The title of this paper consists of "prediction" and "optimization". I think the paper only propose prediction method without optimization?
2. The pre-training loss contains the losses of downstream tasks. Therefore, I think is a multi-stage training strategy rather than transfer learning.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes TEPO, a transfer learning framework for chip design prediction tasks. It combines a GNN (circuit topology) and ViT (layout image) by aligning features on a shared 256×256 grid. It then decomposes model weights via SVD into “gene” (universal) and “class” (task-specific) components (“Learngene”). These components are used to initialize a downstream transformer for new tasks. TEPO is evaluated on CircuitNet-N28 for congestion, DRC, IR-drop, and a new thermal task, and claims better accuracy, faster convergence, and better data efficiency than Xavier initialization and several existing EDA models.

### Strengths
* The SVD-based split of weights into transferable “gene” vs task-specific “class” knowledge is an interesting, explicit formulation of reusable IC design knowledge.
* The multimodal fusion of graph (netlist/topology) and layout image features is reasonable for EDA and is implemented end-to-end.
* TEPO shows faster convergence and better sample efficiency, including on a task (thermal) not seen during pretraining.
* Tables and curves suggest TEPO beats Xavier init and standard GNN/EDA baselines (GCN, GAT, RouteNet, etc.) in final accuracy and convergence speed.

### Weaknesses
1. **Learngene identifiability is under-justified.**
   The paper assumes “top 512 singular directions = universal” and “bottom 256 = task-specific,” but gives no theory or ablation to prove that split is actually semantic, unique, or robust.

2. **Gating is under-specified.**
   The paper refers to a gating mechanism that turns gene/class knowledge on or off per task, but never gives the actual gating function, loss, or how leakage between the two is prevented.

3. **Limited baselines and no statistics.**
   TEPO is only compared against Xavier for transfer initialization, and does not compare to other transfer / flow-tuning / knowledge-transfer frameworks in EDA. All results are single numbers: no std devs, no significance, no repeated trials.

4. **Single dataset, tiny data regime, possible leakage.**
   All results are on CircuitNet-N28 with ~100 train / 20 test designs. There’s no cross-validation, no discussion of overfitting control, and no evaluation on a different process node or dataset. This weakens generalization claims.

5. **Reproducibility gaps.**
   Key training details (optimizer, LR, batch size, epochs, early stopping, seeding) are missing. The fusion step (“Flatten by Position” putting GNN node features onto a 256×256 grid to align with ViT output) is described conceptually, but collision / aggregation rules are not specified, so it’s not fully reproducible.

6. **Missing related work positioning.**
   The paper does not seriously compare or contrast with recent multimodal / transfer / foundation-style EDA models (e.g., NetTAG, DeepGate4, FlowTuner, etc.), which weakens novelty framing.

**Key questions for authors**

* Show ablations: why 512/256? What happens if you change the split?
* Give the exact gating equation and training procedure.
* How did you prevent overfitting with only ~100 designs?
* How exactly are GNN features mapped onto the 256×256 layout grid when multiple nodes land in the same cell?
* Do results transfer to any dataset or node other than N28?
* Please report variance / error bars and full hyperparameters.

### Questions
refer weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
