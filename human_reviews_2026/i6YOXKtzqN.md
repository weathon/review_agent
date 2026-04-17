# Decoupled Diffusion Models for Efficient Spatio-Temporal Graph Forecasting

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Graph-based diffusion models suffer from a critical computational bottleneck, limiting their use in practical applications such as spatio-temporal graph forecasting. We argue that this inefficiency stems from the fusion of information propagation and feature transformation within standard GNNs. In this paper, we introduce a design principle that decouples these two operations, enabling a highly efficient and linear architecture. Instantiating this principle, Decoupled Spatio-Temporal Diffusion Model (DSTD) leverages the principle alongside a dynamic multi-scale aggregation mechanism to achieve remarkable performance. On widely-used spatio-temporal graph forecasting benchmarks, DSTD not only outperforms existing probabilistic methods but also surpasses top-performing deterministic models, while demonstrating a significant reduction in inference time. Our results validate that decoupling is a powerful and effective strategy for building scalable and high-performing generative models for graph-structured data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a decoupled spatio-temporal diffusion model (DSTD) for probabilistic traffic forecasting. The key idea is to decouple the propagation and transformation steps in GNNs to reduce computational cost during iterative diffusion denoising. Experiments on two benchmark datasets demonstrate competitive accuracy and significantly improved efficiency over existing diffusion-based baselines.

### Strengths
The proposed decoupling design is conceptually simple yet effectively improves inference efficiency.

### Weaknesses
- The evaluation is limited to only two medium-scale datasets and lacks validation on large-scale benchmarks such as [1]. Given that the claimed contribution centers on efficiency and scalability, experiments on larger datasets are necessary to substantiate the generalizability of the approach.

[1] Largest: A benchmark dataset for large-scale traffic forecasting. 2023.

### Questions
- In Table 4, the “–Decoupling” variant achieves a lower CRPS (0.152) than the full model (0.165), although slightly higher MAE/RMSE. Yet, the text in Sec. 4.4 claims that removing decoupling leads to a degradation “in predictive accuracy and efficiency.” Could the authors clarify how they interpret this apparent inconsistency between deterministic and probabilistic metrics? Does the decoupled design trade off distributional sharpness for point accuracy?
- The proposed decoupling (Eqs. 11–12) separates propagation and transformation, using a fixed normalized adjacency $A$ for parameter-free message passing and a 1×1 conv for feature projection, while intentionally removing nonlinear activations. From a GNN perspective, this formulation is equivalent to stacking linear graph filters $A^L W$, which are known to be limited in expressive power and unable to model high-frequency components or distinguish non-isomorphic substructures. Could the authors clarify whether this purely linear design sacrifices representational capacity compared with standard GCNs?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces DSTD (Decoupled Spatio-Temporal Diffusion Model), a diffusion-based framework for efficient spatio-temporal graph forecasting. The key idea is to decouple information propagation and feature transformation within graph neural networks, addressing the computational bottleneck that limits existing diffusion-based graph models. The propagation is implemented by a non-parametric GCN and the transformation is implemented by a 1-D convolution. DSTD further employs a dynamic multi-scale aggregation mechanism to adaptively combine information from different receptive fields and incorporates position-aware embeddings to encode node-specific biases.

### Strengths
(1) The studied problem is important and the task is well-motivated. As the author mentioned, typical graph-based diffusion models would require many steps of graph propagation and feature transformation, which can be time-consuming. In that sense, studying how to speed up such a graph-based diffusion model would be of great importance.

(2) The proposed model is technically sound, though with incremental architecture change compared with previous ones.

### Weaknesses
(1) The main idea is  ``Efficient'' graph-based diffusion model, why not analysis the complexity of different method in the main paper?

(2) Following the first points, in the experiment part, the author mainly compared the inference speed of different method, how about the training, and how about the parameter of different models?

(3) As for the experiement results, why the proposed model outperforms deterministic models in MAE and RMSE. It would be better to have deeper analysis on why, rather than only write "The ability to outperform specialized deterministic models on several key metrics, while also providing rich, uncertain forecasts, underscores the significant advantages of the proposed approach." 
The proposed model use the diffusion loss, which aims to learn the underlying distribution (via minimizing VLB) rather than directly minimizing RMSE in deterministic models. In that sense, how can the proposed model outperform deterministic models?

### Questions
please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a decoupled spatio-temporal diffusion model (DSTD) that separates information propagation and feature transformation in GNN-based denoising networks to improve efficiency. While experiments show speedup and good accuracy on METR-LA and PEMS-BAY.

### Strengths
1. Clear motivation and solid writing.
2. Efficiency improvement is practically meaningful.

### Weaknesses
1.  The core idea of decoupling propagation and transformation has been explored in prior works, the novelty is limited to applying it to diffusion models.
2.  No comparison with other decoupled GNNs to show whether the proposed design offers superior representational quality.
3.  The architecture is quite standard, without new diffusion mechanisms or theoretical analysis.

### Questions
1. How would DSTD perform compared to using existing simplified GNNs (e.g., LightGCN) as the denoising backbone?
2. The ablation study (Table 4) shows that removing decoupling results in only a minor performance drop, whereas removing linearization or dynamic aggregation leads to much larger degradation. This seems to suggest that the main accuracy gains stem from these architectural components rather than from the decoupling principle itself. Could the authors clarify how these results support the claim that “decoupling propagation and transformation” is the key contribution?
3. The proposed framework relies on a fixed and static adjacency matrix to capture spatial dependencies. However, in real-world spatio-temporal systems such as traffic networks, spatial relations are often highly dynamic — for example, sensors may fail, new roads may appear, or traffic patterns may shift over time. How would the proposed model handle such structural changes?

### Soundness
3

### Presentation
3

### Contribution
2
