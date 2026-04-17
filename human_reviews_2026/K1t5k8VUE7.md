# HTMformer: Hybrid Time and Multivariate Transformer for Time Series Forecasting

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4

## Abstract
Transformer-based methods have achieved impressive results in time series forecasting. 
However, existing Transformers still exhibit limitations in sequence modeling as they tend to overemphasize temporal dependencies. This incurs additional computational overhead without yielding corresponding performance gains. 
We find that the performance of Transformers is highly dependent on the embedding method used to learn effective representations.
To address this issue, we extract multivariate features to augment the effective information captured in the embedding layer, yielding multidimensional embeddings that convey richer and more meaningful sequence representations. 
These representations enable Transformer-based forecasters to better understand the series.
Specifically, we introduce Hybrid Temporal and Multivariate Embeddings (HTME). 
The HTME extractor integrates a lightweight temporal feature extraction module with a carefully designed multivariate feature extraction module to provide complementary features, thereby achieving a balance between model complexity and performance.
By combining HTME with the Transformer architecture, we present HTMformer, leveraging the enhanced feature extraction capability of the HTME extractor to build a lightweight forecaster.
Experiments conducted on eight real-world datasets demonstrate that our approach outperforms existing baselines in both accuracy and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces HTMformer, a model that integrates a custom-designed Hybrid Temporal and Multivariate Embedding (HTME) module. HTME integrates a temporal feature extraction module with a multivariate feature extraction module to provide complementary features.

### Strengths
1. The method is very simple but effective. Different from works focusing on model architecture design, this paper focuses on the embedding modeling.

### Weaknesses
1. The integration of temporal and multivariate features represents a well-trodden path in multivariate time series forecasting. The work presented in this paper does not offer distinctly new insights or conceptual advancements beyond this established paradigm, resulting in a lack of compelling novelty.

2. The HTME module is positioned as the core contribution of this work. However, its architectural design appears to be a straightforward composition of existing feature extraction techniques. The paper would benefit from a clearer elaboration of the core design principle or the key innovation that distinguishes HTME from a mere combination of established components.

3. In Table 1, why not apply HTME to models like PatchTST to observe its performance improvement? Baselines like Informer and Reformer are too old.

4. A high-quality embedding strategy should possess general applicability across different model architectures. Is the proposed HTME compatible with non-Transformer models, such as DLinear?

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
The manuscript proposes HTME, a feature-extraction and embedding-generation module, and builds HTMformer on top of it to model dependencies across variable dimensions of multivariate time-series embeddings using a Transformer encoder.

### Strengths
The proposed HTMformer is evaluated on standard benchmark datasets and reports promising accuracy compared with recent state-of-the-art baselines.

The HTME extractor is designed to capture local temporal dependencies and produce a fused representation (temporal + spatial) that serves as input to the Transformer encoder.

The overall architecture is encoder-only and relatively lightweight, which is attractive for long-horizon forecasting settings.

### Weaknesses
The description of HTME is not sufficiently detailed. A step-by-step explanation of the computations is needed to make the core contribution reproducible.

The rationale for fusing temporal and spatial representations before the Transformer is unclear. It is not evident why summation is the best fusion operator, nor how it affects the disentangling of temporal vs. variable-wise patterns.

The use of GRU inside HTME is under-motivated, especially given that the main model is Transformer-based; it is not explained why a Transformer (or even a lightweight temporal attention) is not used in this stage.

The current HTME illustration (Figure 3) does not fully illustrate the computation described in the text and should be revised.

### Questions
(a) HTMformer is missing from Figure 1.

(b) Provide explicit dimensions for all components in Equations (2), (3), and (4). This is the core part of the contribution. Also, clarify why an additional linear layer is used to project from $dim$ to $D$ in Eq. (4) instead of directly projecting the flattened vectors to $D$.

(c) Explain the motivation for summing $D_{\text{out}}$ and $V_{\text{out}}$. Does this operation beneficially entangle temporal and spatial features, or would a decoupled design (two Transformers, one temporal and one variable-wise, followed by fusion) be more appropriate?

(d) Please add an ablation that replaces the GRU in Eq. (6) with a Transformer-based alternative to verify that GRU is indeed the better choice.

(e) Section 3.3 can be made more concise, since it mainly reuses existing models and may distract from the contributions (HTME).

(f) The paper states that HTMformer changes the complexity from $O(L^2)$ to $O(N^2)$. Please discuss the case where $N$ is greater than $L$ and whether the proposed formulation still offers computational advantages.

(g) In Figure 4, replace training time with FLOPs to make comparisons hardware-agnostic.

### Soundness
3

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
3

### Summary
The paper proposes HTMformer, a novel time series forecasting framework using a Hybrid Temporal and Multivariate Embedding (HTME) module. HTME disentangles temporal and multivariate features by separately extracting (1) temporal patterns via patch-based convolution and linear projection, and (2) multivariate dependencies through a GRU-based pipeline (patch->linear->GRU->convolution), which are then fused with learnable weights. Integrated with an "inverted input" Transformer encoder, HTMformer achieves state-of-the-art performance on 8 benchmark datasets (e.g., Traffic, Solar-Energy) for both short- and long-term forecasting, while reducing training time and GPU memory usage. Ablation studies confirm the superiority of the hybrid design over single-feature baselines, and the method generalizes well across Transformer variants.

### Strengths
1.	The proposed HTME module separates temporal and multivariate feature extraction while maintaining their complementarity. This design not only draws on PatchTST's focus on local temporal patterns but also reflect iTransformer's emphasis on the relationships between variables.
2.	Experiments across eight widely used benchmarks (Weather, Traffic, Electricity, ETTh2, Solar, and multiple PEMS datasets), including comparisons, ablation studies, and evaluations of HTME integrated into various Transformer variants. This demonstrates the method’s stability and scalability, consistently outperforming baseline models.
3.	HTMformer achieves 2~3× faster training speed and uses only 20%~45% of the GPU memory compared to other methods.

### Weaknesses
1.	The proposed HTMformer is built on the inverted architecture of iTransformer, and does not propose a new backbone structure or attention mechanism, which lacks paradigm-level innovation.
2.	While the paper empirically shows that the multivariate-only variant (HTMformerV2) performs poorly on temporally dominant datasets (e.g., Traffic, ETTh2) but better on highly correlated ones (e.g., PEMS), it provides no causal explanation for when and why each module is effective. The authors do not link dataset characteristics (e.g., variable number, forecastability, periodicity) to module performance.
3.	The author overemphasizes the role of multivariate and weakens the foundation of time modeling. However, experiments (Table 4) show that time dependence is still the main reason for performance, but the paper narrative can be misleading.
4.	There are no comparative experiments to show why early decoupling time and multivariate is a better inductive bias. The author also did not point out the rationality of early decoupling from the perspective of decoupling theory or information bottlenecks.

### Questions
Could the authors provide a more principled or analytical justification for the HTME design? For example, could a simple mathematical formulation (e.g., between the embedding structures of HTME and iTransformer’s MLP embedding) help illustrate how HTME better preserves cross-variable temporal dynamics? Offering even a high-level analytical perspective would clarify the conceptual motivation behind HTME and help distinguish it from prior component-wise engineering improvements.

### Soundness
3

### Presentation
2

### Contribution
2
