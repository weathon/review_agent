# FlowNet: A Generic Independent and Interactive Model for Streamflow Forecasting

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Streamflow forecasting plays a crucial role in water research for flood prevention, water resource management, or climate resilience. However, it is a challenging task due to complex hydrological system interactions, human interventions and global climate change.  In this paper, we introduce FlowNet, a \emph{unique local global interactive modeling} framework, which is capable of effectively predicting multiple hydrology stations with varied input climate features and data availability at the same time. The key idea of FlowNet is to contruct \emph{independent} prediction models for each station from its local data and from its adjacent neighbors via a hydrological-related directed graph before letting these models to \emph{iteratively} and \emph{interactively} adjust each other to maximize their prediction agreements. This helps to reduce uncertainty, thus improving their accuracy. Additionally, FlowNet dynamically captures inter-station relationships via its directional and delay-aware graph reconstruction method. As a generic framework, FlowNet can be used with any existing Deep Learning (DL) backbone models such as RLinear, PatchTST or iTransformer. However, we also introduce another backbone, called Disentangled Multiscale Cross-attention Transformer (DMCT), to capture the multiscale seasonality-trend information for further performance boost. Extensive experiments on 3 large datasets, including LamaH (with 425 hydrology stations in Europe), CAMELS (672 stations in USA) and  MRB (with 26 gauge stations in the Mekong River Basin), show that FlowNet significantly outperforms 18 state-of-the-art (SOTA) prediction methods in terms of MAE, RMSE, and NSE.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
FlowNet proposes a multi-scale framework for streamflow forecasting. The manuscript proposes to consider the spatial connection when forecasting streamflow. Rather than using a single model for all stations, first, local models are trained to each station separately. This stage is designed to handle varying available local training data. Second, using a predefined graph, the local predictions between the stations are adjusted globally and iteratively via training models. The original graph used in the global stage is refined by measuring correlations to correct the links between the stations. FlowNet is built using the proposed backbone: Disentangled Multiscale Cross-attention Transformer (DMCT), that is designed to capture the multi-scale seasonality-trend information. The manuscript evaluates the proposed framework on 3 benchmarks and compares it to other backbones.

### Strengths
- The framework is clearly described with mathematical formulation. hyperparameters details are described.
- Evaluation on 3 datasets is comprehensive.
- It is promising to consider spatial connections between points for streamflow forecasting.

### Weaknesses
- The method is highly dependent on the graph definition and it does not scale well with graph n x n (L136). In other words, the model can't work on large-scale data and it is limited to small scale dataset e.g., few hundreds of stations.
- What actually effect the streamflow is also meteorological data in the catchment area and near the station. This is ignored by the framework. Furthermore, what is the point of connecting channel like temperature in the streamflow direction? In reality this should be bidirectional.
- Wrong assumption about reality because streamflow cant be obtained in real time and it is used by the framework, since it relies on it heavily.
- Per-station and cross-station models are very expensive to build (L201-202). This is counter intuitive to what we do in ML. For example, "L210-211: we train two sets of cross-station models for each station Si, including the inflow and outflow models", so each station will have 3 models to be trained as I understood. What we do is usually we leverage a single generalized ML model that is trained with a lot of data and let the model learn the correlations by itself rather than building different local models. You might want to look at previous works [[1](https://hess.copernicus.org/articles/28/4187/2024/hess-28-4187-2024.html)].
- It is not clear how the baselines are trained. i.e., the description of the baselines and how they are trained and finetuned is missing.
- State-of-the-art baselines are missing e.g., [[2](https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html), [3](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023WR036170), [4](https://arxiv.org/abs/2505.22535), [5](https://essopenarchive.org/users/810569/articles/1227435-high-resolution-national-scale-water-modeling-is-enhanced-by-multiscale-differentiable-physics-informed-machine-learning), [6](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR035337)]. At least [[2](https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html)] should be included. The baselines in the manuscript should be works for streamflow forecasting following proposed methodologies in previous works rather than different ML building blocks.
- Table 6: looking at the standard deviation and numbers, the improvement compared to some baselines with random seed is negligible. 
- Ablation study, Fig. 3: the improvement is negligible
- Effects of different components of DMCT. The paper has some unjustified claims (L428-431). Some proposed component are redundant e.g., multi-scale (M) in Table 2. In my view, there is also no reason why the model should not work without a specific type of normalization. Most models for streamflow do not necessary use InstanceNorm.

### Questions
- L45-49: this is not true, static features are available globally and are not hard to obtain.
- Line140: where does the streamflow value come from? We can get them from the dataset but in reality, these values are not available in real time.
- How does per-station forecast work? Is it in parallel? It is highly inefficient.
- L204-205: do you finetune the parameters for each station? If yes this is high inefficient, if no this will lead to sup-optimal results. This is why we use one model usually.
- L208: loss function like MSE is not optimal for streamflow i.e., the loss function should consider extremes to account for river flood forecasting.
- I struggle to understand the relation between Eq. 1 and 2? $y_{inflow}$ and $y_{outflow}$ can be zero and the loss will be perfectly fine?
- Missing literature and baselines (see weaknesses).

**Minor**:
- Line740-741: I thought the model should work with an inconsistent spatial representation.
- L166: better to use $\hat{y}$ as ground truth, it is more common to void confusion.
- Eq 7 is incorrect > new variables need to be renamed.
- L345: I see more than 8 baselines. I think you mean 18.

### Soundness
1

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
4

### Summary
This paper proposes FlowNet, the first independent and interactive modeling framework for streamflow prediction. By employing a well-designed local-global interaction scheme and a Disentangled Multiscale Cross-attention Transformer (DMCT), the method achieves advanced performance across three large benchmark datasets, demonstrating its strong potential for real-world hydrological forecasting.

### Strengths
1. The topic of flow forecasting is highly practical and relevant, as it can help mitigate uncertainty caused by climate change.
2. The proposed framework is flexible and can be adapted to various data sources and model types.
3. The experiments are thorough and extensive, convincingly demonstrating the effectiveness of the proposed approach.

### Weaknesses
1. Although FlowNet is novel as a framework, the proposed DMCT appears to mainly reuse existing attention mechanisms with relatively simple adaptation and normalization.
2. The Graph Links Reconstruction Module seems could be simplified by using weighted relationships between stations (e.g., distance-based weights) rather than binary adjacency in the downstream flow graph.
3. The ablation study on the local-global interaction scheme shows limited improvement. For instance, most NSE gains are below 0.01, and FlowNet even underperforms the global-only setting on Phung H. and Ban D. stations. More analysis is needed to explain these results.
4. The design of training multiple independent models for each station increases computational and memory costs, which may limit scalability.

Minor Issue:
- In line 367, the reference to “Figure 1” should be corrected to “Figure 2.”

### Questions
In Section 2.2, the paper defines a global loss to minimize differences between the ground truth $y^i$ and local prediction $\hat{y}^i$, as well as between global prediction $\hat{y}^i_{Global}$ and local prediction $\hat{y}^i$. Why not directly minimize the difference between the global prediction $\hat{y}^i_{Global}$ and the ground truth $y^i$? This seems more straightforward.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
FlowNet is a general framework designed for multivariate spatiotemporal runoff prediction. It introduces a local-global interactive modeling strategy, which allows individual station models to maintain their independence while mutually correcting predictions through iterative integration (Global Consensus). This approach enhances the consistency and robustness of predictions, making it particularly suitable for sets of stations with irregular input features and data lengths. Furthermore, FlowNet incorporates a directional and delay-aware graph reconstruction method to optimize the modeling of spatial relationships. Additionally, it proposes a decoupled multi-scale cross-attention Transformer (DMCT) as a backbone network for efficiently capturing temporal features.

### Strengths
1. FlowNet natively supports the processing of irregular datasets across monitoring stations — for example, when the length of historical records or the dimensionality of input features is inconsistent. This flexibility is of considerable practical importance in hydrological applications. 

2. Comprehensive Baselines and Experiments: The paper employs three challenging large-scale hydrological datasets and conducts comparisons with up to 18 state-of-the-art methods of different architectures (including statistical, MLP, RNN, CNN, Transformer, and GNN approaches), demonstrating extensive empirical advantages.

### Weaknesses
1. **Relevance and Citation of Baseline Selection:** While a substantial number of baseline models are included, some—such as Informer, Autoformer, and FEDformer—are not directly discussed in the Related Work section regarding their application or limitations in hydrological contexts. It is recommended to briefly explain the rationale for including these models in the Related Work or experimental setup, ensuring that all baselines have been referenced in hydrology or time series forecasting.
    
2. **Clarity of the Ablation Study:** The current ablation results are primarily presented in graphical form (e.g., Figure 3). However, due to the relatively small performance differences, visual representations may not precisely reflect the quantitative contribution of each module. To enhance rigor, it is strongly recommended to supplement with detailed ablation performance tables for the core components of FlowNet—local learning, graph reconstruction, and global interaction—across all major datasets (LamaH-CE, CAMELS, MRB). Tabular data would allow clearer and more accurate assessment of the gain from each module.

### Questions
1. **Comparison with Large Time Series Models (TS-LLMs):** Given rapid advances in time series forecasting, large pre-trained models such as TimesFM, TimeGPT, and TabPFN have emerged as new state-of-the-art baselines. Have the authors considered including these as additional and more challenging baselines for comparison? This would further strengthen the demonstrated competitiveness of FlowNet.
    
2. **Computational Complexity and Efficiency:** Although FlowNet employs independent lightweight models, the global interaction phase is iterative. How does FlowNet’s actual overhead in inference time and training time compare to that of single end-to-end GNN models, such as ResGAT or AGCLSTM?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
**Summary**  
This paper proposes **FlowNet**, a flexible and generalizable framework for multivariate spatio-temporal streamflow forecasting across multiple gauge stations. It introduces an **interactive local-global modeling strategy**, where each station is modeled independently in a local phase and then iteratively refined via cross-station interactions in a global phase. A novel **DMCT (Disentangled Multiscale Cross-attention Transformer)** is also proposed as a backbone model to capture multiscale temporal patterns. The method is evaluated on three hydrological datasets and shows superior performance over 18 SOTA baselines.

### Strengths
**Strengths**  
- **Extensive and convincing experiments**: The paper compares FlowNet with a wide range of strong baselines (e.g., Transformers, GNNs, LSTMs) across three datasets (LamaH, CAMELS, Mekong) and multiple horizons. Results consistently show FlowNet outperforms others in NSE, RMSE, and MAE.  
- **Ablation studies and robustness checks**: Ablations validate the contribution of each component (local/global phases, graph reconstruction, DMCT modules), and multiple random seeds are used to ensure stability.  
- **Handles heterogeneous data**: FlowNet accommodates varying input lengths and feature sets across stations, which is a practical advantage in real-world hydrological systems.

### Weaknesses
**Weaknesses**  
- **Poor readability in methodology**: The method description is overly technical and difficult to follow, with inconsistent notation and a lack of high-level intuition. Variable definitions are scattered and not unified, hindering clarity.  
- **Limited novelty in integration**: The local/global phase learning and DMCT module feel like a straightforward combination of existing ideas (interactive learning + multiscale decomposition) rather than a deeply integrated innovation. The novelty appears incremental.  
- **Scalability concerns**: FlowNet requires training separate per-station and cross-station models for each link in the graph. With hundreds or thousands of stations, the computational cost grows sharply, severely limiting its applicability to large-scale river networks. This is only briefly mentioned but not adequately addressed.

### Questions
1.  **Readability:** The methodology is dense and hard to follow due to inconsistent notation. Can the authors provide a clearer, more unified presentation of variables and core concepts?

2.  **Novelty & Integration:** The framework feels like a composition of a local-global scheme and a DMCT backbone. What is the key synergistic novelty beyond this combination?

3.  **Scalability:** The requirement for O(N²) models seems prohibitive for large networks. What is the formal computational complexity, and what strategies are proposed for scaling to real-world, large-scale basins?

### Soundness
2

### Presentation
3

### Contribution
2
