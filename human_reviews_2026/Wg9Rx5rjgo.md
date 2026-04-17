# ASTGI: Adaptive Spatio-Temporal Graph Interactions for Irregular Multivariate Time Series Forecasting

- Decision: Accept (Poster)
- Scores: 6, 4, 2

## Abstract
Irregular multivariate time series (IMTS) are prevalent in critical domains like healthcare and finance, where accurate forecasting is vital for proactive decision-making. However, the asynchronous sampling and irregular intervals inherent to IMTS pose two core challenges for existing methods: (1) how to accurately represent the raw information of irregular time series without introducing data distortion, and (2) how to effectively capture the complex dynamic dependencies between observation points. To address these challenges, we propose the Adaptive Spatio-Temporal Graph Interaction (ASTGI) framework. Specifically, the framework first employs a Spatio-Temporal Point Representation module to encode each discrete observation as a point within a learnable spatio-temporal embedding space. Second, a Neighborhood-Adaptive Graph Construction module adaptively builds a causal graph for each point in the embedding space via nearest neighbor search. Subsequently, a Spatio-Temporal Dynamic Propagation module iteratively updates information on these adaptive causal graphs by generating messages and computing interaction weights based on the relative spatio-temporal positions between points. Finally, a Query Point-based Prediction module generates the final forecast by aggregating neighborhood information for a new query point and performing regression. Extensive experiments on multiple benchmark datasets demonstrate that ASTGI outperforms various state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses two fundamental challenges in Irregular Multivariate Time Series Forecasting (IMTSF): 1) information distortion caused by data preprocessing like interpolation, and 2) the inability of static interaction structures to capture complex, dynamic dependencies. To tackle these issues, the paper proposes a novel framework named ASTGI. The framework preserves raw information by directly embedding each discrete observation into a learnable spatio-temporal space. It then adaptively constructs a causal graph for each point to capture dynamic relationships. Finally, it performs prediction through relation-aware message passing on these graphs. Experimental results on four public datasets show that ASTGI significantly outperforms a wide range of state-of-the-art baselines.

### Strengths
1.	It solves the information distortion problem in irregular time series modeling by directly mapping discrete observations to a learnable continuous space, avoiding any preprocessing.
2.	Its data-driven, adaptive graph construction mechanism discards all prior assumptions about interactions, enabling the model to capture highly complex and dynamic dependencies across time and variables.
3.	It seamlessly unifies the prediction task into its neighborhood aggregation framework, showing a high degree of conceptual consistency and design elegance that many other methods lack.
4.	It consistently outperforms a large number of strong baselines across multiple diverse benchmarks, providing powerful evidence of the framework's superior performance and broad applicability.

### Weaknesses
1.	An analysis of the model's computational complexity and its scalability on larger-scale datasets is not provided, making it difficult to assess its deployment potential in resource-constrained environments.
2.	The motivation for using a unified hyperparameter K across different modules is not sufficiently clarified; setting them separately might offer further optimization for neighborhood aggregation at different stages.
3.	After multi-layer graph propagation, the initial observation $x_i$'s magnitude information could be diluted, affecting final regression accuracy. The paper lacks discussion on mitigating this risk.

### Questions
1.	Regarding the model's computational efficiency, what is the theoretical computational complexity of constructing the adaptive graph for each observation point, and are there practical strategies to optimize this process?
2.	Regarding the hyperparameter K, what was the design consideration for using the same value for both the information propagation and the final prediction stages?
3.	If the K-nearest neighbors in the learned space are all in the future, the causal mask results in zero neighbors. How does the model handle this "neighbor starvation"?
4.	Why use a separate scoring network for the final prediction, instead of reusing the one from the propagation phase? What's the advantage of this design?

### Soundness
3

### Presentation
4

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
The paper proposes ASTGI, a novel framework for Irregular Multivariate Time Series Forecasting (IMTSF), which addresses two core challenges: (1) avoiding information distortion caused by interpolation or alignment, and (2) capturing complex dynamic dependencies between observations. ASTGI introduces a learnable spatio-temporal embedding space where each observation is represented as a point. It then adaptively constructs causal interaction graphs via nearest-neighbor search and performs relation-aware message propagation to model dependencies. A query-based prediction module is used to forecast future values. Extensive experiments on four public datasets show that ASTGI outperforms state-of-the-art baselines, and ablation studies confirm the importance of each component.

### Strengths
1. IMTSF is an interesting topic. The idea of representing IMTS as points in a learnable spatio-temporal space and adaptively constructing causal graphs effectively bridges the gap between raw data representation and dynamic dependency modeling.
2. The methodology is clearly described, and the authors have provided code and detailed experimental setups, which supports reproducibility.
3. The paper includes experiments on four diverse datasets (healthcare, climate, activity) and compares against 12 strong baselines. The results are convincing and statistically sound.

### Weaknesses
1. The connections between the two key challenges are not clearly explained.
2. In the Candidate Neighborhood Identification module, the choice of Euclidean distance in the spatiotemporal coordinate space lacks sufficient justification.
3. The technical contributions of the Neighborhood-adaptive Graph Construction and Spatiotemporal Dynamic Propagation modules are limited.
4. The Related Work section lacks an overarching overview.
5. There is a typo in Equation (8): $a_{q,i}$ should be $s_{q,i}$.

### Questions
1. What is the relationship between raw information representation and dynamic dependency modeling in IMTS?
2. Regarding Candidate Neighborhood Identification:
(1) Why is distance measured in the coordinate space rather than the feature space?
(2) Why is Euclidean distance used instead of Cosine similarity?
(3) Why is normalization not applied? Does the embedding norm influence performance?
(4) Why are points with distant coordinates discarded? Could this remove long-range dependencies?
(5) How is the top-k operation differentiated? Might it suppress useful neighbors during early training stages?
3. What is the novelty of the Spatiotemporal Dynamic Propagation module compared to existing message-passing mechanisms in Graph Neural Networks?
4. Both the raw interaction score in Equation (3) and the message in Equation (4) incorporate the relative positional embedding $p_i - p_j$. How do their roles differ?
5. In Line 322, why does such a separation improve flexibility and accuracy?
6. Can the authors provide visualized analyses of the spatiotemporal points in the embedding space?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes ASTGI, a framework that models irregular multivariate time series by representing each observation as a spatio-temporal point and dynamically constructing adaptive neighborhood graphs for message passing and forecasting. 
The core idea is to treat each timepoint as a node and learn relations based on temporal and channel embeddings, followed by adaptive graph propagation and query-based prediction. Experimental results on several IMTS benchmarks demonstrate improvements over prior methods.

### Strengths
- The overall framework is clearly described and modularized, including point representation, adaptive neighbor discovery, and dynamic propagation.
- The authors conduct thorough ablation studies to justify several architectural design choices.

### Weaknesses
- A central concern is that the conceptual contribution is significantly overlapped with HyperIMTS (Li et al., ICML 2025). HyperIMTS also models each observation timepoint as a node and constructs a relational structure on top of it using temporal and channel information, enabling information aggregation over learned irregular-TS graph topology. Although the exact mechanisms differ (hypergraph vs nearest-neighbor graph), the core idea (treating each observation as a graph node and learning adaptive relational structure in irregular time series) is shared. The authors cite HyperIMTS in Introduction, but do not compare its performance in Experiment, which is problematic for a fair positioning of contributions.
- The paper motivates adaptive graph construction as a key novelty and advantage. However, the experimental section only evaluates forecasting performance and ablations, without providing any qualitative or quantitative analysis of the learned graph structures.

### Questions
Could the authors report training and inference wall-clock time and GPU memory comparison between ASTGI and key baselines (e.g., CRU, Warpformer, tPatchGNN, HyperIMTS if added)? Nearest-neighbor search among N nodes generally incurs at least $O(NK)$, and naive top-K selection may approach $O(N^2)$ if not optimized. For typical IMTS settings with thousands of points per sample, this can become computationally heavy.

### Soundness
2

### Presentation
3

### Contribution
2
