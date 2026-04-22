# PRISM: A Hierarchical Multiscale Approach for Time Series Forecasting

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 4, 8

## Abstract
Forecasting is critical in areas such as finance, biology, and healthcare. Despite the progress in the field, making accurate forecasts remains challenging because real-world time series contain both global trends, local fine-grained structure, and features on multiple scales in between. Here, we present a new forecasting method, PRISM (Partitioned Representation for hIerarchical Sequence Modeling), that addresses this challenge through a learnable tree-based partitioning of the signal. At the root of the tree, a global representation captures coarse trends in the signal, while recursive splits reveal increasingly localized views of the signal. At each level of the tree, data are projected onto a time-frequency basis (e.g., wavelets or exponential moving averages) to extract scale-specific features, which are then aggregated across the hierarchy. This design allows the model to jointly capture global structure and local dynamics, enabling both reconstruction and forecasting. Experiments across benchmark datasets show that our method outperforms state-of-the-art methods for forecasting and also requires less runtime and memory. Overall, our results demonstrate that hierarchical time-frequency decomposition provides a lightweight and robust framework for forecasting multivariate time series.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents PRISM, a novel hierarchical multiscale model for time-series forecasting. PRISM constructs a binary tree over time while performing frequency decomposition (using wavelets or similar filters) at each node. It jointly learns temporal and frequency hierarchies and uses learnable importance scores to weight different frequency bands. By optimizing both forecasting and reconstruction losses, PRISM achieves strong interpretability, robustness, and state-of-the-art performance on multiple benchmarks.

### Strengths
1. The learnable importance scores provide insights into which frequency components drive predictions, adding transparency to the model’s behavior.
2. The paper is well-structured and easy to follow.

### Weaknesses
1. The binary partitioning strategy is manually defined rather than data-adaptive; this might limit flexibility for non-stationary or irregularly sampled signals.
2. **Dependence on pre-defined transforms**. The method relies on fixed wavelet or FFT bases. Learned or adaptive frequency decompositions could potentially capture more expressive features.
3. The benchmarks are still limited to standard ones. More various or big datasets should be put in place to demonstrate the effectiveness of the proposed method.
4. While the model provides interpretable frequency weights, the paper lacks formal evaluation or case studies illustrating interpretability in applications.

### Questions
1. How sensitive is the model to the choice of frequency basis (e.g., wavelet type)?
2. Could the tree depth be learned dynamically based on data complexity rather than fixed by design?
3. How does PRISM perform on highly irregular, non-periodic time series (e.g., event-driven or sparse data)?

### Soundness
3

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
4

### Summary
The paper proposes PRISM, a time-series forecaster that builds a unified time–frequency hierarchy. Concretely: the model (i) performs binary time partitioning with overlap to form a tree, (ii) applies a time–frequency decomposition (default: Haar DWT) at each node, (iii) computes band importance weights via summary statistics → 2-layer MLP → softmax, and (iv) optimizes a joint loss that couples forecasting (future) with reconstruction (past).
Experiments on 8 datasets × 4 horizons report SOTA or competitive results (best MSE in 17/32 settings; best MAE in 18/32), with extensive ablations showing the contribution of the tree encoder, wavelets, importance MLP, reconstruction loss, and residual connections. The motivation is that real-world series exhibit multi-scale behavior (global trends, local fluctuations, and intermediate scales), so hierarchical representations should align long-term structure and fine-scale variability.

### Strengths
* Clear problem framing and gap statement. The paper argues that prior work typically builds hierarchy in only time or only frequency, or mixes domains without a reconstructable shared hierarchy.
* Coherent architecture. Overlapped binary splits (time) + band partition (frequency) + learnable band routing + an explicit reconstruction path form a consistent design.
* Broad empirical coverage and ablations. Results across 32 settings, with component-wise ablations showing 5–14% average performance drops when removing key pieces (tree depth, wavelets, importance MLP, reconstruction loss, residual connections).
* Efficiency and interpretability. Training-time comparisons (e.g., ETTh1–96: 10 epochs in 65s) and band-importance visualizations support practical utility and model introspection.

### Weaknesses
* (Primary) Limited conceptual novelty relative to recent multiscale “decompose–mix” lines. The high-level philosophy—multiscale decomposition and mixing—strongly overlaps with recent TimeMixer-style approaches. The paper does cite such work in Related Work (e.g., Ref. [20]), but the manuscript does not clearly establish a qualitative leap beyond “engineered combination” of known ideas (time hierarchy + frequency filters + learned weighting + auxiliary reconstruction).
Claimed distinction is a reconstructable, shared time–frequency tree, yet the empirical section lacks head-to-head, controlled comparisons designed to isolate scenarios where this specific design strictly dominates competing multiscale methods.
* Wavelet superiority appears under-analyzed. Table-2 shows average gains over FFT/EMA/DoG/MCD, but there is no conditioned analysis clarifying when wavelets lose/win based on spectral characteristics, periodicity, or multivariate correlations.
* Robustness in realistic settings is thin. The paper focuses on public benchmarks; it lacks systematic tests under missing values, anomalies, distribution shift/drift, or longer non-stationary horizons.
* Hyperparameter sensitivity is under-reported. No systematic sweep over overlap (o), tree depth, or number of bands (K) to reveal accuracy/time/memory trade-offs and boundary effects of the cross-fade concatenation.

### Questions
1. Differentiate from TimeMixer-style work with controlled, apples-to-apples tests. Under identical pipelines/tuning/resources, can you provide direct, multi-dataset head-to-head results and analyses showing where and why the reconstructable time–frequency tree and band-importance routing deliver significant, consistent gains? Please include long-context, non-stationary, and low-resource regimes.
2. When are wavelets the right choice? Offer a data-property ↔ filter mapping (e.g., by periodicity, noise spectrum, cross-channel dependencies). If possible, include learned basis experiments to test whether fixed Haar is limiting or optimal across conditions.
3. Hyperparameter sensitivity. Provide thorough sweeps for overlap (o), tree depth, and band count (K), reporting accuracy/time/memory and any bleeding/edge artifacts due to overlap and cross-fade.
4. Role of the reconstruction loss. Visualize the bias–variance trade-off between reconstruction and forecasting (e.g., gradients/importance by band as the reconstruction-loss weight varies). Do larger reconstruction weights ever suppress informative high-frequency bands?
5. Robustness. Add evaluations for missingness, anomalies, covariate shift, and long-term drift, comparing to strong linear/transformer/multiscale baselines.

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
3

### Summary
This paper makes progress on unifying forecasting architectures using temporal hierarchies with methods using frequency modeling. The architecture splits the time series into temporal segments as a binary tree and applies a frequency filtering step at each hierarchy, along with residual connections at the frequency filtering steps. The final derived representations at the base of the binary tree are merged together using learned weights and a FFN is applied to generate the final predictions. The representations learn better due to the auxiliary reconstruction loss.

Experimental results on popular benchmark datasets show improved performance compared to most baselines except for the D-PAD baseline which performs closely to the proposed method.

### Strengths
- The paper presents an interesting technique to enhance hierarchical temporal architectures with frequency filtering. Frequency filtering seems to be an essential component of time series forecasting helping identify cyclical patterns in the dataset.
- Experimental results show that the method performs strongly compared to compared to most baselines, and performs closely to D-PAD which is a more complicated architecture.
- The paper is well presented and the ideas are quite clear. The experimental results are strengthened by the ablation study.

### Weaknesses
- The main weakness of the paper is that the method performs closely to D-PAD and isn't a significant improvement compared to D-PAD, however D-PAD is a much involved architecture compared to the proposed method.
- While, the paper evaluates on the most popular univariate datasets, some more complex datasets could be a newer addition such as M4 and wikipedia. While they are multi-variate datasets, they could help prevent overfitting of techniques on the existing datasets.

### Questions
- Why are the residual connections only in the frequency filters module? Are they not useful in the other layers?

### Soundness
4

### Presentation
4

### Contribution
3
