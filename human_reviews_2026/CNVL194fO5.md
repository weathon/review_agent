# Are Global Dependencies Necessary? Scalable Time Series Forecasting via Local Cross-Variate Modeling

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Effectively modeling cross-variate dependencies is a central, yet challenging, task in multivariate time series forecasting.
While attention-based methods have advanced the state-of-the-art by capturing global cross-variate dependencies, their quadratic complexity with respect to the number of variates severely limits their scalability. In this work, we challenge the necessity of global dependency modeling. We posit, through both theoretical analysis and empirical evidence, that modeling local cross-variate interactions is not only sufficient but also more efficient for many dense dependency systems.
Motivated by this core insight, we propose VPNet, a novel architecture that excels in both accuracy and efficiency. VPNet's design is founded on two key principles: a channelized reinterpretation of patch embeddings into a higher-level variate-patch field, and a specialized VarTCNBlock that operates upon it. Specifically, the model first employs a patch-level autoencoder to extract robust local representations. In a pivotal step, these representations are then re-conceptualized as a 2D field constructed over a "variates × patches" grid. The VarTCNBlock then applies depthwise 2D convolutions across this field to efficiently capture local spatio-temporal patterns (i.e., cross-variate and temporal dependencies simultaneously), followed by pointwise convolutions for feature mixing. This design ensures that the computational complexity scales linearly with the number of variates. Finally, variate-wise prediction heads map the refined historical patch representations to future ones, which are decoded back into the time domain. Extensive experiments demonstrate that VPNet not only achieves state-of-the-art performance across multiple benchmarks but also offers significant efficiency gains, establishing it as a superior and scalable solution for high-dimensional forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes VPNet, a multivariate time-series forecasting framework that models localized cross-variate interactions via depthwise 2D convolutions on a variate–patch representation. The authors argue that global dependency modeling is not necessary and introduce the Local Sufficiency Hypothesis as theoretical justification. Experiments on eight public datasets show improvements over recent baselines.

### Strengths
1. The paper provides a clear and well-motivated perspective questioning the necessity of global dependency modeling in multivariate forecasting. By framing the problem through the Local Sufficiency Hypothesis, the work contributes a conceptual shift that has relevance for both theory and practical model design.
2. The proposed Variate–Patch Field representation is intuitive yet effective, allowing variable-wise and temporal patterns to be captured jointly using simple depthwise 2D convolutions. This design reduces computational complexity from quadratic to linear with respect to the number of variables while preserving modeling capacity.
3. The empirical evaluation is thorough, covering diverse datasets with significantly different dimensionalities and dynamics. The model consistently outperforms competitive baselines, particularly in high-dimensional settings where efficiency matters most.
Empirical results are strong and consistent across multiple datasets.

### Weaknesses
1. Insufficient empirical support for the Local Sufficiency Hypothesis.
The central assumption that cross-variate dependencies are predominantly local is not thoroughly validated using real-world correlation or interaction patterns. The justification relies more on theoretical reasoning than on dataset-driven evidence, which weakens the persuasiveness of the hypothesis.
2. Lack of discussion on scenarios where locality may be insufficient.
While the model performs well on the evaluated datasets, it is unclear how it behaves in settings with strong long-range or sparse inter-variable dependencies. Without an analysis of potential failure cases or applicability boundaries, the generality of the proposed approach remains uncertain.
3. Limited interpretability analysis regarding learned dependency structure.
Given that the core contribution concerns how the model captures local cross-variate relationships, the paper would benefit from visualizations or interpretability studies (e.g., patch importance, receptive field inspection). These insights are necessary to substantiate the claims about the model’s learned dependency patterns.

### Questions
1. The Local Sufficiency Hypothesis is motivated theoretically under random or dense-dependency assumptions.  Could the authors provide empirical analyses of variable correlation / Granger causality / covariance decay across datasets to verify that real-world dependencies are indeed predominantly local?
2. Can the authors identify conditions under which VPNet fails or requires adaptation? A discussion on when local modeling is insufficient would help clarify the scope of applicability.
3. Since VPNet essentially enforces locality through convolutional receptive fields, how does it compare to local-window attention or strided/sparse attention variants on the same datasets?
4. The model claims to capture local cross-variate structure. Can the authors provide patch-level saliency / variable importance / receptive field visualization to show what dependencies the model actually learns?
5. The patch autoencoder is claimed to improve robustness to non-stationarity. Do the authors have shifted-train/test evaluations (e.g., seasonal shift, regime shift in traffic/energy data) to demonstrate this effect explicitly?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper challenges the common assumption that multivariate forecasting requires global cross-variate modeling. It proposes the Local Sufficiency Hypothesis—that, in dense dependency regimes, a small local neighborhood across variates is enough for accurate prediction. Building on this, the authors introduce VPNet, which (i) encodes sequences into patch embeddings with a lightweight overcomplete autoencoder, (ii) channelizes them into a 2D variate–patch field, and (iii) processes this field with stacked VarTCNBlocks, yielding linear complexity in the number of variates. Extensive experiments on eight benchmarks report state-of-the-art (SOTA) or competitive results with strong efficiency.

### Strengths
1. The Local Sufficiency Hypothesis is a useful lens to revisit the global-vs-local trade-off, with a simple probabilistic argument that motivates local kernels.
2. The variate–patch field + VarTCNBlocks pipeline is conceptually clean and easy to implement. The channelization step makes the locality idea actionable.
3. The proposed design keeps computation linear in the number of variates while still modeling cross-variate signals—addressing a pain point of global attention models.
4. Evaluation spans 8 common datasets (Weather, Traffic, Electricity, Solar-Energy, ETTh1/2, ETTm1/2) with standardized horizons and protocol; results are competitive or SOTA.

### Weaknesses
1. The theory assumes random variate permutations when quantifying the probability that a local window contains an informative neighbor. Real systems often have structured (and non-exchangeable) dependencies; the proof does not address such structure nor provide guarantees beyond dense regimes.
2. While the paper claims robustness to different orderings in one setting, the practical definition of “local neighbors” depends on variate ordering. More systematic analysis (beyond a small-kernel, shallow-stack case) is needed, including learned or data-driven orderings and lag-aware neighborhoods.
3. The strongest gains appear on high-dimensional datasets (Electricity/Traffic). On lower-dimensional ETT, improvements are smaller and sometimes only “competitive,” suggesting locality helps most in dense regimes; the paper could better quantify when global modeling is still useful.

### Questions
1. Can VPNet **learn** variate orderings or local neighborhoods (e.g., via learned permutations, dynamic kernels, or dilation) rather than relying on fixed layouts?
2. Do the authors have stress tests where inter-series dependencies are sparse/heterophilous or dominated by long-range cross-variate relations?
3. The paper gives a probabilistic guideline for kernel selection; can the authors show empirical calibration of that rule (Eq. 20) on real datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles multivariate forecasting by questioning the need for global cross-variate attention and proposes VPNet, which targets local cross-variate interactions for better scalability. The authors offer theoretical and empirical support that dense dependency systems can be modeled effectively with locality. They also propose a specialized VarTCNBlock—depthwise 2D convolutions to capture joint spatio-temporal (cross-variate + temporal) structure, which haslinear complexity in the number of variates. Experiments across multiple datasets claim state-of-the-art accuracy with notable efficiency gains.

### Strengths
1. The motivation to reduce the complexity of capturing cross-variate dependencies is sound. The authors investigate whether we really need to capture global dependencies considering all variates. 
2. The authors tried to provide some theoretical analysis for this local sufficiency hypothesis, which has some merits. 
3. The efficiency of the proposed method seems good compared with other baselines, even iTransformer (which only considers cross-variate dependencies without explicitly considering time dependencies)

### Weaknesses
1. I feel that the theoretical analysis of the proposed Local Sufficiency Hypothesis is not very convincing. My concern is mainly on the orders of variates. Please see the later question I raise on the "finite local neighborhood".
2. The experiments are not comprehensive enough in my opinion. There are several more comprehensive benchmarks proposed since 2025, e.g., fev-benchmark [1] and Gift-Eval [2]. I would suggest to have some results on those benchmarks and investigate how it compare with other baselines. 

References:

[1] fev-bench: A Realistic Benchmark for Time Series Forecasting

[2] GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation

### Questions
1. The title in the manuscript is different from that of the submission page. I think it should be revised. 
2. I wonder how we define the "appropriately chosen finite local neighborhood" in Local Sufficiency Hypothesis. The variates naturally do not have orders and are permutation-invariant. Although the ablation study provides some results on the effect of variate reordering. I feel that might not be valid on all datasets. Some orders are better than others on some datasets. However, I do not observe some patterns/insights on the results - e.g., which order we should use for which dataset. 
3. I am curious how performance may change if we vary the input length. Because in my view, the variate locality may heavily depend on the input length. For example, if we really have a short input length, it may not be able to capture the dependencies.

### Soundness
2

### Presentation
3

### Contribution
2
