# Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 4

## Abstract
Time series forecasting has attracted significant attention in the field of AI. Previous works have revealed that the Channel-Independent (CI) strategy improves forecasting performance by modeling each channel individually, but it often suffers from poor generalization and overlooks meaningful inter-channel interactions. Conversely, Channel-Dependent (CD) strategies aggregate all channels, which may introduce irrelevant information and lead to oversmoothing. Despite recent progress, few existing methods offer the flexibility to adaptively balance CI and CD strategies in response to varying channel dependencies. To address this, we propose a generic plugin xCPD, that can adaptively model the channel-patch dependencies from the perspective of graph spectral decomposition. Specifically, xCPD first projects multivariate signals into the frequency domain using a shared graph Fourier basis, and groups patches into low-, mid-, and high-frequency bands based on their spectral energy responses. xCPD then applies a channel-adaptive routing mechanism that dynamically adjusts the degree of inter-channel interaction for each patch, enabling selective activation of frequency-specific experts. This facilitates fine-grained input-aware modeling of smooth trends, local fluctuations, and abrupt transitions. xCPD can be seamlessly integrated on top of existing CI and CD forecasting models, consistently enhancing both accuracy and generalization across benchmarks. The code is available [https://github.com/Clearloveyuan/xCPD].

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes xCPD, a lightweight plugin module designed to enhance multivariate time series forecasting by modeling fine-grained, frequency-aware channel-patch dependencies through graph spectral decomposition and adaptive routing. xCPD introduces spectral channel-patch embedding, frequency-based grouping, and a MoE routing mechanism. Experiments across 9 benchmarks demonstrate consistent improvements when integrating xCPD into both channel-independent and channel-dependent backbones.

### Strengths
1. xCPD can be plugged into various architectures (linear, transformer, convolutional) with relatively low computational cost.
2. Evaluations cover both long-term and short-term forecasting, along with ablations and zero-shot tests.
3. Theorems 4.1 and 4.2 provide formal justification for the spectral construction.

### Weaknesses
1. Questionable novelty claim: The paper claims prior methods like TimeFilter [1] and CM [2] assume static inter-channel dependencies. However, both papers explicitly model **fine-grained**, **adaptive**, and **time-varying** relationships through learned attention or context-dependent graphs. This undermines the motivation that xCPD uniquely addresses “time-varying” dependencies.
2. The structure, especially the Channel-Patch Filtering with MoE, is highly reminiscent of TimeFilter’s[1] adaptive filtering mechanism. The main distinction—the spectral grouping—feels incremental rather than fundamentally novel.
3. In Table 6, the MSE/MAE values are worse than those reported in the TimeFilter[1] paper on the same datasets and horizons. This discrepancy needs to be explained.

[1] TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting

[2] Partial channel dependence with channel masks for time series foundation models

### Questions
1. The paper claims that prior methods assume static dependencies, but TimeFilter and CM both model dynamic ones. Could the authors clarify this claim?
2. Is there any insight into what the learned spectral groups correspond to in temporal terms?
3. The framework essentially reinterprets channel dependencies in the spectral domain, but without clear evidence that this yields fundamentally different learned relationships among low-, mid-, and high-frequency bands.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed a new method facilitating the modeling of cross-channel correlations multivariate time series. The proposed method is built based on the limitation of how existing multivariate time series forecasting methods handle dynamic correlations between channels, and introduce a graph composed of channel-patches to represent correlations between patches. A universal plugin that can be added to existing forecasting methods is introduced, and experiments show that it can bring performance uplift to existing methods.

### Strengths
1. A universal plugin is introduced based on the proposed novel technique of constructing correlation graphs of patches in multivariate time series. Extensive experiments prove its effectiveness at improving the performance of existing multivariate time series forecasting methods.
2. The proposed method is described in detail and thus should be highly reproducible.

### Weaknesses
1. The presentation of the background and challenges can be improved. While discussing existing methods, the paper claims that there are limitations in how existing multivariate time series forecasting methods handle cross-channel correlations. However, the paper mainly uses vague descriptions such as "coarse-grained methods treat each channel as a monolithic unit, missing nuanced interactions between channel segments", which could use some further explanation. There are also some claims that don't seem to be fully correct. For example, one claim that existing methods "failing to capture how inter-channel dependencies evolve dynamically", yet that doesn't seem to be the case for existing methods that dynamically calculate attention weights between channels. In conclusion, it is unintuitive to me why existing methods cannot effectively model correlations in multivariate time series.
2. Similarly, the explanation of the motivation behind the proposed method can be improved. Right now it is unintuitive why the proposed method can effectively tackle the limitations faced by existing methods, and how it is fundamentally different from existing designs. For example, the fundamental difference between building a graph of patches versus existing methods that calculate attention weights between patches (which essentially is a fully-connected graph).

### Questions
Can the authors further explain the motivation of this paper, so that the novelty of their proposed method is more intuitive?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces xCPD, a generic plugin that adaptively routes channel-patch dependencies in multivariate time series forecasting through a graph spectral decomposition framework. Unlike previous channel-partial or CI/CD approaches, xCPD conducts frequency-domain analysis using a learned, shared graph Fourier basis, classifying channel-patch nodes by spectral energy into low-, mid-, and high-frequency bands. It employs a mixture-of-experts (MoE) routing mechanism at the patch level to enable fine-grained, dynamic selection of inter-channel interactions. The plugin is model-agnostic, fits atop various existing backbones, and demonstrates consistent improvement in forecasting accuracy and generalization across a breadth of benchmarks and domains. Extensive empirical results, ablations, qualitative illustrations, and theoretical justifications are included.

### Strengths
1、The proposal to model dependencies at the channel-patch level, routed adaptively by frequency band via graph spectral decomposition, provides a fresh and well-argued approach to balancing robustness and expressivity in MTSF, beyond existing CI/CD/CP tactics.

2、The framework is underpinned by rigorous spectral graph theory, with precise mathematical formulations (e.g., Theorems 4.1 and 4.2 on shared basis and energy response, substantiated in the appendix with detailed proofs).

3、The methodology is described with mathematical precision, from input patch embedding (Equation block in 4.1), spectral grouping (Section 4.2, Theorem 4.2), and MoE-based routing (Section 4.3, detailed algorithmic equations). The description in Section 4 rigorously motivates and supports the approach.

### Weaknesses
1、Some equations and algorithm formulations are somewhat terse or overloaded. For instance, the definition of the adjacency matrix in Section 4.1 blends inner products and node similarity without explicit regularization or clarification about self-connections; there is little discussion on normalization/range scaling (see equation defining $\boldsymbol{A}^{t}$, Page 3).

2、While Figure 1 and the methodological text (Pages 3-4) present the channel-patch notion as a powerful primitive, more care is needed in formally defining the edges: Are patches temporally contiguous? Are there constraints to prevent trivial, densely connected graphs? Is $k$ in $k$-NN constant or adaptive? This lack of detail may hinder reproducibility and may even affect rigor in settings with many variables and timesteps.

3、The MoE routing for patch-level expert filtering, described with stochastic noise and cumulative softmax thresholds (Section 4.3), is algorithmically sound, but its practical behavior is underexplored. For example, the decision threshold $\tau$ and stochastic term $\epsilon$ may have non-trivial impact—these parameters are not deeply analyzed for sensitivity beyond a basic experiment. Additionally, while the adaptive routing is well-motivated, more insights or visualization of selected expert patterns across datasets/backbones would strengthen claims.

### Questions
1、Can the authors provide more precise formalization for channel-patch edge construction? Are there practical constraints on $k$ in the $k$-NN, and how does patch size/overlap affect results, especially in high-dimensional settings (e.g., Traffic, Electricity datasets)?

2、What is the sensitivity of performance and expert utilization (e.g., routing diversity, collapse) to the threshold $\tau$ and noise parameter $\epsilon$? Is there evidence that these parameters are stable across dataset/backbone combinations, or do they require careful tuning?

3、Have the authors considered adaptively selecting the number of frequency bands (rather than fixed to three), e.g., via a learned clustering or continuous assignment, and how would this affect interpretability or accuracy?

4、Can the authors provide statistics or visualizations of the empirical spectral gap across datasets? In scenarios with weak or noisy structure, what is the observed effect on performance and stability of the shared basis?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a generic plugin xCPD that can adaptively model the channel-patch dependencies from the perspective of graph spectral decomposition to address the problem in multivariate time series forecasting (TSF). This plugin can be added to an existing TSF model, improving its performance accordingly.

### Strengths
1. The idea of studying channel relationships is interesting. 

2. The experimental settings in this paper are extensive, covering long-term, short-term, and zero-shot forecasting. 

3. The writing of this paper is good, which is easy to read.

### Weaknesses
1. This plugin introduces lots of computations, but the improvement in the experiment seems to be trivial. 

2. In the experiment, this paper does not include existing CP methods, e.g.. Qiu et al., 2024, Chen et al., 2024,  Hu et al., 2025b, Lee et al., 2025.

3. The module design seems to be straightforward, and the design is not very novel.

### Questions
1. Why does this paper not include existing CP methods in comparison? 

2. In the efficiency evaluation, it does not compare the time difference between the original model and the original+xCPD method.

### Soundness
2

### Presentation
2

### Contribution
2
