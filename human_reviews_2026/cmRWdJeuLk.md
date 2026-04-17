# TwinsFormer: Revisiting Inherent Dependencies via Two Interactive Components for Time Series Forecasting

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Due to the remarkable ability to capture long-term dependencies, Transformer-based models have shown great potential in time series forecasting. However, real-world time series usually present intricate temporal patterns, making forecasting still challenging in many practical applications. To better grasp inherent dependencies, in this paper, we propose TwinsFormer, a novel Transformer-based framework utilizing two interactive components for time series forecasting. Unlike the mainstream paradigms of plain decomposition that train the model with two independent branches, we design an interactive strategy around the attention module and the feed-forward network to strengthen the dependencies via decomposed components. Specifically, we adopt dual streams to facilitate progressive and implicit information interactions for trend and seasonal components. For the seasonal stream,  we feed the seasonal component to the attention module and feed-forward network with a subtraction mechanism. Meanwhile, we construct an auxiliary highway (without the attention module) for the trend stream under the supervision of seasonal signals. In this way, we can avoid the model overlooking inherent dependencies between different components for accurate forecasting. Our interactive strategy, albeit simple, can be adapted as a plug-and-play module to existing Transformer-based methods with negligible extra computational overhead. Extensive experiments on various real-world datasets show the superiority of TwinsFormer, which can outperform previous state-of-the-art methods in terms of both long-term and short-term forecasting performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents a new Transformer-based architecture that innovatively integrates time series decomposition with interactive learning. The proposed dual-stream design and interactive mechanism represent an advancement in modeling temporal dependencies.

### Strengths
1. The dual-stream design with explicit interaction between trend and seasonal components is novel and well-executed. The interactive module that enables information flow between decomposition branches addresses a genuine limitation in existing decomposition-based methods.

2. The generalization bound analysis provides valuable theoretical insights into why the decomposition-interaction mechanism leads to improved performance, elevating the work beyond purely empirical contributions.

3. The thorough ablation experiments in Table 3 effectively validate the importance of each component, particularly demonstrating the value of the subtraction mechanism and interactive module over simpler alternatives.

### Weaknesses
1. While the paper demonstrates that the interactive module improves performance, it provides limited insight into what specific information is being exchanged between the trend and seasonal branches. A more detailed analysis of the interaction dynamics would strengthen the contribution.

2. The work relies on a simple moving average for decomposition initialization. The paper will benefit from exploring how sensitive the model is to different decomposition techniques and whether more sophisticated decomposition methods could yield further improvements.

### Questions
1. Can you provide more insight into what types of information are typically transferred between the seasonal and trend branches through your interactive module? Are there specific temporal patterns that trigger stronger interactions?

2. Have you experimented with more advanced decomposition techniques beyond moving average? Does the performance gain primarily come from the interaction mechanism or could it be limited by the decomposition quality?

### Soundness
3

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
5

### Summary
TwinsFormer, a Transformer-based architecture for time series forecasting that specifically takes into account interactions between seasonal and decomposed trend components, is presented in this paper.  TwinsFormer creates interactive modules where trend and seasonal representations are entwined via attention and feed-forward mechanisms, such as gating and subtraction/complementary transformations, in contrast to current dual-branch approaches that function independently on decomposed parts.  Empirical results show that the model performs at the state-of-the-art level on most tasks when tested against 13 real-world benchmarks for both short-term and long-term forecasting.  Visual diagnostics such as component effect figures and results tables are presented along with ablation studies, compatibility analysis, and theoretical justifications (including generalization bounds).

### Strengths
1. Motivated Issue and Explicit Restrictions of Previous Research:  The authors critically analyze a common flaw in Transformers' time series decomposition, specifically the underutilization of the interaction between trend and seasonal components (see Figure 1 for a visualization and Section 1 for an explanation).  This solves a practical gap in the field.

2. Strong Empirical Assessment:  To compare against a wide range of extremely competitive baselines, including Transformer-based and linear/TCN approaches, extensive experiments are carried out on a variety of benchmarks (see Table 1 for long-term forecasting and Table 2 for short-term forecasting).  TwinsFormer consistently outperforms or performs on par with the best models, according to the results, particularly on difficult multivariate settings.  Several datasets and prediction lengths are examined.

3. Clarity of Presentation: Both theorists and practitioners can understand the work thanks to its combination of mathematical explanation, architectural diagrams (Figure 2), component visualizations, performance curves (Figure 4), and hyperparameter sensitivity plots (Figure 7).  Every visual component is closely related to a central claim or conclusion.

### Weaknesses
1. The paper omits comparison with closely related recent work (e.g., DESTformer, Modeling Temporal Symmetry, Beyond Trend and Periodicity) that also adopt trend–seasonality decomposition and dual-component designs. Without a careful contrast of assumptions, architectures, and results, it is hard to judge whether the claimed contribution is genuinely novel or an incremental extension, which weakens originality and completeness.

2. Theoretical analysis (e.g., Rademacher complexity in Section 3.3) is plausible, but empirical support for real-world generalization is loosely connected. Beyond the MSE gap in Fig. 4, there is no systematic evaluation on out-of-distribution data or failure scenarios. Robustness studies under noise, missingness, and distribution shift, as well as transfer settings, are missing.

3. Methodology is under-specified. For the moving-average decomposition, the kernel window, padding effects, and failure modes (irregular sampling, abrupt trend changes) are not detailed, raising even data-leakage concerns. The Interaction Module lacks intuition and analysis for why the four MLPs (α,β,γ,μ) are chosen and how they are trained. Performance may depend heavily on such low-level choices, which calls robustness and generality into question.

4. The “Limitations” section is superficial. It does not systematically address conditions where the dual-stream strategy can be suboptimal, such as weak seasonality, corrupted observations, or richer multi-scale structures. Concrete failure cases, complexity–performance trade-offs, and plans to improve scalability and robustness should be articulated.

### Questions
1. Can the authors use the same evaluation protocol to directly compare TwinsFormer with DESTformer's dual-component framework on at least two or three key benchmarks? 
2. How does the moving average kernel for decomposition perform on datasets with weak or non-existent trend/seasonality, or on data that is non-stationary or irregularly sampled?

### Soundness
2

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
5

### Summary
This paper designs two interactive components for information mining of time series data.

### Strengths
The solution proposed in this article is simple and easy to understand.

### Weaknesses
•  I understand that the residuals contain interactive and other useful components, but this may result from the decomposition method itself, such as STL being insufficient. If a more thorough decomposition method were used, such as EMD or wavelet decomposition, this issue might be alleviated.

•  From Figure 1, the authors claim that the decomposed trend and seasonal components achieve better cointegration and fluctuation consistency, but there is no theoretical verification. The proportion of residual allocation may also be correlated with the inherent characteristics of the data.

• The method lacks sufficient innovation.

(1) The decomposition + Transformer framework has been widely explored, and a simple “subtraction residual” is not enough to support the claimed novelty.

(2)   The Interactive Module is very similar to FiLM (Feature-wise Linear Modulation, Pérez et al., 2018) and essentially a conditional affine transformation. 

(3)  Equations (8)–(10) only demonstrate signal conservation rather than proving a causal improvement in dependency modeling; direct residual connections can also ensure conservation.

•  Although the experimental results appear strong, the input length of 96 may affect the decomposition quality. Therefore, I remain cautious about the claimed superiority and look forward to the public release of the code.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes TwinsFormer, a Transformer forecasting framework that explicitly models trend and seasonal components via two interacting streams. Unlike prior “plain decomposition” approaches with independent branches, it injects interaction at both the attention and feed-forward stages: the seasonal stream uses a subtraction mechanism inside attention/FFN, while the trend stream follows an auxiliary highway (no attention) trained under seasonal supervision. The module is claimed to be plug-and-play with negligible overhead and to deliver SOTA results on multiple real-world datasets for both short- and long-term horizons.

In my opinion, it investigates an important topic in time series models and proposes valuable solutions.

### Strengths
1. The motivation of this submission on investigating seasonal and trend components in Transformer-based time series models is sound since many of work have this seasonal and trend decomposition.
2. The theoretical analysis has merits. 
3. The experiemental results show that the proposed TwinsFormer is outperforming other baselines on both long-term and short-term forecasting.

### Weaknesses
1. The datasets used in the experiments seems a bit outdated in my opinion. I understand that those datasets are commonly used in time series community. But there are some studies critize these datasets and since 2025, there are some growing benchmarks in the community (e.g., [1] and [2]). I think if we have some results on those benchmarks, it can greatly improve the quality of experiments and be more convincing. 

See Questions for others. 

References: 

[1] fev-bench: A Realistic Benchmark for Time Series Forecasting. 

[2] GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation

### Questions
1. I think that the methodology part miss one formula for $F_T$. From the figure shown, I suppose $F_T = E^1_T + E^2_T$? 
2. For the ablation 2, does "swap trend and seasonal components" mean that use the trend component for attention instead? 
3. In ablation study, it shows that using addition skip connections performs pretty bad. I wonder if we can obtain some insights or reasons behind that. I feel that might be particularlly of interest.

### Soundness
3

### Presentation
3

### Contribution
3
