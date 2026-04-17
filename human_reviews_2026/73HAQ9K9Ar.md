# FACT: Frequency-Aware Channel-Guided Multivariate Time Series Forecasting

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Forecasting Multivariate Time Series (MTS) requires capturing complex intra-channel dynamics and evolving inter-channel dependencies. However, existing methods often struggle to disentangle meaningful signals from inter-channel noise and intricate interaction patterns. To address this, we propose a novel framework that operates entirely in the frequency domain, modeling inter-channel relationships at the component level. Our approach first dynamically decomposes each time series into its constituent frequencies. An Adaptive Band Decomposition mechanism then identifies and isolates the most salient frequency components, simultaneously filtering noise and enhancing computational efficiency. This allows our model to capture time-varying inter-channel dependencies with high fidelity. Furthermore, our learning objective effectively balances accuracy against regularization constraints for both computational efficiency and interpretability. Extensive experiments on diverse, real-world datasets demonstrate that our method achieves competitive performance. Code is available at this repository: \url{https://anonymous.4open.science/r/FACT}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a new method for multivariate time series forecasting. The paper identifies the limitation of existing methods as the modeling of cross-channel correlations. The paper then proposes a solution that focus on modeling cross-channel correlations at the channel-frequency cell level with both magnitude and phase. Experiments demonstrate the effectiveness of the proposed method.

### Strengths
A novel method is proposed to specifically tackle the challenge of multivariate time series forecasting identified by the paper, with multiple specific designs that are described in detail.

### Weaknesses
1. The experimental setting can be expanded to provide more insights into the performance behavior of the proposed method, for example, ablation studies.
2. The presentation of the motivation and explanation of design choices are rather technical and can be elaborated further to improve their intuitiveness.
3. A minor complaint is the paper has obvious typesetting issues and is not following the standard ICLR template. The authors probably should check their LaTeX compile errors.

### Questions
Could the authors elaborate more on the challenge of "channel-frequency cell level" modeling and how their specific module designs are effective at tackling the challenge?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper focuses on the core challenge of modeling channel interactions in multivariate time series. It identifies that existing methods mostly process correlations at the original channel dimension level, often struggling to balance noise suppression with preserving effective information, especially in high-dimensional or long-sequence tasks where they either lose fine-grained mechanisms or introduce high computational complexity.

### Strengths
1. Multivariate time series forecasting is important to various domains.

2. There are quite a few nice illustrations.

3. This work focuses on an important problem that could have real-world applications.

### Weaknesses
1. As a paper submitted to ICLR 2026, the baselines used for comparison are relatively outdated, lacking evaluations against recent works from 2025. This undermines the credibility and persuasiveness of the experimental results.

2. In Table 1, the proposed method performs worse than SOFTS in most cases. Such performance does not appear sufficient for publication at a top-tier venue like ICLR.

3. The methodological description is unclear and poorly organized, making it difficult for readers to understand the overall workflow and specific implementation details.

4. The paper does not reach the required 9-page limit and lacks several essential experiments, such as ablation studies and parameter sensitivity analyses, which makes the experimental validation incomplete and less rigorous.

### Questions
1. As a paper submitted to ICLR 2026, why are the baselines used for comparison relatively outdated, without including recent works from 2025? Doesn’t this weaken the credibility and persuasiveness of the experimental results?

2. In Table 1, why does the proposed method perform worse than SOFTS in most cases? Can such performance be considered sufficient for publication at a top-tier venue like ICLR?

3. Why is the methodological description so unclear and poorly organized, making it difficult for readers to understand the overall workflow and specific implementation details?

4. Why does the paper fail to reach the required 9-page limit and omit essential experiments such as ablation studies and parameter sensitivity analyses? Doesn’t this make the experimental validation incomplete and less rigorous?

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
4

### Summary
This paper proposes FACT, a frequency-aware and channel-guided framework for multivariate time series forecasting. FACT operates in the frequency domain and captures meaningful signals from inter-channel noise and intricate interaction patterns at the component level. Extensive experiments on real-world datasets demonstrate its competitive performance.

### Strengths
1. FACT Lifts channel interaction modeling from original signals to the frequency-component level, enabling fine-grained and physically meaningful analysis.
2. The integration of explicit magnitude coherence and phase offset modeling with corresponding regularization terms provides an approach to capturing meaningful signals and improving forecasting accuracy.
3. Extensive experiments on real-world datasets demonstrate consistent competitive performance, with particularly strong results on Solar-Energy and Weather forecasting tasks.

### Weaknesses
1. I can’t find the ablation results in this paper. The critical contribution of component modeling remains insufficiently quantified. 
2. The model lacks comparison with recent frequency models to better understand the advancement.
3. The experimental design fails to properly validate the plugin capability, as all results present FACT as an end-to-end model rather than demonstrating its integration as a plugin component with existing architectures.
4. The writing quality impedes understanding, with numerous sections exhibiting unclear expression and disorganized structure that obscure methodological contributions.

### Questions
1. The manuscript seems require significant formatting revisions to comply with ICLR submission guidelines.
2. Table 5 contains citation errors in its caption ("The results are taken from SOFTS and iTransformer (?)").

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FACT (Frequency-Aware Channel-Guided Multivariate Time Series Forecasting), a framework that models multivariate time series by decomposing them into frequency components using learnable Gaussian masks. The model leverages a channel masking mechanism to identify salient frequency components, incorporates channel priors that reflect amplitude and phase coherence, and employs complex attention/fusion modules. Regularization terms tie interpretability to accuracy during training.

### Strengths
1. Well-motivated frequency-level interaction modeling: The proposed approach recognizes that not all frequency bands are equally informative for prediction and that genuine inter-channel dependencies often exist at the frequency component level, going beyond typical channel mixing or global spectral reweighting.

2. Model-agnostic plug-in capability: FACT can be used as a drop-in frontend for multiple backbone architectures (Transformer, MLP, or Linear) and maintains gains across backbones.

### Weaknesses
1. Lack of clarity: The article lacks clarity regarding how the relationship between channel and frequency is modeled.

2. Poor performance: The selected baseline is insufficient, and the proposed method does not outperform the 2024 baseline in any average MSE results of the datasets.

3. Absence of Ablation and Sensitivity Analyses: The manuscript lacks both ablation studies and sensitivity analysis. Consequently, the efficacy of the proposed modules and the overall viability of the method cannot be validated.

### Questions
1. What is the actual computational overhead (cost of Inference time and training time performance, max GPU memory, number of parameters and MACs) of FACT relative to the strongest baselines (e.g., PatchTST, iTransformer, FreTS) on the datasets?

2. Could the author explain the phenomenon of the proposed model consistently underperforming across all average Mean Squared Error (MSE) results? Additionally, we recommend considering the inclusion of more recent and relevant baselines for comparison, such as DUET[1], TimeFilter[2], and TQN[3].

3. How robust are results to the choice of regularization strengths ($\lambda_{coh}, \lambda_{phase}$)? Is there any risk of over-regularizing and thereby compromising prediction?

[1] DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting. KDD2025

[2] TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting. ICML 2025

[3] Temporal query network for efficient multivariate time series forecasting. ICML 2025

### Soundness
2

### Presentation
1

### Contribution
2
