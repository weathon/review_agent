# Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
While many advances in time series models focus exclusively on numerical data, research on multimodal time series, particularly those involving contextual textual information, remains in its infancy. With recent progress in large language models and time series learning, we revisit the integration of paired texts with time series through the Platonic Representation Hypothesis, which posits that representations of different modalities converge to shared spaces. In this context, we identify that time-series-paired texts may naturally exhibit periodic properties that closely mirror those of the original time series. Building on this insight, we propose a novel framework, Texts as Time Series (TaTS), which considers the time-series-paired texts to be auxiliary variables of the time series. TaTS can be plugged into any existing numerical-only time series models and effectively enable them to handle time series data with paired texts. Through extensive experiments on both multimodal time series forecasting and imputation tasks across benchmark datasets with various existing time series models, we demonstrate that TaTS can enhance multimodal predictive performance without modifying model architectures. Our Code is available at https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the TaTS framework, a novel approach for integrating contextual textual information with numerical time series data to significantly enhance forecasting accuracy. The authors propose a method that projects auxiliary variables from paired texts to augment the original time series representation, theoretically underpinned by the Platonic Representation Hypothesis, which posits a convergence of multimodal representations into shared spaces. The central contribution lies in the demonstrated effectiveness of this multimodal fusion: empirical results across nine diverse real-world domains consistently show that TaTS achieves an average of approximately 14% improvement in forecasting Mean Squared Error (MSE), justifying the minor incurred computational overhead of around 8% in training time.

### Strengths
1. Significant Empirical Improvement: The framework demonstrates a substantial and consistent performance gain (around 14% MSE reduction) across a wide array of domains (e.g., Economy, Climate, Health), indicating high robustness and generalizability.

2. Sound Theoretical Grounding: The methodology is well-motivated by the Platonic Representation Hypothesis, providing a strong conceptual basis for unifying textual and numerical time series representations.

3. Practical Efficiency Analysis: The paper includes a necessary trade-off analysis, clearly quantifying that the high performance benefits come with only a marginal and acceptable increase in computational overhead (around 8%).

### Weaknesses
1. Interpretability of Textual Influence: While the model is effective, the specific mechanism by which the projected auxiliary variables qualitatively impact or modulate the time series dynamics is not deeply explored. More detailed visualization or ablation studies on the how the text is woven into the temporal narrative would strengthen the paper.

2. Scalability to Text Length/Complexity: The paper does not thoroughly discuss the framework's scalability or performance behavior when dealing with significantly longer or structurally complex paired texts (e.g., full documents vs. short captions).

3. Sensitivity to Text Quality: A crucial missing component is an analysis of how the model performs when the paired text is noisy, weakly correlated, or intentionally irrelevant, which would better demonstrate the robustness of the feature extraction mechanism.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces TaTS, a simple framework that integrates timestamped text with numerical time series by converting text embeddings into auxiliary features for standard models. It is motivated by Chronological Textual Resonance (CTR), which captures periodic alignment between textual and numeric signals, and defines TT-Wasserstein to quantify this alignment. TaTS works with existing architectures without modification, improving forecasting and imputation performance across multiple datasets while adding minimal computational cost.

### Strengths
- Evaluated across multiple datasets and diverse time-series backbones, supporting robustness and generalizability.
- Clear, well-structured framework: CTR (motivation), TaTS (mechanism), and TT-Wasserstein (diagnostic) are distinctly defined and complement each other.
- Plug-and-play simplicity: TaTS adds a small projector to turn per-timestamp texts into features and works with existing TS backbones without architecture changes.

### Weaknesses
- CTR/TT-Wasserstein focus on magnitude-only spectral alignment, favoring stable periodicities and largely ignoring phase—so they can undervalue datasets where texts are leading or lagging indicators or where alignment is time-varying/non-stationary, even if the texts are predictive.
- Limited discussion of TS→text (inverse direction): brief related-work mention and a single zero-shot ChatTime baseline; no deeper analysis or dedicated experiments.
- Limited native multimodal baselines: ChatTime is the only truly native multimodal method evaluated, while others are adapted uni-modal TS models; MM-TSFLib is further extended for imputation since it doesn’t support it natively.

### Questions
Please address the identified weaknesses and limitations noted above.

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
This paper presents TaTS, a plug-in framework that treats timestamp-aligned texts as auxiliary variables for time-series models. By embedding the texts, projecting them through a small MLP, and concatenating them to numeric inputs, the approach allows standard forecasters and imputers to exploit textual signals without changing their architectures. The work is motivated by Chronological Textual Resonance (CTR), the observation that paired texts often share periodic structure with the target series, and it introduces TT-Wasserstein to quantify this alignment. Extensive experiments across varied datasets and backbones indicate consistent gains, particularly when TT-Wasserstein suggests strong alignment. The computational overhead is modest, which enhances the practical appeal of the method.

### Strengths
- The paper introduces a simple, model-agnostic plug-in that treats text embeddings as auxiliary variables, simplifying integration into existing pipelines while keeping computational overhead low.
- The work provides an interpretable alignment rationale via CTR and the TT-Wasserstein metric, which helps explain when and why text improves performance and guides practical deployment.
- Extensive experiments with clear ablations (shuffle/drop text, encoder swaps, multiple backbones and datasets) demonstrate stable gains.

### Weaknesses
- The central mechanism is projection followed by concatenation, which may be outperformed by comparably simple baselines, such as gated residuals or lightweight cross-attention, unless comparisons are conducted under strict parameter matching.
- The definition and estimation of TT-Wasserstein lack sufficient statistical treatment; sensitivity to windowing, normalization, and frequency resolution is not examined, and confidence intervals are absent.
-  The data-splitting procedure is not described in adequate detail, creating uncertainty about potential information leakage from contemporaneous or retrospective texts.
- The framework offers no automatic protection against negative transfer when CTR is weak, which could undermine robustness in noisy or weakly aligned settings.

### Questions
- How are splits constructed per dataset to ensure texts at time t do not reveal contemporaneous or near-future outcomes, and are audits/filters applied for retrospective or outcome-summarising language?
- Which FFT, windowing, and normalization settings are used for both modalities, how sensitive are results to these choices, and can confidence intervals be reported for the correlation?
- How does TaTS compare with (a) a small gated residual that can down-weight text channels and (b) a parameter-matched cross-attention/FiLM block?
- Has weighting or dropping text channels based on TT-Wasserstein been tested to mitigate negative transfer when alignment is weak?
- What are the training/inference wall-clock costs for different text encoders (e.g., GPT-2, BERT, LLaMA), and what rule of thumb should practitioners use to choose an encoder?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces TaTS, a new multimodal time series forecasting benchmark compatible with any time series forecasting model. TaTS embeds the textual descriptions of time series, projects them using an MLP, and then combines them with the original time series to create a new input for forecasting models. The motivation behind this architecture is that time series paired with text often exhibit periodic patterns that reflect the dynamics of the numerical time series. Additionally, the authors introduce TT-Wasserstein, a metric designed to evaluate the alignment between the time series and the associated text.

### Strengths
1. The proposed architecture is model agnostic and delivers good improvement of the base model.

2. Incorporate textual information into time series if timely and interesting.

### Weaknesses
1. The paper lacks comparison with other multimodal algorithms. For example, the authors don't compare their approach with TimeMMD, which is also model-agnostic and offers similar improvements over unimodal models. There are also several other multimodal algorithms available that could serve as valuable comparisons.

2. The method requires access to a textual description for each timestamp, which may be difficult to obtain in real-world applications.

### Questions
1. How did you choose the competitors and why not incorporating multimodal algorithms?

2. How the method can adapt if not enough textual descriptions are not available for each time stamp if there are not alignment between time series and text or at least it is not known ?

3. it seems the information from the text is not complementary but redundant information with the time series as captured by the low TT-Wasserstein. In this redundant information case one would expect the text not to improve the result and expect more an improvement when the information is complementary and critical for the forecast?

### Soundness
2

### Presentation
3

### Contribution
2
