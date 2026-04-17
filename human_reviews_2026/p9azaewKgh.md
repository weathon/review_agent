# From Images to Signals: Are Large Vision Models Useful for Time Series Analysis?

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Large Vision Models (LVMs) are emerging tools for transferring cross-modal knowledge to time series, but this potential is not well understood. This work addresses the gap by investigating LVMs for both high-level (classification) and low-level (forecasting) tasks. Our aim is to not only assess whether LVMs can succeed, but also reveal why they succeed or fall short. Through a comparative benchmark covering 4 LVMs, 8 imaging methods, 18 datasets, and 26 baselines, we identify the strengths and limitations of LVMs, as well as strategies for adapting them to time series modeling. Our findings indicate while LVMs are effective for time series classification, they face notable challenges in forecasting - the best LVM forecaster is limited to specific model types and imaging methods, exhibit biases toward forecasting periods, and struggle to leverage long look-back windows. We hope our findings can serve as both a cornerstone and a practical guide for advancing LVM- and multimodal-based solutions to different time series tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper benchmarks LVMs for time-series analysis by converting time-series data into image representations. It systematically evaluates 4 LVMs (ViT, Swin, MAE, SimMIM), 8 imaging techniques, and 18 datasets across classification and forecasting tasks. The authors analyze pretraining paradigms, imaging methods, decoder vs. encoder roles, and context length sensitivity.

### Strengths
1. First comprehensive study to jointly analyze LVMs across TSC and TSF using diverse image encodings.
2. Provides a valuable reference for researchers exploring multimodal or vision-inspired time-series models.

### Weaknesses
1. Most of the forecasting results rely on UVH/MVH imaging applied to relatively periodic datasets. It’s not clear how well these conclusions hold for more irregular, noisy, or non-periodic signals.
2. The claim that decoders matter more than encoders in forecasting is interesting, but the current evidence is mostly correlational. A simple decoder-swapping or partial-finetuning experiment could make this stronger.
3. The authors note that performance drops for longer look-back windows, potentially due to image resolution limits, but this isn’t experimentally verified. It’s unclear whether the weaknesses observed (e.g., poor long-context forecasting) are intrinsic to LVMs or simply due to architectural mismatch, because the approach reuses vision models mostly as-is, with only changes to heads or decoders. 
4. The central assumption that converting time series into 2D image structures enables vision transformers to generalize to sequential modeling is interesting but not fully justified. Time and space are fundamentally different dimensions; the study doesn’t discuss what inductive biases are gained or lost by this transformation. For example, local temporal continuity is often disrupted in GAF/UVH encodings, and the benefit of spatial locality learned by ViTs is not clearly transferable to time-dependent modeling.
5. The paper tests many imaging techniques (GAF, RP, UVH, etc.), but it remains unclear why certain encodings align better with LVM features. There’s little analysis of the representational geometry, e.g., whether these encodings preserve correlations, periodicities, or value scales in a meaningful way. The success of UVH/MVH feels more empirical.

### Questions
1. How robust are the UVH/MVH findings to non-periodic datasets or those with exogenous inputs (e.g., with irregular seasonality or abrupt regime shifts)?
2. Have you tested simple augmentations (phase shifts, randomized boundaries) to mitigate the UVH bias? You have shown evidence that UVH introduces a copy-period shortcut during forecasting, leading to biased predictions. Did you experiment with simple augmentations to disrupt the implicit periodic alignment? I am wondering whether the bias is structural to the imaging method or sensitive to data alignment choices.
3. The paper concludes that forecasting success in self-supervised LVMs (MAE, SimMIM) mainly stems from pre-trained decoders rather than encoders. Could decoder-swapping experiments (e.g., MAE encoder + SimMIM decoder and vice versa) validate this more directly? Did you observe consistent trends when fine-tuning only the decoder versus only the encoder?
4. The authors report that forecasting performance declines when the input length exceeds roughly 1k steps, hypothesizing this stems from fixed image resolution constraints. Have you tried multi-scale or variable-resolution inputs (e.g., tiled or pyramid representations) to test whether LVMs can better exploit extended contexts when provided with hierarchical image patches?

### Soundness
2

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
3

### Summary
This paper conducts the comprehensive study on the utility of Large Vision Models for time series analysis, covering classification and forecasting tasks. It evaluates 4 LVMs, 8 imaging methods, 18 datasets, and 26 baselines. Key findings show LVMs excel at TSC by leveraging pre-trained semantic recognition capabilities, with GAF as the optimal imaging method. However, TSF poses greater challenges, self-supervised LVMs such as MAE paired with UVH imaging perform best but suffer from period bias and limited long look-back window utilization. Ablation analyses confirm LVMs transfer pre-trained knowledge, capture temporal order, and rely more on decoders for forecasting. While LVMs trade higher computational cost for superior TSC performance, their TSF effectiveness is constrained by task-specific limitations, providing foundational insights for multimodal time series research.

### Strengths
1. It systematic explore large vision models for time series classification and forecasting, covering diverse models, imaging methods, datasets, and baselines to fill existing research gaps.
2. In-depth mechanism analysis reveals key insights and quantifies temporal pattern capture, providing essential support for future optimizations.
3. It identifies optimal task-specific configurations and targeted fine-tuning strategies, enabling efficient real-world implementation.
4. Rigorous benchmark comparisons  and analysis experiments ensure reliable conclusions, establishing a credible reference for subsequent research.

### Weaknesses
1. The study mentions that Large Vision Models have numerous limitations in time series forecasting tasks. However, leveraging the feature extraction capability of LVMs for TSF and integrating them with time series through cross-modal fusion may help avoid visual limitations and improve overall performance. The current work lacks an investigation into cross-modal fusion involving visual modalities.
2. The study evaluates 8 imaging methods, and from Table 16, UVH achieves the best performance while MVH ranks second. Other imaging methods perform significantly worse—frequency-domain methods such as Wave and STFT do not yield better results through imaging. Does this indicate that frequency-domain graphs are unsuitable as inputs for LVMs? For MVH and UVH, why does MVH underperform UVH, and would the results differ for datasets with variable dependencies? Additionally, could simultaneous input of MVH and UVH bring performance gains?
3. The study lacks comparisons with state-of-the-art large time series models (e.g., Chronos-2, Moirai2), which may undermine the comprehensiveness of LVMs’ performance evaluation.

### Questions
See Weakness.

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
This paper presents a comprehensive benchmark study investigating the effectiveness of Large Vision Models (LVMs) for time series analysis. The authors evaluate four LVMs (ViT, Swin, MAE, SimMIM) across two representative tasks: time series classification (TSC) and time series forecasting (TSF). Through extensive experiments with 8 imaging methods and 18+ baselines on 18 datasets, the paper finds that LVMs excel at TSC but face significant challenges in TSF due to limitations in encoder utilization, inductive biases, and long look-back window handling. The work provides detailed ablation studies and practical insights for adapting LVMs to time series tasks.

### Strengths
(1) The paper covers a substantial experimental scope (4 LVMs, 8 imaging methods, 18 datasets, 26 baselines) with well-designed ablation studies (RQ1-RQ10). The breadth of analysis is commendable and provides genuine value to the community.
(2) The discovery that pre-trained decoders contribute more than encoders in TSF (RQ8) is genuinely interesting and counterintuitive. The analysis of the period-based imaging bias (RQ9) with formal characterization (Lemma 1) provides actionable insights.
(3) The temporal perturbation experiments (RQ6, Table 4) effectively demonstrate that LVMs do capture temporal information, strengthening claims about their utility beyond pattern matching.
(4) The paper is generally well-organized with clear research questions guiding the narrative. Figures 1-3 effectively summarize the methodology and key findings.

### Weaknesses
(1) The paper is purely an empirical benchmark study without methodological contributions. While valuable, such studies typically require either exceptional insights or novel proposed solutions. 
(2) The paper identifies what fails (encoders, long windows) but provides limited mechanistic understanding of why.
(3) The claim that forecasting is "low-level" and requires numerical inference needs deeper investigation beyond decoder architecture.
(4) The connection to recent multimodal approaches (mentioned briefly) deserves more engagement.
(5) Using ImageNet-derived normalization for time series images may introduce bias. Ablation on normalization strategy is absent.
(6) Why variate-independence assumption for all tasks? For TSC, joint multivariate modeling might capture interactions.

### Questions
(1) Decoder Importance (RQ8): Can you provide more analysis on what the decoders learn that aids TSF? Attention visualizations or learned representations would be illuminating. Is this specific to MAE/SimMIM architectures or general?
(2) Periodic Bias Generalization: Lemma 1 is elegant for UVH, but how do other imaging methods induce biases? Can similar formal analysis be provided for GAF or MVH? Does this explain why they underperform for TSF?
(3) Why limit to 8 TSF datasets? Larger evaluation (similar to TSC's 10 datasets) would strengthen claims. Would results change with higher-dimensional time series (e.g., multivariate > 20 dimensions)?
(4) The recommendation to fine-tune only norm layers for TSF seems counterintuitive. Is this an artifact of small datasets or a fundamental limitation of LVMs for low-level tasks?

### Soundness
3

### Presentation
3

### Contribution
3
