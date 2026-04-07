=== CALIBRATION EXAMPLE 47 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is clear and reflects the core contribution. The abstract succinctly states the problem (expensive dense annotations for LTA), the proposed solution (first weakly-supervised method using only transcripts), and summarizes the key components (temporal alignment, cross-modal attention, encoder-decoder). It correctly claims novelty and mentions competitive results on three benchmarks. The abstract is well-aligned with the paper's content.

### Introduction & Motivation
The introduction effectively motivates the need for weakly-supervised LTA by highlighting the cost and scalability limitations of dense annotations. It clearly distinguishes prior work (Zhang et al., 2021) that still relies on some temporal labels, establishing a clear gap for a transcript-only approach. The five contributions are precisely stated. The problem definition is clear: using only ordered action lists (transcripts) without timing.

### Related Work
Comprehensive and well-organized. It appropriately covers Temporal Action Segmentation (TAS), Action Anticipation, and Long-Term Action Anticipation (LTA), correctly positioning the work as the first fully weakly-supervised method for dense LTA using only transcripts. It acknowledges relevant weakly-supervised TAS methods and sequence-to-sequence alignment techniques (CTC, DTW). The relation to prior LTA methods is clear.

### Methodology
This is the core technical section, but several critical points require clarification and raise concerns:

1. **Inference without transcripts**: The model uses cross-modal attention and a temporal alignment module during training, both of which require the transcript. At inference, the transcript is not provided. The paper states the model must "implicitly estimate" the boundary \(k^*\) and observed pseudo-labels. It is unclear how the model performs this estimation without the transcript. Does the temporal alignment module run at inference? If not, how does the model segment the observed interval to inform anticipation? This creates a potential train-test mismatch that must be explained.

2. **Cross-modal attention at inference**: The cross-attention layer uses transcript embeddings and a mask derived from pseudo-labels. Since the transcript is unavailable at inference, is this layer simply skipped? If so, the features used during inference are not enriched by transcript context, which may degrade performance. The paper should clarify the inference pipeline and whether any compensatory mechanism exists.

3. **Pseudo-label noise propagation**: The temporal alignment module (ATBA) generates soft pseudo-labels. The quality of these labels, especially near boundaries, is critical for supervising both segmentation and anticipation. No analysis is provided on the robustness to alignment errors or how noise affects the anticipation decoder. An ablation with oracle alignment (or ground-truth pseudo-labels) would help quantify this sensitivity.

4. **Anticipation decoder and CRF details**: The description of the anticipation decoder is somewhat vague. It outputs "descriptors S" decoded to action classes. How are the number of future segments determined? The CRF loss (Eq. 6) is applied to the future action sequence \(Y_{LTA}\). However, \(Y_{LTA}\) is derived from the transcript's future part, which is a sequence of action symbols without durations. The CRF operates on a sequence of symbols, but the decoder also predicts durations via a separate head. The interaction between the symbol sequence and duration prediction is not fully explained. Additionally, the CRF transition matrix is learned; how is it initialized and regularized without ground-truth transitions?

5. **Duration loss rationale**: The duration loss uses class-wise priors estimated from the observed segment's predictions. This assumes that action durations are consistent across the video, which may not hold (e.g., "cutting" may have variable duration). The loss may reinforce biases from noisy pseudo-labels. An ablation without this loss would be informative.

6. **Progressive training stability**: The three-stage training (pre-training, alignment+segmentation, full end-to-end) is described, but no justification is given for the chosen epoch counts (10, 30, then full). Sensitivity to these hyperparameters and the risk of error accumulation across stages are not discussed.

Despite these concerns, the overall architecture is innovative and integrates several weakly-supervised techniques in a principled manner.

### Experiments & Results
The experimental setup is standard and appropriate. Results in Table 1 show that TbLTA outperforms the prior weakly-supervised baseline (Zhang et al., 2021) by a large margin and is competitive with fully supervised methods, which is impressive. On Breakfast, it even surpasses some supervised methods at certain observation ratios. On 50Salads, performance is lower but still competitive, reflecting the dataset's greater complexity. Results on EGTEA (Table 2) show competitive performance on rare classes, suggesting transcript supervision helps with class imbalance.

However, several issues weaken the empirical evaluation:

1. **Missing baseline**: A critical baseline is absent: a two-stage approach where a weakly-supervised TAS method (e.g., ATBA) generates pseudo-labels for the full video, and then a supervised LTA model is trained on these pseudo-labels (using the observed segment as input and future pseudo-labels as target). This would isolate the benefit of joint training versus simply using pseudo-labels as a substitute for ground truth.

2. **Incomplete ablation study**: Ablations focus only on CTC loss and cross-attention. The impact of other key components—CRF, duration loss, temporal alignment module, and the use of class tokens—is not quantified. For instance, how much does the CRF contribute to temporal coherence? Does the duration loss actually improve duration prediction accuracy? An ablation on the loss weights (\(\gamma_1, \gamma_2, \gamma_3\)) would also be informative.

3. **Evaluation of pseudo-label quality**: No metrics are provided for the quality of the generated pseudo-labels (e.g., frame-wise accuracy or F1 score on the training set). This is important to understand the upper bound of performance if pseudo-labels were perfect.

4. **Stochastic results**: The paper mentions stochastic results in the supplement but does not include them in the main text. Since stochastic anticipation is an important direction, at least a summary should be provided in the main paper for context.

5. **Statistical significance**: Results are averaged over splits, but no measures of variance (e.g., standard deviation) are reported, making it difficult to assess the stability of the results.

### Limitations & Broader Impact
The paper lacks a dedicated limitations section. The conclusion briefly mentions duration estimation as a challenge, but other limitations are not discussed:
- Dependence on pre-extracted I3D features, limiting end-to-end learning from raw video.
- Assumption that transcripts are perfectly accurate and follow the exact video order; real-world transcripts may have omissions, errors, or paraphrasing.
- The method is designed for procedural activities with a linear narrative; it may not handle activities with branching or complex temporal dependencies.
- The computational cost of the progressive training scheme and multiple losses is not analyzed.
- Broader impact is only implicitly positive (reducing annotation cost). Potential negative societal impacts (e.g., surveillance) are not mentioned, though they are minimal for this work. A short discussion would be appropriate.

### Writing & Clarity
The paper is generally well-written, but the methodology section is dense and occasionally ambiguous, as noted above. The figures (referenced but not included in the text snippet) are likely helpful. There are minor typos (e.g., "OBJETIVE" in Section 3.2). Overall, the writing is acceptable for ICLR, but the technical clarifications are necessary for full understanding.

## Overall Assessment
This paper presents a novel and timely contribution: the first weakly-supervised method for dense Long-Term Action Anticipation using only video transcripts. The proposed TbLTA integrates several innovative components, including temporal alignment for pseudo-label generation, cross-modal attention, and a combination of CTC and CRF losses. The results are impressive, showing competitive performance with fully supervised methods on standard benchmarks, which strongly supports the claim that transcript-based supervision is a viable and scalable alternative.

However, the paper has significant weaknesses in clarity and empirical rigor. Key aspects of the inference pipeline are unclear, the ablation studies are incomplete, and critical baselines are missing. The lack of a limitations section is also a drawback. Addressing these concerns—especially by clarifying the inference procedure, adding the two-stage baseline, and expanding the ablation analysis—would substantially strengthen the paper.

Given the novelty, strong results, and potential impact on reducing annotation burden, the paper is likely above the acceptance bar for ICLR **if the authors can adequately address these issues in a rebuttal**. The core contribution is solid, but the presentation needs refinement.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces TbLTA, the first weakly-supervised method for dense Long-Term Action Anticipation (LTA) that uses only video transcripts (ordered action lists without timing) as supervision. The model employs a temporal alignment module to generate frame-level pseudo-labels, enriches video features via cross-modal attention with transcript semantics, and trains with a combination of CTC, CRF, and duration losses. Experiments on Breakfast, 50Salads, and EGTEA demonstrate competitive performance compared to fully supervised baselines, particularly on Breakfast and rare action classes.

### Strengths
1. **Novel Contribution**: This is the first work to address dense LTA using only transcript-level supervision, significantly reducing annotation cost and opening a new research direction. The paper clearly positions itself against prior fully- and semi-supervised methods.
2. **Comprehensive Methodology**: The proposed architecture integrates multiple innovative components—temporal alignment for pseudo-label generation, cross-modal attention for feature grounding, and a combination of alignment, segmentation, and anticipation losses (CTC, CRF, duration)—into a coherent framework tailored for weakly-supervised LTA.
3. **Rigorous Evaluation**: Extensive experiments on three standard benchmarks (Breakfast, 50Salads, EGTEA) show that TbLTA achieves competitive results, sometimes surpassing fully supervised methods on Breakfast (e.g., at 30% observation). The ablation studies systematically validate the importance of key components like the CTC loss and cross-attention.
4. **Clear Presentation**: The paper is well-structured, with a clear problem definition, detailed methodology, and thorough experimental setup. Figures 1 and 2 effectively illustrate the framework and training pipeline.

### Weaknesses
1. **Performance Limitations on Complex Datasets**: While competitive on Breakfast, results on 50Salads lag behind state-of-the-art supervised methods (e.g., ActFusion). The paper acknowledges that longer videos with denser action distributions and frequent transitions challenge the temporal alignment and duration estimation.
2. **Duration Prediction Remains Challenging**: The proposed affinity-based duration loss relies on learned priors from observed segments, which may not generalize well to unseen actions or highly variable durations. Qualitative results (Figure 3) show that duration estimation is a clear weak point.
3. **Dependence on Noisy Pseudo-Labels**: The model relies on pseudo-labels generated by an alignment module (ATBA). Errors in this alignment, especially during early training, can propagate and limit performance, as hinted by the need for a progressive training scheme.
4. **Architectural Complexity**: The integration of multiple components (alignment module, cross-attention, multiple loss terms) introduces training complexity and hyperparameter tuning overhead (e.g., balancing coefficients γ1, γ2, γ3). This may hinder reproducibility and practical adoption.
5. **Limited Evaluation on EGTEA**: Evaluation on EGTEA is restricted to verb prediction (not full verb-noun actions) and shows a performance gap on overall mAP compared to supervised methods, though the method excels on rare classes.

### Novelty & Significance
The paper presents a novel and significant contribution by being the first to demonstrate that dense long-term action anticipation can be performed effectively with only transcript-level supervision. This weak supervision paradigm substantially reduces annotation cost and increases scalability, aligning with ICLR’s emphasis on innovative and practical research. The work bridges the gap between weakly-supervised action segmentation and anticipation, showing that high-level semantic guidance can capture procedural regularities needed for long-horizon forecasting.

### Suggestions for Improvement
1. **Enhance Duration Modeling**: Investigate more robust duration prediction mechanisms, such as integrating external temporal priors, using probabilistic models (e.g., Gaussian processes), or leveraging language models to infer typical action durations from textual descriptions.
2. **Improve Pseudo-Label Robustness**: Explore iterative pseudo-label refinement strategies (e.g., self-training with confidence thresholds) or ensemble alignment methods to reduce noise and error propagation, especially for complex datasets like 50Salads.
3. **Simplify and Streamline the Architecture**: Consider reducing the number of loss components or unifying them into a more elegant objective. Provide clearer guidance on hyperparameter selection (e.g., loss weights) to aid reproducibility.
4. **Conduct Deeper Failure Analysis**: Include a detailed analysis of failure cases, particularly on 50Salads, to identify whether errors stem from alignment, duration prediction, or semantic misunderstanding. This would strengthen the paper’s diagnostic insights.
5. **Extend Evaluation**: Report results on the full verb-noun action labels for EGTEA and include additional benchmarks (e.g., Epic-Kitchens) to demonstrate broader applicability and robustness.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Strong weakly-supervised segmentation + simple anticipation baseline**: The paper lacks a baseline where a state-of-the-art weakly-supervised action segmentation method (e.g., ATBA) segments the observed portion, followed by a standard decoder (LSTM/Transformer) for anticipation. Without this, it is unclear whether the gains stem from the novel architecture or merely from using pseudo-labels.
2. **Ablation on temporal alignment methods**: The core reliance on ATBA for pseudo-label generation is not ablated against other alignment techniques (e.g., DTW, CTC). The performance may be highly sensitive to this choice, undermining claims of a general weakly-supervised framework.
3. **Feature robustness test**: Experiments only use I3D features. Testing with features from vision-language models (e.g., CLIP) is critical to assess whether the cross-modal attention module truly leverages semantic alignment or is dependent on a specific feature type.

### Deeper Analysis Needed (top 3-5 only)
1. **Pseudo-label quality analysis**: There is no quantitative evaluation of the pseudo-labels' accuracy (e.g., MoC/F1 against ground truth on a validation set). Since the entire training depends on these labels, their quality directly determines the credibility of the final anticipation results.
2. **Error analysis by action type and duration**: The paper notes duration prediction is challenging but does not break down errors by action frequency (rare vs. frequent) or duration variability. This analysis is essential to understand the method's limitations and whether it truly generalizes.
3. **Cross-attention effectiveness**: Beyond ablation scores, there is no analysis showing what the cross-attention learns. Quantitative measures (e.g., alignment between attention weights and pseudo-labels) and qualitative examples are needed to verify that the module provides meaningful grounding.

### Visualizations & Case Studies
1. **Pseudo-label vs. ground truth timelines**: Visualizations comparing generated pseudo-labels (for both observed and future intervals) with ground truth annotations would reveal alignment accuracy and error propagation into anticipation.
2. **Cross-attention heatmaps**: Showing which video frames attend to which transcript actions would demonstrate whether the module captures semantically relevant regions or learns spurious correlations.
3. **Failure case visualizations**: Displaying examples where anticipation fails (e.g., wrong action order, severe duration errors) with analysis would highlight the method's boundaries and inform future improvements.

### Obvious Next Steps
1. **Incorporate a more advanced duration model**: The duration prediction relies on simple class priors from observed segments. A learnable duration model conditioned on context should have been explored to improve future segment estimation.
2. **Comprehensive stochastic evaluation**: The paper mentions stochastic results in supplementary material but does not compare them thoroughly with state-of-the-art stochastic supervised methods. This is a missed opportunity to show the method's ability to model uncertainty.
3. **Scale to larger, more complex datasets**: To convincingly argue for scalability, experiments on a larger dataset (e.g., COIN) with longer and more diverse activities are necessary, not just the standard benchmarks.

# Final Consolidated Review
## Summary
This paper introduces TbLTA, the first weakly-supervised method for dense long-term action anticipation (LTA) that uses only video transcripts (ordered action lists without timing) as supervision. It integrates a temporal alignment module to generate frame-level pseudo-labels, cross-modal attention to ground video features with transcript semantics, and a combination of CTC, CRF, and duration losses. Experiments on Breakfast, 50Salads, and EGTEA demonstrate competitive performance with fully supervised methods, particularly on Breakfast and rare action classes.

## Strengths
- **Novel contribution**: First work to tackle dense LTA using only transcript-level supervision, significantly reducing annotation cost and opening a new research direction in scalable video understanding.
- **Comprehensive methodology**: The architecture coherently integrates multiple weakly-supervised techniques—temporal alignment, cross-modal attention, and a combination of alignment, segmentation, and anticipation losses—tailored for the LTA task.
- **Competitive empirical results**: On standard benchmarks, TbLTA achieves performance competitive with, and in some settings superior to, fully supervised methods (e.g., on Breakfast at 30% observation) and shows particular strength on rare action classes in EGTEA.

## Weaknesses
- **Inference pipeline clarity**: The paper does not clearly explain how the model operates at inference without the transcript. Specifically, it is ambiguous how the temporal alignment module and cross-modal attention are used (or not) during inference, creating a potential train-test mismatch and obscuring how the observed segment is segmented to inform anticipation.
- **Reliance on noisy pseudo-labels without quality analysis**: The model depends on pseudo-labels generated by the temporal alignment module, but there is no quantitative evaluation of their accuracy (e.g., frame-wise MoC/F1 on a validation set) or analysis of how alignment errors propagate to the anticipation decoder. This undermines understanding of the method's robustness.
- **Incomplete ablation study**: Ablations only cover the CTC loss and cross-attention, omitting the contribution of other key components (e.g., the CRF, duration loss, temporal alignment method, class tokens). Without these, it is unclear which components are essential for the reported performance.
- **Missing critical baseline**: A two-stage baseline—where a state-of-the-art weakly-supervised action segmentation method generates pseudo-labels for the full video, followed by a supervised LTA model trained on these pseudo-labels—is absent. This baseline is necessary to isolate the benefit of the proposed joint training framework versus simply using pseudo-labels as a substitute for ground truth.
- **Weak duration prediction**: The affinity-based duration loss relies on class-wise priors estimated from the observed segment's predictions, which may be noisy and not generalize well to future segments. Qualitative results confirm duration estimation remains challenging, and no ablation validates the loss's effectiveness.
- **Limited evaluation on EGTEA**: Evaluation is restricted to verb prediction, not the full verb-noun action labels, leaving the method's performance on fine-grained actions unexplored.
- **Stochastic results relegated to supplement**: The paper mentions stochastic results in the supplementary material but does not include them in the main text, missing an opportunity to fully demonstrate the method's ability to model uncertainty and compare with stochastic supervised methods.

## Nice-to-Haves
- More detailed error analysis by action type and duration variability to better understand failure modes and limitations.
- Visualizations comparing generated pseudo-labels to ground truth, cross-attention heatmaps, and failure cases to provide qualitative insights into model behavior.
- Extension to larger, more complex datasets (e.g., COIN) to demonstrate scalability and robustness beyond the standard benchmarks.
- Investigation of more advanced duration models that condition on context rather than relying solely on class-wise priors.
- Ablation on different temporal alignment methods (e.g., DTW, CTC) to assess sensitivity and generalizability of the pseudo-label generation.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Progressive training stability**: The three-stage training scheme with specific epoch counts is described without hyperparameter sensitivity analysis, but the scheme works and is not a core flaw; detailed justification is not required.
- **Architectural complexity**: While the model integrates many components and loss terms, this is a trade-off for performance and does not constitute a critical weakness for a research paper.
- **Dependence on pre-extracted I3D features**: This is standard practice in the field; the paper does not claim end-to-end learning from raw video.
- **Assumption of perfect transcripts**: The problem setting explicitly assumes transcripts are given; handling noisy or paraphrased transcripts is outside the paper's scope.
- **Linear narrative assumption**: The method is designed for procedural activities, which matches the benchmarks used; handling branching narratives is not required for the stated contribution.
- **Computational cost analysis**: Not provided, but such analysis is not a core requirement for acceptance at ICLR.

## Novel Insights
The paper demonstrates that high-level semantic supervision from transcripts can effectively guide long-term action anticipation, capturing procedural regularities without dense frame-level annotations. It shows that weak supervision can be competitive with full supervision in certain settings, challenging the notion that dense labels are indispensable for dense anticipation. The work bridges weakly-supervised action segmentation and anticipation, suggesting that narrative temporal structure provides a robust signal for forecasting future actions.

## Suggestions
- Clarify the inference pipeline in the methodology section, explicitly stating how the model segments the observed interval without the transcript and whether the cross-attention layer is used (or skipped) at inference.
- Include a quantitative analysis of pseudo-label quality (e.g., frame-wise accuracy on a validation set) and discuss how alignment errors might affect anticipation performance.
- Expand the ablation study to include the CRF, duration loss, temporal alignment module, and class tokens to better understand each component's contribution.
- Add the two-stage baseline (weakly-supervised TAS + supervised LTA) to the experimental comparison to isolate the benefit of joint training.
- Integrate a summary of the stochastic results in the main text, including comparisons with stochastic supervised methods, to showcase the method's ability to model multiple plausible futures.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 4.0]
Average score: 3.0
Binary outcome: Reject
