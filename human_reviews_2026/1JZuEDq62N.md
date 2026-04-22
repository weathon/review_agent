# Probing Rotary Position Embeddings through Frequency Entropy

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
Rotary Position Embeddings (RoPE) are widely used in Transformers to encode positional information in token representations, yet the internal frequency structure of RoPE remains poorly understood. Previous studies have reported conflicting findings on the roles of high- and low-frequency dimensions, offering empirical observations but no unifying explanation. In this paper, we present a systematic framework that bridges these disparate results. We introduce Frequency Entropy (FE), a metric that quantifies the effective utilization of each RoPE frequency dimension, and we provide an analysis of how RoPE’s sinusoidal components contribute to model representations on a per-dimension basis. Based on an analysis of the Llama-4 model, which incorporates both RoPE and NoPE layers, we find that the periodicity captured by FE appears in RoPE layers but not in NoPE layers. Furthermore, FE identifies dimensions in which energy concentrates under RoPE. These characteristics are observed across the spectrum rather than being confined to specific dimensions. Moreover, attenuating extreme-entropy dimensions at inference yields downstream accuracy that is statistically indistinguishable from the baseline, with modest perplexity improvements on average, suggesting that such dimensions are often redundant. Overall, FE provides a simple, general diagnostic for RoPE with implications for analysis and design.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
1. SpectrumFE aligns with distinct band-limited patterns in l2-norm maps.
2. SequenceFE is sensitive to periodic structures along the token axis.
3. From SpectrumFE, the shallow NoPE layer (layer 3) exhibits more frequency bands than the corresponding RoPE layer. Conversely, from SequenceFE, the NoPE layer lacks rotating pairs, indicating an absence of clear periodic oscillations.
4. NoPE may therefore suppress the periodic structures characteristic of RoPE while amplifying specific frequency bands.
5. Dimensions with τ < 0.2 in SpectrumFE appear to contribute positively to model performance, suggesting that these frequency bands are important functional components.
6. Dimensions with τ > 0.4 may be redundant or unnecessary for model performance.
7. Together, these results suggest that dimensions where SpectrumFE becomes an outlier and where SequenceFE decreases may not contribute meaningfully to performance and could be removed without loss.

### Strengths
* The authors introduce two novel metrics, SpectrumFE and SequenceFE, to analyze the frequency characteristics of RoPE embeddings.
* They identify a variety of interesting phenomena associated with these metrics.
* Ablation studies are performed to assess the effects of dampening or removing specific frequency bands.
* The paper attempts to reconcile previously conflicting findings regarding frequency importance in RoPE.

### Weaknesses
* The practical impact of the findings appears limited. The only direct link to model performance is provided by the ablation study (Section 5.3), which shows that removing dimensions where SpectrumFE becomes an outlier or where SequenceFE decreases does not degrade performance. This limits the applicability and relevance of the results for researchers and practitioners.
* The authors’ explanation of prior conflicting results is anecdotal rather than empirical. No targeted experiments are provided to demonstrate that earlier discrepancies arose from confounding factors. The claim that their findings “resolve the confusion of previous research” lacks sufficient supporting evidence.
* It remains unclear what SpectrumFE and SequenceFE truly measure. The paper states only that SpectrumFE aligns with band-limited l2 patterns and SequenceFE captures periodicity along the token axis-insufficient for understanding their mechanistic or theoretical grounding.
* Overall, the study focuses primarily on how these metrics interact with the model, rather than revealing new, interpretable mechanisms that explain how RoPE or LLMs function at a deeper level.

### Questions
1. Could the authors provide greater intuition for what SpectrumFE and SequenceFE actually measure? How do these metrics relate to underlying model mechanics? Examples or analogies could greatly enhance interpretability.
2. How might these findings be useful for researchers and practitioners? Could the authors elaborate on how the results should inform future research directions or guide the design of LLM architectures?
3. The author write: "We expect FE to function as a practical, model-independent diagnostic tool for position coding." Could the authors expand on this? What decisons or actions would a practitioner take based on FE measurements?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Frequency Entropy (FE), a novel metric to analyze how different frequency dimensions in Rotary Position Embeddings (RoPE) are utilized in Transformers. Through systematic analysis of models like Llama-4, the authors show that FE identifies two key structures: frequency bands (via SpectrumFE) and periodic dimensions (via SequenceFE). They find that while frequency bands are essential for model performance, periodic dimensions are often redundant and can be attenuated without harming downstream task accuracy, offering a unified explanation for previously conflicting findings and providing a practical diagnostic tool for positional encoding design.

### Strengths
1. Novel Metric and Framework. Introduces Frequency Entropy (FE)—a new quantitative metric to analyze RoPE's frequency-wise utilization, unifying previously conflicting observations about high- vs. low-frequency roles. The dual metrics (SpectrumFE and SequenceFE) offer a systematic, model-agnostic diagnostic tool.
2. Rigorous and Multi-Model Validation.Strong empirical foundation with experiments across multiple architectures (Llama-4, Llama-3, Gemma-2, Qwen-3), probing both queries and keys. Combines entropy analysis with intervention (Weighted RoPE) to validate functional importance of frequency bands and periodicity.
3. Well-Structured and Accessible. Clear problem framing, method description, and visualizations (entropy scatter plots, norm heatmaps) make complex frequency dynamics interpretable. Appendices extend analysis to keys and all layers, enhancing reproducibility.
4. Resolves Prior Conflicts and Informs Design. Resolves contradictions in prior work by showing frequency bands (low SpectrumFE) are essential, while periodic dimensions (low SequenceFE) are often redundant. Offers practical implications for model efficiency and positional encoding design.

### Weaknesses
1. Experiments are conducted primarily on the Wikitext-103 dataset. To fully support the claim that FE is a "general diagnostic," the analysis should include diverse text datasets to verify whether the findings generalize beyond specific tasks.
2. The paper establishes correlations between FE values and model behaviors but does not rigorously prove causality. For instance, while attenuating low-SequenceFE dimensions doesn't hurt performance, it remains unclear if this is because they are truly redundant or if the model compensates via other mechanisms. A more controlled ablation would strengthen the causal claim.
3. The paper studies Llama-4's iRoPE but does not disentangle the separate effects of interleaved NoPE layers and RoPE frequency scaling. An ablation study comparing would clarify which component drives the observed entropy shifts and performance outcomes.

### Questions
1. The sequence lengths in the paper are mostly set to 4096. Have you tried any other lengths, or does this length yield the most prominent effect?
2. An in-depth analysis of the FE features (such as Figure 2) is mainly conducted for "head 0". Is it necessary to systematically extract samples from different heads at different layers to observe whether the conclusions are consistent?
3. The downstream evaluation relies on benchmarks (e.g., HellaSwag, MMLU) that primarily evaluate general knowledge and reasoning. Could the "no significant difference" conclusion be undermined on other tasks specifically designed to assess long-context understanding?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Frequency Entropy as a metric to study the utilization of RoPE frequency dimensions. The authors perform experiments on Llama-4 to compare the periodicity of RoPE and NoPE layers. The authors introduce Spectrum Frequency Entropy and Sequence Frequency Entropy to evaluate both which frequency components are present and how periodic the energy fluctuations are in each dimension.

### Strengths
- The methodology is interesting and provides a quantitative way to study RoPE utilization, as compared to prior more qualitative works
- The figures provide nice visualizations of the analysis and results
- The weighted RoPE experiments provide interesting insights on functional relevance, showing that suppressing low-Spectrum-FE dimensions worsens performance.

### Weaknesses
- The experiments are done on LLama-4 which interleaves RoPE and NoPE layers, so comparisons between RoPE and NoPE are not layer-matched. It would be interesting to see even at a smaller scale the differences between FE for a model trained with and without RoPE at a specific layer.
- The takeaways to me are not clear. In particular:
- The SpectrumFE results for RoPE and NoPE seem quite similar at later layers. It seems like the main difference is at the earliest layers. Do you have insights on the impact of this early-layer difference? Would a model with no RoPE layers exhibit the same late-layer behavior?
- What are the practical takeaways of the Weighted RoPE intervention?

### Questions
See weaknesses. In addition, prior work mentions that ablating low-frequency RoPE dimensions impacts long-context performance. I'd be interested to see the Weighted RoPE intervention in a similar setting to see if you could provide additional insights to long-context.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides an analysis of RoPE in terms of Frequency Entropy (FE), or the Shannon entropy of the power spectrum. Two measures are used in this analysis: Spectrum FE which captures the active components and Sequence FE which quantifies the regularity of energy in each dimension. These quantities are normalized to provide scale-free measures. An analysis of RoPE in Llama-4 is presented. This analysis shows that RoPE and NoPE layers behave differently, but that earlier NoPE layers show bands under Spectrum FE similar to RoPE. This effect disappears at deeper layers. The paper also describes experiments to test the redundancy of RoPE dimensions with outlier FE by down-weighting them. Results show that low entropy dimensions (frequency bands) are important to model perplexity, though downstream tasks show similar performance between standard and weighted models.

### Strengths
The paper provides an interesting, theoretically grounded analysis of RoPE embeddings. It provides new insights into the contribution of RoPE on model performance, specifically distinguishing between periodic signals and high energy bands. The explanations are clear and the experimental evidence helps to solidify the analysis.

### Weaknesses
The choice to study only Llama-4 reduces the impact and generality of this paper. It is noted that NoPE layers in this model may attenuate periodic structures and emphasize frequency bands. If I understand, Spectrum and Sequence FE analysis has not been applied to a RoPE-only model like Gemma 7B or Llama-3. This would help to clarify the effect of NoPE in Llama 4.

### Questions
Did you consider attenuating outlying low or high-frequency components only? This could help strengthen the case that these specific component regions are not responsible for model performance.

### Soundness
3

### Presentation
4

### Contribution
3
