# CC-Time: Cross-Model and Cross-Modality Time Series Forecasting

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 6, 2

## Abstract
With the success of pre-trained language models (PLMs) in various application fields beyond natural language processing, language models have raised emerging attention in the field of time series forecasting (TSF) and have shown great prospects. However, current PLM-based TSF methods still fail to achieve satisfactory prediction accuracy matching the strong sequential modeling power of language models. To address this issue, we propose Cross-Model and Cross-Modality Learning with PLMs for time series forecasting (CC-Time). We explore the potential of PLMs for time series forecasting from two aspects: 1) what time series features could be modeled by PLMs, and 2) whether relying solely on PLMs is sufficient for building time series models. In the first aspect, CC-Time incorporates cross-modality learning to model temporal dependency and channel correlations in the language model from both time series sequences and their corresponding text descriptions. In the second aspect, CC-Time further proposes the cross-model fusion block to adaptively integrate knowledge from the PLMs and time series model to form a more comprehensive modeling of time series patterns. Extensive experiments on nine real-world datasets demonstrate that CC-Time achieves state-of-the-art prediction accuracy in both full-data training and few-shot learning situations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes CC-Time, a Cross-Model and Cross-Modality framework for time series forecasting that integrates the strengths of pre-trained language models and time-series-specific models. Traditional forecasting approaches either focus on numerical modeling of temporal patterns or exploit PLMs for sequence understanding, but each alone is limited. CC-Time bridges this gap by combining both paradigms and leveraging semantic knowledge from text to model complex temporal and channel correlations.The framework contains a PLM branch, a time-series branch, and a cross-model fusion module. The PLM branch introduces a cross-modality modeling mechanism, combining time-series data with automatically generated channel text descriptions to help PLMs capture both temporal dependencies and semantic relationships among channels. The time-series branch, built on a transformer structure, models fine-grained numerical temporal dynamics. The cross-model fusion block adaptively integrates multi-level features from both branches using attention and gating mechanisms, yielding a unified representation that captures both semantic and quantitative aspects of time series data.
Experiments conducted on real-world datasets from diverse domains such as energy, weather, and traffic show that CC-Time achieves state-of-the-art prediction accuracy under both full-data and few-shot settings.

### Strengths
This paper is original in proposing a cross-model and cross-modality framework that unites pre-trained language models with time-series-specific architectures for forecasting. The idea of using automatically generated channel text descriptions to inject semantic knowledge into time-series modeling is interesting. The paper is well-structured. Its comprehensive experiments across multiple datasets and settings demonstrates strong generalization and robustness.  In terms of clarity, the paper is clearly written, well-organized, and supported by informative figures that make the architecture and motivation easy to follow. In all, this paper opens new directions for multimodal and cross-domain modeling in time-series research.

### Weaknesses
* The paper does not provide a detailed hyperparameter sensitivity analysis, leaving some uncertainty about the robustness and stability of the proposed framework across different parameter settings.
* The paper demonstrates that CC-Time captures richer correlations, but remains a little bit unclear how these align with real-world semantics or physical dependencies among channels.

### Questions
* What is the computational overhead of the two branches？

* Since the model relies on automatically generated channel text descriptions, how does the quality of these descriptions affect performance?

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
4

### Summary
This paper introduces CC-Time, a dual-branch architecture that integrates large language models (LLMs) and time-series-specific models for time series forecasting (TSF). The model addresses two key questions: (1) how to enable pre-trained language models (PLMs) to jointly capture temporal dependencies and variable correlations, and (2) how to leverage the complementary strengths of LLMs and traditional time-series models through adaptive fusion.To this end, the authors propose Cross-Modality Learning to enhance correlation modeling between numerical and textual representations, and a Cross-Level Fusion (CLF) block to integrate features from different representational levels. Experiments on nine real-world datasets show consistent performance improvements in both full-data and few-shot settings.

### Strengths
1. Novel integration of PLMs and TS models: The paper presents a thoughtful framework that unites semantic (LLM-based) and numerical (Transformer-based) modeling. The design of the CLF block reflects careful consideration of cross-representational learning in time series forecasting.

2. Innovative cross-modality correlation modeling: Incorporating text-based variable descriptions and a correlation extractor allows the model to capture both global and local dependencies from semantic and numeric perspectives.

3. Strong empirical validation: Experiments cover a broad range of datasets and forecasting horizons, consistently showing superior results over LLM-based and TS-specific baselines. Few-shot evaluations further demonstrate the potential of LLMs for low-data forecasting.

4. Comprehensive component analysis: The paper provides ablation studies on cross-level fusion and cross-modality correlation modules, along with model-depth and freezing analyses that strengthen the empirical soundness of the claims.

### Weaknesses
1. Lack of explicit modality alignment between time-series embeddings and PLM semantic space: The paper directly feeds time-series embeddings into the pre-trained PLM without introducing any explicit alignment constraint between the numerical and linguistic modalities. This raises concerns about whether the frozen PLM can effectively interpret unaligned numeric encodings, especially since no contrastive or projection-based objective is applied to bridge the representational gap. As a result, the semantic structure within the PLM may not correspond to the statistical dynamics of the time-series features, potentially limiting the PLM’s contribution during fusion.

2. Limited interpretability of the correlation modeling process: While the paper proposes both a correlation extractor and PLM-based correlation layers to capture global and local dependencies across variables, the modeling results are not visually or quantitatively analyzed. 

3. Insufficient ablation to isolate the PLM branch’s contribution: The proposed Cross-Model Fusion integrates semantic correlations from the PLM with numerical representations from the time-series model. However, the paper lacks an ablation experiment isolating the PLM branch’s effect, which is essential to determine whether the semantic representations learned by the PLM genuinely enhance forecasting performance. Without such analysis, it is unclear whether the observed gains primarily come from the PLM, the time-series backbone, or their interaction. This omission weakens the causal interpretability of the architecture’s design claims.

### Questions
1. Would it be beneficial to introduce an explicit alignment objective (e.g., contrastive loss or learned projection) to map time-series embeddings into the PLM’s semantic space? If the PLM remains largely frozen, how does it meaningfully process such unaligned numerical representations?

2. Can the authors provide visualization or quantitative analysis (e.g., inter-channel correlation maps or attention weight distributions) to validate the correctness and interpretability of the extracted correlations?

3. Could the authors include a control experiment where the PLM branch is removed or replaced with a lightweight MLP to confirm that the semantic correlations extracted by the PLM provide measurable improvements to the TS branch’s predictive accuracy?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper works on time series forecasting (TSF) and presents CC-Time, a dual-branch framework integrating LLM and traditional TSF models. CC-Time addresses two core questions: which time series features PLMs can model, and whether PLMs alone suffice for TSF.
CC-Time adopts cross-modality learning, which combines time series and text (that describe channels), to capture temporal dependencies and channel correlations. Further, it introduces cross-model fusion (CMF) block to adaptively integrate knowledge from both branches. Extensive experiments have been conducted to validate the effectiveness of proposed modules.

### Strengths
1. It's reasonable to incorporate the ability of LLMs into traditional TSF models in a multimodal manner.
2. It's interesting to use ChatGPT to describe each channel, making the essence and functionality of each channel more clear, which may help to better model channel-wise correlations and improve interpretability.
3. The CMF block seems to be novel and reasonable.
4. Extensive experiments have been conducted to validate the effectiveness of modules. Particularly, this paper works on full-data and few-shot settings to provide robust evaluation.

### Weaknesses
1. A discussion between CC-Time and existing multimodal TSF methods (that also use both time series and textual data) is strongly recommended, which would make the contribution of this work more prominent.
- What are the differences between the constructed textual input, in terms of both method and motivation.
- Prompt length.
- Why such textual input can make your multimodal data fusion unique (compared to existing methods like Time-LLM, TimeCMA).
2. The computational cost of each module (particularly the attention-related modules), and efficiency analysis are recommended.
3. In Table 3 and Appendix F.1, the reasons why introducing larger LLMs cannot further boost the performance of TSF is not clear.
4. This paper captures channel-wise correlations near the input end, and TimeCMA puts similar-purpose module after data fusion (i.e., near the output end), an experiment (conducted under the same conditions) studying the positions of channel-wise correlation capturing is missing.

### Questions
N.A.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CC-Time, a framework for time series forecasting that attempts to leverage Pre-trained Language Models (PLMs). The stated contributions are: 1) Cross-Modality learning, using time series data and corresponding textual descriptions (semantic and statistical) to model channel correlations; and 2) Cross-Model fusion, via a CMF Block, to integrate features from the PLM branch and a dedicated Time Series (TS) branch.

### Strengths
1. The paper attempts a novel cross-modal fusion approach, and the idea of using PLMs to process channel correlations is explored. The design of the Cross-Model Fusion (CMF) Block is architecturally complex, utilizing multiple attention mechanisms to integrate information from the two heterogeneous branches.

2. Given the model's complexity, the authors have conducted numerous ablation studies. The experiments in Figure 3, Figure 9, and Appendix E attempt to demonstrate the necessity of the model's key components (though, as noted in the weaknesses, key ambiguities remain).

3. The model reports SOTA or competitive results on multiple datasets. The inclusion of comparisons against time series Foundation Models in a few-shot setting (Section 4.2 and Table 11) is a relevant experimental point, but the validity of these results is questionable given the methodological flaws.

### Weaknesses
1. **(Most Serious Issue) Dependency on Text Source and Generalizability**: The paper's primary methodological flaw is its dependency on an external LLM. While using an LLM to auto-generate text (Appendix A) solves the problem of missing text modalities in existing datasets, it builds a part of the model's performance on an uncontrolled, external black-box tool.

   * **Concerns about Text Quality**: We are concerned about the quality of the LLM-generated text.
   * **Concerns about Specific Datasets**: For a dataset like **Traffic** (with 862 channels), where channels represent sensors at different locations, can an LLM generate meaningful and **differentiated** descriptions for all 862 sensors? If the LLM just produces 862 copies of "This is a traffic sensor," the cross-modal innovation becomes meaningless.
   * **Question**: Could the authors provide examples of the (LLM-generated) text descriptions for several different channels from the **Traffic** dataset in their rebuttal?
   * **Motivation**: Overall, the motivation behind this article is somewhat strange. It seems that the text was added simply to introduce multimodality, and the quality of such text information is questionable.


2. **Source of Performance Gain: Semantics vs. Statistics**: Closely related to point 1, it is a reasonable inference that the performance gains may come more from the **"Statistical Information"** in the text description, rather than the "Semantic Description".

   * For datasets like Traffic, if (as we suspect) the "semantic descriptions" for different channels are highly similar or generic, the primary source of information for the PLM branch to differentiate channels becomes the "statistical information".
   * If this is the case, the novelty of this paper would be severely diminished, reducing the contribution to "a method for fusing statistical priors as auxiliary features into a time series model," rather than cross-modal semantic fusion.

3. **Ambiguous Ablation for Weakness #2**: **Appendix F.3 (Figure 9)** is critical for clarifying point 2, but its description is ambiguous.

   * **Question**: Can the authors explicitly confirm whether the "w/o Text" condition refers to: (A) removing *both* the semantic description and the statistical information, or (B) removing *only* the semantic description but *retaining* the statistical information?
   * This ambiguity also applies to the other experiments in that figure ("Add Noise", "Random Text"): are these interventions applied only to the semantic portion, or to both? Clarifying this is essential to evaluate the true contribution of "semantics".

4. **Necessity of the PLM Auxiliary Loss**: The model only uses the output of the TS branch during inference (Section 3.4), meaning the PLM loss ($\\mathcal{L}\_{plm}$) primarily functions as an auxiliary training objective.

   * \*\*Appendix F.2 (Figure 8)\*\*tests the loss weight $\\lambda$ and finds $\\lambda=0.6$ to be optimal, which does suggest the loss is useful.
   * **Question**: However, to clearly prove its "necessity," the most critical ablation would be the result for $\\lambda = 0$ (i.e., completely removing this loss). Can the authors provide the experimental data for $\\lambda = 0$?

5. **Weak Conclusion from CKA Analysis**: The CKA analysis in **Figure 5** is interesting, but the conclusion drawn is unconvincing.

   * The authors find that the CKA value of CC-Time is intermediate between the two model classes (PLM-based and TS-specific) and therefrom infer that the model captures "appropriate" complex features.
   * An intermediate CKA value is somewhat "inevitable" as the model is, by design, a hybrid (or average) of the two feature types. This result is not surprising.
   * **Question**: The authors fail to provide a convincing analysis of the causal link between an intermediate CKA value and low error, beyond simple correlation. Why is being "in the middle" necessarily "appropriate" or "superior"? The current analysis reads more as a phenomenon-observation rather than a deep insight.

6. **Misleading Terminology ("Cross-Model")**: The term "Cross-Model" is potentially misleading. In the community, this often implies the integration or interaction of multiple independent models (e.g., multiple expert PLMs). However, the paper's architecture involves only *one* PLM and *one* TS-specific model. Therefore, the fusion is more accurately a "Cross-Paradigm" (PLM vs. TS) fusion, not a "Cross-Model" fusion in the plural sense. The term inflates the architectural complexity.

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2
