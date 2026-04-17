# When Does Multimodality Lead to Better Time Series Forecasting?

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Recently, there has been growing interest in incorporating textual information into foundation models for time series forecasting. However, it remains unclear whether and under what conditions such multimodal integration consistently yields gains. We systematically investigate these questions across a diverse benchmark of 16 forecasting tasks spanning 7 domains, including health, environment, and economics. We evaluate two popular multimodal forecasting paradigms: aligning-based methods, which align time series and text representations; and prompting-based methods, which directly prompt large language models for forecasting. Our findings reveal that the benefits of multimodality are highly condition-dependent. While we confirm reported gains in some settings, these improvements are not universal across datasets or models. To move beyond empirical observations, we disentangle the effects of model architectural properties and data characteristics, drawing data-agnostic insights that generalize across domains. Our findings highlight that on the modeling side, incorporating text information is most helpful given (1) high-capacity text models, (2) comparatively weaker time series models, and (3) appropriate aligning strategies. On the data side, performance gains are more likely when (4) sufficient training data is available and (5) the text offers complementary predictive signal beyond what is already captured from the time series alone. Our study offers a rigorous, quantitative foundation for understanding when multimodality can be expected to aid forecasting tasks, and reveals that its benefits are neither universal nor always aligned with intuition.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates whether and under what conditions the textual modality benefits time series forecasting. Aligning-based methods and prompting-based methods are summarized for experiments. Experimental results reveal that the benefits of textual modality are highly condition-dependent for time series forecasting.

### Strengths
1. This paper explores the contribution of textual modality and large language models (LLMs) to time series forecasting, with particular attention to the effect of model size.

2. The experimental evaluation is extensive and well-organized.

3. The paper covers a wide range of domains, enhancing the comprehensiveness of the analysis.

### Weaknesses
1. The existing work CM2TS [1] has already investigated similar questions regarding cross-modality modeling for time series. However, this paper does not provide a proper citation or discussion to clarify how its contributions differ from or extend CM2TS.

2. Only text/language is involved as an external modality. Thus, this study is a dual-modality or cross-modality analysis, rather than a multimodality analysis. Please ensure the authenticity.

3. Some experimental results (e.g., Sections 4.2 and 4.5) are interesting but appear to depend heavily on hyperparameter configurations of methods. As such, the observed improvements may correspond to local optima. It would be better to combine the theoretical proof and the absolutely global optimal experimental results to verify your assumption.

4. The code and implementation details are not available, which limits reproducibility and independent validation of the findings.

[1] Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era, IJCAI 2025.

### Questions
1. It is difficult to understand Table 1. How are PatchTST, DLinear, and Chronos aligned with LLMs? Which alignment strategies were applied in each case?

2. For each LLM-based method, have you fine-tuned the best hyperparameters individually, or did you use a unified setting across models? Please clarify the procedure.

3. The paper discusses the benefit of multimodality, but it seems that CM2TS [1] has already explored a similar topic. Could the authors clarify how this work differs from or extends the contributions of CM2TS?

[1] Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era, IJCAI 2025.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work, the authors evaluate various existing models for multimodal time series (MMTS) forecasting across 16 datasets under two paradigms: alignment-based and prompt-based methods. The paper provides detailed analyses that can inform the broader understanding of this research field. However, the most critical aspect—namely, the definition of “multimodal time series” -- is not clearly explained.

### Strengths
- The authors evaluate multiple existing methods using two pipelines (alignment-based and prompt-based) across 16 diverse datasets.
- The paper provides a comprehensive analysis of different modeling strategies and offers detailed experimental results and insights.

### Weaknesses
### 1. **The Definition of “Multimodal” Time Series**

I am very concerned about the formulation of **multimodal** time series in this paper.

In Lines 034–044, the authors categorize six “MMTS” methods into two types: (I) alignment-based and (II) prompt-based. 

However, I believe that **multimodal learning inherently implies semantic alignment** between modalities. For instance, I think it difficult to perceive any semantic alignment between numerical time series data and a textual statement such as *“Tomorrow there will be a meeting between the US and Canada.”* These two sources do not describe the same underlying content; rather, they convey fundamentally different types of information. Hence, such data and methods are better described as **multi-source** or **multi-factor** forecasting approaches, rather than traditional multimodal learning.

**Speech–text** is a good example to illustrate what true semantic alignment means. The alignment between modalities presupposes that they represent the *same underlying content* (e.g., the same utterance in two modalities) rather than *different kinds of information*. In contrast, most time series language models, e.g., [4],  do not share this semantic correspondence. 

For multimodal time series (MMTS), recent works [1, 2, 3] have explored more semantically grounded formulations by transforming raw time series into frequency, visual, or textual representations—e.g., spectrograms or pattern images—and learning alignment in these shared spaces. These directions capture the essence of multimodality much better.

If the authors intend to redefine or extend the concept of MMTS, I strongly suggest that they **explicitly discuss what constitutes a multimodal time series and what does not**. As it stands, I disagree with the paper’s implicit definition of “multimodal time series,” which appears more closely related to multi-source or multi-factor data integration rather than genuine multimodal learning.


[1] Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting

[2] Teaching Time Series to See and Speak: Forecasting with Aligned Visual and Textual Perspectives

[3] GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images

[4] One Fits All: Power General Time Series Analysis by Pretrained LM

### 2. **Benchmark**

Considering the conceptual gap between conventional multimodal learning and the “MMTS” formulation in this paper, I believe it is necessary for the authors to provide a more comprehensive clarification of the semantic relationship between time series and textual modalities in the chosen datasets. Moreover, recent studies have demonstrated that vision-language models (VLMs) can effectively comprehend and reason over time-series data, highlighting the importance of discussing these conceptual distinctions in greater depth.

### Questions
For my question, please refer to the “Weaknesses” section.

Additionally, although the authors provide extensive experimental details in Appendices A, B, and C, following ICLR’s reproducibility guidelines, it is recommended that the authors include a dedicated “Reproducibility Statement” section before the References.

### Soundness
2

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
4

### Summary
This work systematically addresses the fundamental question of when and why multimodality improves forecasting, from both model and data perspectives. By evaluating 16 benchmarks and two key paradigms (alignment-based and prompting-based), the study offers essential insights into choosing encoders and fusion strategies for building powerful multimodal forecasting models.

### Strengths
1. The research is underpinned by a clearly defined and compelling motivation.

2.  The work itself is pioneering and addresses a problem of considerable importance.

3. The article is exceptionally well-written, and the experimental section is systematically conducted, with results presented in a clear and convincing manner.

### Weaknesses
Please refer to the **Questions**.

### Questions
The paper validates through synthetic and real-world datasets: MMTS is effective only when the text provides complementary predictive signals not contained in the time series. However, this conclusion is based on static evaluation scenarios (i.e., the "complementarity" between text and time series in experiments is fixed, such as whether the text contains trend shift information in synthetic data being a predefined condition). In real-world scenarios, the complementarity of text often changes dynamically (for example, in economic time series forecasting, a piece of news may contain complementary information before a policy is released but becomes redundant after the policy is implemented; in medical monitoring, the complementarity of clinical notes dynamically evolves with the patient's condition stability). Existing MMTS models all adopt fixed fusion strategies (such as fixed late fusion or early fusion) and are unable to determine in real time whether the text possesses complementarity and adjust the fusion intensity. Therefore, how can we design an MMTS model that can dynamically quantify the complementarity intensity between text and time series (e.g., based on metrics such as novelty in temporal patterns or semantic relevance of text) and adaptively switch fusion strategies (e.g., increasing text weight when complementarity is high, or reverting to unimodal mode when complementarity is low)? Can such a dynamic mechanism break through the performance ceiling of existing static fusion models in real-world scenarios (such as real-time economic forecasting or intensive care time series monitoring)?

### Soundness
3

### Presentation
3

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
The paper "When Does Multimodality Lead to Better Time Series Forecasting?" conducts a systematic investigation into the effectiveness of multimodal time series (MMTS) forecasting by integrating textual information. The authors evaluate two dominant paradigms: alignment-based methods (fusing time series and text representations) and prompting-based methods (directly using LLMs for forecasting). Through a comprehensive benchmark spanning 16 datasets across 7 domains (e.g., health, economics), the study reveals that multimodal improvements are highly conditional and not universal. Key contributions include:

Demonstrating that MMTS methods do not consistently outperform unimodal baselines, challenging common assumptions.

Providing insights into how model capacity (e.g., text encoder size, time series model strength) and data characteristics (e.g., training data size, text complementarity) influence performance.

Offering data-agnostic guidelines via controlled experiments, such as synthetic data analyses, to generalize findings beyond specific benchmarks.

The paper emphasizes that multimodality is most beneficial when text provides complementary signals not captured by time series alone, and it encourages more cautious, data-driven approaches in future MMTS research.

### Strengths
Quality: The experimental design is thorough, covering 16 datasets, multiple model families (e.g., Chronos, BERT, LLMs), and diverse alignment strategies. The synthetic data approach is particularly strong for isolating key variables.

Clarity: The writing is accessible, with clear explanations of methods and results. Visualizations (e.g., scatter plots showing performance trends) effectively communicate complex findings.

Significance: The paper provides actionable guidelines for researchers and practitioners, potentially reducing wasted effort on ineffective multimodal integrations. Its focus on data characteristics beyond model architecture broadens the impact.

### Weaknesses
The study is limited to text and time series; excluding other modalities (e.g., images in retail forecasting) may reduce generalizability to broader multimodal settings.

While datasets are diverse, they may not capture all real-world challenges (e.g., ultra-long sequences or low-resource domains). Including more extreme cases could strengthen the conclusions.

The evaluation of prompting-based methods relies on current LLMs (e.g., GPT-4, Claude), which evolve rapidly; however, this is mitigated by testing multiple models and versions.


Some recent work addresses these problems by converting text into code or by involving human intervention to improve alignment (e.g., https://arxiv.org/abs/2505.15354, https://arxiv.org/pdf/2506.13705). We encourage future studies to demonstrate and benchmark such approaches as well.

### Questions
How might the inclusion of other modalities (e.g., images or audio) affect the conclusions? Could the guidelines be extended to multimodal settings beyond text?

Could the findings apply to streaming or online learning scenarios where data arrives incrementally?

Based on the results, are there specific domain invariants (e.g., healthcare vs. finance) where multimodality is consistently beneficial or ineffective?

Some recent work addresses these problems by converting text into code or by involving human intervention to improve alignment (e.g., https://arxiv.org/abs/2505.15354, https://arxiv.org/pdf/2506.13705). Please also include these kinds of methods—and any others that have been overlooked—to ensure a more complete and rigorous evaluation. I will raise additional points as further work and study are incorporated.

### Soundness
4

### Presentation
3

### Contribution
3
