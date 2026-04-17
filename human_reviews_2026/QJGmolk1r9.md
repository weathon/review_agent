# LLM-based Online Time Series Forecasting with Frequency-driven Pattern Recognition

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Online Time Series Forecasting (OTSF) task has been consistently studied due to its practicality in multiple domains. Considering the sequential and evolving nature of time series, OTSF models must be robust to distribution shifts and possess long-term adaptability for practical scenarios. However, existing research falls short due to lack of explicit handling of time series patterns and limitations of memory buffer-based retrieval strategies. In this paper, we propose a novel LLM-based online time series forecaster, called LLM4OT, which excels not only in continuous distribution shifts, but also in extended online scenarios. Our main idea can be summarized in two points: (1) By representing time series as a combination of frequency bases, and encoding the knowledge of each basis into prompts that guide the data distribution, our model can effectively adapt to unobserved patterns. (2) By collaboratively employing pretrained LLM with time series backbone, we enhance the model’s adaptation to data-scarce online scenarios. Additionally, we provide text-based descriptions that the LLM can easily understand, enriching the sparse data and maximizing the LLM’s adapting ability without requiring training. Our extensive
experiments on various real-world datasets demonstrate superiority and practicality of LLM4OT in various scenarios, including cross-dataset scenarios that maximize distribution shifts and scenarios with an extended online phase. Our code is available at https://anonymous.4open.science/r/LLM4OTSF-38FE/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel LLM-based framework LLM4OT for online time series forecasting (OTSF) that integrates frequency-domain pattern learning and language-driven adaptation. The framework represents time series as combinations of frequency bases to capture underlying temporal patterns, incorporates a pre-trained LLM aligned with a time-series backbone network, and leverages text-based recent pattern descriptions to enhance adaptation under data-scarce online scenarios. The proposed framework is evaluated on multiple real-world datasets across different domains against three categories of baselines—static forecasting models, LLM-based models, and online forecasting methods. Experiments, including ablation studies, cross-dataset robustness tests, and extended online phase analyses, comprehensively demonstrate that LLM4OT achieves superior adaptability and robustness under continuous distribution shifts.

### Strengths
(a) The paper presents a novel formulation of online time series forecasting (OTSF) by integrating large language models (LLMs) with frequency-domain pattern recognition. The idea of decomposing time series into frequency bases and encoding these into prompts offers a creative way to represent evolving temporal dynamics beyond traditional buffer-based or dual-stream OTSF designs.

(b) The paper also introduces text-based recent pattern descriptions, allowing pre-trained LLMs to exploit semantic priors and adapt effectively to data-scarce online scenarios without additional training. The combination of frequency-based pattern embeddings, aligned time-series representations, and text-based recent pattern descriptions provides a new perspective on achieving adaptability under continuous distribution shifts.

(c) Extensive experiments across multiple real-world datasets (ETT, Weather, ECL, Traffic) and ablation studies (including cross-dataset and extended online-phase evaluations) support the proposed framework. The results demonstrate improvements over both traditional OTSF models (e.g., FSNet, OneNet, DSOF) and recent LLM-based time series baselines.

### Weaknesses
(a) In the framework description, the paper only briefly mentions that the pattern embedding, text embedding, and aligned time-series embedding are concatenated and then fed into the pre-trained LLM. However, it does not clearly explain how the text representation embedding is generated or whether any additional information (such as task prompts, positional encodings, or structural tokens) is included after concatenation. These details should have been clarified earlier in the framework section. The current description makes it difficult to understand how the inputs are constructed and how the LLM processes them.

(b) According to Appendix F.5, the prompt bank is trained jointly with the backbone and alignment module during the training phase and then frozen during the online phase. However, this design choice is only mentioned later in the appendix instead of being clearly described in the main framework section. It should have been explicitly introduced earlier to help readers understand how the prompts are constructed and integrated into the overall architecture.

(c) Although Appendix F.2 and Table 8 report online update times and the number of parameters updated, the analysis remains limited to a single dataset and focuses only on per-step update efficiency. The framework depends on large-scale LLMs (e.g., Llama-7B or 13B), but the paper provides no discussion of the overall computational cost, inference latency, or memory footprint. Without details on GPU utilization, runtime scalability, or hardware configurations, it is difficult to assess the true efficiency and deployability of LLM4OT in realistic online forecasting scenarios. A more comprehensive resource and scalability analysis would be necessary to substantiate the claimed efficiency.

### Questions
(a) Could the authors clarify how the text representation embedding is generated before being concatenated with the pattern and aligned time-series embeddings?

(b) When concatenating the three embeddings (pattern, text, and aligned time-series), is any normalization, positional encoding, or token-type distinction applied before feeding them into the LLM? If not, how does the model differentiate between heterogeneous sources of information?

(c) Regarding the prompt bank, could the authors explain on how initialization strategy was chosen, and whether the prompts share the same embedding space as the pre-trained LLM tokens? This information seems crucial for reproducibility.

(d) The paper claims that LLM4OT is efficient during the online phase, however, the reported analysis (Appendix F.2 and Table 8) only covers update speed on a single dataset. Could the authors provide more comprehensive runtime statistics such as total training time, GPU memory usage, or inference latency?

### Soundness
3

### Presentation
2

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
This paper proposes an online learning method based on LLM and frequency component activation.

### Strengths
1、Introducing LLM into online learning is a rather interesting approach.

### Weaknesses
1、Why are the results corresponding to the blue modules in Figures 1a and 1b the same? Is the data segmentation ratio in 1a consistent with that in 1b?

2、The definition of cross-domain prediction is not clearly defined. If ETTh1 is used as the training set and ETTh2 is used as the online set, is it still necessary to perform online learning based on the data of ETTh2? Or should we only train on ETTh1 and perform inference directly on ETTh2?

3、Frequency domain decomposition operations and operations based on frequency components as basis vectors are quite common in the time series domain.

4、A prediction length of 1 is rare in real-world scenarios, and is updating part of the model's parameters based on each sample too frequent? How can the time interval between model inference calls be balanced with the frequency of model updates?

5、Cross-domain forecasting recommendations are compared with a range of current time series baseline models, including zero-shot and few-shot operations.

6、The definitions of online learning of time series and time series forecasting are not very clear to me, and the description in this article makes it even more difficult for me to understand.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LLM4OT, a framework for online time series forecasting that integrates large language models with frequency-domain pattern recognition. It introduces frequency-driven prompts to encode transferable temporal patterns and an LLM-enhanced adaptation module that converts recent time-series trends into textual descriptions for semantic reasoning. By combining spectral decomposition with LLM priors, LLM4OT achieves robust forecasting under distribution shifts and long-term online scenarios, outperforming existing state-of-the-art methods on multiple real-world benchmarks.

### Strengths
The paper is well-written and clearly organized, with thorough explanations of both the model design and the motivation behind each component. The methodology is presented in a way that is easy to follow, with clear distinctions between the training and online adaptation phases.

The experimental evaluation is extensive and convincing, covering multiple real-world datasets, diverse baselines (including traditional, online, and LLM-based models), and comprehensive ablation studies. These results strongly support the effectiveness and robustness of the proposed LLM4OT framework.

Overall, while the conceptual novelty lies mainly in combining frequency-domain prompting with LLM-based reasoning, the work distinguishes itself through its systematic empirical validation, detailed analysis, and clarity of presentation, making it a solid and well-executed contribution to online time series forecasting.

### Weaknesses
The primary weakness of this paper lies in its limited conceptual novelty. Both of the core ideas—leveraging frequency-domain representations and incorporating LLMs for sequence modeling—have been explored in prior works. While the proposed combination of frequency-driven prompting and LLM-based adaptation is practically useful, the paper does not sufficiently justify why these two components need to be integrated or what complementary benefits the frequency domain provides to LLM reasoning. The argument that frequency prompts help encode transferable temporal patterns remains mostly empirical, without deeper theoretical or analytical discussion. A stronger motivation or ablation directly comparing LLM4OT to time-domain prompting or alternative representation schemes (e.g., wavelet or trend-seasonality decomposition) would help clarify this design choice.

Moreover, the paper lacks a direct comparison with strong recent baselines, particularly the TimeMixer series, which are currently leading architectures for both static and online forecasting. Since these models also capture multi-scale temporal dependencies and exhibit strong generalization under distribution shifts, omitting them makes it difficult to assess the relative advancement of LLM4OT. Including such comparisons, or at least providing analytical discussion of the differences in design and performance trade-offs, would make the empirical evaluation more convincing.

Overall, while the work is well-executed and experimentally thorough, it would benefit from a clearer theoretical justification for the use of frequency-domain prompting with LLMs and a more comprehensive comparison against state-of-the-art forecasting frameworks to better position its originality and impact.

### Questions
The paper proposes combining (a) frequency-based pattern embeddings derived from DFT/DWT and (b) an LLM with text descriptions of recent behavior. Conceptually, both ingredients individually have precedents in the literature (frequency-domain decomposition for forecasting, and LLM-augmented forecasters). Could you elaborate on why these two components must be used together and why this specific pairing is essential for online adaptation? In other words: is there a principled reason that an LLM cannot adapt equally well using only time-domain representations and recent numeric context, without frequency prompts?

### Soundness
2

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
3

### Summary
The authors improve upon current online time series forecasting methods by representing time series as combination of frequency bases and connecting a pretrained LLM with a time series backbone.

### Strengths
- original idea
- reasonably well written
-

### Weaknesses
- unclear integration and roles of LLM and time series backbone
- unclear handling of distribution shifts occuring over time in the online period
- very long supplementary material, making the paper itself not self-contained

### Questions
Which roles are played by the LLM and by the time series backbone?
Do you handle distribution shifts in the online period and how?

### Soundness
2

### Presentation
2

### Contribution
2
