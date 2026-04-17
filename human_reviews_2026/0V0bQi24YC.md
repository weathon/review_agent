# Video-LevelGauge: Investigating Contextual Positional Bias in Video Language Models.

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
Large video language models (LVLMs) have made notable progress in video understanding, spurring the development of corresponding evaluation benchmarks. However, existing benchmarks generally assess overall performance across entire video sequences, overlooking nuanced behaviors such as contextual positional bias, a critical yet under-explored aspect of LVLM performance. We present **Video-LevelGauge**, a dedicated benchmark designed to systematically assess positional bias in LVLMs. We employ standardized probes and customized contextual setups, allowing flexible control over context length, probe position, and contextual types to simulate diverse real-world scenarios. In addition, we introduce a comprehensive analysis method that combines statistical measures with bias pattern recognition to characterize bias. Our benchmark comprises 438 manually curated videos spanning multiple types, yielding 1,177 high-quality multiple-choice questions and 120 open-ended questions, validated for their effectiveness in exposing positional bias. Based on these, we evaluate 27 state-of-the-art LVLMs, including both commercial and open-source models. Our findings reveal significant positional biases in many leading open-source models, typically exhibiting head or neighbor-content preferences. In contrast, commercial models such as Gemini 2.5 Pro show impressive, consistent performance across entire video sequences. Further analyses on context variation, context length, model scale, and multi-modal reasoning provide insights for mitigating bias and guiding model enhancement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors address the problem of **contextual positional biases** in large video-language models (LVLMs), where the content of a clip is interpreted inconsistently depending on its position within a video. While this issue has been explored in language models, it has not been studied in multimodal video-language settings, making this work both unique and novel. The paper introduces **Video-LevelGauge**, a benchmark designed to evaluate contextual positional biases in LVLMs across diverse tasks and video types. Beyond standard accuracy, it proposes novel **statistical metrics** to quantify such biases. Using these metrics, the authors further characterize **morphological patterns** in LVLMs, enabling the identification of specific phenomena underlying positional bias. Extensive experiments and analysis conducted with Video-LevelGauge provide valuable insights to guide future LVLM development and contextual positional bias mitigation. Overall, the paper is well-written and presents notable contributions to a previously underexplored area of LVLM research.

### Strengths
- Video-LevelGauge incorporates a comprehensive coverage of six video-language tasks spanning diverse video types, including egocentric, media, and synthetic videos, enabling extensive evaluation of existing LVLMs for contextual positional biases across a multitude of settings.
- Video-LevelGauge introduces three statistical metrics: $P_{mean}$, which captures the magnitude of positional bias, and $P_{ran}$ and $P_{var}$, which measure the volatility of model behavior. Together, these metrics provide a holistic assessment of positional biases in LVLMs.
- The proposed morphological recognition provides a grounded approach to identifying potential root causes of contextual positional biases, providing a framework to diagnose and characterize positional biases in LVLMs.
- Thorough analysis is performed on Video-LevelGauge using the proposed metrics across multiple factors, including context length and video type, revealing several insightful observations that could inform future efforts toward mitigating these biases.

### Weaknesses
- For the positional bias metric, the relative score is intuitive and easy to understand. However, greater details could be provided on what the score entails: How is $RS_i$ computed if $S_{meta} = 0$, and what does $RS_i$ reflect in this case? Additionally, for the five types of morphological patterns, several categories are not straightforward to understand. In particular, the morphological phenomena of “lost in the middle”, “neighbor bias”, and “volatile” could be somewhat confusing. I would suggest for the authors to perhaps provide additional elaboration for each of these types in Section 3.4, and possibly how they correlate with model performance.
- Although the authors validated against multimodal information leakage within each probe instance, the paper does not seem to address potential leakage between the probe clip and background video(s) during evaluation. Background videos with similar contexts to the probe clip could lead to cross-video leakage, allowing the LVLM to answer the query correctly by referring to background video content, which may be unintended.
- Evaluation results for individual tasks (e.g., OCR, AP, etc.) only report $P_{ran}$, which measures the worst-case variation, but omit $P_{mean}$, which captures the average performance across instances. Including $P_{mean}$ would provide a clearer view of the overall positional bias for each task.

### Questions
- “$\nearrow \text{if MSE}_1 \leq 3 \text{ and }k > 0.5$” [Appendix A.5] - Could the authors elaborate what the neighbor bias entails, and how the trend reflected by this condition correlates with this bias?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a benchmark suite for evaluating contextual positional bias of large video language models (LVLMs) considering long contexts. The data collection involves three stages,
1. QA Generation: They collect videos from existing video-text resources including ego-centric videos, and generate frame-wise captions using GPT-4o. These captions are combined with crowd-sourced task definitions, and then an LLM generates question-answer pairs.
2. QA Refinement: The generated question-answer pairs are filtered by GPT-4o, considering hallucinations and cases where the question does not require visual information in order to be answered. Human validators filter out the invalid question-answer pairs before proceeding to next step.
3. Distractors: LLMs generate distractors (i.e., incorrect answer choices) for the collected question-answer pairs, later, again refined by human validators.

Additionally, the benchmark is divided into 7 sub-categories, which are OCR, Attribute Perception, Object Reasoning, Count Problem, Relationship Recognition, Action Reasoning, and Instructed Description. The benchmarked models are tested under different conditions, which are denoted as customized contexts. The set of customized contexts within this work include multiple videos inputs, long videos, multimodal interleaved input and lastly template video with ImageNet mean pixel values. This work introduces 3 different metrics to measure the positional bias, where all 3 metrics are centered around the relative score (RS) measure. These 3 metrics are, average relative score ( $P_{mean}$ ), difference between maximum and minimum relative score ( $P_{ran}$ ), and the variance in relative scores ( $P_{var}$ ). The evaluations on the proposed benchmark include 27 LVLMs including both open-weight and proprietary models where the model scale is ranging up to 108 billion parameters. Further studies investigates the effect of context type, context length and model size on positional bias.

### Strengths
- A novel benchmark for assessing the positional bias in video-language models.
- Focuses an important aspect which could help to assess robustness of LVLMs under different positional settings, which could help further developing more trusthworthy LVLMs.
- Detailed experimentation: 27 models, further beneficial studies on the effect of different design choices.

### Weaknesses
- The presentation needs to be improved,
	- In Figure 3, the entire set of sub-tasks should be illustrated. This is not feasible.
	- In Figure 3, the tasks should be renamed by following the terminology already exist in the literature. I don't understand why someone should pose these tasks as *reasoning* tasks, where they are *recognition* tasks in fact. For instance, please see [Fig 1](https://arxiv.org/pdf/2306.13394) in MMU work to see the difference between reasoning and recognition type of tasks. So, the tasks should be renamed as,
		- Object Reasoning -> Object Recognition
		- Count Problem -> Object Counting (because actions can be counted also as well)
		- Action Reasoning -> Action Recognition
		- Attribute Perception -> Attribute Recognition (for consistency)
	- In Fig1(a), what do the numbers next to the tick and x marks?
	- Fig 2 is cluttered, I think it would be better if the only metric is positional variance.
- Human refinement process remains entirely opaque. There are no details on this process, even how many human validators participated in the refinement process.
- The proposed metric is not a standard metric used to evaluate the models, so I think this part should be expanded in the main text. This is such an important element of this paper but it only takes up 20 lines in the main text currently.
	- For instance, why should one use the relative score metric, and why should the standalone accuracy should be in the denominator?
	- What is the position $i$ ? What is the unit, seconds, or frame? Is this position absolute or relative? Are these positions randomly sampled for each individual example? Do these positions guarantee that the question-answer pairs remain valid for the video with custom context?
	- The morphological recognition (MR) term is confusing because morphology is a term which exists in NLP literature. Additionally, MR term currently seems unclear in the main text, and it is not motivated well enough.

### Questions
- I think it would be good to show visual examples of customized contexts in the appendix part.
- There is some typo on Fig2: GTP-4o-latest should be GPT-4o-latest.
- There is also white text on the last figure (Fig. 22) which can be barely seen on the background: `"Please output the questions and reference answers in the following JSON format: [ {'question': 'xxx', 'answer': 'xxx’}, {'question': 'xxx', 'answer': 'xxx’},"`

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Video-LevelGauge, a benchmark exploring positional bias in large video language models (LVLMs). The benchmark covers a range of models for comprehensive evaluation and reveals that existing open-sourced ones suffer significant positional bias, exhibiting unstabe video understanding affected by the position of target content. The paper also provides a deep analysis on models’ behavior across divers perspectives, such as context length and model scales.

### Strengths
- The paper tackles an important and underexplored problem in video understanding — positional bias in LVLMs. Overall, the paper is well written and easy to follow.
- The benchmark is well structured, featuring diverse videos and mechanisms that prevent single-frame or blind-bias predictions. The positional bias metric is clearly defined and suitable for evaluation.
- The paper provides extensive experimental results across 27 leading LVLMs with in-depth analyses, offering valuable insights into model behavior.

### Weaknesses
- While the benchmark design and evaluation criteria are solid, the results and analyses feel somewhat lukewarm and unsurprising. In Section 4.2, the main conclusion that positional bias tendencies vary across models and depend largely on training methods and model scale is expected. It is natural that larger models exposed to diverse video data perform better, and this conclusion could likely apply to other evaluation criteria beyond positional bias. I would appreciate a deeper, more tailored analysis of *why* such biases arise and *what* model characteristics contribute to distinct bias patterns (e.g., MR types). For instance, why do MiniGPT4-Video and InternVL3 (8B) show head preference? Are their training datasets skewed toward early-frame cues? 
- Consequently, it seems somewhat trivial that longer videos are more challenging and lead to lower performance. However, it remains unclear whether this truly amplifies positional bias or if the observed drop is simply due to increased video complexity.
- The paper primarily identifies the problem of positional bias but does not propose or discuss concrete directions for mitigation. Including such discussion would make the contribution more complete and valuable for future work.
- (Minor) Discussing inconsistent and biased video understanding found in prior works [1, 2] could also strengthen this analysis.

**References** 

[1] A Closer Look at Temporal Sentence Grounding in Videos: Dataset and Metric, arxiv 2021

[2] On the Consistency of Video Large Language Models in Temporal Comprehension, CVPR 2025

### Questions
See the Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Video-LevelGauge, a benchmark for evaluating contextual positional bias in LVLMs by inserting standardized probes into different positions of video contexts. The benchmark systematically assesses 27 LVLMs, revealing that commercial models exhibit less positional bias than open-source ones. The study also explores the effects of context type, length, and model size on positional bias, providing insights for future improvements.

### Strengths
- Novel Contribution: Highlights contextual positional bias, an underexplored issue in LVLMs.
- Well-Designed Benchmark: Includes standardized probes, flexible context configurations, and comprehensive metrics.
- Comprehensive Evaluation: Thorough analysis of 27 LVLMs, revealing actionable insights on bias patterns.

### Weaknesses
- No Mitigation Strategies: While the paper introduces contextual positional bias, it does not propose or evaluate methods to mitigate it.
- Lack of Task-Specific Examples: The paper does not provide clear illustrations of how positional bias affects specific tasks, making the findings less interpretable.
- Narrow Scope: Focuses solely on positional bias without addressing other LVLM limitations, such as hallucination or temporal reasoning errors.

### Questions
Please see Weaknesses for details.

### Soundness
2

### Presentation
2

### Contribution
2
