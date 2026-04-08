## Human Reviewer 1

### Summary
This paper introduces SPORTU, a new benchmark designed to evaluate the capabilities of Multimodal Large Language Models (MLLMs) in sports understanding and reasoning. SPORTU consists of two components: SPORTU-text, focusing on text-based reasoning, and SPORTU-video, focusing on video-based reasoning. The authors evaluate various LLMs on both components, revealing limitations in complex reasoning tasks.

### Strengths
1. The proposed dataset could be useful for the community.
2. Both close and open-sourced models are evaluated.
3. Metrics are studied with human verification.

### Weaknesses
1. The reviewer is concerned about the lack of diversity and coverage of the dataset because of the limited prompt templates and number of samples. 

2. Implementation could be possibly flawed. 
- The error in Figure 6 looks suspicious and makes the reviewer wonder whether the model is called correctly or not. 
- The reasoning prompt asks the model to first generate answer and then reasoning, which is not optimal since the model's final answer cannot benefit from the reasoning process.
- It is known that LLM usually prefers its own answer so it is important to understand G-eval' quality with different LLMs as the rater.

Minor:
L821 typos of "Section ??"

### Questions
Please check weakness for details.

### Soundness
1

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper provides a multimodal dataset (text and slow-motion video) for evaluating (multimodal) LLM capabilities in the sports domain.

### Strengths
- A multimodal new dataset for sports domain (with multiple sports) and well annotated by experts; the dataset should be helpful for the research communities

- A well prompting capabilities to show the limitation of current LLM capabilities on the dataset. 

- Evaluating several reasonable public or private LLM models.

### Weaknesses
- Yet another vertical dataset for LLM
- It's helpful but marginal to expand the technical depth for the community
- not clearly identified what current models failed.

### Questions
- The video quality for the datasets
- For each sports type, the video are biased to certain views or events?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
In the AI+Sports area, existing works are limited to restricted kinds of sports, absence of explanations, or lack of reasoning on rules, and it proposes SPORTU consisting of SPORTU-text and SPORTU-video to boost understanding more sports with rules understanding. SPORTU-text evaluates models on rule comprehension and strategy understanding in the pure text domain and SPORTU-video evaluates models on understanding both video details and rules in the video domain.
It evaluates LLMs and MLLMs on SPORTU-text and SPORTU-video, revealing their limitations in complicated sports questions.

### Strengths
1. It proposes SPORTU-text and SPORTU-video to boost understanding more sports with rules understanding in text and video domains.
2. It analyzes the views, reasoning prompts, sport types, the error types, which are comprehensive.
3. The writing is clear.

### Weaknesses
1. A benchmark aims to evaluate certain abilities and give some insights. The paper does not deeply discuss why the models have different performances and does not give advice on how to resolve the problem of understanding videos and reasoning on rules.

2. Prompt strategy in LLM can also be tested on MLLM when evaluating on SPORTU video benchmark to see how the reasoning process influences MLLM.

3. It's not very clear if the questions in this dataset can comprehensively detect the models' abilities to understand sports.

4. The Pearson correlation between humans and the other metrics is low. Many are near 0.

### Questions
1. How do you split the SPORTU-text questions into rules-related, strategy-related, and scenario-related? What is your basis?

2. What are the results on rule/strategy/scenario, respectively, on sportu-text?

3. How is the error analysis in 5.1 conducted? Is there a definition for each error type?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper prensents SPORTU, a comprehensive Sports Understanding Benchmark that integrates both text-based and video-based tasks to evaluate models’ sports reasoning and knowledge application capabilities. Based on this benchmark, this paper tests the capability of existing open-source or close-source  models.

### Strengths
1. As a sport domain understanding benchmark, the proposed  SPORTU combines text-based and video-based tasks to assess models' sports reasoning and knowledge application abilities.

2. The evaluation setting is comprehensive including the direct prompting, chain-of-thought (CoT) prompting. In addition, few-shot promoting is also applied in SPORT-text evaluation.

### Weaknesses
1. Unclear motivation. The authors should clarify the differences between the proposed SPORTU and existing sport domain understanding benchmarks. Although discussions have been made in introduction and related work section together with Table 1, it is still unclear why the introduced features , for example, slow motion, multi-camera angles are important. More discussions and visualizations are needed.

2. Missing details in dataset construction. There exist some unclear details in the dataset construction. For example, how to guarantee the multi-camera setting? Is it achieved simply by human annotator check? In addiation, the proposed SPORTU contains both the multi-choice and open-ended question, how are these two categories divided?

3. More advanced evaluation methods should be applied. For example, ST-LLM [1], qwen-vl [2]

4. The paper writing should be polished. Some references are missing, for example "Section ??" in Line 821. The quotation mark error in '”Why is it a foul in the video?”' in Linee 482.

[1] ST-LLM: Large Language Models Are Effective Temporal Learners
[2] Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4