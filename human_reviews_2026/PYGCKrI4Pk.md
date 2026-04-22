# PHORECAST: Enabling AI Understanding of Public Health Outreach Across Populations

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Understanding how diverse individuals and communities respond to persuasive messaging holds significant potential for advancing personalized and socially aware machine learning. While Large Vision and Language Models (VLMs) offer promise, their ability to emulate nuanced, heterogeneous human responses, particularly in high stakes domains like public health, remains underexplored due in part to the lack of comprehensive, multimodal dataset. We introduce PHORECAST - Public Health Outreach REceptivity and CAmpaign Signal Tracking), a multimodal dataset curated to enable fine-grained prediction of both individual-level behavioral responses and community-wide engagement patterns to health messaging. This dataset supports tasks in multimodal understanding, response prediction, personalization, and social forecasting, allowing rigorous evaluation of how well modern AI systems can emulate, interpret, and anticipate heterogeneous public sentiment and behavior. By providing a new dataset to enable AI advances for public health, PHORECAST aims to catalyze the development of models that are not only more socially aware but also aligned with the goals of adaptive and inclusive health communication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a multimodal dataset, PHORECAST, to enable the study of individual and community responses to public health messaging. The work uses this dataset to train and evaluate vision-language models on the task of predicting human reactions based on demographic and psychometric profiles, which is a direction of inquiry that has not been explored with this type of data before.

### Strengths
The dataset contains a wide range of features, including participant demographics, personality traits from the Big Five Inventory, locus of control measures, and responses to public health materials. These responses are captured through multiple modalities, such as Likert-scale ratings on emotions and free-form text. The paper demonstrates the dataset's utility by fine-tuning existing vision-language models and measuring performance on response prediction. The experiments presented include ablation studies that investigate the contribution of different feature sets, providing an examination of the factors that influence model predictions.

### Weaknesses
One concern is the reported model performance on certain demographic subgroups. The paper indicates that for some underrepresented groups, such as individuals identifying as Non-Binary or Agnostic, prediction accuracy decreases after fine-tuning on the PHORECAST dataset. This decrease is observed in both models tested. This outcome raises questions about the model learning biases present in the data distribution. The paper acknowledges that disparities persist but does not offer an in-depth analysis of why performance degrades for these specific groups or what the implications of this degradation are. Another area for consideration is the generalizability of the findings, as the dataset is composed of U.S.-based, English-speaking participants, a limitation the paper states.

### Questions
What is the hypothesis for the observed decrease in prediction accuracy for certain underrepresented demographic groups after fine-tuning?

Are there any plans for future data collection efforts to expand the dataset to include participants from outside the United States or non-English speakers?

Could the evaluation of free-form text responses be expanded in the main paper to include the distributional and discriminator-based metrics that are mentioned as being in the appendix?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PHORECAST, a large-scale dataset designed to enable computational modeling of public health opinions, reasoning, and emotional responses. The dataset contains 7,527 survey responses collected from adult participants in the United States, spanning seven public-health topics (e.g., vaccination, smoking, drug use, diet, mental health). For the data collection process, each participant rated their agreement, reasoning, and emotional stance on multiple health statements, providing both Likert-scale ratings and free-text rationales.

### Strengths
* This paper introduces a novel dataset that combines structured ratings and natural-language rationales in public health, a sensitive real-world domain.
* This paper detailed the data collection process, cohort demographics, and transparent methodologies.
* This paper connects multiple subjects with ML research, including social psychology, public-health communication, etc.
* The evaluation of the dataset across multiple LLMs presents the effectiveness of the dataset thus offers clear starting points for future research.

### Weaknesses
* As noted in the limitations, the dataset is U.S.-only and English-only, which weakens claims of generalizability.
* The data collection participant recruitment process is not clearly described. The recruitment platform and sampling frame remain unclear, making it hard to evaluate potential socioeconomic and regional biases.
* It remains unclear if there's external or expert validation through the data collection and validation process. Attention checks or consistency metrics are not reported.
* For Table 2, it remains unclear why different demographic groups are highlighted differently.
* From Table 2, the population across different demographic groups is not balanced, where some subgroups have very small sample sizes, which leads to potential concern about a fair conclusion for all future use.
* Since the public health opinion evolves rapidly, while the paper presents a great model performance over categories, the model's robustness remains unclear in future use of the dataset.

### Questions
* How were participants recruited? Are they compensated? Are there any quality response selection processes? How are the responses selected?
* Are health topics assigned evenly across participants, or are some (e.g., vaccination) overrepresented?
* Are self-reported emotions verified for internal consistency or compared against any external annotation?
* How were fairness metrics computed? Are there any statistical tests performed?
* Are there plans to collect multilingual or longitudinal extensions of PHORECAST to address cultural and temporal bias?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents PHORECAST, a multimodal dataset of over 1,000 participants and 30,000+ responses to public health campaigns, capturing demographic, personality, and psychographic factors. The dataset enables modeling of individual and group reactions to health messaging. Using PHORECAST to fine-tune vision-language models (Llama 3.2-Vision, Gemma 3), the authors show obvious improvements in predicting emotional and behavioral responses. The work provides a valuable benchmark for studying socially aware and demographically grounded AI models in public health contexts.

### Strengths
1. Addresses the underexplored challenge of modeling human receptivity to public health messaging, bridging AI, behavioral science, and social forecasting.

2. PHORECAST combines visual stimuli, detailed demographics, Big Five personality traits, and free-text responses, enabling nuanced, multidimensional modeling of human behavior.

3. The dataset is collected from real society and the collection process is scientific and valid. Includes >1,000 diverse participants and >30,000 responses with careful control of demographics and psychometrics, providing a robust empirical foundation.

4. Benchmarks two VLMs (Llama 3.2-Vision, Gemma 3) with fine-tuning and ablation studies that reveal the predictive roles of personality, context, and visual cues.

5. Examines model performance across demographic subgroups, exposing disparities and offering a framework for fairness-aware behavioral modeling.

### Weaknesses
1. The model selection is somewhat limited, only use two VLMs. I would suggest adopt more models, such as different model families like Qwen-VL, also different model sizes. If condition permits, adopt at least one sota commericial models to showcase the sota results.

2. The ablation study only using single Llama model, which makes all the conclusion are somewhat restricted only to this model. I also didn't see the whole ablation study in your appendix. For example, the author claim the education matters the most (table 4), is this consistent with Gemma? 

3. The citation format is wrong, please check your latex or template. E.g. line 81-102. Also the related work should also included demographic simulation. Besides, the related work is not extensive explained, the social simulation, demographic simulation and VLM for social science are not included. 

4. In table 3, the author show that after fine-tuning, the model get more semantic alignment with human's response. Based on the figure 18 in appendix, the model get shorter answer but the opinion is not aligned with golden opinion. Specifically, the first case, the golden opinion is negative, but after finetuning, Llama's opinion is poistive. The author should conduct human evaluation to valid whether it's meaningful improvement. So far, I'm not convinced by this metric improvement. 

5. The paper provides limited discussion of common failure modes or examples of model misinterpretation in free-form responses.

6. The dataset is restricted to U.S.-based, English-speaking participants, which limits its cultural and linguistic generalizability. While this is also important, but I agree that the first work in this domain can be focus only on one country.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors present a dataset of perceived opinions of public health messaging and the ability for current VLMs to understand and infer them. Authors present various evaluation experiments to understand the impact of each individual component of data collected.

### Strengths
The authors conduct a large-scale user-study and account for various differences in demographics

### Weaknesses
# Relevance:
While the dataset collected presents novel insights about public health, however the current draft or subsequent explanations do not describe the relevance to an ML community

# Benchmark:
- I couldn't find the exact prompt/template used to be able to understand how the benchmarking was conducted. Understanding the prompt configuration will help enhance the validity of the results
- Authors evaluate the effect across Llama and Gemma, but the choice of models in need to be better justified

### Questions
Added in weakness

### Soundness
3

### Presentation
3

### Contribution
3
