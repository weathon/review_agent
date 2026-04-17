# Where in the World? A Vision-Language Benchmark for Probing Model Geolocation Skills Across Scales

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Vision–language models (VLMs) have advanced rapidly, yet their capacity for image-grounded geolocation in open-world conditions, a task that is challenging and of demand in real life, has not been comprehensively evaluated. We present WhereBench, a comprehensive benchmark for VLM image geolocation that evaluates visual recognition, step-by-step reasoning, and evidence use. WhereBench comprises 810 globally distributed images across two complementary geolocation scales: WhereCountry (i.e., 500 multiple-choice question-answering, with country-level answer and panoramas) and WhereStreet (i.e., 310 fine-grained street-level identification tasks requiring multi-step reasoning with optional web search). 
For evaluation, we adopt the final-prediction metrics: location accuracies within k km (Acc@k) for coordinates and hierarchical path scores for textual localization. Beyond this, we propose to explicitly score intermediate reasoning chains using human-verified key visual clues and a Shapley-reweighted thinking score that attributes credit by each clue’s marginal contribution.
We benchmark 12 state-of-the-art VLMs with web searching tools on our WhereBench and report different types of final answer accuracies as well as the calibrated model thinking scores.
We reveal that web search and reasoning do not guarantee improved performance when visual clues are limited, achieving only an overall 56.3% with the best SOTA model. These findings highlight not only the promise but also the persistent challenges of models to mitigate bias and achieve robust, fine-grained localization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces WhereBench, a new benchmark to evaluate the geolocation capabilities of Vision-Language Models (VLMs). The benchmark is split into two tasks: WhereCountry, a coarse-grained multiple-choice task, and WhereStreet, a fine-grained, reasoning-based task. The paper's primary contribution is a novel "Shapley-reweighted thinking score," designed to evaluate the faithfulness of a model's reasoning process by checking its use of human-verified visual clues. The authors benchmark 12 VLMs, finding that closed-source models dominate, web search provides mixed results, and models exhibit strong regional biases.

### Strengths
1. The primary strength is the introduction of a novel and potentially useful "thinking score" metric, reweighted by Shapley values, to evaluate the process of reasoning rather than just the final answer's accuracy.
2. The paper's experiments provide a nuanced analysis of when web search and deeper reasoning can help or, counter-intuitively, hurt localization performance depending on the task's nature.

### Weaknesses
1. The central claim that "there remains a lack of a fair and comprehensive benchmark that evaluates... faithfulness"  is incorrect. Several such benchmarks have been proposed at a much larger scale, including GeoChain (Yerramilli et al., 2025) and Gaea (Campos et al., 2025), which the paper fails to cite or compare against.
2. The benchmark dataset is extremely small, with only 810 total examples (500 for WhereCountry, 310 for WhereStreet), making it difficult to draw statistically significant or generalizable conclusions.
3. The WhereStreet task has a severe, built-in geographic bias, as it is sourced only from English- and Chinese-language videos. This is a major design flaw that limits the benchmark's diversity and directly leads to the "regional bias"  reported as a finding.
4. The paper's annotation protocol is underspecified. It fails to mention if inter-annotator agreement was measured for the 861 key clues, or how ambiguities between the 7 annotators were resolved. This undermines the reliability of the ground-truth clues used for the novel "thinking score."
5. The paper misinterprets its own results regarding WhereCountry. The claim that the task is "less reasoning-intensive"  is likely incorrect; it is more probable that the models' poor visual perception is the bottleneck, preventing them from knowing what to search for or reason about, a common failure mode in multimodal tasks.
6. The methodology for the novel Shapley-reweighted score is underspecified. The paper states $v(S)$ is estimated by "prompting the judge (Gemini-2.5-Pro) for all $2^{|C|}$ subsets" but never states the average number of clues $|C|$, which could make this computationally infeasible. The value function $v(S)$ ("achievable answer quality") is also highly subjective and not clearly defined.
7. The paper validates its answer-scoring judge but provides no such validation for its clue-in-reasoning judge. It is unclear how the judge reliably distinguishes a clue that was truly "used" from one that was merely "mentioned."The paper's regional bias finding may be based on confounding variables. It is unclear if the performance gap is due to model training data bias or other factors, such as a different density/type of visual clues (e.g., Roman vs. Chinese script) in the two biased data sources.

### Questions
1. What was the inter-annotator agreement (IAA) for the 861 verified key clues, and what protocol was used by the 7 annotators to resolve disagreements?
2. What is the average number of clues $|C|$ per sample, and how exactly is the "achievable answer quality" score for $v(S)$ defined and prompted to the judge?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces WhereBench, a novel benchmark designed to evaluate the image-grounded geolocation capabilities of Vision-Language Models (VLMs) across different geographic scales. The benchmark is comprised of two distinct tasks: WhereCountry, a multiple-choice question-answering task with 500 panoramic images for coarse, country-level localization, and WhereStreet, a more challenging task with 310 images requiring precise, street-level localization that often necessitates multi-step reasoning and the use of external tools like web search. 

A key contribution of this work is the introduction of a novel evaluation metric, the "thinking score," which moves beyond simple accuracy to assess the quality of a model's reasoning process by measuring its ability to identify and utilize relevant visual cues.  The authors conduct a comprehensive evaluation of 12 state-of-the-art VLMs on WhereBench, analyzing performance across various settings (e.g., with and without tool use, varying reasoning lengths). The results reveal several key insights, including a significant performance gap between closed-source and open-source models, the surprising ineffectiveness of web search in certain scenarios, and strong regional biases in model performance.

### Strengths
1. **Important and Well-Motivated Problem:** The paper addresses a challenging and practically relevant problem. Image geolocation has numerous real-world applications, from aiding in search and rescue operations to enhancing content moderation systems.  The lack of a comprehensive benchmark for this task represents a significant gap in the literature, which this work effectively addresses.

2. **Thorough Experimental Suite:** The paper provides a rigorous and wide-ranging experimental evaluation, testing a diverse set of 12 VLMs across various settings (e.g., with and without tool use, varying reasoning lengths).

### Weaknesses
1. **Marginal Novelty and Missing Citations:** The paper's novelty is somewhat diminished by its close resemblance to recent works like GeoChain and GeoReasoner. These papers explore similar themes of geolocation reasoning with VLMs, yet they are not cited in the related works section. Acknowledging and differentiating this work from these existing papers is crucial for establishing its unique contribution. 

2. **Limited Scale of the Dataset:** While well-designed, the WhereBench dataset is relatively small, with only 500 images for WhereCountry and 310 for WhereStreet.  This limited scale raises concerns about the statistical significance and generalizability of the findings. For a benchmark paper, a larger and more diverse dataset would be necessary to provide a more robust and reliable evaluation of model performance. 

3. **Flawed Assumption about Web Search:** The finding that web search does not consistently improve accuracy is presented as a surprising result.  However, this is expected given that current web search APIs used by VLMs do not support visual search (i.e., searching with the image itself). The models are limited to searching with textual keywords derived from the image. If these keywords are incorrect or insufficient, the web search will only add noise to the context, potentially degrading performance.  This limitation of the underlying tool, rather than a deficiency in the models' reasoning, is a more likely explanation for the observed results.

4. **Dataset Bias and Lack of Transparency:** The WhereStreet dataset is heavily skewed towards Asia, which could introduce regional biases into the evaluation. Additionally, the paper is unclear about the source of the images used in the WhereCountry dataset.  Providing more transparency about the data sourcing and composition would improve the benchmark's credibility.

5. **Lack of Qualitative Analysis:** The paper would be significantly strengthened by a more in-depth qualitative analysis of the models' reasoning processes. While the "thinking score" is a good first step, providing concrete examples of successful and unsuccessful reasoning chains would offer deeper insights into the models' strengths and weaknesses.

### Questions
1. Could you elaborate on the novelty of WhereBench in the context of recent works like GeoChain and GeoReasoner? A direct comparison would help to clarify your paper's unique contributions.

2. Given the limited size of the dataset, have you considered any statistical methods to validate the significance of your findings?

3. Could you clarify the source of the images used in the WhereCountry dataset?

4. Could you point to common failure modes of some models?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents WhereBench, a benchmark to evaluate how well vision–language models can perform visual geolocation and reason about geographic context. The dataset contains 810 images split into two subtasks: WhereCountry (coarse, 500 samples) and WhereStreet (fine, 310 samples). The benchmark measures both localization accuracy and reasoning quality, using human-verified visual clues and a Shapley-based thinking score. Twelve recent models are evaluated, with and without web search. Results show that explicit reasoning and retrieval do not reliably improve localization, and that models exhibit strong regional biases. The work highlights a persistent gap between general reasoning and grounded geographic understanding.

### Strengths
Originality: The paper fills a clear gap in multimodal evaluation by targeting geographic reasoning across scales. It complements recent reasoning-focused datasets like GeoChain, which emphasize step-by-step inference. WhereBench instead measures end-to-end performance, providing a complementary perspective.

Quality: The benchmark is carefully designed, with globally balanced sampling, reasoning-clue annotations, and transparent evaluation metrics. The Shapley-weighted thinking score is an interesting way to assess reasoning faithfulness.

Clarity: The paper is well organized, clearly written, and supported by good figures and visual examples. The reasoning and bias analyses are easy to follow.

Significance: The findings reveal key weaknesses in current multimodal systems, especially their bias toward familiar regions and limited grounding in fine-grained cues like vegetation, signage, or lighting. The benchmark will likely serve as a useful testbed for future models aiming for spatial or embodied reasoning.

### Weaknesses
The dataset is relatively small (810 samples), which limits the reliability of regional generalization and fine-grained comparisons.

The thinking-score metric relies on LLM judgment, and it is unclear how consistent it is across judge models or prompts. A small human correlation study would strengthen the claim.

The error analysis is mainly qualitative. More quantitative breakdowns (e.g., cue failures, retrieval errors) would improve interpretability.

The discussion of regional bias is insightful but short. It would help to quantify which regions or cultural features cause the largest degradation.

Some of the results are descriptive rather than hypothesis-driven. Clearer framing of expected effects from reasoning or retrieval would make the conclusions stronger.

### Questions
1. How reproducible is the Shapley-based thinking score across different LLM judges or seeds?
2. Did you measure agreement between human reasoning annotations and model thinking scores?
3. Could reasoning-supervised datasets like GeoChain help improve model performance on WhereBench?

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
4

### Summary
WhereBench is a benchmark with 810 images designed to evaluate Vision Language Models (VLMs) on image-based geolocation capabilities. It consists of two tasks - WhereCountry and WhereStreet. WhereCountry consists of 500 country-level identification MCQ problems, and WhereStreet comprises 310 street-level identification questions that are evaluated on final answer and a multi-step reasoning process. The WhereCountry samples were filtered from an initial pool of 8,041 images from GeoComp dataset using an open-source model to remove simple case and low-information images. The WhereStreet samples were extracted frames from 503 public geolocation videos, transcribed by Gemini-2.5-Pro to identify 861 visual clues, and manually verified by 7 PhD volunteers. The authors use a novel Shapley-reweighted thinking score that evaluates the model's intermediate reasoning process to check if it uses human provided visual clues. WhereBench was evaluated on 12 VLMs with Gemini-2.5-Pro achieving the best performance. Results also show that web search and deeper reasoning do not consistently improve performance on WhereCountry task but provide a small gain on WhereStreet task.

### Strengths
1. The evaluation method proposed by authors is an interesting way of quantifying how a model uses visual evidence. 
2. The selection of MCQ options using UN geoscheme and UN regional groups is clever for making the choices harded for WhereCountry.
3. The benchmark design allows us to compare a variety of models including tool calls and web search models.

### Weaknesses
1. The evaluation pipeline relies heavily on Gemini-2.5-Pro for estimating the Shapely values for all clue subsets. This makes the evaluation expensive and limits its reproducibility.
2. While the related works section talks briefly about geolocation benchmarks, the section feels incomplete and does not position WhereBench against other works. The authors should use this section to motivate the problem better.
3. Most models are able to achieve very high performance on this task which makes me question about the need of this benchmark.

### Questions
1. What was the annotation process for the 861 key visual clues?
2. The filtering process tries to remove simple examples, could this have introduced a bias?

### Soundness
3

### Presentation
3

### Contribution
2
