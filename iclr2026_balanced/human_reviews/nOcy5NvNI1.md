## Human Reviewer 1

### Summary
Existing Benchmarks often mirror the capabilities of models to evaluate and understand how models
perform on tasks of interest. These benchmarks contain short and simple instructions. Authors present ,
ECHO framework for constructing benchmarks using 35000 prompts created from real world (social
media data- Twitter) and used GPT4o for image generation to identify SoTA model capabilities and its
alternatives , design metrics , complex task which was absent in existing or previous benchmarks. Curated
prompts were filtered by filtered posts by designing the relevant keywords and relevance of the post text.
Further online discussion may have messy conversation which leads to incorrect dataset collection . To
avoid this authors have used LLMs to turn every messy conversation into one clean, self-contained data
sample that includes: the prompt, the responses, and how good it is. After text refinement , Multimedia
(Images) requires VLM (Qwen 2.5 VL). After Analysis , 20% Data marked as High Quality
(Benchmarking) , 66% Data as Medium Quality (Analysis). Due to policy of twitter and openai some
posts were refused to download or generate and further data removed manually by the author . Following
types of models used such as Unified Models , LLM+Diffusion , Image Editing Models. Instead Human
Evaluation for large scale authors used VLM as a Judge. As a secondary Validation , authors present five
expert raters with outputs of 8 models for 200 samples and ask the annotators to rank the output from best
to worst for both image to image and image to text splits. Authors framed ECHO could help to
differentiate model performance in the fine grained ways such as Color Shift Magnitude , Face Identity
Similarity , Visual structure such as object positioning or human pose.

### Strengths
1. Paper primary goal is to create a new generative model structured dataset involving users sharing
interesting prompts and outputs, novel task ideas, or commentary on model behavior.
2. Authors present , ECHO framework for constructing benchmarks using 35000 prompts created
from real world (social media data- Twitter).
3. Designed several specialized automated metrics: color shift magnitude, face identity similarity,
structure distance, and text rendering accuracy.

### Weaknesses
1. Community perceptions and discussion topics evolve over time. Benchmarks derived from a
snapshot of community feedback may not adapt quickly, risking obsolescence or misalignment
with current priorities unless actively maintained.

2. Since social media discussions are shaped by active communities, the collected prompts and
feedback may reinforce prevailing stereotypes, biases, or misconceptions. This can lead to
benchmarks that unintentionally favor certain demographic groups, styles, or cultural contexts,
thereby limiting fair assessment across diverse user groups.
3. Leveraging social media data introduces vulnerability to manipulation, such as spam posts,
coordinated misinformation, or artificially amplified feedback, which could skew the benchmark
toward certain failure modes or artificially elevate the perceived performance.

### Questions
1. The success of your approach relies heavily on LLMs and CV models for classification and
extraction. How sensitive are these models to inaccuracies or biases, and what measures are in
place to validate their outputs? Could errors in automated classification impact the reliability of
the derived metrics?
2. Are there classes of failures or use cases that ECHO systematically misses due to its reliance on
social media posts or community feedback?
3. Are there risks that users might intentionally or unintentionally manipulate community feedback
(e.g., spamming certain prompts or promoting specific outputs) to bias the benchmark? How does
your framework mitigate or detect such scenarios?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper introduces ECHO (Extracting Community Hatched Observations), a framework for constructing adaptive, data-driven benchmarks for image generation models based on real-world user interactions on social media. Motivated by the observation that existing evaluation datasets lag behind rapidly evolving generative capabilities, the authors propose leveraging posts from platforms such as Twitter/X to capture emergent tasks, user prompts, and qualitative feedback surrounding newly released models (e.g., GPT-4o Image Gen). ECHO systematically extracts multimodal data, comprising textual prompts, reference images, and community commentary, using a pipeline that integrates large language and vision–language models (LLMs/VLMs) to filter, contextualize, and structure these inputs into benchmark-ready samples.

Empirically, the authors curate a dataset of over 35,000 social media posts and develop a benchmark subset containing approximately 1,700 text-to-image and image-to-image tasks. Using this dataset, they evaluate eight leading generative models, showing that ECHO better distinguishes performance differences than conventional benchmarks such as GEdit or CompBench. Beyond quantitative evaluation, the framework also transforms recurring user observations (e.g., color shift, identity drift, text rendering errors) into measurable diagnostic metrics, providing a mechanism to align benchmarking with authentic user concerns. The paper concludes that such community-grounded, continuously updatable benchmarks can serve as a scalable and dynamic alternative to static evaluation paradigms for assessing progress in generative modeling.

### Strengths
1. The paper articulates a timely problem: benchmarks for generative models cannot keep pace with emerging user behavior. The introduction (pp. 1–2) effectively grounds this issue using the example of “Ghiblification” — a community-invented use case of GPT-4o that no prior benchmark captured. This framing convincingly motivates ECHO.

2. The ECHO pipeline (Figure 2) is the paper’s technical core: Collect relevant social posts, Reconstruct context across replies, Process multimodal data (text + images + screenshots), Filter and classify for quality and benchmarking. The pipeline is modular, well-illustrated, and generalizable. The multimodal LLM+VLM processing steps (using GPT-4o and Qwen-2.5-VL) are particularly novel — they turn noisy social media content into structured benchmark samples.

3 Empirical Validation: The authors benchmark 8 models (open-source and proprietary) across the new dataset. 

4 Turning user complaints into quantitative metrics (color shift, face identity, text rendering) is an interesting idea. Figure 8 (page 9) demonstrates these metrics’ power to confirm qualitative observations (e.g., GPT-4o’s “yellow tint” or identity drift). This “closing the loop” contribution elegantly connects community perception and model evaluation.

### Weaknesses
1. ECHO’s exclusive reliance on Twitter/X introduces substantial platform- and demographic-specific biases. As acknowledged in *Appendix A*, trending phenomena such as the “Ghibli style” disproportionately influence the sample composition, leading to a skewed task distribution that may compromise benchmark representativeness and fairness. Consequently, models optimized for highly visual or viral content may appear to perform better under this framework. Furthermore, the pipeline is evaluated solely on Twitter/X, without empirical validation across alternative social platforms (e.g., Reddit, Discord, or YouTube), thereby limiting the generalizability of the proposed approach and undermining its claim to universality.

2. Although ECHO is described as “re-runnable,” the paper lacks sufficient methodological transparency for full replication. Critical details concerning data acquisition—such as API endpoints, scraping protocols, temporal sampling strategies, and filtering heuristics—are not disclosed. While Figure D.1 enumerates example keywords, key parameters governing data retrieval, LLM-based relevance scoring, and post-filtering thresholds remain unspecified. This omission impedes reproducibility and may also raise compliance concerns regarding Twitter/X’s data use policies. Moreover, the reuse of user-generated content (including images and textual comments), even in anonymized form, poses potential ethical and legal challenges under data protection regulations such as the General Data Protection Regulation (GDPR).

3. Despite the use of an ensemble of evaluators (GPT-4o, Gemini, and Qwen), the low Kendall’s τ correlation with human judgments (τ ≈ 0.10–0.12) indicates that current VLM-as-a-judge paradigms remain insufficiently reliable for fine-grained assessment of generative quality. Additionally, although approximately 35,000 posts are collected, only ~1,700 high-quality samples are retained for benchmarking. This limited subset, especially when contrasted with larger-scale datasets such as DiffusionDB (≈14 million prompts), raises concerns regarding statistical robustness and the stability of model performance differences. It remains unclear whether the reported distinctions reflect genuine capability gaps or are artifacts of small-sample variance.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper introduces a framework ECHO, that constructs benchmarks by directly probing social media posts from platforms like twitter. The framework is applied to GPT 4o Image Gen, and thereby a collection of 35000 prompts are curated, which are those that are directly used by real-life users. Resultantly, the benchmarks consists of complex and realistic tasks, not included in previous benchmarks. By applying such prompts, state-of-the-art generative models are re-evaluated, and the discovered indicators from the posts help evaluate the models further.

### Strengths
1. The benchmark is unique, and is of utmost importance, as it ties the general public opinion with model performance. 
2. The authors have put in substantial efforts to extract the prompts from the complex tweet chains.
3. The framework uncovers significant model failures observed by the users, as shown in Fig. 4.

### Weaknesses
1. While I agree that the large-scale images generated from the prompts cannot be manually evaluated, the VLM-human correlation seems quite low, raising questions on the evaluations.
2. It is unclear how many images have been generated by each generative model for evaluation. 
3. The benchmark consists of several tasks, while Fig. 6 summarizes them. It would be good to capture how the different models on the individual tasks, as virtual try on, novel view synthesis are themselves quite significant as tasks.
4. Fidelity, Faithfulness and Diversity - the three most important metrics related to text-to-image generations should have been discussed in more detail.
5. [minor] The resulting observations are similar to the expected results - the closed-source models generally outperform the open-sourced ones. However, that being said, the curated prompts seem useful, as they are obtained from real-life users.

### Questions
1. How many images per model did the authors evaluate on the constructed benchmark?
2. Did the authors try the traditional fidelity and faithfulness metrics? Wherever applicable, how diverse are the generations by the different models? This question is important as the prompts are unique in the proposed benchmark.
3. Counting and hallucinations have been raised by users as failure cases, and these have been well-established problems in literature. Did authors try measuring them, especially for newer models like FLUX Kontext, GPT etc?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
4

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper addresses the growing gap between the rapid progress of image generation models and the static nature of existing benchmarks. It introduces ECHO, where instead of relying on predefined and manually curated tasks, the framework builds its challenges by mining social media posts about new image models (the case study for this paper is GPT-4o Image Gen) to construct an in-the-wild, re-runnable benchmark. It then details an LLM and VLM-based processing pipeline, where the authors extract structured <input, output, feedback> tuples from messy social media threads and curate a benchmark of around 35K samples. The paper demonstrates that ECHO captures creative, complex, and evolving tasks not present in existing benchmarks, provides stronger differentiation among models, and motivates the development of new quantitative metrics (e.g., colour shift, text rendering accuracy, identity preservation) based on community feedback.

### Strengths
1. The paper's central premise (benchmarks must dynamically evolve with model capabilities and user behaviours) is a highly significant and timely contribution. The idea of using community-generated evidence from social media as a source is a novel solution. 
2. The paper is clearly written, visually rich, and easy to follow. The proposed pipeline (ECHO) is detailed and well-motivated. The analysis demonstrates how ECHO surfaces novel tasks, uncovers model failure modes, and differentiates state-of-the-art models.
3. It is an important contribution to how we evaluate image generation models. The framework could inspire new adaptive benchmarking protocols in other modalities. 
4. Traditional benchmarks for image generation often rely on abstract metrics that correlate poorly with human preference. ECHO’s feedback-derived metrics aim to capture human-relevant error dimensions.

### Weaknesses
1. Although ECHO collects a large and diverse dataset, the quality control process heavily depends on LLM and VLM filtering. The paper does not provide any quantitative validation on how accurate the pipeline is. How do the authors handle bias or noisy samples in their final benchmark? The fill-in-the-blank prompts (Section 3.3) could also introduce hallucinated prompts. 
2. The authors analyze bigrams to assess the linguistic diversity of the benchmark, but this is a coarse view of the benchmark. A more informative approach would be to use an LLM to cluster each datapoint into broader task categories, which will be more meaningful to understand the benchmark distribution. It would also reveal potential imbalances, there could be an overrepresentation of certain tasks, which is not currently captured in the analysis.

### Questions
1. The framework currently relies on (presumably) Western-centric, English-language social media. How feasible or what challenges would arise to extend ECHO to non-English or region-specific platforms and mitigate this sampling bias. 
2. Can the authors provide a more granular breakdown of the benchmark's composition, as suggested in W2. An analysis of task clusters would be far more insightful than bigrams and would help identify which specific capabilities ECHO is truly testing and how balanced it is.
3. Based on this analysis, which tasks in the benchmark are most difficult (i.e., highest failure rate for all models) and which are comparatively easier? This would be invaluable for guiding future research.

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
6

### Confidence
4