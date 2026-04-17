# UI2V-Bench: An Understanding-based Image-to-video Generation Benchmark

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Generative diffusion models are developing rapidly and attracting increasing attention due to their wide range of applications. Image-to-Video (I2V) generation has become a major focus in the field of video synthesis. However, existing evaluation benchmarks primarily focus on aspects such as video quality and temporal consistency, while largely overlooking the model's ability to understand the semantics of specific subjects in the input image or to ensure that the generated video aligns with physical laws and human commonsense. To address this gap, we propose UI2V-Bench, a novel benchmark for evaluating I2V models with a focus on semantic understanding and reasoning. It introduces four primary evaluation dimensions: spatial understanding, attribute binding, category understanding, and reasoning. To assess these dimensions, we design two evaluation methods based on Multimodal Large Language Models (MLLMs): an instance-level pipeline for fine-grained semantic understanding, and a feedback-based reasoning pipeline that enables step-by-step causal assessment for more accurate evaluation. UI2V-Bench includes approximately 500 carefully constructed text–image pairs and evaluates a range of both open source and closed-source I2V models across all defined dimensions. We further incorporate human evaluations, which show strong alignment with the proposed MLLM-based metrics. Overall, UI2V-Bench fills a critical gap in I2V evaluation by emphasizing semantic comprehension and reasoning ability, offering a robust framework and dataset to support future research and model development in the field.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work introduces UI2V-Bench, a new benchmark designed to evaluate Image-to-Video generative diffusion models from the perspective of semantic understanding and reasoning, rather than only visual quality or temporal smoothness. The benchmark assesses four key capabilities and contain ~500 curated text–image pairs. Two evaluation pipelines based on Multimodal Large Language Models (MLLMs) are proposed: an instance-level pipeline for fine-grained semantic assessment and a feedback-based reasoning pipeline enabling step-by-step causal evaluation. Experiments on multiple open-source and commercial I2V models, along with human evaluation, confirm that the MLLM-based metrics correlate well with human judgment.

### Strengths
1. This work addresses an underexplored area—evaluating reasoning ability in image-to-video (I2V) generation. Existing benchmarks largely overlook semantic understanding and causal reasoning, making this contribution particularly valuable to the community.

2. Significant effort was invested in image collection and dataset construction, including sourcing high-quality images from the web to support a diverse set of evaluation scenarios.

### Weaknesses
1. Although the benchmark covers four dimensions (spatial understanding, attribute binding, category understanding, and reasoning) and 19 fine-grained sub-dimensions, the dataset size (~500 text–image pairs) may be insufficient considering the breadth of subcategories and reasoning types examined. Given the huge diversity and variability of real-world I2V scenarios, certain sub-dimensions (e.g., temporal changes, natural environment reasoning) are represented by only a small number of cases. This may constrain the benchmark’s ability to generalize and stress-test models under long-tail or compositionally complex situations.
2. While the benchmark introduces numerous sub-dimensions (e.g., emotion, object-holding, texture, etc.), the criteria distinguishing these subcategories are not always well formalized. Many examples remain simple or binary (e.g., “the blue ball rolls, the others stay still” for object attributes, or “birds with high positions fly away, birds with low positions stay still” for spatial understanding). Without clearer annotation guidelines or explicit difficulty levels, it is not obvious whether failure cases stem from misunderstanding of semantics or from prompt ambiguity.

### Questions
Refer to weakness

### Soundness
3

### Presentation
3

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
This paper introduces UI2V-Bench, a novel benchmark designed to evaluate Image-to-Video (I2V) generation models with a specific focus on semantic understanding and reasoning capabilities, aspects largely overlooked by existing benchmarks. The authors identify a critical gap where current I2V evaluations primarily focus on video quality and temporal consistency, failing to assess a model's ability to interpret image semantics, bind attributes, understand categories, or perform logical reasoning conforming to physical laws and human commonsense.

UI2V-Bench proposes four primary evaluation dimensions: spatial understanding, attribute binding, category understanding, and reasoning. These dimensions are further broken down into 9 aspects and 19 fine-grained sub-dimensions. To facilitate evaluation, the paper develops two Multimodal Large Language Model (MLLM)-based evaluation methods: an instance-level pipeline for fine-grained semantic understanding (spatial, attribute, category) and a feedback-based reasoning pipeline for step-by-step causal assessment. The benchmark includes approximately 500 carefully constructed text-image pairs and evaluates a range of both open-source and closed-source I2V models. Crucially, human evaluations are conducted, demonstrating strong alignment with the proposed MLLM-based metrics. The authors conclude that UI2V-Bench provides a robust framework and dataset to guide future research and development in I2V models by emphasizing comprehension and reasoning.

### Strengths
1. The paper successfully identifies and addresses a crucial limitation in current I2V evaluation, shifting the focus from purely perceptual quality to semantic understanding and reasoning. This is a significant step forward for the field. The introduction of four distinct dimensions (spatial, attribute, category, reasoning), further broken down into 19 sub-dimensions, offers a highly granular and comprehensive assessment of I2V model capabilities.
2. The two proposed MLLM-based evaluation pipelines (instance-level and feedback-based) are creative and leverage the strengths of modern MLLMs for automated, fine-grained assessment, which is essential for scalability. The feedback-based reasoning pipeline is particularly novel in guiding MLLMs through causal inference.
3. The validation of MLLM-based metrics against human judgment, showing high correlations (Kendall's tau and Spearman's rho), significantly bolsters the credibility and reliability of UI2V-Bench. The benchmark evaluates both open-source and commercial I2V models, providing insights into the current state of the art and highlighting areas for improvement. The dataset of ~500 text-image pairs is tailored to specific challenges. The paper clearly describes the construction of the benchmark, the evaluation pipelines, and the experimental setup, making it easy for other researchers to understand and potentially build upon.

### Weaknesses
1. While "carefully constructed," a dataset of approximately 500 text-image pairs, when distributed across 4 main dimensions and 19 sub-dimensions, might be considered relatively small. This could limit the diversity and statistical robustness for some niche scenarios within each sub-dimension, potentially leading to an incomplete picture of a model's true capabilities across all possible variations.
2. The heavy reliance on MLLMs for evaluation, while validated, introduces a dependency on the capabilities and potential biases of these underlying models. The paper mentions "textual biases" (line 349) but could further elaborate on how the feedback-based pipeline fully mitigates this, or if there are inherent limitations that still exist due to MLLM behavior. The performance of the benchmark itself is tied to the evolving capabilities of MLLMs.
3. While system prompts are provided, more detailed examples or a rubric for how the "Result Judge" (for instance-level) and the final scoring (for reasoning) precisely convert MLLM outputs/responses into a quantitative score would enhance transparency and reproducibility. For instance, what constitutes a "correct" comparison or a "successful" chain of questions leading to a score?
4. Running evaluations using multiple advanced MLLMs and specialized video-LLMs (VideoRefer, Tarsier) could be computationally intensive and incur significant API costs for closed-source MLLMs. This might pose a barrier for researchers with limited resources who wish to use the benchmark extensively.

### Questions
1. Could the authors elaborate on the process of constructing the ~500 text-image pairs? What specific strategies were employed to ensure diversity and representativeness across all 19 sub-dimensions, especially given the relatively small total number of samples? Were there any specific challenges in creating complex reasoning cases?
2. The paper mentions "textual biases" for MLLMs. Could the authors provide a more in-depth discussion on these biases and how the feedback-based reasoning pipeline specifically addresses or mitigates them beyond just generating a chain of questions? Are there known failure modes or types of reasoning where current MLLMs still struggle, even with this feedback mechanism?
3. What is the estimated computational cost (e.g., GPU hours, API calls for commercial MLLMs) for running a full evaluation of a single I2V model across the entire UI2V-Bench? This information would be very valuable for researchers planning to utilize the benchmark.
4. For the "Result Judge" component in the instance-level evaluation and the final score aggregation in the feedback-based reasoning, could the authors provide more concrete examples or a detailed rubric of how MLLM outputs are translated into quantitative scores? This would improve the transparency and reproducibility of the evaluation process.
5. Are there plans to expand the dataset size in future iterations of UI2V-Bench, particularly for sub-dimensions that might have fewer examples, to further enhance its statistical robustness and coverage?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a new benchmark for image-to-video generation that tests whether a model can reason about plausible events. The evaluation metrics span four perspectives: spatial understanding, attribute binding, category understanding, and reasoning. The judge is an MLLM-driven agent system.

### Strengths
1. The paper is well-written with a clear storyline. The structure of the paper is well designed.
2. The research topic of I2V evaluation is timely, important, and motivated.
3. The evaluation metrics cover sufficiently many perspectives.

### Weaknesses
1. The heavy dependency on using MLLMs as judges can be inaccurate. I understand that the paper already has a few user studies (Section 5.4 human evaluation), but that is relatively vaguely described. What are the qualifications of the selected users? How does the author team guarantee that the users are sufficiently independent to the authors? 
2. More principled video evaluation metrics that utilize symbolic structures will be more convincing (e.g., the consistency and dynamics metrics presented in WorldScore (ICCV 2025)).

### Questions
The weakness is listed above, and my initial rating for the paper is borderline rejection (4). My final rating will be conditioned on the soundness of authors' rebuttal.

### Soundness
2

### Presentation
2

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
This paper introduces U12V-Bench, a new benchmark for evaluating Image-to-Video (I2V) models. It focuses on a model's ability to understand the input image and perform reasoning, rather than just assessing low-level video quality. The benchmark tests four key dimensions: Spatial Understanding, Attribute Binding, Category Understanding, and Reasoning. It uses innovative MLLM-based methods for automated evaluation, and the results show a strong correlation with human judgment.

### Strengths
The main strength is its novel and important focus on semantic understanding and reasoning, which addresses a clear gap in existing I2V evaluation. The four evaluation dimensions are well-chosen and comprehensive. The two proposed evaluation pipelines are clever; the instance-level pipeline allows for fine-grained checks, and the feedback-based "question chain" method for reasoning is particularly effective. The high correlation with human evaluation strongly validates their automated metrics. The planned release of the dataset is a valuable contribution to the community.

### Weaknesses
The current dataset size of ~500 samples is a good start but could be expanded in the future for even better coverage and robustness. The evaluation pipelines, especially the one involving segmentation and video LLMs, seem computationally heavy. It would be helpful to briefly discuss the computational cost or efficiency. There is a minor terminology inconsistency in the "Attribute Binding" dimension between "Textures" in the table and "material" in the text.

### Questions
1. Do terms "Textures" (in Table 1) and "material" (in Section 3.1.2) refer to the same concept?

2. Do authors have a specific timeline for the public release of the benchmark dataset and evaluation code?

3. Could authors provide a rough estimate of the computational cost required to run the full evaluation suite on a single model?

### Soundness
2

### Presentation
3

### Contribution
2
