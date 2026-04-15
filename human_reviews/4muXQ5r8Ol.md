# A-Bench: Are LMMs Masters at Evaluating AI-generated Images?

- Decision: Accept (Poster)
- Scores: 6, 8, 5, 3

## Abstract
How to accurately and efficiently assess AI-generated images (AIGIs) remains a critical challenge for generative models. Given the high costs and extensive time commitments required for user studies, many researchers have turned towards employing large multi-modal models (LMMs) as AIGI evaluators, the precision and validity of which are still questionable. Furthermore, traditional benchmarks often utilize mostly natural-captured content rather than AIGIs to test the abilities of LMMs, leading to a noticeable gap for AIGIs. Therefore, we introduce **A-Bench** in this paper, a benchmark designed to diagnose *whether LMMs are masters at evaluating AIGIs*. Specifically, **A-Bench** is organized under two key principles: 1) Emphasizing both high-level semantic understanding and low-level visual quality perception to address the intricate demands of AIGIs. 2) Various generative models are utilized for AIGI creation, and various LMMs are employed for evaluation, which ensures a comprehensive validation scope. Ultimately, 2,864 AIGIs from 16 text-to-image models are sampled, each paired with question-answers annotated by human experts. We hope that **A-Bench** will significantly enhance the evaluation process and promote the generation quality for AIGIs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper study the multimodal LLM's ability in the context of image evaluation. Instead of studying the effectiveness of certain LLM-based metrics, This work aims to identify whether multimodal LLM (LMM) are truly capable of evaluating AI-generated images through a question answering benchmark. Proposed a benchmark that contains 2,864 set of questions which can be categorized into 6 categories, involving semantic understanding and quality perception. After benchmarking a total of 18 LMMs and comprehensive analysis, the authors came up with a conclusion that LMMs are still not masters at evaluating AI-generated images.

### Strengths
S1) The paper is well-written and well-organized. All the relevant details are included in the main paper and appendix. The evaluating method also seems rigorous.

S2) A good pitch in studying LMMs directly through question answering instead of studying the effectiveness of certain LMM-based metrics. This helped to shed a light on the true capabilities of current LMM-based image evaluation metrics.

### Weaknesses
W1) The paper would have provided more insights if the authors also studied the reasoning to verify if the LMMs truly understand how to evaluate for each categories (i.e. did the reasoning fully explain the choice made by the LMM?). This might help to explain the gap between the performance of LMMs and humans. I suggest conducting a study on small subset for each categories and see how the reasoning was aligned to the choice made.

W2)  It would also be desirable to see what kind of images LMM evaluate poorly across each categories in AIGI. A more detailed of diversity analysis on the AIGI dataset is required. E.g. for Basic Recognitions, how much portion of the questions are regarding recognitions of animal, human, or artifacts? Are these LMMs doing poorly on particularly certain type of objects?

### Questions
Q1) What is SRCC/PLCC in the introduction paragraph? It seems this abbreviation is never explained in the paper.

Q2) In section 4.1, "It’s worth noting that the instruction prompt might slightly differ for different LMMs according to the official setting." Why the instruction prompt is slight different for different LMMs? How will it impact the performance of LMMs?

Q3) For the human annotators, how are they recruited? What kind of training were they given? How many instances are labelled by each human annotator? I am also interested in the total time required to build this benchmark.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
- Human evaluations are the gold standard for evaluating generative models especially text-to-image (T2I) models. However, they are expensive. An alternative is using automatic metrics and Large Multi-modal Models (LMMs) are a popular choice.
- LMMs are trained on real images and AI-generated Images (AIGIs) are out of domain for LMMs questioning their reliability as evaluation models. 
- This work proposes A-Bench a diagnostic benchmark for assessing the reliability of LMMs for evaluating AIGIs. 
- A-Bench consists of two subsets 1) A-Bench P1 to evaluate the text faithfulness or prompt adherence of T2I models and 2) A-Bench P2 to evaluate the quality of the generations.
- Authors samples 2864 AIGIs from 16 open and closed source T2I models. For each generation, they sourced human experts to annotate question-answer pairs and computed the accuracies of popular proprietary and open-source LMMs. 
- Authors report that 1) Proprietary LMMs are better than open-source counterparts for text faithfulness, 2) proprietary LMMs perform as well as humans on simple enough prompts and 3) LMMs are not good models to evaluate the generation quality of AIGIs.

### Strengths
- Authors address a very important problem: Are current LLM/LMMs good enough to be used as judges for generative models? This line of research can provide valuable insights to train better LMMs for understanding AIGIs.
- A-Bench along with standard LMM evaluation benchmarks provide a complete picture of an LMMs capability to understand both real and AI generated images.
- The paper is well written and very easy to follow containing all the details necessary for reproduction.
- The experimental section is exhaustive with comparisons provided for both proprietary and open-source LMMs.

### Weaknesses
- I didn't find any major weakness with this work.

### Questions
- I would recommend authors check out SelfEval [1] as another source of evidence that external models cannot be reliable for evaluating T2I models. Please discuss it if relevant.
- In my experience there is a huge variance to the responses provided by  LLMs/LMMs. Did the authors compute variance of the scores or perform any statistical significance studies?
- L272 controversial -> counter factual
- In the introduction (L112-L117), in my opinion, authors should provide some numbers to make the point that LMMs are still not masters at evaluating AIGIs. Right now authors state that "there remains a considerable gap and significant room for improvement". Instead providing some numbers can make it more straightforward.

[1] Sai Saketh Rambhatla, Ishan Misra, SelfEval: Leveraging the discriminative nature of generative models for evaluation

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a benchmark designed to assess the efficacy of large language models (LLMs) in evaluating AI-generated images (AIGI). As the field increasingly depends on LLMs for this evaluation—sidestepping the high costs and time commitments of traditional user studies—quantifying the quality and reliability of LLM-based assessments is essential. While it's generally accepted that LLM evaluations fall short of human assessments, this paper provides a systematic analysis of the performance gap across various LLMs, comparing open-source and closed-source models to human evaluations.

The benchmark defines several key metrics within two primary dimensions: Semantic Reasoning and Quality Perception. Using this framework, the study measures the performance of multiple LLMs, revealing a substantial disparity between human judgment and LLM performance in AIGI evaluation.

### Strengths
1. The benchmark is undoubtedly useful. Given the growing reliance on LLMs to evaluate various AI-generated content like images, having a comprehensive, quantitative benchmark that assesses the effectiveness of LLMs in evaluation is highly valuable.
2. The paper tries to objectively define the underlying metrics of evaluation.
3. The benchmark development involved a rigorous process, starting with user studies to establish a baseline, followed by testing various LLMs, which adds credibility and depth to the analysis.
4. While the findings align with expectations, quantifying the gap between human and LLM performance is a valuable contribution. It enables the research community to approach improvements in this field with a more data-driven perspective, facilitating measured, progressive advancements.

### Weaknesses
1. While the metrics cover several important facets of semantic reasoning, they lack a rigorous scientific foundation, raising questions about whether they capture the full scope of semantic understanding as implicitly perceived by humans. Specific dimensions of semantic reasoning, such as cultural nuances, or emotional depth, may be missing from the current metrics, which could impact the holistic evaluation of AI-generated images. As such, while the comparisons of different LLMs using these metrics provide intriguing insights, it remains questionable whether these metrics are robust enough to serve as a truly holistic benchmark for evaluating semantic reasoning in AI-generated images.

2. The number of images used (~2,000) feels arbitrary and may be insufficient to capture the nuanced aspects of reasoning and quality perception required for a comprehensive evaluation. Expanding the dataset to around 5,000–10,000 images, with careful attention to diversity across image types and contexts, could improve the robustness of the analysis. Additionally, it would be helpful for the authors to provide a rationale for this dataset size or acknowledge any limitations they faced in scaling up.

### Questions
1. It would strengthen the paper if the authors could provide scientific backing for the proposed metrics, citing sources that systematically define each measure. Specific areas where additional references might be valuable include the validity of semantic reasoning components and quality perception dimensions, to help ensure that the chosen metrics align with established frameworks in the field.

2. Providing further details on the image selection process would clarify the robustness of the benchmark. Specifically, information on the criteria for image selection, the diversity of image types, and the distribution of different content categories would offer valuable context. If possible, outlining how these factors impact the benchmark's representativeness, and validity could further enhance transparency.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
Due to the existing evaluation models' inability to effectively assess the performance of AIGI tasks, more and more researchers are turning to LMMs for evaluating the quality of generated images. The authors question this approach and design a framework consisting of seven dimensions focused on high-level semantic understanding and low-level quality evaluation to assess the quality of AIGI. By manually annotating 2864 different image quality issues, the authors compare the evaluation performance of multiple open-source and closed-source LMMs and contrast these with human evaluation results, summarizing numerous shortcomings of LMMs in the AIGI quality assessment task.

### Strengths
1. The authors manually annotated a dataset containing 2864 image quality issues, which contributes to the development of AIGI evaluation.
2. The authors evaluate AIGI quality from high-level semantic aspects like counting and low-level aspects like distortion, providing valuable insights for subsequent general AIGI task evaluations.
3. The paper's A-Bench includes the evaluation performance of multiple LMMs, offering guidance for researchers who wish to use LMMs for AIGI quality assessment.

### Weaknesses
1. Although A-Bench includes multiple LMMs, it lacks some of the latest SOTA models. Better models such as QWEN-VL2 and MiniCPMv2.6 can be found from opencompass. The paper does not specify the versions of gpt4o used, such as gpt-4o-2024-08-06 or gpt-4o-2024-05-13, which is crucial for future researchers.
2. The AIGI models used to generate the dataset are somewhat outdated, lacking relatively advanced image generation models such as SD3, PixArt, Flux, etc. Currently, the more outstanding AIGI models often embed large language models, which might significantly impact the evaluation conclusions.
3. The questions are all manually generated, which is certainly good. However, this makes the evaluation dataset difficult to expand and might lose value as AIGI models rapidly evolve. It would be better if the questions could be designed based on the text prompts of T2I models.
4. Compared to previous work, The paper's main contribution, i.e., high-level semantic question answering, is not strongly related to AIGI and does not seem necessary to research specifically in the AIGI context.
5. Two-thirds of the low-level semantic question-answering data come from other datasets, reducing the paper's contribution.
6. The paper's findings are somewhat unremarkable. It is obvious that closed-source LMMs perform better than open-source ones, and some other findings, such as the LMMs' insufficient perception of distortion, have already been mentioned in works like Q-Bench.

### Questions
1. Overall, the paper adds the evaluation of LMMs' high-level semantic cognition for AIGI, in addition to previous work using LMMs to assess image generation quality. However, it does not highlight the difference between AIGI tasks and conventional cognition tasks. Could the authors elaborate on this further?
2. Generally, the authors focus more on evaluating the perceptual capabilities of LMMs, but these perceptual capabilities are more inclined towards the low-level aspects for AIGI tasks. Could the authors further elaborate on the differences between the low-level perceptual aspects of AIGI and some earlier works?

### Soundness
2

### Presentation
2

### Contribution
1
