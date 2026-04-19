# Textual Aesthetics in Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 3

## Abstract
Image aesthetics is a crucial metric in the field of image generation. However, textual aesthetics has not been sufficiently explored. With the widespread application of large language models (LLMs), previous work has primarily focused on the correctness of content and the helpfulness of responses. Nonetheless, providing responses with textual aesthetics is also an important factor for LLMs, which can offer a cleaner layout and ensure greater consistency and coherence in content. In this work, we introduce a pipeline for aesthetics polishing and help construct a textual aesthetics dataset named TEXAES. We propose a textual aesthetics-powered fine-tuning method based on direct preference optimization, termed TAPO, which leverages textual aesthetics without compromising content correctness. Additionally, we develop two evaluation methods for textual aesthetics based on text and image analysis, respectively.Our experiments demonstrate that using textual aesthetics data and employing the TAPO fine-tuning method not only improves aesthetic scores but also enhances performance on general evaluation datasets such as AlpacalEval and Anera-hard.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper pioneers the exploration of textual aesthetics in Large Language Models (LLMs). The authors propose an aesthetic refinement process and create a dataset named TEXAES. Based on this dataset, they introduce a textual aesthetics optimization fine-tuning method called TAPO, which aims to enhance textual aesthetics without compromising content accuracy. Additionally, the paper develops two textual aesthetics assessment methods, one based on text analysis and the other on image analysis, to evaluate the aesthetics of LLM outputs. Experimental results indicate that using the TEXAES dataset and TAPO fine-tuning method not only improves aesthetic scores but also enhances model performance on general evaluation datasets such as AlpacalEval and Anera-hard.

### Strengths
1. The paper is the first to investigate textual aesthetics in LLMs, introducing the TEXAES dataset and TAPO fine-tuning method, providing a new direction for the aesthetic optimization of LLMs.
2. The paper empirically validates the effectiveness of the TEXAES dataset and TAPO method, demonstrating not only improved aesthetic scores but also enhanced model performance.
3. The paper develops two assessment methods based on text and image analysis, offering tools for a comprehensive evaluation of the aesthetics of LLM outputs.

### Weaknesses
1. The paper does not elaborate on the quantitative standards for textual aesthetics, that is, what constitutes good textual aesthetics, and how the consistency among human evaluators in textual aesthetics assessment is ensured.
2. From the examples shown in Figure 4, the paper's concept of textual aesthetics seems to involve only line breaks, bold fonts, and highlighting key points, which are relatively simple and may lack long-term research value.
3. The paper does not clearly explain how to determine if texts with better textual aesthetics are better understood, and how to ensure that human evaluators are making judgments based on content comprehension rather than just neat formatting.
4. The paper uses GPT-4o for automated assessment, but as a machine, GPT-4o processes only code/symbols/characteristics, and its suitability for evaluating textual aesthetics is questionable.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on textual aesthetics in large language models (LLMs). It first highlights the importance of textual aesthetics, which has been less explored compared to image aesthetics despite the widespread use of LLMs.
Contributions:
1. Dataset Construction:
• Developed an aesthetic data generation pipeline leveraging GPT - 4o for aesthetic polishing.
• Constructed the first aesthetic dataset in the LLM domain, TEXAES, with 50,390 prompts data.
2. Fine - Tuning Method:
• Proposed a textual aesthetics - powered fine - tuning method, TAPO, based on direct preference optimization. It uses the Plackett - Luce model with adjustable optimization weights to better leverage TEXAES and enhance aesthetic fine - tuning performance while preserving general performance.
3. Evaluation Methods:
• Developed two evaluation pipelines for textual aesthetics: one based on text and the other based on images.
• Validated the effectiveness of TEXAES and TAPO through extensive experiments. Fine - tuned LLaMA series models using TEXAES and TAPO and compared their aesthetic scores with state - of - the - art LLMs. Also, employed human experts for professional evaluation. Results showed improvements in aesthetic scores and general capabilities on some benchmarks.

### Strengths
Originality
• The paper shows originality in addressing textual aesthetics in LLMs, an area that has received less attention compared to image aesthetics. The construction of the TEXAES dataset and the proposed TAPO fine-tuning method are novel contributions.
Quality
• The research methodology appears to be of good quality. The construction of the dataset through an aesthetic polishing pipeline and the use of appropriate evaluation methods (text-based and image-based) demonstrate a systematic approach.
Clarity
• The paper is generally clear in its presentation. The introduction effectively sets the context and the importance of textual aesthetics. The methods section explains the dataset construction, fine-tuning method, and evaluation pipelines in a understandable manner.
Significance
• The work is significant as it fills a gap in the study of LLMs by focusing on textual aesthetics. The proposed techniques have the potential to improve the aesthetic quality of LLM outputs, which can enhance user experience and the usability of these models in various applications.

### Weaknesses
Dataset Limitations
• While the construction of the TEXAES dataset is a significant step, it may have limitations. The dataset is built based on a filtered version of UltraFeedback, and there could be potential biases introduced during this process. For example, the responses in UltraFeedback might already have a certain style or pattern that could limit the diversity of the aesthetic preferences captured in TEXAES.
Evaluation Complexity
• The evaluation methods, although comprehensive with text-based and image-based scoring, could be further refined. The use of GPT - 4o as a judge in both text and image evaluations might introduce some subjectivity and reliance on a single model. There could be a need for more diverse evaluation metrics or a more objective way to combine the text and image evaluations to get a more accurate assessment of textual aesthetics.
Generalizability
• The experiments are mainly focused on the LLaMA series models. It is not clear how well the proposed methods (TEXAES and TAPO) would generalize to other LLMs. There is a need for more extensive experiments across different types of LLMs to demonstrate the broader applicability of the techniques.

### Questions
1. Dataset Construction Details:
• Could the authors please provide more details about the filtering process used to create TEXAES from UltraFeedback? How were the responses selected and what criteria were used to ensure a diverse range of aesthetic preferences?
2. Evaluation Metric Objectivity:
• Given that GPT - 4o is used as a judge in both text and image evaluations, how can the authors ensure the objectivity of the evaluation metrics? Are there any plans to explore alternative evaluation methods or to combine multiple evaluation models to reduce subjectivity?
3. Generalizability to Other LLMs:
• The experiments focused on LLaMA series models. What are the authors' expectations or initial findings regarding the generalizability of the proposed methods (TEXAES and TAPO) to other LLMs? Have any preliminary tests been conducted on different models?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
In this paper, the authors tackle the problem of textual aesthetics in LLMs. To this end, they introduce a new dataset named TEXAES. In addition, they proposed Textual Aesthetics Preference Optimization (TAPO) method. The experiments show that the proposed method performs well on the benchmark datasets.

### Strengths
+ The paper is well organized. 
+ The problem of textual aesthetics in LLMs is interesting.

### Weaknesses
-I have doubts about the textual aesthetics scores. The scores should be decided by human, not by ChatGPT.
-The proposed textual aesthetics-powered training actually aims to predict the scores as close as ChatGPT, not human. 
-The authors did mention 3 evaluators, 2 graduate students and one professor. First, the number of evaluators is too small. Second, there is no information about the evaluations, for example, age, background, first language, and expertise.

### Questions
Why did the authors use ChatGPT to provide aesthetics scores? Why not human?

Why is there no information about the evaluators? Is it fair for the data collection?

### Soundness
3

### Presentation
2

### Contribution
2
