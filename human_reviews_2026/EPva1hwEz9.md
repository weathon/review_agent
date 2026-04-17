# Towards Trustworthy Dermatology MLLMs: A Benchmark and Multimodal Evaluator for Diagnostic Narratives

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Multimodal large language models (LLMs) are increasingly used to generate dermatology diagnostic narratives directly from images. However, reliable evaluation remains the primary bottleneck for responsible clinical deployment. We introduce a novel evaluation framework that combines DermBench, a meticulously curated benchmark, with DermEval, a robust automatic evaluator, to enable clinically meaningful, reproducible, and scalable assessment. We build DermBench, which pairs 4000 real-world dermatology images with expert-certified diagnostic narratives and uses an LLM-based judge to score candidate narratives across clinically grounded dimensions, enabling consistent and comprehensive evaluation of multimodal models. For individual case assessment, we train DermEval, a reference-free multimodal evaluator. Given an image and a generated narrative, DermEval produces a structured critique along with an overall score and per-dimension ratings. This capability enables fine-grained, per-case analysis, which is critical for identifying model limitations and biases. Experiments on a diverse dataset of 4500 cases demonstrate that DermBench and DermEval achieve close alignment with expert ratings, with mean deviations of 0.251 and 0.117 (out of 5) respectively, providing reliable measurement of diagnostic ability and trustworthiness across different multimodal LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a benchmark for evaluating multi-modal LLMs on the task of generating dermatology diagnostic narratives from images. Furthermore, as judging such narratives efficiently (e.g. automated pipelines) and accurately is challenging, authors develop an evaluator that can score such narratives at scale with strong alignment to expert human judges. The paper leverages a combination of supervised fine-tuning and reinforcement learning to train the evaluator on the proposed benchmark.

### Strengths
- The motivation of the paper is clear and has practical merit. 

- The benchmark is developed with the supervision of medical professionals, improving the reliability of the evaluation framework.

- The evaluator is a significant practical contribution that may enable more robust and trustworthy MLLM-based pipelines in dermatology.

### Weaknesses
- The paper doesn't provide sufficient details on human evaluations. How many experts have scored each narrative? What is the agreement between experts on the assigned scores?

- More details and ablation would be necessary on the training strategy used for evaluator training. In particular there is no study on the necessity of the RL stage. Could we obtain a comparable model with SFT-only without the added complexity? Furthermore, the reward seems to be easily hackable: it is only calculated over valid dimensions that can be parsed. Wouldn't this incentivize the model to generate invalid scores for categories it is unsure about and only score the easiest dimensions?

- The paper does not define the key evaluation dimensions. Even though some of them are self-explanatory (e.g. Accuracy), but others would require more explanation (e.g. what does Safety mean in this context?).

Minor comments:
- Line 264 mentions $x$ however it does not appear in the equation. 
- Line 265: I'm unsure what authors mean by "sentinel" here.
- Line 361: I would argue that evaluating on samples that were not in the training set is not to "avoid overfitting" but simple common sense practice in machine learning.

### Questions
- How is expert agreement considered when creating the benchmark?
- Is the RL stage necessary to obtain a performant evaluator? How is it supported? How is the specific form of the reward supported?
- How are the evaluation dimensions defined?

Minor:
- Wouldn't the results in Table 2 be biased towards Gemini models, as this model family has been used to generate the dataset?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to address the bottleneck of reliably evaluating AI-generated diagnostic narratives in dermatology. The authors propose an evaluation framework consisting of two main contributions: (i) DermBench: A benchmark dataset composed of 4,000 real-world dermatology images paired with expert-certified diagnostic narratives. This benchmark evaluates candidate narratives from models using an LLM-judge, which scores them against a certified reference text across six clinically-grounded dimensions: Accuracy, Safety, Medical Groundedness, Clinical Coverage, Reasoning Coherence, and Description Precision; and (ii) DermEval: A reference-free multimodal evaluator. Given only a dermatology image and a candidate narrative, DermEval is trained to generate a structured textual critique and assign scores for the same six dimensions.

### Strengths
The paper is well-motivated by the lack of datasets and methods that benchmark trustworthiness in generative LLMs for dermatology. This is a critical problem, given that today LLMs are widely used to help generate medical descriptions/captions for images, and there is no reliable measurement or evaluation method in this domain. This paper is among the first to bring this issue to attention.

### Weaknesses
- Before moving on to the method, it is important to standardize the 6-dimensional criteria on the reliability of generated text by LLMs. Specifically, what it means for a narrative to be "accurate, safe, grounded, comprehensive, coherent and precise"? What do you mean by accuracy and precision? What is the difference between accuracy being rated 3 vs. 4? On one hand, it's important to make sure that the dermatologist and the LLM to be evaluated share the same rubric for assessment. If this is not established, then it puts the entire framework's reliability at stake. On the other hand, these standardized criteria are critical for future users and practitioners as a reference or consensus, which is important for implementation purposes as a benchmark.
- Clarity-wise, I found that the method section really hard to follow. There are multiple types of text described in the work: image caption, pseudo-CoT output, hierarchical CoT output, evaluation text label, certified reference texts, candidate text, and gold reference text, along with various prompts at different stages. It is hard to keep track of the use and purpose of each textual input and output. I'd advise the authors to put them together in an overview figure to illustrate the use of each text input and output.
- Experiment-wise, there are just two results presented in the work, Table 1 and Table 2 (since Table 2 and Figure 6 are essentially the same results presented in different ways). Aside from the limited quantity, only Table 1 presents MAE results, aiming to demonstrate that DermBench and DermEval align well with dermatologist assessments on generated diagnostic narratives across the 6 criteria. But this result alone insufficiently establishes "the alignment of DermBench and DermEval with expert judgments" as claimed in Sec. 4.2.
- The certification process involves clinicians revising an existing model's output rather than writing a reference narrative from scratch. This process risks anchoring the "gold standard" texts to the style, structure, and potential biases of the initial model used for drafting. This is acknowledged in the limitations but remains a noteworthy weakness in the data's construction and is under-discussed quantitatively.

### Questions
- The weaknesses outlined above point to some fundamental issues with problem formulation consensus, method clarity, and experiment quality, which must be fully established before assessing the novelty and effectiveness of the proposed method.

- It seems like CoT is an important part of the method, but related literature in this line of work is poorly cited. Please add more details about how the proposed method is relevant to CoT development.

- The paper presents DermBench as a primary contribution, but the experiments (Table 1) show it is significantly less aligned with human experts than DermEval. This undermines its value as a standalone evaluation tool and reframes it as primarily a data-generation artifact used to train DermEval. Can the author discuss more about the performance difference and other potential applications of DermBench?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a comprehensive framework for evaluating MLLMs in dermatology, focusing on the generation of trustworthy diagnostic narratives from images. The authors identify that existing evaluation methods are inadequate for clinical settings, as they are either too generic (text similarity) or too expensive and slow (expert grading).

### Strengths
1. The framework moves beyond simple accuracy to evaluate narratives on six well-defined dimensions that are critical for clinical trustworthiness, such as safety and reasoning coherence.
2. The process for creating "certified references" is rigorous, involving generation followed by a human-in-the-loop process where board-certified dermatologists review and revise narratives until they achieve a perfect score. This ensures a high-quality gold standard.

### Weaknesses
1. DermBench uses a human-certified reference, but the final scoring is still performed by an LLM-judge. This introduces a potential layer of abstraction and bias. Although the alignment tests show this works well, the system's ultimate "ground truth" for benchmarking still relies on a model's judgment rather than direct human scoring of the candidate narratives.
2. The certified references are initially drafted by an MLLM before being revised by clinicians. This process might inadvertently anchor the style, structure, or content of the references to the generating model's biases. A process where references are drafted from scratch by clinicians would be the gold standard, though admittedly far less scalable.
3. The reliance on board-certified dermatologists for scoring and revision is a core strength but also a bottleneck. The cost and time required could make it difficult to expand this high-quality methodology to much larger datasets or other complex medical domains.
4. The authors acknowledge that their reliance on the DermNet dataset may limit diversity. Expanding to multiple image sources, including data from different demographic populations and acquisition settings, would be crucial for ensuring the robustness and fairness of both the benchmark and the evaluator.
5. Inter-Rater Reliability is not very clear. The paper mentions that physician scoring can be "noisy." Was any inter-rater reliability analysis (e.g., Cohen's Kappa or Krippendorff's Alpha) conducted among the dermatologists who provided the scores? Quantifying this would help contextualize the reported alignment errors of DermBench and DermEval.
6. DermEval showed a lower mean absolute error to expert ratings than DermBench (0.18 vs. 0.36). This is a significant finding. Do the authors have a hypothesis for why the reference-free evaluator is more aligned with experts than the reference-based one? Could it be that the LLM-judge in DermBench struggles with the comparative task, while DermEval learns the direct mapping from (image, text) to score more effectively?
7. Which specific LLM was used as the final LLM-judge in the DermBench evaluation pipeline? Its capabilities are critical to the benchmark's reliability. Was it the same model used for generating parts of the dataset (e.g., Gemini 2.5 Pro)?
8. For the alignment test in Section 4.1, 4500 narratives were generated and scored by dermatologists. Were the 500 images used for this test completely separate from the 4000 images used to build the training set for DermEval and the references for DermBench?
9. The aggregate MAE scores are compelling. Are there specific types of cases or clinical conditions where DermEval's alignment with experts is weaker? An error analysis could reveal limitations and guide future improvements.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new dermatology-focused benchmark DermBench and a corresponding evaluation model DermEval. DermBench pairs 4,000 real-world dermatology images with expert-certified diagnostic narratives and uses an LLM-based judging system to score candidate narratives across clinically grounded dimensions. This enables consistent and comprehensive evaluation of multimodal models in dermatologic diagnosis. For individual case-level evaluation, the authors propose DermEval, a reference-free multimodal evaluator that, given an image and a generated diagnostic narrative, produces structured evaluations including both overall and dimension-wise scores.

### Strengths
1. Prior work focusing on LVLM performance evaluation in dermatology is limited, and this study addresses this gap by providing both a benchmark and an evaluator tailored to this field.

2. The evaluation dimensions cover multiple clinically meaningful aspects beyond simple accuracy, which provides a more holistic understanding of model behavior.

### Weaknesses
1. The clarity of Figure 2 is unacceptable. The font size in the figure is significantly smaller than that of the main text, and even at 200% magnification, the text remains unreadable. This seriously affects the readability and professionalism of the manuscript.

2.  Since DermEval is designed to function without reference annotations, how is its evaluation reliability ensured? Comparing only against human ratings is insufficient. The paper lacks experiments that demonstrate the evaluator’s trustworthiness compared with closed-source evaluators or other reference-based methods. Given that evaluation reliability is central to the paper’s contribution, this is a critical omission.

3. The number of experiments and figures is insufficient. The main text contains only two tables and one figure, two of which simply present evaluation results of existing models on the proposed benchmark. Furthermore, the paper does not clearly explain how the proposed framework outperforms other benchmarking or evaluation pipelines in methodology or practical utility.

### Questions
1. Data Source and Ethics:

Where were the dermatology images sourced from?
If human subject data were involved, were ethical approvals obtained? The AC should pay special attention to this aspect.

2 .Human Evaluation Details:

The paper lacks sufficient information about human raters. How many dermatologists participated in the evaluation process, and what were their qualification levels? How was inter-rater consistency ensured to guarantee credible human supervision?

3. Ethical Use of Closed-Source Models:

The paper mentions using closed-source commercial models (e.g., Gemini) during the image annotation stage. Were the patient images used in this step processed under ethical approval and with informed consent?

4. Model Hallucination Correction:

During the first stage of data annotation, the pipeline relies heavily on Gemini for generating image captions. How were potential hallucinations or clinically incorrect outputs manually reviewed or corrected to ensure data quality?

### Soundness
2

### Presentation
2

### Contribution
2
