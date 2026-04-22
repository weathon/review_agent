# VideoMathQA: Benchmarking Mathematical Reasoning via Multimodal Understanding in Video

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 4, 2

## Abstract
Mathematical reasoning in real-world video presents a fundamentally different challenge than static images or text. It requires interpreting fine-grained visual information, accurately reading handwritten or digital text, and integrating spoken cues, often dispersed non-linearly over time. In such multimodal contexts, success hinges not just on perception, but on selectively identifying and integrating the right details from a rich and noisy stream of content. To this end, we introduce VideoMathQA, a benchmark designed to evaluate whether models can perform such temporally extended cross-modal reasoning on videos. The benchmark spans 10 diverse mathematical domains, covering videos from 10 seconds to over 1 hour. We employ graduate-level experts to ensure high quality, for over 920 man-hours of annotation. To reflect real-world scenarios, questions are designed around three core reasoning challenges: direct problem solving, conceptual transfer, which requires applying learned methods to new problems; and deep instructional comprehension, involving multi-step reasoning over extended explanations and partially worked-out solutions. Each question includes multi-step reasoning annotations, enabling fine-grained diagnosis of model capabilities. Through this benchmark, we establish an evaluation framework for models that must reason, rather than merely perceive, jointly ground concepts across visual, audio, and textual modalities, across temporally extended mathematical problem settings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces VideoMathQA, a new benchmark designed to evaluate mathematical reasoning in videos. The authors argue that this task presents unique challenges not found in static-image or text-based benchmarks, requiring models to interpret fine-grained visual information, read text, and integrate spoken cues that unfold non-linearly over time.

### Strengths
1. The effort invested in data quality is a major strength. The three-stage annotation pipeline (Video Selection, QA Annotation, Step-wise Reasoning) is exceptionally rigorous. Using "expert science graduates" and "involving different annotators for verification at each stage" ensures high fidelity. The fact that this process required "115 person-days" demonstrates a significant and laudable effort.
2. The experimental setup is thorough. The paper evaluates a wide range of models using four distinct strategies: standard Multiple-Choice (MCQ), a more robust Multi-Binary (MBin) evaluation, Chain-of-Thought (CoT) vs. direct answering , and a detailed Step-wise Reasoning Evaluation.
3. The analysis provides clear insights into current model limitations including: (a) A large gap between human performance and the best model (b) The error analysis (c) Temporal Analysis and (d) The ablations on subtitles and frame sampling.

### Weaknesses
1. The primary limitation, which the authors correctly identify in the appendix, is the size of the dataset (420 video-question pairs). While the quality is extremely high, this small scale limits its utility for training and raises questions about the statistical robustness of the evaluation.
2. The benchmark appears to be English-centric. The paper does not specify the languages covered, but the examples (e.g., in Figure 1) and discussion imply an English-only dataset. This limits its applicability for evaluating mathematical reasoning in other languages, where instructional content and terminology may differ.

### Questions
See Weaknesses.

### Soundness
4

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
3

### Summary
This paper introduces VideoMathQA, a benchmark designed to evaluate multimodal mathematical reasoning in real-world videos, addressing the limitations of existing static image/text-based math benchmarks that lack support for temporal extension, dynamic visuals, and cross-modal integration. The benchmark comprises 420 manually curated video-question pairs spanning 10 mathematical domains and video durations from 10 seconds to over 1 hour. Each pair includes expert-annotated multi-step reasoning (2,945 total steps) with timestamps, enabling fine-grained evaluation of intermediate inference.

### Strengths
1. VideoMathQA fills a gap by focusing on temporally extended cross-modal reasoning for math, an underexplored area in existing video benchmarks.  
2. The benchmark leverages graduate-level experts to create detailed step-wise reasoning with timestamps. 
3. The four evaluation strategies address limitations of traditional MCQ and provide nuanced insights.    
4. The authors systematically investigate factors impacting performance (model size, video duration, subtitles, frame sampling) and conduct error analysis across 7 categories, offering actionable guidance for model improvement.

### Weaknesses
1. With only 420 video-question pairs, the benchmark may lack sufficient diversity to generalize across all real-world math instructional scenarios. The high annotation cost raises concerns about scalability.

### Questions
1. Given the high annotation cost, what semi-automatic or crowdsourcing strategies are you exploring to scale VideoMathQA? Could synthetic data or data augmentation techniques be integrated without compromising the "real-world" essence of the benchmark?
2. Models perform poorly on some categories, such as topology and graph theory. Could you provide qualitative examples of why these domains are more challenging for current models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduce VideoMathQA, a new benchmark focus on cross-modal reasoning on video modality. This dataset is annotated by raduate-level experts with a long annotation process. It fills the gap of high quality math reasoning dataset in the video modailty scope.

### Strengths
1. VideoMathQA tests different reasoning scenarios: Direct Problem Solving, Conceptual Transfer, and Deep Instructional Comprehension, which supports hierarchical evaluation and training of models with varying levels of reasoning.
2. The annotation process uses expert manpower to ensure annotation quality, thereby contributing to the benchmark community that current primarily relies on model auto generation.
3. It supports a comprehensive evaluation to MLLM on Math Reasoning with the four question type: MCQ, MBin, Cot vs Direct, and step-wised.

### Weaknesses
1. Data scale: The benchmark only contains 420 videos.
2. Lack the SOTA Proprietary Models' performace: the paper hasn't test GPT-5 and Gemini 2.5-pro
3. The absence of cross-modal ablation experiments makes it difficult to quantify the actual contributions of each modality to mathematical reasoning.
4. The paper use Qwen-3-4B as the judge model for Stepwise Reasoning Evaluation, but this is a relatively small size model. So the model's ability in evaluations is questionable, and it also lacks comparative experiments with human eval to demonstrate its feasibility.

### Questions
1. The overall dataset is not large. After distributing 420 videos across 10 categories, some categories contain as few as 17 videos (topology and graph theory each account for 4%). Within a category, videos are further distributed across different lengths. As a result, certain categories have only single-digit counts of videos at specific lengths. Could this introduce bias and error?
2. In Fig2 a), the sum of all proportion is **112%**, how could this happen?

### Soundness
3

### Presentation
4

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
VideoMathQA introduces a benchmark for math problem solving in instructional videos, bridging traditional text/image-based QA and full multimodal video reasoning. It includes a few thousand curated QA pairs from formula-rich YouTube videos, each with aligned frames, audio, transcripts, and formulas across 10 math domains (geometry, calculus, statistics, etc.). Each question features step-by-step solutions (chain-of-thought) for detailed evaluation. The benchmark targets three scenarios: Direct Problem Solving, Conceptual Transfer, and Deep Instructional Comprehension. Baselines with models like Math-LLaVA, MathBLIP, and Video-CoT reveal a large gap from human performance, underscoring the difficulty of integrating visual, textual, and temporal reasoning in mathematical contexts.

### Strengths
1. Introduces the first benchmark for math reasoning in instructional videos, capturing dynamic visual and spoken content absent in prior static text/image datasets.

2. Built through 920+ hours of expert annotation, covering 420 problems (~4.5K QA pairs) with aligned formulas, video timestamps, and full step-by-step solutions.

3. Spans 10 math domains and varied video styles (lectures, tutorials, handwritten, slides), testing both short-term perception and long-range reasoning.

4. Measures both answer accuracy and chain-of-thought alignment, includes conditions with/without subtitles, and offers detailed error categorization for model failures.

5. Evaluates multiple vision-language models (e.g., Qwen-VL, InternVL, Math-LLaVA), showing large performance gaps to human accuracy, confirming the benchmark’s difficulty and diagnostic value.

### Weaknesses
1. The five-option format simplifies scoring but allows guessing or elimination, limiting evaluation of free-form reasoning and creativity.
2. Some Conceptual Transfer questions may be solvable without watching the video, letting models rely on prior knowledge rather than true video understanding.

### Questions
1. Is there any analysis of the model performance without any video and only given the question?
2. Is there any analysis on how reliable the evaluation scores are? is the model following the video at all, or having the video distracting the model from the correct solution?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a benchmark called videoMathQA which contains math/reasoning questions about materials presented in a video. The author claims that this work provides realistic reasoning challenges and they highlight that they use 920 hours of human labour for the annotation.

### Strengths
1. The presentation of the paper is clear and the paper writing is good.

2. The human labels might be useful as a point of comparison.

### Weaknesses
1. __Limited technical contribution__: 

1.1. While the authors claiming that they have filled the gap of video-based reasoning in math or very specific domains, there is work in [1] which have (1). videos in math, biology and various other scientific domains; and (2). question with high-quality human annotations which include reasoning. Therefore, the contribution might be incremental. A discussion and comparison should be made.

[1] Sun et al. "video-SALMONN-o1: Reasoning-enhanced Audio-visual Large Language Model". ICML 2025.
 
1.2. The size of the dataset is questionable especially when there are only 420 MCQs. The authors should justify whether 420 questions are enough to draw statistically significant conclusions when comparing different models. Especially when 10 sub-categories are given, as shown in Table 1, there are a lot of systems having the same score because there are not enough data samples. This makes analysis in the results part unconvincing because the difference might just be getting a couple more questions right. \
I would encourage the authors to theoretically compute the number of samples needed to have statistically significance in their results.

1.3. Statistics and verification of human annotator quality missing. Statistics and evidence should be provided to show whether the annotators are reliable, such as inter-annotator agreement for their reasoning steps etc. I do not see a reason why we need the number of human-hour since the efficiency of human annotators can vary a lot.

2. __Lack of justification for the need of Human Reasoning__:

While a lot of efforts have been made to human reasoning annotation, it is unclear whether those are helpful performance indicators. While the authors try to do a comparison using LLM-as-a-judge. I have the following concerns regarding this part:
  - I do not understand the results stated in Appendix C (Note that without referring to Appendices, the justification for this is missing in the main paper). Proper metrics, e.g. Pearson Correlation Coefficient at sample level should be given instead of saying the performance metrics are close to each other. Per-sample PCC is suitable here because we are assigning scores and comparing the absolute values of the scores. 
  - An easier and more objective way would be to perform rollout and check the rate of correctness to assign scores to each reasoning step. This is standard in reasoning literature and is more controllable and convincing to me than using LLM-as-a-judge, especially with such a small LLM. I wonder why the authors did not try this.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
1
