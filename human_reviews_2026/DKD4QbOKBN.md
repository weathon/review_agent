# EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Most existing benchmarks for egocentric vision understanding focus primarily on daytime scenarios, overlooking the low-light conditions that are inevitable in real-world applications. To investigate this gap, we present EgoNight, the first comprehensive benchmark for nighttime egocentric vision, with visual question answering (VQA) as the core task. A key feature of EgoNight is the introduction of day–night aligned videos, which enhance night annotation quality using the daytime data and reveal clear performance gaps between lighting conditions. To achieve this, we collect both synthetic videos rendered by Blender and real-world recordings, ensuring that scenes and actions are visually and temporally aligned. Leveraging these paired videos, we construct EgoNight-VQA, supported by a novel day-augmented night auto-labeling engine and refinement through extensive human verification. Each QA pair is double-checked by annotators for reliability. In total, EgoNight-VQA contains 3658 QA pairs across 90 videos, spanning 12 diverse QA types, with more than 300 hours of human work. Evaluations of the state-of-the-art multimodal large language models (MLLMs) reveal substantial performance drops when transferring from day to night, underscoring the challenges of reasoning under low-light conditions. Beyond VQA, EgoNight also introduces two auxiliary tasks, day–night correspondence retrieval and egocentric depth estimation at night, that further explore the boundaries of existing models. We believe EgoNight-VQA provides a strong foundation for advancing application-driven egocentric vision research and for developing models that generalize across illumination domains.  The code and data can be found in https://github.com/dehezhang2/EgoNight

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents EgoNight, a new dataset and benchmark for low-light egocentric video understanding. EgoNight has three subsets, (1) a synthetic video dataset rendered by Blender, where the authors can have more fine-grained control of the scene and the lighting conditions, (2) an aligned day-night real video dataset collected by having video recorders walking around Sofia, Bulgaria at similar speed and route during the day and the night, and (3) a subset of the Oxford day-and-night video dataset. For each subset, a set of questions and corresponding answers are generated for each of the 12 type of questions, and these generated questions and answers are verified by human annotators. In addition to VQA, the benchmark also includes day-night correspondence retrieval, temporal localization, and night-time depth estimation as additional tasks for evaluating the performance of multimodal LLMs under low-light egocentric video conditions.

### Strengths
- The paper is well written and easy to understand. The figures are clear and provides a great summarization to the dataset construction pipeline. The paper provides abundant examples from the dataset and both qualitative and quantitative results (some in the supplementary material).
- The new real video recordings in the dataset (especially EgoNight-Sofia) can be a good contribution to the community, especially its aligned day-night recordings captured through the video-guided recording strategy.
- The day–night performance gap illustrated in figure 5 is very significant, highlighting the necessity for a low-light egocentric video dataset and benchmark.
- In addition to VQA, the benchmark also include a few additional tasks that have decent practical value, and allows the user to evaluate the capabilities of MLLMs in low-light egocentric videos more holistically.

### Weaknesses
- The paper lacks sufficient discussion regarding the human verification stage. Since most of the ground-truth answers are initially generated from multimodal LLMs, the quality of the human verification stage directly determines the quality of the dataset. Therefore I think the paper should provide more information about the human verification stage, including detailed guidelines to the human annotators and reporting inter-annotator statistics. It would be even better if a subset of the data can be annotated by humans without the help of LLMs to check whether there is bias in the human verification process.
- There are a few ethical concerns regarding the dataset. Details in the ethical concerns section below.

### Questions
I wonder whether the authors have did some more controlled experiment with the synthetic data generator to explore why there is such a significant gap in performance between the daytime and nighttime videos. What factors (e.g. illumination, motion blur, location of light source) contribute most to the performance degradation? What are the main failure modes in low light conditions (e.g. spatial confusion, missing small objects)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces EgoNight, the first comprehensive benchmark dataset designed to evaluate egocentric vision understanding, particularly for Multimodal Large Language Models (MLLMs), under challenging nighttime and low-light conditions. The authors identify that existing egocentric datasets focus almost exclusively on well-lit daytime scenarios, ignoring a critical operational domain for real-world applications like personal assistants. The core contribution is the creation of a benchmark suite featuring day-night aligned videos, which is crucial for two reasons: (i) It enables a rigorous and fair comparison of model performance, isolating the performance drop caused by illumination changes. (ii) It facilitates a novel "day-augmented auto-labeling" pipeline, where the clear daytime videos are used to help generate high-quality annotations for the ambiguous nighttime videos.

The EgoNight dataset is sourced from three streams:
- EgoNight-Synthetic: 50 perfectly aligned pairs from a Blender simulation.
- EgoNight-Sofia: 20 real-world pairs captured with a novel video-guided recording protocol to ensure "decent alignment".
- EgoNight-Oxford: 20 unaligned nighttime videos from an existing dataset to test generalization.

Upon this data, the authors build three benchmarks:
- EgoNight-VQA (the core task): 3,658 human-verified question-answer pairs spanning 12 diverse QA types, including novel tasks like "lighting recognition" and "non-common-sense reasoning".
- Day-Night Correspondence Retrieval: Evaluates spatial and temporal matching across illumination conditions.
- Egocentric Depth Estimation: Uses the synthetic data's ground-truth depth to test models at night.

Extensive experiments on over 10 state-of-the-art MLLMs (including GPT-4.1 and Gemini 2.5 Pro) reveal a substantial performance drop from day to night (e.g., 32.8% on Synthetic, 25.0% on Sofia), demonstrating that current models are not robust to low-light conditions and validating the difficulty and necessity of this new benchmark.

### Strengths
- The paper tackles a critical and previously overlooked gap in egocentric vision. The benchmark (EgoNight) is the first of its kind and will be a valuable community resource.

- The paper's use of both perfectly-aligned synthetic data and realistically-aligned real-world data provides a robust and comprehensive foundation for evaluation.

- High-Quality Annotation Pipeline: The "day-augmented auto QA generation" followed by over 300 hours of human verification ensures the reliability of the core VQA benchmark, which is especially difficult for low-visibility data.

- Comprehensive Evaluation: The paper presents not just one VQA task, but a suite of three benchmarks (VQA, Retrieval, Depth).

### Weaknesses
- The scale of real-world data is pretty small. Only 20 day–night pairs in EgoNight-Sofia dataset.

- The quality of simulation data is uncertain. Both qualitative and quantitative studies are missing.

### Questions
- Could the authors provide more analysis and examples of the synthetic data?

### Soundness
2

### Presentation
2

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
This paper introduces EgoNight, a new benchmark suite designed to evaluate egocentric vision understanding in challenging nighttime or low-light conditions. The authors identify this as a critical but overlooked area, as most existing benchmarks focus on daytime scenarios.
The primary contributions are threefold:
1. A Novel Dataset: The EgoNight dataset combines three sources: a synthetic dataset with perfectly aligned day-night video pairs (EgoNight-Synthetic), a real-world dataset with carefully recorded, temporally aligned day-night pairs (EgoNight-Sofia), and an unaligned real-world night video dataset for diversity (EgoNight-Oxford). The day-night alignment is a core feature, enabling controlled analysis of model performance under different illumination.
2. A Comprehensive Benchmark Suite: The main task is Visual Question Answering (VQA), featuring 3,658 human-verified QA pairs across 12 diverse types, including novel tasks like lighting recognition and non-common-sense reasoning. The suite is complemented by two auxiliary tasks: day-night correspondence retrieval and egocentric depth estimation.
3. Some Empirical Analysis: The authors evaluate a wide range of state-of-the-art Multimodal Large Language Models (MLLMs), including closed-source models like GPT-4.1 and Gemini. The results consistently show a substantial performance drop from day to night across all models and tasks, highlighting that current systems are not robust to illumination changes.

### Strengths
1. Addresses a Relevant Problem: The paper highlights the practical and important challenge of model robustness to illumination changes, a common failure mode for vision systems in real-world applications.
2. Significant Data Collection Effort: The authors have clearly invested a substantial amount of effort in collecting and annotating a new dataset, particularly in creating the temporally aligned real-world videos (EgoNight-Sofia). This resource, if made public, could be of some use to the community.

### Weaknesses
1. Limited Novelty: The work feels incremental. It takes the well-established paradigm of egocentric VQA and applies it to the nighttime domain. This is more of a domain generalization study on a new dataset than a novel research contribution. Prior work like the Oxford Day-and-Night dataset already provided a basis for nighttime egocentric vision, and benchmarks like EgoCross have already explored cross-domain challenges for egocentric VQA. The current work is a straightforward combination of these existing concepts.
2. Potential for Dataset Contamination and Bias: The VQA pairs were bootstrapped using GPT-4.1. This creates a significant risk that the dataset is "contaminated" with the stylistic and factual biases of the model used to generate it. Consequently, the benchmark may unfairly favor models that are architecturally or stylistically similar to GPT-4.1, making it an unreliable tool for impartially assessing a diverse set of MLLMs.
3. Flawed and Unreliable Evaluation: The use of an LLM-as-a-Judge for open-ended VQA is a major methodological weakness. This evaluation strategy is not yet considered robust or unbiased in the research community. Without calibration against human judgments, the reported accuracy numbers are not trustworthy and cannot be reliably used to compare models. This undermines the paper's core empirical claims.
4. Presentation Can Be Further Addressed: The paper suffers from some grammatical errors and lacks the polish expected for a top-tier publication. The writing style is often convoluted, making it difficult to parse key details of the data collection and annotation pipeline.

Here are some mistakes i listed, but more: 

Page 2, Line 70-71: "This motivates us to investigate egocentric vision at night, with an emphasis on complex scene understanding and reasoning tasks."
Issue: The sentence structure is a bit convoluted. "With an emphasis on" feels like an add-on. A clearer phrasing might be: "This motivates us to investigate egocentric vision at night, focusing on complex scene understanding and reasoning tasks."

Page 2, Line 74-75: "A central challenge in constructing such a benchmark lies in obtaining suitable video sources that capture the characteristics of nighttime environments, as well as developing annotation methods that ensure high labeling quality."
Issue: The phrase "as well as developing" creates a slightly unbalanced structure. A more parallel construction would be better: "...lies in obtaining suitable video sources... and developing annotation methods..."

Page 2, Line 83-84: "However, in practice, collecting perfectly aligned day-night pairs in the real world is highly nontrival."
Issue: "Nontrival" is not a standard English word. It should be "non-trivial." This is a clear spelling/typographical error.

Page 5: Starting from 3.2, there's always a space at the beginning of the paragraph, which is not aligned with the ICLR template.

### Questions
Please address the weakness in the above. Some additional questions: 

1. The paper's main argument rests on its novelty as the "first comprehensive benchmark for nighttime egocentric vision." However, given existing work like the Oxford Day-and-Night dataset, could you clarify what makes your contribution fundamentally different, rather than just an extension in scale and task diversity? 

2. Regarding the scope of the benchmark: The paper presents EgoNight as a suite for evaluating existing models. However, the true value of a new benchmark often lies in its ability to drive the development of new models. Have the authors considered creating and releasing a dedicated training set? Providing a training split would fundamentally increase the benchmark's impact, allowing the community to build and fine-tune models specifically for the challenges of nighttime egocentric vision. Without a training set and a baseline evaluation of a model fine-tuned on it, the novelty and contribution of the work could be seen as limited to just a test set. 

3. The paper's analysis is too simple. While it successfully demonstrates that models fail in low-light conditions, it provides almost no insight into why they fail or how the community should go about fixing them. The key takeaway is a problem statement, but it leaves researchers with no clear direction. For example, is the core issue a lack of diverse, low-light data in the models' massive pre-training datasets, or can this deficit be effectively overcome with targeted fine-tuning (post-training) on a smaller, specialized dataset like EgoNight? The paper does not investigate this. The analysis in Figure 5b shows that perception-oriented tasks (e.g., object/text recognition) degrade more severely than reasoning tasks (e.g., navigation). This is a good observation, but the paper stops there. It fails to diagnose whether this is due to the vision encoder failing to extract meaningful features from noisy pixels (an issue with the pre-trained foundation) or the language model's inability to handle uncertain visual inputs (an issue that might be solved via fine-tuning). The analysis simply presents the final scores without diagnosing the root cause of the failure.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces EgoNight, a benchmark focused on egocentric vision understanding in nighttime. It consists of three data sources: (i) EgoNight-Synthetic, 50 ideally aligned egocentric pairs with varying illumination levels using a simulator; (ii) EgoNight-Sofia, 20 pairs of real-world egocentric videos with spatially and temporally aligned day–night counterparts; (iii) EgoNight-Oxford,  20 nighttime videos from the Oxford Day-and-Night dataset

### Strengths
- While most egocentric datasets are overwhelmingly focused on well-lit, daytime scenarios, this paper provides valuable nighttime egocentric data and thus has high significance for real-world applications.
- The proposed benchmark includes a comprehensive list of QA tasks, along with three critical tasks (day-night correspondence retrieval, temporal localization, egocentric depth estimation) that are unique for nighttime videos.
- The paper is well-written and easy to follow with carefully designed figures.

### Weaknesses
- The EgoNight-Sofia (real-world) dataset is the most valuable and novel component in the proposed benchmark. However, it is quite small with only 20 video pairs. The vast majority of QA pairs (2029 of 3658) come from the synthetic dataset, which may not capture the full complexity of real-world sensor noise and lighting artifacts.
- The day-augmented annotation pipeline could introduce a bias. It risks generating questions about details that are perfectly visible in the day video but are genuinely invisible or non-inferable in the night video. Is this considered in the human verification stage?
- The evaluation uses an "LLM-as-a-Judge" (GPT-4.1) to score answers, which might introduce bias towards its own outputs. GPT4.1 achieving the highest accuracy under this metric is less convincing.

### Questions
- Could the authors explain the human verification process for the day-augmented annotation? Specifically, how were QA pairs handled if the answer was only derivable from the day video and not the night video?
- Are there any strategies to address the synthetic-to-real gap in EgoNight-Synthetic?
- In Figure5, is NonC in the egoNight-Syn bar plot a typo? Could the authors use the same color for the same task to improve readability?
- What is the performance of reasoning on MLLM-generated captions of the videos?

### Soundness
3

### Presentation
3

### Contribution
3
