# MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Recent progress in the reasoning capabilities of multimodal large language models (MLLMs) has empowered them to address more complex tasks such as scientific analysis and mathematical reasoning. Despite their promise, MLLMs' reasoning abilities across different scenarios in real life remain largely unexplored and lack standardized benchmarks for evaluation. To address this gap, we introduce MMR-Life, a comprehensive benchmark designed to evaluate the diverse multimodal multi-image reasoning capabilities of MLLMs across real-life scenarios. MMR-Life consists of 2,646 multiple-choice questions based on 19,108 images primarily sourced from real-world contexts, comprehensively covering seven reasoning types: abductive, analogical, causal, deductive, inductive, spatial, and temporal. Unlike existing reasoning benchmarks, MMR-Life does not rely on domain-specific expertise but instead requires models to integrate information across multiple images and apply diverse reasoning abilities. The evaluation of 37 advanced models highlights the substantial challenge posed by MMR-Life. Even top models like GPT-5 achieve only 58% accuracy and display considerable variance in performance across reasoning types. Moreover, we analyze the reasoning paradigms of existing MLLMs, exploring how factors such as thinking length, reasoning method, and reasoning type affect their performance. In summary, MMR-Life establishes a comprehensive foundation for evaluating, analyzing, and improving the next generation of multimodal reasoning systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MMR-Life, a multimodal multi-image reasoning benchmark focusing on real-life scenarios. The benchmark contains  2,676 multiple-choice questions covering seven reasoning categories. Evaluation of state-of-the-art models shows deficiency in performance, and current reasoning enhancement strategies do not work too well on the benchmark.

### Strengths
- The paper is well-written and easy to follow, with an abundance of examples.
- The benchmark contains interesting challenges and performs a pretty thorough evaluation of sota models, showing their deficiency.
- The exploration of reasoning enhancement strategies provides some interesting insights.

### Weaknesses
- I appreciate the rich appendix section and accompanying analysis. But these examples are most likely cherry-picked, which is totally fine for presentation. It would be good to include a link to an anonymously hosted repository containing the full dataset so reviewers can get a more unbiased, holistic view of data quality.
- Related to above: Some of the problems in the appendix are of low quality. 
    - For example, on page 65, the ground truth explanation states that the answer should be "water sport/racing". Not all images in the question are water sports (e.g., there is horse racing), so it is perfectly fine to select E as the correct answer as well. 
    - On page 45, the question states "My friend already owns this pair of shoes", but four pairs of shoes are shown. This is rather confusing. Also, I don't think this question makes any sense. The ground truth, C, looks similar to the first pair in terms of color and the second in terms of style.
    - On page 77, I do not even understand the question. It seems to contain two completely separate questions?
    - On page 37, even after zooming in, I do not see where there is milk on the table.

The point I want to make is that the selected questions probably represent some of the higher-quality ones, yet some of them are still not totally reasonable for models to answer, so I think giving reviewers access to the full dataset is essential for quality control.
- The paper mentions a 600-question validation set. But I think it only appears once in the paper in the context of human performance. Model performance is reported on the whole set, but human performance is only on the validation set. They may not be directly comparable. What is the process for constructing this validation set? What is model performance on this subset?
- The paper claims the questions cover real-life scenes, but many are in fact pretty artificial or unlikely to come across in real-life (e.g., pages 40, 41 -- synthetic images; pg 60, 61 -- scientific and domain-grounded; pg 48, 49 -- cartoon scenes). These still carry relevance and assess multimodal reasoning, but weakens the paper's focus on real-life scenes.

### Questions
The paper says that "We generate question-answer pairs using either automatic synthesis or manual annotation, depending on the task type". What is the percentage of machine- and human-generated problems? I see there is a task description section in the appendix, but I think it is also important to clearly outline sources of questions (i.e., human/LLM-generated) for each task.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces MMR-Life, a large-scale benchmark designed to evaluate multi-image reasoning abilities of multimodal large language models (MLLMs) in real-world scenarios. It includes 2,676 multiple-choice questions derived from 19,367 real-life images, spanning seven reasoning types. Unlike previous benchmarks that rely on domain-specific knowledge or synthetic geometric objects, MMR-Life emphasizes integrating information across multiple real-world images. Experiments with a broad range of open and proprietary models reveal certain insights and gaps relative to human performance, underscoring the need for further progress in multi-image reasoning.

### Strengths
1. The paper tackles an important and timely problem - evaluating whether MLLMs can effectively handle multi-image, vision-based reasoning tasks, a topic of clear relevance to the ICLR community.
2. The MMR-Life benchmark covers a broad range of reasoning types across diverse real-world image contexts, enabling fine-grained analysis of model strengths and weaknesses and offering a well-structured evaluation framework.
3. The evaluation is thorough and well-rounded, including a diverse set of both open and proprietary models. Importantly, the authors also examine the impact of different training and inference methods, such as Chain-of-Thought, Self-Consistency, Best-of-N, and GRPO, adding an additional analytical dimension.
4. The paper provides insightful analyses of performance patterns across reasoning categories and includes an error taxonomy that helps characterize model limitations and failure modes.
5. The paper is clearly written, logically structured, and comprehensive in its literature review, making it accessible and informative to readers.

### Weaknesses
1. The benchmark largely repurposes existing datasets (Appendix C.1), which limits its novelty. The primary contribution lies in the reorganization and categorization of these datasets by reasoning type rather than in introducing new data or task formulations.
2. The evaluation omits several important baselines, including representative MLLMs (e.g., LLaVA, InstructBLIP) and non-MLLM approaches such as supervised CNNs or few-shot/meta-learning methods (Prototypical Networks, SNAIL, MetaBaseline).
3. The human performance evaluation (Section 3.1) appears shallow and exhibits high variance. Each of the 600 validation questions seems to have been solved by only one annotator, making it unclear how results would generalize across individuals. Moreover, the comparison to models could be more balanced if model performance were reported on the same subset of questions used for human evaluation.
4. The claims of a substantial human–model performance gap (e.g., Figure 1) seem overstated. As shown in Table 3, GPT-5 actually outperforms humans on three reasoning categories (Analogical, Deductive, Inductive), with humans leading in the remaining four.
5. The Spatial reasoning category, identified as the most challenging, has already been explored in prior work - particularly in the SPACE benchmark [Ramakrishnan, 2025]. This overlap somewhat limits the unique contribution and potential utility of MMR-Life for advancing spatial reasoning evaluation.

Ramakrishnan, Santhosh Kumar, et al. "Does spatial cognition emerge in frontier models?" International conference on learning representations. 2025.

### Questions
1. Table 2: Why is the source of MMR-Life marked solely as A (Annotated)? Given the benchmark heavily draws from existing datasets, shouldn’t it also be labeled as E (Existing datasets)?
2. Can the authors clarify why human performance falls below GPT-5 in the Analogical, Deductive, and Inductive categories? Does this reflect genuine model advantages, human weaknesses, or potential biases in the evaluation setup?
3. How can the insights derived from the benchmark inform future model architecture design or prompting strategies to enhance multi-image reasoning?
4. Were the authors able to verify whether the benchmark’s source datasets were not included in the training data of the evaluated open models?
5. As noted, Keye-VL-1.5-8B and InternVL3.5-8B perform worse than random guessing. Can the authors explain this behavior? Does it stem from output formatting issues, context-length limitations, or other factors?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces MMR-Life, a benchmark for evaluating multi-image and multimodal reasoning in realistic scenarios. It builds 2,676 multiple-choice questions from 19,367 real-world images, covering seven reasoning types and 21 tasks. The authors construct the dataset through automatic question generation plus human filtering, then evaluate 37 vision–language models to compare with human performance. Results show a clear gap between models and humans, especially on spatial and temporal reasoning, so the benchmark is intended to serve as a diagnostic testbed for future multimodal reasoning model

### Strengths
1. The benchmark uses 2,676 MCQ questions built from 19,367 real-world images, with many questions requiring information integration across several images, so models cannot rely on single-image shortcuts.
2. Reasoning tasks are organized into seven well-defined types (abductive, analogical, causal, deductive, inductive, spatial, temporal) and 21 tasks, providing a structured view of where current models fail.
3. The analysis across reasoning types shows differentiated difficulty profiles, which makes the benchmark useful as a diagnostic tool rather than only a leaderboard.

### Weaknesses
1. The paper analyzes the effect of longer thinking traces only at the overall level, lack discussion by break down reasoning type, it is unclear whether “longer thinking ≠ better” holds uniformly across tasks.
2. Human-level performance is based on a subset (validation questions), whereas model results are reported on the full benchmark, which makes the human–model comparison not fully aligned and may overstate the human gap.
3. Error analysis stays relatively shallow and mostly descriptive, could reveal deeper insights, guiding future model improvements.

### **Minor comments**
1. Figure 6 font size is a bit small
2. In Figure 48 the chosen option seems defensible from the question phrasing.
3. Table 2 lists MMMU as a comparison point, but MMMU is not a single-image benchmark.

### Questions
Please refer to weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
