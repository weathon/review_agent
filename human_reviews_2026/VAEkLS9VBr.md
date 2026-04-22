# SpaCE-Eval: A Benchmark for Real-World Multi-Modal Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Multi-modal Large Language Models (MLLMs) represent a significant advancement in artificial intelligence. Among the growing capabilities exhibited by MLLMs, abilities to understand and reason in real-world environments stand out as particularly vital as a fundamental prerequisite for a wide array of real-world applications. The current methods for evaluating MLLMs often fall short in their ability to comprehensively assess these crucial capabilities. However, being able to reason on complex environment-scale spaces, for example, room spaces, building spaces, and even urban spaces, and to predict the future and plan actions, is essential for humans and various autonomous agents to survive in the real physical world. To address these gaps, we propose a visual-question-answering benchmark, **SpaCE-Eval** (**Spa**tial Reasoning, **C**ommonsense Knowledge and **E**nvironment Interaction) in the real world, designed to evaluate some of MLLM’s most important reasoning abilities in real-world environments. As the name suggests, it challenges the models to reason on complex spatial scenarios, invoke commonsense knowledge of the physical world, and interact with the environment. The dataset consists of all new diagrams purposefully produced by humans, where diagram-question pairs are meticulously refined and selected through a rigorous pipeline. Additionally, with the benchmark, we evaluate a selection of leading MLLMs, both proprietary and open source. The results suggest that a significant enhancement of MLLMs in reasoning in the real physical world is necessary to realise more advanced general artificial intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
SpaCE-Eval introduces a new visual question answering (VQA) benchmark designed to evaluate multi-modal large language models (MLLMs) on real-world spatial reasoning, commonsense knowledge, and environment interaction. The dataset features 1,139 high-quality, human-created diagram-question pairs spanning diverse spatial scales (from objects to urban spaces) and includes both textual and visual answer options. Extensive evaluation of leading MLLMs reveals significant gaps in spatial reasoning and simulation, especially as spatial scale increases, highlighting the need for more advanced real-world reasoning capabilities in AI models.

### Strengths
- Real-World Focus & Diversity: Covers a wide range of spatial scales and real-world scenarios, moving beyond object-level or synthetic tasks
- Rigorous Data Curation: All diagrams are newly created by human experts, eliminating data contamination and ensuring visual diversity.

### Weaknesses
- Need for artifacts: The paper identifies gap in MLLMs, provides a benchmark to help evaluate but should also provide basic artifacts like reasoning traces and training corpus that can be used to do SFT/RL and mitigate this gap, and further confirm the paper's hypothesis.
- Expand Task Formats: While broad in spatial scale, incorporating open-ended, multi-step, or interactive tasks (e.g., navigation, planning, or manipulation) to better reflect real-world reasoning demands would help the benchmark remain strong and relevant.
- Diagnostic Error Analysis: Provide more granular, qualitative analysis of failure cases, especially for abstract and large-scale spatial reasoning, to guide model and dataset improvements.

### Questions
Flags identified and covered above in weakness section

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
4

### Summary
This work proposes an expert-designed and peer-reviewed dataset aimed at evaluating the spatial reasoning abilities of various LLMs. This dataset comprises real-world VQA tasks across a range of contexts. Model outcomes show that for real-world spatial reasoning and use of LLMs in complex environment-scale spaces and creation of autonomous agents that can survive in the real world, these models require a great deal of enhancement and improvements to the reasoning process, and this work establishes this dataset as a human-verified benchmark to assess the same.

### Strengths
1. The dataset is well-curated, with a clear and comprehensive process for design, review, and modification.
2. The newly-designed nature of these problems eschews concerns about data leakage
3. Table 1 is a comprehensive comparison against similar datasets
4. The analysis section provides a useful look at model performance patterns and interpretations.

### Weaknesses
1. The data curation and review process are vital to the creation and success of this dataset. While the methodology is reasonable, there is no discussion of the soundness and reliability of this process - especially regarding the following;
- 1.1. The instructions provided for the design process, any rubrics that were followed, etc
- 1.2. Rubric for the review process, and how sounds this process was in terms of annotator agreement. This is especially relevant to the meta-annotators.
- 1.3 Examples of questions that did not pass quality control.

### Questions
Suggestions:
1. This work would benefit from demonstrating the question creation, review, and modification process for examples in the appendix, including questions that did not pass quality control
2. Fig 2 contains a typo in the diagram: SpeCE -> SpaCE
3. An exploration of how the quality control and review process was kept consistent and sound, and how annotator agreement was measured to ensure no bias or high variance in quality decisions, would lend more credibility to the curated dataset

Questions:
1. Given the small set of problem creators (51) all from the same background (university students), how do you ensure that problems are diverse not only in context and content, but also complexity?

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
The paper introduces SpaCE-Eval, a new visual-question-answering benchmark designed to evaluate the real-world reasoning abilities of Multi-modal Large Language Models (MLLMs). The dataset consists of novel, human-drawn diagrams to prevent data contamination and is structured into three categories: Spatial Reasoning, Commonsense Knowledge, and Environment Interaction. Evaluations across various MLLMs reveal that the benchmark is highly challenging, especially in the spatial reasoning domain where models perform poorly. The results highlight a significant gap between current model capabilities and human-level spatial understanding, underscoring the need for improved reasoning in complex physical environments.

### Strengths
1.  **Strong and Clear Motivation:** The paper does an excellent job of identifying a critical and timely gap in existing MLLM evaluation. It correctly argues that reasoning in complex, environment-scale spaces is a fundamental prerequisite for real-world applications like robotics and autonomous agents, and that current benchmarks often fall short in this area.


2.  **Proactive Mitigation of Data Contamination:** The core methodological decision to create the entire dataset from brand-new, human-drawn diagrams is a significant strength. This approach directly addresses the pervasive problem of data contamination, where models may have already seen test images during pre-training, ensuring a more honest evaluation of their reasoning abilities.


3.  **Comprehensive and Well-Defined Task Taxonomy:** The benchmark is thoughtfully structured into three distinct and meaningful categories: **Spatial Reasoning (SR)**, **Commonsense Knowledge (CK)**, and **Environment Interaction (EI)**. Breaking these down further into twelve subcategories provides a fine-grained framework for analyzing model capabilities and pinpointing specific areas of weakness.


4.  **Novel Emphasis on Multi-Scale Spatial Reasoning:** A key contribution is the benchmark's explicit focus on reasoning across multiple spatial scales—from objects and rooms to buildings and urban contexts. This is a crucial aspect of real-world intelligence that is often overlooked in object-centric datasets, and SpaCE-Eval makes it a central part of its evaluation.


5.  **Focus on Abstract and Schematic Interpretation:** While the use of diagrams limits its "real-world" photographic realism, it is a strength in its own right. The ability to interpret abstract representations like floor plans, maps, assembly instructions, and infographics is a vital cognitive skill. This benchmark is specifically designed to test this form of abstract visual reasoning.

### Weaknesses
1.  **Superficial Analysis of Model Failures:** The paper correctly identifies that models fail at spatial simulation and abstract reasoning. However, the analysis in Section 4.3 is brief. It doesn't deeply investigate *why* these failures occur. Is it a limitation of the vision encoder, the language model's reasoning capabilities, the vision-language alignment, or something else?


2.  **Dataset Balance and Composition:** Figure 2 shows the dataset composition. "Building Space" accounts for nearly 50% of the data by scale. This heavy imbalance could mean the overall performance scores are disproportionately influenced by model capabilities on this specific scale, potentially masking poorer performance at other scales like urban or object.


3.  **In-Depth Bias Analysis:** Beyond a brief mention of linguistic shortcuts, the paper lacks a thorough analysis of potential biases in the dataset. This could include:
    *   **Answer Distribution Bias:** Is the correct answer (A, B, C, D) uniformly distributed? The paper claims it is, but doesn't provide the final distribution.
    *   **Content Bias:** Are certain objects, architectural styles, or cultural contexts overrepresented?
    *   **Negative Examples ("Probing"):** Including questions where the answer is "cannot be determined from the image" to test if models are hallucinating or guessing.

### Questions
1.  **Clarity of Category Definitions:** What are the precise, mutually exclusive definitions for the subcategories? For instance, how do you differentiate a task requiring "Spatial Interpretation" (reasoning about perspectives) from one requiring "Space Association" (associating spaces across different views)?


2.  **Meaning of CLIP Similarity Score:** The paper reports a mean CLIP similarity of 0.654 to argue for the novelty of the diagrams. How should this value be interpreted? Without a baseline (e.g., similarity scores within other VQA datasets or between random web images), it is difficult to judge whether 0.654 indicates true novelty or moderate similarity.


3.  **Reliability of the Automated Judge:** The paper uses GPT-4o-mini to verify model predictions that are not exact string matches. What was the measured accuracy of this automated judge? How was it validated to ensure it does not have its own biases or make systematic errors in judgment, which could affect the reported accuracy of the models being tested?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SpaCE-Eval, a new visual question answering (VQA) benchmark designed to evaluate the reasoning capabilities of Multi-modal Large Language Models (MLLMs) in real-world environments. The authors argue that existing benchmarks often fall short by focusing on object-scale understanding and static scenes, neglecting the complex, multi-scale reasoning required for practical applications. SpaCE-Eval addresses this gap with three core categories: Spatial Reasoning (SR), Commonsense Knowledge (CK), and Environment Interaction (EI). The benchmark consists of 1,139 high-quality image-question pairs based on brand-new, human-created diagrams to prevent data contamination. A comprehensive evaluation of numerousSOTA MLLMs on SpaCE-Eval reveals significant limitations, particularly in spatial reasoning, where even the best models perform poorly. The analysis highlights models' difficulties with larger spatial scales, tasks requiring spatial simulation, and reasoning with visual-only options, underscoring the benchmark's value in guiding future MLLM development.

### Strengths
1. The paper's greatest strength is the meticulous construction of the SpaCE-Eval benchmark. By commissioning brand-new diagrams from individuals with design backgrounds, the authors successfully avoid data contamination and create visually diverse and conceptually rich problems. The four-stage quality control process ensures the reliability and difficulty of the questions.
2. SpaCE-Eval is uniquely comprehensive. The three categories which are Spatial Reasoning, Commonsense Knowledge, and Environment Interaction, cover a wide and crucial set of abilities for real-world intelligence. The benchmark's focus on multiple spatial scales (from objects to urban environments) is a significant step beyond existing datasets.
3. The benchmark is demonstrably difficult for even the most advanced MLLMs, with the best model scoring only 56.37%. The paper provides deep insights into specific model failures, such as the stark performance gap between questions with textual vs. visual options, the degradation of performance on larger spatial scales, and the inability to perform mental spatial simulations. These findings are crucial for understanding the current limitations of MLLMs.

### Weaknesses
1. There is a discrepancy between the highest Spatial Reasoning score reported in the abstract (31.75% for claude-sonnet-4) and the one in Table 2 (42.25% for GPT-5). This should be clarified and corrected for consistency.
2. While the quality-over-quantity approach is commendable, the total dataset size of 1,139 examples is relatively small for an evaluation benchmark. When broken down into the twelve subcategories, the number of questions per subcategory can become quite small (e.g., Space Association is ~3.25%, which is only about 37 questions), which may limit the statistical significance of the fine-grained analysis.

### Questions
1. Could you please clarify the inconsistency regarding the top performance in the Spatial Reasoning category? Which value is correct: 31.75% as stated in the abstract or 42.25% as presented in Table 2?
2. Given the modest size of the dataset, do you have plans to expand SpaCE-Eval in the future? A larger number of examples would strengthen the statistical power of the results, especially for the more niche subcategories.
3. The analysis shows a significant performance drop for questions with purely visual options (26.58% accuracy). Is this poor performance uniform across the three main categories (SR, CK, EI), or is it particularly higher in one area, such as Spatial Reasoning, which seems intuitively more visual?

### Soundness
3

### Presentation
3

### Contribution
3
