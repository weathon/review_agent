# ChartNexus: Evaluating Multi-Chart Reasoning Capabilities of Multimodal Large Language Models

- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
While Multimodal Large Language Models (MLLMs) have achieved remarkable success on single-chart question answering tasks, reaching over 90% accuracy on benchmarks such as PlotQA, this apparent success masks a critical limitation. Current models perform poorly on complex, multi-chart reasoning tasks that mirror real-world analytical scenarios. In professional document analysis, users typically integrate information across multiple visualizations within rich contextual frameworks rather than examining isolated charts, which is a capability that remains largely unexplored in existing evaluations. To bridge this gap, we introduce ChartNexus, a novel and challenging benchmark specifically designed to assess multi-chart reasoning capabilities of MLLMs in authentic document contexts. ChartNexus comprises 1,370 carefully curated question-answering pairs derived from 6,793 real-world charts spanning 18 domains, including scientific papers, government reports, and industry analyses. Each question demands complex reasoning skills, such as comparative analysis, sequential information integration, and cross-modal synthesis between visual and textual elements. We design a comprehensive taxonomy featuring 4 high-level difficulty categories and 11 fine-grained sub-categories to systematically evaluate these capabilities. Our comprehensive evaluation of 23 state-of-the-art MLLMs reveals significant performance degradation compared to single-chart benchmarks. While the best commercial model achieves over 90% accuracy on simpler tasks, its performance drops by more than half on ChartNexus. Through systematic failure analysis, we identify critical weaknesses in current models’ ability to maintain working memory across multiple charts, perform cross-modal reasoning, and integrate contextual information effectively. ChartNexus establishes a new frontier for evaluating complex chart understanding capabilities, demonstrating that robust multi-chart reasoning remains an open challenge. Our benchmark and comprehensive analysis provide the research community with essential diagnostic tools to advance the development of more capable and practically useful MLLMs for real-world document analysis scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces ChartNexus, a benchmark for evaluating multi-chart reasoning in multimodal models. The authors collect charts from diverse sources in multiple languages, build a question-generation pipeline, and perform human annotation for question refinement and answer generation. They evaluate a range of models and find that significant gaps remain in multi-chart understanding and reasoning.

### Strengths
- The benchmark is well-curated, incorporating diverse real-world sources in multiple languages and including a human annotation process.
- The combination of multilingual (non-English) data and multi-chart questions is novel and valuable for assessing realistic multilingual chart understanding.
- The breakdown of results by question type and reasoning skill (e.g., numerical, identify, compare, reason) provides diagnostic insight into capability-specific weaknesses.
- The evaluation is comprehensive, covering a broad range of commercial, open-source, and chart-specific models.

### Weaknesses
* Incomplete or inaccurate comparison with other benchmarks:
  * Some relevant multilingual chart understanding datasets (e.g., PolyChartQA) are missing from the comparison.
  * Benchmarks such as CharXiv include unanswerable questions, but the table incorrectly marks this feature as absent.
* The reported inter-annotator agreement of 93.4% could implicitly reflect human performance (if re-annotation is considered as human evaluation), but no explicit human performance baseline is provided. Without this, it is hard to assess the human–model gap, especially for potentially ambiguous or erroneous questions.
* The use of Qwen3-32B as an automated evaluator is reasonable for long-term reproduction, but there is no validation of model–human evaluation consistency. It remains unclear how accurate the judgments are, how well its judgments align with human assessment or whether biases exist across models.For example, for numerical tasks, verification requires explicit calculation of error margins (e.g., 5%), and SEAT-based decomposition introduces subjective interpretation that should be analyzed.
* Several presentation issues reduce professionalism: “Unanswer” -> “Unanswerable”, “Open-End” -> “Open-Ended”, and “GPT-4o and its brothers” (L375) should be revised. Citations for evaluated models are also missing in relevant sections/tables.
* The underperformance of specialized chart models is unsurprising, since (1) most are not trained for multi-chart settings, and (2) prior works such as ChartQAPro and CharXiv already highlight similar limitations even in single-chart setups.
* The discussion on data leakage mitigation is unconvincing. Many data sources (e.g., OECD, Pew, arXiv) are present in existing multimodal pretraining datasets (e.g., MINT-1T, ChartQA). A more rigorous analysis would be required to support claims of leakage avoidance.

### Questions
* The text formatting seems inconsistent with the submission template — can the authors adjust it?
* In Figure 1, should “BenchNexus” (bottom right) be “ChartNexus”?
* What is the exact question for the “Judgment” example in Figure 1?
* There should essentially be 3 settings for chart analysis — (1) a single chart (2) a single chart with multiple subplots (i.e., multi-chart in a single image) and (3) multiple charts (i.e., multiple images). Do all samples in the dataset belong to category 3? If so, I wonder if authors could do an analysis turning all (3) instances into (2) for evaluation? This would help the community understand whether issues stem from multi-chart capabilities or multi-image capabilities.
* The SEAT evaluation prompt appears to be in Chinese, while others are in English. Can the authors perform an ablation on evaluation sensitivity to prompt language?
* Section 3.2 states annotators could create entirely new questions, but Figure 1 suggests only template selection and refinement. Please clarify this inconsistency.
* How many options exist for multiple-choice questions? If there are typically four, why do some models (e.g., SmolVLM, ChartGemma) perform well below random chance (~25%)?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper identifies a significant weakness in Multimodal Large Language Models (MLLMs): while they excel at answering questions about single charts, their performance drastically declines when faced with complex, real-world tasks that require reasoning across multiple charts within a document. To address this evaluation gap, the authors introduce ChartNexus, a novel benchmark built from a large collection of real-world charts from domains like scientific papers and government reports. ChartNexus contains a curated set of questions designed to test multi-chart reasoning skills. A comprehensive evaluation of numerous state-of-the-art MLLMs on this benchmark reveals a substantial performance drop, uncovering key weaknesses in areas like cross-chart working memory and cross-modal reasoning. The study concludes that robust multi-chart understanding remains a major, unsolved challenge for MLLMs.

### Strengths
1. Identifies a Critical Research Gap: The paper successfully highlights a major disconnect between existing single-chart benchmarks and the complex needs of real-world document analysis, moving the field beyond an overemphasis on isolated chart understanding.

2. Novel and Rigorous Benchmark: The introduction of ChartNexus is a key contribution. Its strengths include: 1) Real-World Relevance:It is built from a large corpus of authentic charts from scientific, governmental, and industrial documents. 2) Systematic Design:It features a well-defined taxonomy of reasoning skills and difficulty levels, allowing for nuanced model diagnosis. 3) Complexity and Challenge:The benchmark is demonstrably challenging, causing significant performance drops even in top-tier models.

3. Comprehensive and Conclusive Evaluation:The large-scale evaluation of 23 diverse MLLMs provides strong, empirical evidence for the paper's central claim about the limitations of current models.

### Weaknesses
1. The overrepresentation of bar charts in the benchmark skews the overall evaluation. A benchmark for higher-difficulty tasks should prioritize more complex chart types to ensure a reliable and meaningful assessment of model capabilities.

2. A fine-grained, per-category performance breakdown is required. It is essential to identify if there are specific chart types that the model completely fails to process, revealing the true boundaries of its current abilities.

3. The insights presented are currently unsubstantiated, as they rely solely on textual description. Convincing validation requires dedicated ablation studies and visualizations to provide quantitative and tangible support for these claims.

4. The core distinction between multi-chart and single-chart reasoning must be clarified. The evaluation must go beyond reporting a performance gap and actively diagnose the underlying reasons for it, which is critical for understanding and advancing multi-chart reasoning.

### Questions
please refer to the weaknesses

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
3

### Summary
ChartNexus introduces a benchmark for evaluating multi-chart reasoning in multimodal large language models, focusing on realistic document-level understanding rather than isolated chart interpretation. The dataset combines real-world charts, accompanying text, and human-verified question–answer pairs to test models’ ability to integrate visual and contextual information. Compared to existing single-chart benchmarks, results show a substantial performance decline across all models, indicating that multi-hop and cross-modal reasoning remain unsolved challenges. While top commercial systems outperform open-source and chart-specialized models, they still struggle with numerical precision, contextual integration, and detecting unanswerable questions. Chain-of-thought prompting offers limited gains, effective mainly for numerical reasoning tasks.

### Strengths
1. Chart understanding is an important problem to work on an multimodal models generally struggle with this task.
2. The benchmark is well-constructed and covers many task types and chart types. The automatic annotation is balanced with high manual annotation agreement. 
3. The evaluation covers over 20 models of many different types.

### Weaknesses
It is hard to tell what insights / takeaways are novel from this benchmark vs other ones. The comparison across benchmarks is good to see, but I would want to know what additional signal this benchmark provides. For instance charXiv already identified that commercial models generally outperform open-source models in real-world chart settings. I think this benchmark is good, but it is important to understand what trends it shows that we could not find otherwise.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ChartNexus, a new benchmark dataset intended to evaluate the multi-chart and cross-modal (text + chart) reasoning capabilities of Multimodal Large Language Models (MLLMs). The authors argue that existing benchmarks are mostly restricted to single-chart queries, failing to capture the real-world scenarios. Empirical evaluation of 23 state-of-the-art MLLMs, including GPT-4o, exhibits a dramatic performance drop on this new benchmark.

### Strengths
1. A well-motivated problem, addressing a major concern in existing chart benchmarks that are mostly restricted to single-chart scenarios. 
2. Collection of data from real-world sources.
3. Robust Human-in-the-Loop Annotation.
4. Multilingual extension.

### Weaknesses
1. The paper uses the Qwen3-32B model as the judge to evaluate the correctness of answers. However, the paper did not discuss the judge's accuracy against human scoring. 

2. The abstract claims the benchmark comprises 6,793 real-world charts. However, Table 4 mentions only 3,198 charts. This is a major contradiction, making the dataset construction questionable in a datasets & benchmark-focused work.

3. While the paper reviewed various existing multi-chart benchmarks, it still lacks justification on how their contribution is not just an incremental contribution in comparison to the prior work.

4. Lack of details on what method was applied to measure inter-annotator agreement.

5. Lack of details on how the disagreement in open-ended QA is resolved.

### Questions
Address the weaknesses mentioned above.

### Soundness
2

### Presentation
2

### Contribution
2
