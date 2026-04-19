# DSBench: How Far Are Data Science Agents from Becoming Data Science Experts?

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6, 6

## Abstract
Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) have demonstrated impressive language/vision reasoning abilities, igniting the recent trend of building agents for targeted applications such as shopping assistants or AI software engineers. Recently, many data science benchmarks have been proposed to investigate their performance in the data science domain. However, existing data science benchmarks still fall short when compared to real-world data science applications due to their simplified settings. To bridge this gap, we introduce DSBench, a comprehensive benchmark designed to evaluate data science agents with realistic tasks. This benchmark includes 466 data analysis tasks and 74 data modeling tasks, sourced from Eloquence and Kaggle competitions. DSBench offers a realistic setting by encompassing long contexts, multimodal task backgrounds, reasoning with large data files and multi-table structures, and performing end-to-end data modeling tasks. Our evaluation of state-of-the-art LLMs, LVLMs, and agents shows that they struggle with most tasks, with the best agent solving only 34.12% of data analysis tasks and achieving a 34.74% Relative Performance Gap (RPG). These findings underscore the need for further advancements in developing more practical, intelligent, and autonomous data science agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper constructs DSBench, a benchmark focused on evaluating LM-based data science agents through realistic tasks sourced from ModelOff and Kaggle competitions.

### Strengths
1. I believe the dataset provided by the authors can, to some extent, reflect a model's ability to tackle data science tasks, and even today, it remains quite challenging, serving as a non-trivial evaluation benchmark.
2. The authors conducted some interesting analyses across various dimensions for different models and different data science tasks (e.g., examining task completion rates relative to task release time, the correlation between task difficulty and context length, etc.), offering potential insights into the development of LM as agents.

### Weaknesses
1. **Data Source:**
- All the data analysis tasks mentioned by the authors at line 106 are related to finance. I would appreciate it if the authors could clarify this point in the paper and explain why other types of analysis tasks are not as suitable. This concern arises because ModelOff is actually a global financial modeling competition.
- Similarly, for data modeling, it seems this is also influenced by the fact that there are numerous modeling task competitions on Kaggle.

I wonder if the authors have explored more platforms or data sources and could explain why they were not suitable for evaluating data science agents?
Additionally, in my opinion, competitions are not always the closest representation of the real world. For example, as far as I know, Spider2-V[1] incorporates a lot of tools and software from industrial data pipelines. Could this be a more realistic measure of real-world scenarios?

2. **Evaluation Metrics:**
- I fully understand that collecting and building complex evaluation environments is a considerable engineering task. However, if the evaluation is based solely on competition platforms and existing output-only metrics, it seems that it may not fully capture the comprehensive capabilities of data science agents. This is similar to what the authors mentioned in lines 153 and 154, such as extracting insights, proper data handling, etc.

3. **Need better presentation, especially for some tables and figures:**
- I noticed in Figure 4, some of the models do not have the reported accuracy (does this mean zero accuracy?); while the width of the bars are not set as the same. However, I do not find clear explanation to these.
- I believe transforming Table 5 into a line chart with corresponding accuracy values would clearly illustrate the trend in accuracy over time.

[1] Spider2-V: How Far Are Multimodal Agents From Automating Data Science and Engineering Workflows?, NeurIPS 2024.

### Questions
1. As highlighted in Table 1, What is the purpose of distinguishing tables from data files? What is the difference between tables and data files? I would like the authors to clarify how they treat these two types of data samples during evaluation.

2. I also find the taxonomy somewhat difficult to understand. For example, "tables" and "excels" are categorized separately, and I hope the author can clarify the distinctions between these categories more clearly.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces DSBench, a comprehensive data science benchmark containing 466 data analysis tasks and 74 data modeling tasks sourced from ModelOff and Kaggle competitions. Compared to existing benchmarks, DSBench provides a more realistic evaluation environment, encompassing long-context understanding, multimodal task backgrounds, large data file processing, and multi-table structure reasoning. Through evaluation of state-of-the-art LLMs, LVLMs, and agents, the study finds that they still struggle with most tasks, with the best agent achieving only 34.12% accuracy in data analysis tasks and a 34.74% Relative Performance Gap (RPG) in data modeling tasks.

### Strengths
### 1. Originality & Vision
- Pioneering creation of a comprehensive real-world data science benchmark, analogous to SWE-bench in software engineering, marking a significant step toward AGI evaluation
- First benchmark to integrate both data analysis and modeling tasks in realistic settings with complex data structures and multimodal contexts
- Novel introduction of RPG (Relative Performance Gap) metric that effectively normalizes performance across diverse modeling tasks
- Innovative approach to testing both pure language understanding and tool utilization capabilities
- Make consideration of not only the performance but also the cost

### 2. Technical Robustness
- Rigorous task collection methodology, carefully curated from established platforms (ModelOff and Kaggle) ensuring real-world applicability
- Comprehensive coverage of data science tasks: 466 analysis tasks + 74 modeling tasks, spanning different complexity levels and domains
- Sophisticated evaluation framework that considers:
  * Multi-table reasoning capabilities
  * Long-context understanding
  * End-to-end solution generation
  * Tool integration and utilization
  * Multiple modalities processing

### 3. Practical Significance
- Direct application to real-world data science scenarios, bridging the gap between academic benchmarks and practical challenges
- Clear identification of current AI systems' limitations in data science tasks:
  * Understanding complex data relationships
  * Handling multi-step reasoning
  * Managing tool interactions
- Provides valuable insights for developing more capable data science agents by revealing specific areas where current models fall short
- Sets a new standard for evaluating AI systems' practical data science capabilities, essential for progressing toward AGI

### 4. Research Impact
- Creates a foundation for systematic evaluation of data science capabilities in AI systems
- Enables quantitative comparison of different AI approaches in real-world data science scenarios
- Provides a roadmap for developing more capable AI systems that can handle complex, real-world data science tasks
- Serves as a crucial benchmark for measuring progress toward AGI in the domain of data science

### Weaknesses
### 1. Statistical Rigor in Dataset Scale
- While 540 samples is reasonable given the scarcity of high-quality ModelOff & Kaggle competitions, the paper could benefit from:
  * Reporting confidence intervals through multiple experimental runs
  * Conducting bootstrap analysis to estimate the robustness of performance metrics
  * Providing power analysis to justify the sample size
- The paper could discuss how the current sample size was determined and what would be an ideal size for future extensions

### 2. Evaluation Methodology
- Human baseline performance (22 competitions) could be strengthened by:
  * Including more expert evaluators per task
  * Reporting inter-rater agreement scores
  * Documenting the selection criteria for human experts
- The evaluation process could benefit from:
  * Explicit discussion of potential biases in task selection
  * Analysis of task difficulty distribution
  * More detailed failure case studies with expert annotations

### Questions
In Table 1's comparison section, I suggest reversing the criteria for "Exec. Eval.", "Code only", and "Env. Fix." to negative statements. This way, checkmarks would indicate DSBench's unique advantages where other benchmarks fall short, making it easier for readers to quickly grasp DSBench's contributions. This would better highlight how DSBench addresses limitations in existing benchmarks and make the comparison more intuitive.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
DSBench introduces a benchmark to test the performances of LLM or VLM agents. It contains data analysis and modeling tasks sourced from Kaggle and ModelOff competitions. Compared to other similar benchmarks, the benchmark is more realistic in that it contains data files, tables, and images and provides executable evaluation functions to validate the answers. The paper tests various models and agent systems and found them to pale in comparison to human data scientists.

### Strengths
The paper collects a set of data science challenges from Kaggle and ModelOff. By releasing the code and data, the community can use it to measure the performances of their agent tools. The paper demonstrates through experiments that there exists huge gaps between data science agents and humans, demonstrating that the benchmark is not saturated.

### Weaknesses
In general, the paper is well written. The only concerns relate to the sustainability of the benchmark; by releasing the dataset, there is a high chance that it will be, either intentionally or unintentionally, incorporated into the training data for LLMs or VLMs. It seems like this possibility has not been considered by the authors and their explanation for the correlation between accuracy and year of release (Table 5, L416-L421) is weak. Some critical information is missing and it affects my judgement. I would be happy to change my score if the authors can clear my potential misunderstandings.

### Questions
1. In Table 3, what does "context length" refer to? Is it the number of English words, characters, or tokens?
2. The paper assumes that the LLMs should be able to generate the answers from the data files. Is it possible for the LLMs to know the answers to the questions from pretraining? For example, the answers to the challenge may be discussed in a Reddit forum that has been scraped in pretraining.
3. In Figure 2(b), what are A-F, A-I, and A-D?
4. L265: what is N in \hat{F} = \mathcal{G}(E,N,S,M)?
5. Figure 3: Can you define m0, m1, ..., m17, perhaps as a separate table in the Appendix?
6. Please define the versions of the model used in the main paper instead of relegating it in the Appendix.
7. What are the settings for the models (e.g., temperature)? Are the experiments conducted multiple times? If so, how are they aggregated?
8. For the analyses (e.g., L404 -- L421), it would be nice to state if they have been corroborated by prior research. For example, Qian et al. has also observed that ML agents perform better on older Kaggle challenges and hypothesize that older challenges have more data leakage into pretraining data.
9. Can you provide a unified script to download the datasets, perhaps using an API, process the downloaded data and run the experiments? It would help for readers to verify the results.
10. L406-L409: Figure 4 shows only GPT-4o, AutoGen and Gemini, but your text references Llama3-8B and GPT 3.5.
11. How do I interpret the RPG of human? Doesn't 'human' represent the best human generated answer and RPG compares against the best human?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a new benchmark for automated data science, DSBench.  This
encompasses both excel exercises from ModelOff and kaggle competitions.  DSBench
also includes metrics for evaluating success in both kinds of tasks.  The paper
evaluates several open and closed source LLMs/VLMs.  The key take away is that
SOTA architectures are still far away from achieving human-level performance.

### Strengths
**Originality**: This is chiefly an engineering paper, it introduces no novel techniques or ideas.  DSBench collates existing resources.

**Quality**: The key contribution is DSBench, which I am confident took quite some effort to set up.  All in all, this contribution seems to be of good quality: it encompasses a large number of relevant tasks which go beyond what the literature currently offers.  It also comes with some rather natural metrics.  The interesting bit is the evaluation, which on the one hand shows that DSBench can indeed be used as intended, and on the other points out a performance gap for existing architectures.  The findings are otherwise quite intuitive: newer models perform better, more complex tasks are harder to solve.  The real contribution is the research that DSBench will enable.

Unfortunately, as a non-expert (my work on automating data science predates LLMs), I cannot assess the overlap between DSBench and recent works in this area.

**Clarity**: The text is generally readable, with a few linguistic quirks here and there.  The figures are easy to understand.

**Significance**:  Automated data science is a central topic nowadays.  It is not impossible that DSBench will provide a significant boost to auto-DS.  But again, this depends on overlap with existing work, which I am not overly familiar with.  As a result, I have decided to grade conservatively the contribution aspect of the paper.

### Weaknesses
**Quality**: The only relatively minor issue I'd like to point out concerns Section 3.3, which promises discussing the errors made by LLMs, but does not provide any details, really.  I think this should be amended.

### Questions
Plase see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DSBench, a comprehensive data science benchmark designed to assess the performance of data science agents on real-world tasks. DSBench integrates the challenges from ModelOff and Kaggle competitions, creating a realistic environment that covers 466 data analysis and 74 data modeling tasks. To enable fair comparison, the paper proposes the Relative Performance Gap metric, which normalizes various evaluation metrics for data modeling tasks. Evaluation of state-of-the-art models, including GPT-4o, Claude, and Gemini, reveals that DSBench remains challenging, with significant performance gaps between these models and human capabilities.

### Strengths
**Novelty** 
- This paper offers a fresh contribution to data science by introducing DSBench, a new benchmark that evaluates data science agents under realistic task conditions derived from ModelOff and Kaggle competitions. It pushes the boundaries of traditional benchmarking.

**Quality** 
- The design of DSBench, with its comprehensive task types and Relative Performance Gap (RPG) metric, demonstrates rigor in addressing evaluation inconsistencies across various modeling tasks. 

**Clarity** 
- Task designs, methods, and performance comparisons are clear and well-organized. They contain many details but are not hard to follow.

**Significance**
- DSBench sets a new standard in evaluating data science agents, driving advancements in realistic, end-to-end task performance. Its contributions are useful to future advancements of intelligent, autonomous data science agents.

### Weaknesses
- I'm bit unsure of the robustness and persuasiveness of the RPG metric is valid. It could be beneficial to further assess how well the RPG reflects actual performance across varying data types and task complexities.
- Although Kaggle tasks are highly relevant, they often focus on a narrow subset of domains (e.g., retail, finance).

### Questions
1. Can authors please explain if the metric and dataset are sufficient to cover the real-world diversity that the benchmark aims to address?
2. Explain why RPG is able to reflect actual performance across varying data types and task complexities.

### Soundness
3

### Presentation
3

### Contribution
2
