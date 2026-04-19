# UGMathBench: A Diverse and Dynamic Benchmark for Undergraduate-Level Mathematical Reasoning with Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Large Language Models (LLMs) have made significant strides in mathematical reasoning, underscoring the need for a comprehensive and fair evaluation of their capabilities. However, existing benchmarks often fall short, either lacking extensive coverage of undergraduate-level mathematical problems or probably suffering from test-set contamination. To address these issues, we introduce UGMathBench, a diverse and dynamic benchmark specifically designed for evaluating undergraduate-level mathematical reasoning with LLMs. UGMathBench comprises 5,062 problems across 16 subjects and 111 topics, featuring 10 distinct answer types. Each problem includes three randomized versions, with additional versions planned for release as the leading open-source LLMs become saturated in UGMathBench. Furthermore, we propose two key metrics: effective accuracy (EAcc), which measures the percentage of correctly solved problems across all three versions, and reasoning gap ($\Delta$), which assesses reasoning robustness by calculating the difference between the average accuracy across all versions and EAcc. Our extensive evaluation of 23 leading LLMs reveals that the highest EAcc achieved is 56.3% by OpenAI-o1-mini, with large $\Delta$ values observed across different models. This highlights the need for future research aimed at developing "large reasoning models" with high EAcc and $\Delta = 0$. We anticipate that the release of UGMathBench, along with its detailed evaluation codes, will serve as a valuable resource to advance the development of LLMs in solving mathematical problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces UGMathBench, a new benchmark for evaluating mathematical reasoning capabilities in LLMs at the undergraduate level. The benchmark addresses key limitations of existing evaluation methods by incorporating 5,062 diverse problems spanning 16 subjects, with each problem featuring three randomized versions to help detect potential test set contamination. The authors contribute two new metrics: effective accuracy (EAcc), which measures consistent problem-solving ability across variations, and reasoning gap ($\Delta$), which quantifies reasoning robustness by comparing average accuracy to EAcc. Detailed error analysis showing calculation errors as the primary failure mode and revealing inconsistencies in how models handle different versions of the same problem.

### Strengths
The functional variation approach represents a significant advancement over traditional math benchmarks. By systematically generating variations, it provides a robust mechanism to detect and account for potential test set contamination and overfitting - a critical concern given the widespread training of models on public mathematics content.

Additionally, the benchmark's difficulty is carefully calibrated to the current capabilities of LLMs while maintaining sufficient headroom for future improvements. This thoughtful positioning ensures the benchmark will remain relevant and discriminative as model capabilities advance, providing a stable metric for tracking progress in mathematical reasoning.

The authors offer valuable insights through: granular performance breakdowns across difficulty levels, subject-specific analysis that illuminates relative strengths and weaknesses, detailed examination of error modes that reveals systematic failure patterns (about half of the errors were due to calculation errors). This analysis not only establishes current baseline performance but also provides insights for model improvement and understanding of LLM mathematical capabilities.

### Weaknesses
The observed consistency in the reasoning gap metric (converging to approximately 10 across increasing ACC values) raises important methodological concerns. This finding contradicts established results from previous papers on functional variations and test set contamination, which demonstrated significant performance disparities among models with comparable ACC levels (a pattern typically indicative of overfitting). Two issues warrant discussion:

1. The apparent "reasoning gap" may primarily reflect sampling temperature effects rather than true reasoning capabilities. By definition, expected accuracy (EAcc) will be lower than actual accuracy (AAcc) due to the stochastic nature of the sampling process. This suggests the metric might be predominantly measuring a model's robustness to temperature variation rather than its reasoning capabilities.
2. Models previously demonstrated to overfit on the test set would be expected to exhibit substantially larger reasoning gaps. The absence of this pattern in the current results suggests that either: a) the metric is not effectively capturing reasoning differences, or b) the relationship between overfitting and reasoning capabilities requires re-examination. The authors claim that "These metrics help mitigate the impact of potential test set contamination," but their is no provided evidence that the proposed metrics detect contamination.

These observations suggest the need for additional validation of the metric's ability to meaningfully differentiate between genuine reasoning gaps and artifacts of the sampling process.

Minor:
Figure 1's layout appears overcrowded with difficult-to-read text sizes and fonts. Additionally, I would strongly reccomend against using pie charts as humans are bad at comparing areas.

### Questions
- What was the threshold used for deduplication and how many questions were filtered out?
- How well do students perform on the benchmark? (Assumming this data is availble/not private)
- Is it possible to derive the expected reasoning gap as a function of sampling temperature?
- How do models specifically overfit to one variation effect the reasoning gap?
- Were the "bad questions" found in the error analysis filtered out of the benchmark?

### Soundness
2

### Presentation
3

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
Authors propose a new benchmark namely UGMathBench comprising of undergraduate level math problems for testing mathematical reasoning abilities of LLMs. Authors also propose two metrics namely effective accuracy and rasoning gap and perform extensive experiments with multiple open and closed source LLMs showing that there is still a big scope for improving mathematical reasoning abilities of LLMs.

### Strengths
1) Dataset is comprehensive consisting of 5062 problems including problems from different areas of mathematics.

2) I acknowledge the efforts made by the authors in cleaning the data by performing deduplication ensuring that no problems are repeated and including multiple versions of the problems for thorough evaluation of mathematical reasoning abilities of LLMs.

3) Authors have promised to make the benchmark dynamic as in to include new and more tougher problems in the future

4) The proposed metrics namely effective accuracy and reasoning gap provide an effective way to evaluate the reasoning abilities of LLMs.

### Weaknesses
1) Authors should experiment with more advanced prompting techniques like progressive hint prompting [1]. Progressive hint prompting allows LLMs to use previously generated answer as a hint which allows LLMs to correct/ refine their mistakes. I believe that this experiment would provide crucial insights if LLMs can solve tough mathematical problems with refinement. I will increase my score if authors perform this experiment.

2) Authors mention in line 477 that there are some number of "bad questions" present in the benchmark which may not be suitable for LLMs to solve. Pls mention the number and percentage of bad questions present in the benchmark as it directly effects the quality of the benchmark.

### Questions
what are the limitations of the current work? Pls include a description of limitiations of current work too as it is important

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
5

### Summary
This work underscores the limitations of current mathematical benchmarks due to the rapid advancements in mathematical reasoning capabilities of LLMs and issues of test-set contamination. To address these challenges, the authors introduce UGMathBench, a comprehensive benchmark designed to effectively evaluate these capabilities. UGMathBench consists of over 5,000 problems across various subjects, aiming to provide a fair assessment of LLM performance. It addresses common issues in existing benchmarks by offering a dynamic dataset that can generate different question variations by setting random seeds. Additionally, the authors propose new metrics to evaluate accuracy and reasoning robustness, facilitating a deeper analysis of the challenges LLMs face in mathematical reasoning.

### Strengths
1. **Dynamic Dataset:** UGMathBench provides multiple randomized versions of each problem, allowing for a robust assessment of LLM reasoning abilities. This approach enables a more rigorous evaluation by examining how well models handle variations of the same problem.
   
2. **Diversity:** The benchmark covers a wide range of mathematical topics, ensuring comprehensive subject matter coverage.

3. **Difficulty Level:** UGMathBench includes undergraduate-level problems, challenging LLMs with more advanced content.

4. **Innovative Evaluation Metrics:** The paper introduces metrics like Effective Accuracy (EAcc) and Reasoning Gap (∆) to measure LLM performance. EAcc captures the percentage of problems correctly solved across all problem variations, while the Reasoning Gap quantifies the robustness of reasoning when problems are modified slightly.

5. **High-Quality Problem Collection:** The problems are carefully curated, cleaned, and formatted from an online undergraduate-level homework grading system, ensuring quality and relevance.

6. **Fully Public Benchmark:** UGMathBench is publicly available, enabling LLM developers and users to conduct thorough error analysis.

7. **Detailed Error Analysis:** The benchmark provides insights into the current limitations of LLMs in mathematical reasoning, guiding future directions for improvement.

### Weaknesses
1. **Premise of Problem Variants:** The paper assumes that "a model capable of solving a problem through reasoning should also be able to solve all its variants under variable disturbance." The benchmark achieves this by randomizing numbers in each question to create multiple versions while theoretically maintaining the problem’s difficulty and intrinsic properties. However, simply changing certain values can fundamentally alter the problem. For example, modifying the denominator in $\int_{a}^{b} \frac{x^3}{x^4 - 1} \, dx$ to $\int_{a}^{b} \frac{x^3}{x^4 + 1} \, dx$ requires a completely different solution approach. Similarly, changing the series $\sum_{n=1}^{\infty} \frac{1}{n^2}$, which converges to $\frac{\pi^2}{6}$, to $\sum_{n=1}^{\infty} \frac{1}{n}$ transforms it into the divergent harmonic series. These examples show that certain modifications in mathematical problems lead to essientially different problems, which may not align with the benchmark’s intended assessment of consistent reasoning across equivalent variants.

2. **Precision in Numerical Calculations:** Classifying the example in **Table 12** as a "numerical calculation error" might be overly strict, given that the minor difference between 12305.72 and 12305.670999 likely falls within acceptable rounding tolerance. Unless the benchmark is meant to test extreme precision, such small discrepancies shouldn’t necessarily indicate an error. Accuracy assessments could allow for minor rounding differences or specify a standard precision level (e.g., significant figures or decimal places) for intermediate calculations to ensure fair evaluation.

### Questions
1. How do you ensure that changes to specific values within a problem do not fundamentally alter its nature or required solution approach? Given that small modifications can impact mathematical properties such as convergence and factorization, are there any safeguards or verification steps to confirm that randomized problem variants retain similar problem structures, difficulty levels, and solution approaches as the original?
2. Would it be feasible to incorporate a tolerance level for minor discrepancies in numerical answers, especially those influenced by floating-point precision errors? For instance, a defined rounding tolerance (e.g., within 0.01%) could help accommodate acceptable computational variations and more accurately reflect the model’s reasoning accuracy.

### Soundness
4

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
4

### Summary
The paper introduces UGMathBench, an undergraduate-level math reasoning benchmark for LLMs. THe benchmark includes 5,062 problems across 16 subjects, with randomized variations to prevent test-set contamination and evaluate reasoning consistency. All problems are taken from the authors’ school curricula. The authors also propose a variety of metrics that could theoretically be applied to other benchmarks (with additional work) to enable more robust benchmarking of LLM abilities. Testing 23 LLMs, the highest EAcc achieved was 56.3%, highlighting challenges for current models.

### Strengths
* The authors create a novel undergraduate-level benchmark for math reasoning.
* The proposal of the Effective Accuracy, Average Accuracy, Reasoning Gap, and Robustness Efficiency metrics are interesting, and are hopefully adopted by math benchmarks as a whole to promote more robust benchmarking.
* The results on each of these metrics are described at length, with potential reasons explored for any interesting/unexpected results.
* UGBench is fine grained, with each question being labeled with a main topic and subtopic, enabling a more granular study of various LLMs’ strengths and weaknesses.
* The appendix is robust and includes detailed information about the creation and distribution of the benchmark.

### Weaknesses
* The overall discussion of results feels slightly generic (besides what is driven by the metrics proposed in the paper). It demonstrates a lot of what has already been established (namely, undergraduate math being a challenging task, closed source models performing better than open source, bigger model size = better performance, specialized LLMs’ improved performance over unspecialized ones, etc.) UGBench is fairly uniquely positioned in how granular it is (in terms of each question being associated with a topic/subtopic). I believe this could lead to interesting comparisons between models and which skill sets certain families of models succeed/fail on.

### Questions
* Same as weaknesses — I think there are possibly more interesting analyses that could be conducted given how rich the data in UGMathBench is (as opposed to the usual open vs. small, regular vs. specialized, etc.)

### Soundness
4

### Presentation
4

### Contribution
3
