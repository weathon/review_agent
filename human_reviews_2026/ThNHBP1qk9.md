# CodeInsightBench: A Benchmark for Advanced Code Understanding and Comparison in Large Language Models

- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
While Large Language Models (LLMs) have demonstrated significant progress in coding tasks, their capabilities in nuanced code analysis and deep comprehension remain insufficiently explored. To address this gap, we introduce CodeInsightBench, a comprehensive multilingual benchmark designed to systematically evaluate advanced code reasoning. Built upon real-world Codeforces data, it employs both multiple-choice and open-ended questions to assess three core tasks: Semantic Code Judgment, Debugging Path Tracking, and Code Efficiency Comparison. We conduct extensive evaluations on 22 state-of-the-art LLMs (11 closed-source, 11 open-source),  revealing critical insights into their strengths and limitations. Our results reveal substantial performance gaps across tasks, with closed-source models generally outperforming open-source counterparts. Besides, models fail primarily on large-scale code transformations, indicating fundamental limitations in understanding code evolution logic. Additionally, the results indicate distinct programming language preferences in code efficiency comparison, and show that multiple sampling substantially improves semantic code judgment performance, with Pass@3 achieving 92.71% accuracy compared to 60.57% at Pass@1. By providing a comprehensive and systematic evaluation methodology, CodeInsightBench enables deeper understanding of LLM capabilities in sophisticated code comprehension tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CodeInsightBench, a benchmark for evaluating code LLM’s comprehension and reasoning capabilities. It designs three novel tasks: semantic code judgment, debugging path tracking, and code efficiency comparison. Extensive evaluation shows the lack of deep reasoning capability of existing code LLMs.

### Strengths
1. Timely topic. The benchmarks aims to evaluate “how LLMs understand code”. Code comprehension capability is an important and timely problem. The benchmark provides more angles to evaluate code reasoning in addition to simple tasks like LiveCodeBench.
2. Recency of data. The benchmark is built on top of data from 2024-2025, mitigating the threat of data leakage.
3. Novel tasks. The three tasks in CodeInsightBench are novel, especially Debugging Path Tracking and Code Efficiency Comparison.
4. Solid evaluation. The paper built a large-scale dataset and used 22 models for evaluation. They evaluate 5 metrics for the 3 tasks
5. Presentation. The paper is well-written and easy to follow. I enjoy reading the paper.

### Weaknesses
1. The paper does not discuss in detail why using the 3 tasks. Semantic Code Judgment makes sense as the discriminative capability in terms of distinguishing correct from incorrect implementation can reflect the semantic comprehension capability. But the correlation between sorting the submission order and logical execution tracing is not as straightforward as task 1. 
2. The code efficiency evaluation of task 3 can be affected by factors like various factors, especially when the workload is small, e.g., us-ms level. The physical running time profiling is sensitive to the environment, e.g., running background applications. The variation can easily exceeds the range of 20%. The paper did not either repeat the profiling multiple times, or using more stable metrics like hardware instruction counter or simulators. This weakness poses a significant threat to the evaluation of task 3.
3. The data comes from only 14 medium-difficulty problems. Although the 14 problems contain a large number of user submissions, the paper does not discuss the threat of using only 14 problems, which could affect the generality of the conclusions.
4. In task 1 Semantic Judgment, the model is asked for a probability, but the primary metric is Accuracy. It remains unclear how accuracy is computed from the probability.

### Questions
1. Could the authors provide more justification of task 2 debugging path tracking, i.e., on the correlation between sorting the submission order and logical execution tracing? The correlation is not as straightforward and intuitive as other tasks, since users could experience many back-and-forth due to various reasons when debugging their solutions. The assumption that the users always improve the solutions might not hold.
2. Why do the authors use physical runtime metrics for evaluating task 3? What’s the variance of physical runtime metrics if repeated multiple times? Why don’t the authors use more stable and reliable metrics like gem5 simulators?
3. Table 4 shows that GPT-4-Turbo's performance on this task jumps from 60.57% (Pass@1) to 92.71% (Pass@3). This suggests that the model is often "directionally correct" (e.g., outputting 0.7 when the answer is 1.0, or 0.4 when it's 0.0) but is not calibrated well enough to pass a simple "correct/incorrect" threshold at Pass@1. Could the authors clarify how "Accuracy" is computed from the probability? Is it a simple threshold at 0.5? The high Pass@3 result suggests the task may be "easier" for the model than the Pass@1 accuracy implies, and the core issue is one of calibration. A brief discussion on this point would help readers interpret the Task 1 results.

### Soundness
4

### Presentation
3

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
This paper introduces CodeInsightBench, a new benchmark for testing how well LLMs understand complex code. Instead of just asking models to write code, this benchmark tests if they can “reason” about it. It’s built from Codeforces data and has three core tasks: 1) telling a correct code snippet from a subtly buggy one (Semantic Code Judgment), 2) figuring out the logical order of code submissions in a debugging session (Debugging Path Tracking), and 3) comparing which of two correct solutions is more algorithmically efficient (Code Efficiency Comparison) .

The authors tested 22 different models and found a significant gap: while models like GPT-5 are great at the first task (93.43% accuracy), they all fail badly at the third, efficiency-comparison task (max accuracy of 45.50%). The main takeaway is that today's LLMs are good at surface-level pattern matching but are fundamentally weak at deep algorithmic reasoning.

### Strengths
A Helpful Key Finding: The clearest strength here is the discovery that even the best LLMs (like GPT-5) fail at comparing code efficiency (Task 3). This is a significant result. It strongly suggests that our models are still just very sophisticated pattern matchers, excelling at static, "surface-level" tasks (like Task 1) but failing when "deep" algorithmic reasoning is required.

A High-Quality, Novel Benchmark: The benchmark itself is a great asset. Using real, recent Codeforces submissions is a huge plus, as it means the models haven't likely been trained on this data. The tasks are novel and push beyond standard "code generation" evaluations.



Comprehensive Evaluation: Testing 22 different models gives a very clear "state of the union" on this problem and confirms the gap between closed- and open-source models.

### Weaknesses
My concerns with this paper aren't about what it does, but what it doesn't do.

Limited Workload/Scope: The contribution, while valuable, feels a bit thin. At its core, the work consists of data collection/cleaning from Codeforces and running an evaluation. This is a good start, but it feels more like a benchmark report than a complete research paper.

The Key Insight is Under-Explored: The finding that LLMs fail at efficiency analysis is fantastic, but the paper just reports this finding and moves on. This is precisely the point where the work should have dug in. Why do they fail? Is it because they can't distinguish between O(n^2) and O(n \log n)? Is it a failure in comparative reasoning? The paper identifies a critical problem but offers no diagnosis. This insight is strong, but as presented, it's not quite enough to carry the paper on its own.

Findings Need Verification (e.g., via Training): This leads to my main suggestion. Many of the insights here—especially the behavioral ones—feel like hypotheses that need to be tested. For example, the paper notes a massive label bias in Task 3, where some Qwen models predict label "2" over 95% of the time 7. Is this a fundamental inability to reason, or just a superficial alignment failure on a novel task format? The authors could have tested this. A simple experiment, like finetuning an open model on just 10% of the Task 3 data, would tell us if this bias is easily "trained away" or if it points to a deeper, more stubborn flaw in the model's reasoning architecture.

### Questions
Can you provide a more qualitative analysis of the failures in Task 3? When models fail, what kind of efficiency errors are they making? Are they failing to see the difference between a bubble sort and a merge sort, or are the errors more subtle? A few concrete examples of failures would make the paper's main finding much more powerful.

You tested advanced prompting (CoT, ToT) on Task 1, but why not on Task 3? Task 3 is clearly the most difficult and complex reasoning task here. It seems like the perfect candidate for these more deliberate prompting methods. Did you try this, and if so, what were the results?

### Soundness
2

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
This work propose a CodeInsightBench, a benchmark that assess LLM in a wide range of coding capabilities, in multilingual, and both MCQ and open-ended format. The empirical evaluation highlight the gap of existing LLMs in these tasks.

### Strengths
The benchmark is quite comprehensive and evaluate many capabilities of existing LLMs.

### Weaknesses
- How did the authors prepare the raw data? Did the authors obtain permissions from CodeForces or crawl data through the official API?
- What are some common failure cases made by the LLMs, especially in tasks that all models have low performance?
- Benchmarks for evaluating LLMs have becoming more common, it would be nice for the benchmark to also evaluate agentic solutions.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
CodeInsightBench supports several code understanding tasks: semantic code judgment, debugging path tracking, and code efficiency comparison. Several findings in the paper include: 1. performance gaps across the tasks, 2. closed-source models generally outperform open-source counterparts, 3. long-context code transformations are challenging for all models, 4. distinct programming language preferences in code efficiency comparison, 5. Multiple sampling substantially improves semantic code judgment performance.

### Strengths
1. Well written, no obvious grammatical errors. 
2. The benchmark is large

### Weaknesses
1. Task difficulty concerns: I find the task complexity may be too simple, as in Table 2, GPT-5 can achieve more than 90 on task 1 with only Pass@1. And even GPT-4 can achieve 92.71 on task 1 with Pass@3. Note that Pass@3 is not a large sampling number.
2. Why not try larger Pass@K numbers? Would the results be saturated? For the problems that cannot be solved by large sampling, why can't LLMs solve them?
3. As I find GPT-5 insanely strong on the benchmark, I am worried that there would be data contamination on the test dataset. How do the authors make sure the comparison is fair?
4. On page 7, the authors mentioned that LLMs are weak at Algorithmic Reasoning. And I find the maximum length of decoding is 4K. Would things be different if they try larger sampling lengths?
5. The open-source models they choose are mostly non-reasoning models, and most of them are small-size models (e.g., 7B and 14B). Would the conclusion that open-source models are weaker than closed-source models be different if they try reasoning models with larger CoT length or larger sampling budgets?
6. The experiment settings are unclear. In Section 4.4, what are the hyperparameters of LLM decoding when adopting various advanced systematic CoT strategies?
7. The experiments are incomplete. In Section 4.4, the authors only test different CoT strategies on Task 1 and only on two models. As task 1 is the easiest, I think they should put more effort on harder tasks to explore the potential of CoT.
8. Why models show imbalanced preference on programming languages has not been explained.
9. Table 5 is broken.
10. Incorrect formatting: missing spaces between words and left braces '(', for example, line 156.

### Questions
See Weakness

### Soundness
2

### Presentation
2

### Contribution
2
