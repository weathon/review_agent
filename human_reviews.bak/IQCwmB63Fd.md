# Retrieval Or Holistic Understanding? Dolce: Differentiate Our Long Context Evaluation Tasks

- Decision: Reject
- Scores: 5, 6, 5, 6

## Abstract
We argue that there are two major distinct capabilities in long context understanding: retrieval and holistic understanding. Understanding and further improving LLMs' long context capabilities would not be possible without knowing the tasks' focus categories. We aim to automatically identify retrieval focused and holistic understanding focused problems from suites of benchmarks and quantitatively measure the difficulty within each focus. In this paper, we present the Dolce framework, which parameterizes each problem by $\lambda$ (complexity) and $k$ (redundancy) and assigns to one of five predefined focus categories. We propose to sample short contexts from the full context and estimate the probability an LLM solves the problem using the sampled spans. To find the $\lambda$ and $k$ for each problem, we further propose a mixture model of a non-parametric background noise component and a parametric/non-parametric hybrid oracle component, where we derive the probability functions parameterized by λ and k for both the correct-or-wrong (COW) scenario and the partial-point-in-grading (PIG) scenario. Our proposed methods can identify 0% to 67% of the problems are retrieval focused and 0% to 90% of the problems are holistic understanding focused across 44 existing long context evaluation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes DOLCE, a framework that parameterizes the complexity $\lambda$ and the redundancy $k$ of long context understanding tasks and categorizes the tasks based on these values. For a given task, DOLCE estimates the complexity $\lambda$ as the minimum length of context required to correctly answer the task, based on the correct response rate of LLMs vs. the input context length (number of consecutive units). Since the correct response rate is noisy, $\lambda$ is estimated by dedicated mixture models. The paper reports $\lambda$ and $k$ estimated by DOLCE for 44 tasks in three recent benchmarks (L-Eval, BAMBOO, and LongBench) as well as the categorization results of them.

### Strengths
- The motivation is convincing: improving the LLMs' capability of long context understanding is a key challenge. It can be useful to categorize the types of problems so that improvements can be considered separately for each type of problem.

- It is reasonable to design the proposed model to explicitly incorporate the possibility that LLMs may give correct answers by chance (background noise component) and that even oracle models may give "correct" wrong answers in a given context.

### Weaknesses
W1. The proposed model DOLCE assumes that the minimum number of "consecutive" units (typically sentences) required to correctly answer a question determines the complexity of the question \lambda. However, it seems to me that it is not always necessary to look at consecutive units in order to properly understand the context (there may be irrelevant sentences in between relevant sentences to understanding the context). This assumption appears to be somewhat unreasonable.


W2. As the authors acknowledge, the category mapping given in Fig. 1 is quite subjective and has no objective basis for its validity.


W3. This paper reports a number of estimated results of the complexity $\lambda$ and the redundancy $k$ for various problems as well as the categorization results of the problems based on these values. However, whether these estimation results correctly evaluate the complexity and redundancy of the problems is largely dependent on the qualitative observations on a small number of examples, such as Table 2, and lacks an objective basis. It is understandable that quantitative evaluation is rather difficult, but as discussed in L. Limitation, evaluation by human evaluation verification would be possible. These factors somewhat obscure the validity of DOLCE.


W4. As explained in the introduction section, one of the motivations for DOLCE is to improve LLMs' ability for long context understanding. However, whether it is actually useful for this goal has not been evaluated and is not clear.


W5. Some minor points: 

L178: It would be good to describe the meaning of subscriptions i and j here. As given later in L236, they sould be j-th observation for the i-th problem.

L190: The authors claim that the background noise component is modeled by a "nonparametric" model, but I would like an explanation of what they mean by "nonparametric" here.

### Questions
The biggest problem with this paper would be the lack of objective evidences for the validity of DOLCE for the results of task complexity estimation and categorization and for its usefulness (W2-W4). I would appreciate it if the authors could provide supporting evidences.

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
1

### Summary
The paper presents the DOLCE framework which aims to automatically identify retrieval and holistic understanding focused problems from benchmark suites. 

Proposes to sample short contexts from the full context and estimate the probability an LLM solves the problem using the sampled spans. Introduces a mixture model of a non-parametric background noise component and a parametric/non-parametric hybrid oracle component to find λ and k.

### Strengths
The idea of using a mixture model of background noise and oracle components to estimate these parameters for different scenarios (COW and PIG) is creative.
The theoretical foundation of the DOLCE framework seems solid. The derivations of probability functions for the COW and PIG scenarios based on the assumptions of the mixture model are detailed and mathematically sound.
The experimental setup is comprehensive, using 44 tasks from three benchmark suites.

### Weaknesses
1. The background noise module and the oracle module are combined using an “additive” mixture assumption. However, alternative assumptions such as “multiplicative” or generative assumptions could be explored further.

2. Using MLE as the sole objective might limit the framework’s performance. Alternative formulations of the MLE could be explored, such as giving different weights to samples or observation lengths. 

3. The assumptions made in the DOLCE framework, such as the specific forms of the background noise and oracle components, may limit its generalizability to a wide variety of real-world language tasks. For example, in real-world scenarios, the distribution of relevant information within a long context may not follow the assumed patterns precisely. This could lead to inaccurate categorization of tasks as retrieval or holistic understanding focused.

4. The paper focuses on complexity in terms of span length (λ) and redundancy (k), but it may not fully capture the semantic complexity variation within long contexts. Different semantic relationships and levels of abstraction can significantly impact the difficulty of a task, yet these aspects are not comprehensively incorporated into the framework.

### Questions
see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The manuscript presents a novel framework, DOLCE, which aims to differentiate between retrieval-focused and holistic understanding-focused problems within long context evaluation tasks. The authors argue that understanding these distinctions is crucial for improving the capabilities of large language models (LLMs) in processing extensive textual data. The framework introduces a parameterization approach with λ (complexity) and k (redundancy) to categorize problems into predefined focus categories.

### Strengths
1. The DOLCE framework is innovative in its approach to categorizing long context evaluation tasks, providing a new lens through which to understand and improve LLMs.
2.  The paper employs a rigorous methodology, including a mixture model that combines non-parametric and parametric components, to estimate the parameters λ and k.
3. The authors provide a thorough empirical validation across 44 existing long context evaluation tasks, demonstrating the applicability and effectiveness of the DOLCE framework.

### Weaknesses
1.The main text of this paper has major flaws, it is difficult to understand the relationship between the author's theory and experiments, and the layout is confusing. In addition, retrieval and overall understanding have been mentioned in the larger model and are not innovative.
2. While the paper focuses on long context understanding, it would benefit from discussing the broader implications of the findings on other areas of natural language processing.
3. The authors should compare the DOLCE framework with existing methods for evaluating long context capabilities to highlight its advantages and potential limitations.
4. The paper could address the scalability and efficiency of the DOLCE framework, especially when applied to very large datasets or in real-time applications.
5. The manuscript could benefit from a deeper exploration into how retrieval and holistic understanding interact in the context of LLMs, potentially offering more nuanced insights.
6. A robustness analysis to evaluate how sensitive the framework's categorization is to variations in λ and k would strengthen the paper's contributions.

### Questions
Please refer to weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a framework for understanding and improving long-context capabilities in large language models (LLMs) by distinguishing between two key skills: retrieval and holistic understanding. The authors propose the DOLCE framework, which classifies tasks into focus categories based on complexity and redundancy parameters. Their methods can identify retrieval-focused and holistic-focused problems across 44 long-context evaluation tasks, classifying 0%-67% as retrieval and 0%-90% as holistic understanding-focused.

### Strengths
1. Long context understanding is crucial for enhancing LLMs' performance in real-world applications where retaining and processing extended information is essential.

2. To my knowledge, the framework’s ability to automatically categorize long-context tasks based on their focus—retrieval or holistic understanding—is novel.

### Weaknesses
1. The comparison on the right side of Figure 1 is confusing; it lacks clear explanations of the hyperparameters. Additionally, the description of 𝑘 for balanced and holistic understanding (marked as “any” in the right table) does not align with the illustration on the left side of Figure 1. It would be better to polish this figure.

2. The meaning of 𝑧 in Equation 1 is unclear and should be explicitly defined for better understanding. Please clarify it.

3. There are no theoretical results provided to guarantee the effectiveness of maximum likelihood estimation in this context. It would be great if the authors could provide any optimality analysis for this strategy.

4. The efficiency of observation sampling is not discussed, leaving uncertainty about its computational feasibility and scalability.

5. Lines 456-461 need further clarification regarding the differences between the various probing models and the reasons behind these differences. This would help readers understand the specific roles and implications of each model.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
