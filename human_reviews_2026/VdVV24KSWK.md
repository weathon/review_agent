# SPRIG: Improving Large Language Model Performance by System Prompt Optimization

- Decision: Accept (Poster)
- Scores: 2, 4, 4, 6

## Abstract
Large Language Models (LLMs) have shown impressive capabilities in many scenarios, but their performance depends, in part, on the choice of prompt. Past research has focused on optimizing prompts specific to a task. However, much less attention has been given to optimizing the general instructions included in a prompt, known as a system prompt. To address this gap, we propose SPRIG, an edit-based genetic algorithm that iteratively constructs prompts from prespecified components to maximize the model's performance in general scenarios. We evaluate the performance of system prompts on a collection of 47 different types of tasks to ensure generalizability. Our study finds that a single optimized system prompt performs on par with task prompts optimized for each individual task. Moreover, combining system and task-level optimizations leads to further improvement, which showcases their complementary nature. Experiments also reveal that the optimized system prompts generalize effectively across model families, parameter sizes, and languages. This study provides insights into the role of system-level instructions in maximizing LLM potential.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates how to optimize system prompts rather than task-specific prompts. To address this, the authors propose SPRIG, an edit-based genetic algorithm that incrementally constructs effective prompts from predefined components. SPRIG is designed to enhance model performance in a wide range of general-purpose scenarios. The authors evaluate the optimized system prompts across 47 diverse tasks to test generalizability. Results show that a single optimized system prompt can perform comparably to task-specific optimized prompts, and combining both system- and task-level optimization yields further improvements.

### Strengths
1. The paper shifts focus from task-specific prompts to system-level prompt optimization, which is an underexplored yet practically important area for improving general LLM behavior.

2. The authors evaluate their method across 47 diverse tasks, providing reasonably broad empirical coverage to assess generalization and robustness.

### Weaknesses
1. The optimization approach is relatively standard, and most existing task-level prompt optimization methods could be directly adapted for system prompts without fundamental innovation.

2. Although optimizing across multiple benchmarks is necessary for generalization, the proposed method’s use of 100000 prompts to train a reward model introduces substantial computational overhead—potentially exceeding the cost of directly evaluating prompts during optimization.

3. The comparison with PROTEGI, a two-year-old task prompt optimization method, is insufficient. More recent and advanced prompt optimization frameworks should be included to fairly contextualize the proposed contribution. It is unclear that whether the gain will reduced when using more advanced task prompt optimization method.

4. The most meaningful test for system-level optimization should be generalization to unseen tasks. However, the experiments mainly evaluate on seen or similar tasks, making it unclear whether the proposed system prompt truly transfers to new scenarios.

### Questions
See the weak points.

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
This paper presents SPRIG (System Prompt Refinement for Increased Generalization), a genetic algorithm-based approach for optimizing system prompts that generalize across tasks. Unlike existing work focusing on task-specific prompt optimization, SPRIG aims to find universal system instructions that improve performance across diverse tasks. The authors evaluate on 47 tasks and find that optimized system prompts achieve performance comparable to task-specific optimization, with complementary benefits when combined.

### Strengths
- The paper's proposed SPRIG method provides an efficient solution for the challenging task of searching for optimal system instructions through its genetic algorithm-based design.

- The paper conducts large-scale experimental evaluation, comprehensively validating SPRIG's effectiveness across three models and 47 tasks, while testing generalization capabilities across multiple dimensions including language types and model sizes.

- The paper presents several interesting findings, such as the complementarity between system and task prompt optimization and the varying degrees of benefit from system prompts across different task types, with thorough analysis of these discoveries.

### Weaknesses
- The paper merely transfers genetic algorithms to the prompt optimization task, with an engineering-oriented pipeline design that lacks methodological innovation.
- In SPRIG's design, the reward model's accuracy directly affects optimization quality. As optimization progresses, the prompt distribution may drift from the initial training distribution, leading to decreased prediction accuracy. The paper lacks experimental analysis of the reward model during training, such as comparing calibration curves between reward model predictions and actual benchmark scores across training steps.
- The paper only experiments on instruct models. Scaling experiments on base models and reasoning models would strengthen the claims of generality.
- The selection of genetic algorithm hyperparameters lacks explanation and experimental analysis.
- Ablation studies analyzing the individual contributions of different mutation operations would provide valuable insights into the method's effectiveness.

### Questions
Refer to Weaknesses

### Soundness
2

### Presentation
3

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
While Large Language Models (LLMs) are powerful, their performance is highly sensitive to the prompts. Previous prompt optimization methods mainly focus on descriptive expression of the task instruction or prompt but ignore system prompt. This paper proposes SPRIG, a genetic algorithm focusing on optimizing system prompt, which can be adapted to a suite of similar tasks with consistent system prompt. 
Different from task-specific prompt, a single system prompt can match the performance of prompts tailored to individual tasks. Combining both system and task-level optimization yields better results. Further, the optimized system prompts transfer well across different languages and model families.

### Strengths
* This paper is well-written and easy to understand.
* Concentration on system prompt provides a new perspective, previous methods ignore this general point, which can help to improve a kind of tasks with a single system prompt. 
* The experiments show the effectiveness and validates the importance of the description of the system prompt. 
* The analysis is very thorough and analysis is in-depth. For example, it does make sense that optimization of system prompt could hardly improve on knowledge-intensive tasks.
* It is wonderful that the optimized system prompts transfer well across different languages and model families.

### Weaknesses
* The population size, 9000 is very large, which brings costful overhead. The cost analysis compared with baselines should be investigated, in the aspect of time, tokens, etc.
* The average score improvements are only shown with figures without detailed values. It seems that system prompt only brings marginal improvement (Fig.4). Could the authors give statistic results?
* The generation process of new prompts are simple. This paper just uses prompt optimization methods to optimize system prompt, which seems a little incremental. But it focuses on a point which has been ignored up to present.
* The improvements of task prompts are larger than those of system prompts, so this paper is a little over-claiming.

### Questions
See in the weakness part.

### Soundness
2

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
3

### Summary
This paper introduces SPRIG (System Prompt Refinement for Increased Generalization), a genetic algorithm-based method for optimizing system prompts in large language models. Unlike prior work focusing on task-specific prompt optimization, SPRIG aims to discover general-purpose system-level instructions that improve performance across diverse tasks. The authors build a corpus of 9,000 prompt components across 9 categories, use a reward model to efficiently evaluate prompts, and apply evolutionary operators (mutation, crossover) to iteratively refine system prompts. Experiments across 47 tasks and 3 LLMs show that SPRIG-optimized system prompts achieve performance comparable to task-specific optimization and complement existing methods when combined.

### Strengths
- Systematically addresses system prompt optimization, which has received limited attention compared to task-specific optimization
- 47 benchmarks across 7 categories, 3 model families, multiple languages—one of the most thorough evaluations in prompt optimization literature
- The finding that system and task optimization capture different failure modes (Figure 6b) is insightful and well-demonstrated
- Strong evidence that system prompts transfer across languages better than task-specific prompts (Figure 8b)
- Extensive implementation details, code promised in supplementary materials, and clear documentation of experimental setup

### Weaknesses
- The genetic algorithm approach is standard; the main contribution is application domain rather than methodological innovation
- With Spearman correlation of 0.59, the reward model may introduce significant noise into the optimization process. No analysis of how reward model errors affect final results
- Most comparisons lack confidence intervals or significance tests, making it difficult to assess the reliability of observed differences

- While Figure 6a shows CoT and Behavioral components are selected most frequently, there's limited analysis of why these combinations work or how they interact
- No systematic study of hyperparameters, alternative reward model architectures, or different component categorization schemes
- The optimized prompts (Table 3) are surprisingly simple variations on "think step by step," raising questions about whether simpler search methods might suffice

### Questions
- How sensitive is the final performance to reward model quality? Could you show results with reward models of varying quality?
- Have you compared against simpler baselines like random search or grid search over a smaller set of manually curated prompts?
- Do certain component categories work particularly well together? Is there evidence of synergistic or antagonistic combinations?
- Can you provide qualitative analysis of cases where SPRIG fails? What types of tasks or questions are most resistant to system prompt optimization?
- Why do you think the optimized prompts don't transfer to larger models? Is this because larger models are already closer to optimal, or because they require different strategies?
- Can you provide confidence intervals or significance tests for the main results? How much variance is there across different optimization runs?
- Have you conducted human evaluation of the quality or interpretability of generated responses with different prompts?

### Soundness
3

### Presentation
3

### Contribution
3
