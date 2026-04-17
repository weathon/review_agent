# OptimalThinkingBench: Evaluating Over and Underthinking in LLMs

- Decision: Accept (Poster)
- Scores: 2, 8, 6

## Abstract
Thinking LLMs solve complex tasks at the expense of increased compute and overthinking on simpler problems, while non-thinking LLMs are faster and cheaper but underthink on harder reasoning problems. This has led to the development of separate thinking and non-thinking LLM variants, leaving the onus of selecting the optimal model for each query on the end user. In this work, we introduce OptimalThinkingBench, a unified benchmark that jointly evaluates overthinking and underthinking in LLMs and also encourages the development of optimally-thinking models that balance performance and efficiency. Our benchmark comprises two sub-benchmarks: OverthinkingBench, featuring simple general queries in 72 domains along with simple math problems, and UnderthinkingBench, containing 11 challenging reasoning tasks along with tough math problems. Using novel thinking-adjusted accuracy metrics, we perform an extensive evaluation of 33 different thinking and non-thinking models and show that no model is able to optimally think on our benchmark. Thinking models often overthink for hundreds of tokens on the simplest user queries without improving performance. In contrast, large non-thinking models ``underthink'', often falling short of much smaller thinking models. We further explore several methods to encourage optimal thinking, but find that these approaches often improve on one sub-benchmark at the expense of the other, highlighting the need for better unified and optimal models in the future.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a unified benchmark that jointly evaluates overthinking and underthinking in LLMs and also encourages the development of optimally-thinking models that balance performance and efficiency. This benchmark comprises two sub-benchmarks: OverthinkingBench, featuring simple math and general queries in 72 domains, and UnderthinkingBench, containing 11 challenging reasoning tasks along with harder math problems. The author evaluate 33 different thinking and non-thinking models and show that no model is able to optimally think on our benchmark. Thinking models often overthink for hundreds of tokens on the simplest user queries without improving performance. In contrast, large nonthinking models underthink, often falling short of much smaller thinking models.

### Strengths
This paper introduces a unified benchmark that jointly evaluates overthinking and underthinking in LLMs and also encourages the development of optimally-thinking models that balance performance and efficiency. Different thinking and non-thinking models and show that no model is able to optimally think on our benchmark.

### Weaknesses
1. The paper's central claim is that "no model is able to optimally think." However, this conclusion is drawn from a pool of models that may not represent the current state-of-the-art in reasoning and instruction-following. Top-tier commercial models like OpenAI's GPT-4/GPT-4o, Anthropic's Claude 3 family, and Google's Gemini family are specifically engineered to handle a wide distribution of task difficulties. More commercial models should be considered. 
2. It seams that many other simple benchmarks can work as the no-thinking parts(many instance like MATH500), and many hard benchmarks can work as the thinking parts(like HLE math parts). The authors should highlight more unique features of this benchmark, as it currently appears highly replaceable. 
3. Limited exploration on "optimal thinking". The explored methods are not detailed in the summary, but they might be simple approaches like static routing or basic prompting. More advanced techniques exist, such as dynamic router models that predict the required reasoning depth, or using the LLM itself to decide if it needs to "think" (i.e., use a COT prompt). Without comparing against these more sophisticated strategies, the conclusion that the trade-off is unavoidable seems premature.

### Questions
Just as the statements on weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper evaluates the critical problems of "overthinking" and "underthinking" in long reasoning models. It introduces OptimalThinkingBench, a unified benchmark designed to evaluate the thinking efficiency of these models. The authors conduct a comprehensive evaluation of 33 LLMs on this benchmark, compare the performance of several efficiency optimization methods, and provide a detailed analysis of specific cases. The paper's key finding—that no current model achieves an optimal balance of thinking efficiency—is insightful to this field.

### Strengths
1. Comprehensive Workload and Evaluation: The paper demonstrates a substantial amount of work. The benchmark is constructed through automatic synthesis and filtering existing datasets. The experimental section is thorough, covering a broad spectrum of LLMs and testing the effectiveness of multiple optimization methods. The findings and analysis are also detailed and insightful.
2. Sound Metric Design: The proposed metrics avoid subjective semantic judgments of what constitutes "overthinking" and "underthinking". Instead, they rely on objective, quantifiable, and reproducible measures: thinking token count and final answer accuracy. This approach provides a robust and practical standard for evaluation.
3. Scalable and Diverse Benchmark Construction: The benchmark's construction relies on automated synthesis and filtering, ensures it is extensible for future models and helps prevent dataset contamination. Furthermore, the benchmark's broad domain coverage facilitates a more holistic evaluation of model performance across diverse tasks.

### Weaknesses
1. Lack of formal definitions: The paper relies heavily on an intuitive understanding of "overthinking" and "underthinking" rather than providing precise, formal definitions. This ambiguity can hinder a clear grasp of the exact problem being solved and makes it difficult to establish firm boundaries for what qualifies as each behavior.

2. Potential for dataset bias in UnderthinkingBench: The construction of the UnderthinkingBench is filtered based on the performance gap between two specific models (Qwen3-1.7B and Qwen3-235B-A22B) . This reliance on a specific pair of models may introduce significant dataset bias, potentially making the benchmark overly tailored to the strengths and weaknesses of these models. Furthermore, the "performance-only" evaluation approach may amplify this bias, the developer can not capture the true nature of the cognitive failure and is less conducive to developing genuine solutions.

3. Limited analysis of root causes: While the paper provides robust metrics for what is happening (evaluation), it offers limited analysis into why models overthink or underthink. The evaluation neither reveal the root causes of these inefficiencies, nor provide clear, actionable directions for model optimization. Given that "overthinking" and "underthinking" are widely discussed problems, a paper introducing a benchmark would be stronger if it also provided deeper diagnostic insights, moving beyond scoring to explain the underlying mechanisms.

### Questions
1. There appears to be a discrepancy in the paper. Line 209 states the filtering threshold $\lambda=0.1$, while line 744 states $\lambda=0.3$. Could the authors clarify which value was actually used for the dataset construction?
2. What's the reasoning for selecting $t_{max} = 1000$ as the maximum thinking length for the $AUC_{OAA}$ calculation? The experimental results (e.g., in Table 1) show that several models generate far more than 1000 tokens on average for simple queries.

### Soundness
4

### Presentation
4

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
The paper proposes a benchmark unifying the overthinking and underthinking for LLMs and corresponding metrics to evaluate the LLM performance on accuracy and efficiency.

### Strengths
1. The benchmark jointly evaluates LLM performance on both simple and challenging tasks. This assessment reveals whether LLMs can adaptively allocate reasoning effort based on task difficulty, thus achieving an optimal balance between accuracy and efficiency.
2. The paper proposes the overthinking-adjusted accuracy, which presents the token-efficiency of reasoning LLM more precisely
3. The paper conducts experiments on a wide range of LLMs, which provide a comprehensive and solid evaluation of the current model performance.

### Weaknesses
1. The underthinking dataset is constructed based on performance differences between small reasoning models and large non-reasoning models. However, the evaluation method and the reason for the selected performance threshold are not clearly explained. Additionally, the construction process does not account for the absolute performance of each model. If the reasoning model also performs poorly on a given task, there could be factors beyond reasoning capability, such as context misunderstanding or knowledge gaps, causing the failures. Performance on such tasks may not precisely reflect reasoning capabilities.
2. The information provided by precision, recall, and F1 scores is limited. Overthinking is a complex behavior that can arise from different underlying causes. The current metrics only assess response accuracy with and without token length constraints, but cannot reveal how responses evolve over time. For instance, on simple questions where LLMs can derive answers in a few tokens, it remains unclear how the model behaves after generating the correct answer. Understanding this evolution is crucial for improving LLMs' optimal thinking capabilities.

### Questions
1. What is the advantage of using this unified benchmark compared to combining two existing benchmarks that separately target overthinking and underthinking?

### Soundness
3

### Presentation
3

### Contribution
3
