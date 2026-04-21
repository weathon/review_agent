# Quantifying Generalization Complexity for Large Language Models

- Avg Score: 6.25
- Decision: Accept (Poster)
- Scores: 5, 6, 8, 6

## Abstract
While large language models (LLMs) have shown exceptional capabilities in understanding complex queries
and performing sophisticated tasks, their generalization abilities are often deeply entangled with 
memorization, necessitating more precise evaluation.
To address this challenge, we introduce Scylla, a dynamic evaluation framework that quantitatively measures the generalization abilities of LLMs. Scylla disentangles generalization from memorization via assessing model performance on both in-distribution (ID) and out-of-distribution (OOD) data through 20 tasks across 5 levels of complexity.
Through extensive experiments, we uncover a non-monotonic relationship between task complexity and the performance gap between ID and OOD
data, which we term the generalization valley.
Specifically, this phenomenon reveals a critical threshold---referred to
as critical complexity---where reliance on non-generalizable behavior peaks, indicating the
upper bound of LLMs' generalization capabilities.
As model size increases, the critical complexity shifts toward higher levels of task complexity, 
suggesting that larger models can handle more complex reasoning tasks before over-relying on
memorization.
Leveraging Scylla and the concept of critical complexity, we benchmark 28 LLMs including 
both open-sourced models such as LLaMA and Qwen families, and closed-sourced models like Claude and 
GPT, providing a more robust evaluation and establishing a clearer 
understanding of LLMs' generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces SCYLLA, a dynamic evaluation framework that quantifies the generalization abilities of LLMs by distinguishing between memorization and true generalization across a spectrum of task complexities. SCYLLA assesses models on both in-distribution and out-of-distribution data across various complexities, revealing a "generalization valley". Results from evaluating 28 LLMs highlight the effects of model size on generalization, offering insights into the threshold at which LLMs shift from generalizable to memorized behavior.

### Strengths
1. The proposed method is straightforward and accessible, with clearly illustrated figures that enhance understanding.
2. The study discusses a critical issue by evaluating LLMs' generalization capabilities, providing valuable insights into the models' potential and limitations.
3. The experiments are comprehensive and detailed, offering sufficient details to ensure reproducibility.

### Weaknesses
1. The study does not examine how different prompt techniques might impact model performance. If a model can handle more complex OOD problems under few-shot prompts, does that indicate it possesses generalization capabilities?
2. The method of generating OOD tasks by replacing numbers may be overly simplistic, potentially limiting its effectiveness. This approach might not produce genuinely OOD questions or represent all OOD types, as the model could easily recognize and substitute numbers. The tasks may be too similar in structure to ID tasks.
3. Some conclusions, such as the presence of the generalization valley, may lack novelty. Since all models perform well on simple tasks but poorly on complex ones, the existence of the valley and the rightward peak shift might be predictable.

### Questions
1. Could simple prompt techniques significantly enhance base model performance and reduce the peak in performance gaps?
2. The following study [1] also examines the relationship between OOD challenges and model size and complexity.

[1] ALCUNA: Large Language Models Meet New Knowledge

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel evaluation framework, SCYLLA, designed to assess the generalization capabilities of large language models (LLMs) across varying task complexities. The framework is scalable, dynamic, knowledge-light, and memorization-aware, addressing limitations of existing evaluation methods. 
Through extensive experiments on 20 tasks across 5 complexity levels, the authors identify a non-monotonic relationship between task complexity and the performance gap between in-distribution (ID) and out-of-distribution (OOD) data, termed the "generalization valley." 
This reveals a critical complexity threshold where reliance on non-generalizable behavior peaks, indicating the upper bound of LLMs' generalization capabilities. The study shows that as model size increases, this critical complexity shifts to higher task complexities, suggesting larger models can handle more complex tasks before over-relying on memorization. The paper benchmarks 28 LLMs, including both open-sourced and closed-sourced models, providing a robust evaluation and deeper understanding of LLMs' generalization abilities. Additionally, a new metric is proposed to reward models with strong OOD generalization and penalize overfitting to ID data.

### Strengths
- The idea that the evaluation dataset should be generated dynamically is interesting and can be useful for promoting the accuracy of the evaluation. 
- The performance gap between ID and OOD data is also an interesting and novel indicator.

### Weaknesses
- The paragraph of Line 300 is not very clear. "From these generated responses, we extract test examples and designate them as ID test data". How could you *extract* test examples? I thought all the generated examples form a candidate pool for test data. What are "the individual components within these examples"? Given that you generate no less than 10k examples, why only 256 are selected in the end?
- Can "ID/OOD data" selected by one model, i.e., mistral-7b, be regarded as ID/OOD data for another model, e.g., GPT-4?
- The scope of your study is narrowed down to (algorithm and numerical-related) reasoning task. However the claim is a little bit of a wider range and is simply referred to as "generalization" as a whole. 
- The discussions (e.g., section 4.4 and 4.5) are mainly observed evidence while lack of more insights. E.g., we can easily induce that larger model size brings better generalization, however, there are some other interesting conclusions you may draw by comparing different open-source models that are fine-tuned on code/math or not. Currently only Qwen2.5-Math/Coder-7B is tested but I believe there are other models like deepseek which has different versions of instruction tuning data.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an analysis of the generalization abilities of LLMs. To conduct this analysis, they have created a framework, SCYLLA, composed of tasks with different levels of algorithmic complexity. They use this framework to quantify the balance between generalization and memorization in LLMs, by comparing the results on in-distribution and out-of-distribution samples. From the accuracy difference of the two, they conclude that there exists a generalization valley and a critical complexity in the modes. They present a Generalization Score to measure this phenomenon. Finally, they use probe tasks to analyze the complexity that LLMs use to solve these problems.

### Strengths
- New Evaluation Framework (SCYLLA). They present a new evaluation framework that can be used to measure to evaluate the generalization performance of LLMs. 
- They uncover and measure the Generalization Valley Phenomenon. According to the experiments conducted on several models, there seems to be a consistent gap between their ID and OOD evaluations. And this gap has a peak (critical complexity) that shifts to the right as model size increases in open-source models. 
-  They conduct a large set of experiments and evaluate a good amount of well-known models.

### Weaknesses
- Dependence on approximate ID data. Due to the inability to access pre-training data, it is hard to estimate in-distribution data. The approach presented is a workaround that only focuses in the number generation. They do not present the estimated distribution for the models and their difference with their out-of-distribution data. 
- As far as I understand, they focus on tasks that require mathematical operations. They do not show a possible generalization to other kind of tasks. Adding other type of tasks could have benefitted the overall quality of the paper.

### Questions
In general it is a great paper and you have performed a great analysis from the results obtained. I have the following questions from the paper: 

- When creating the OOD sample, do you create 1 for each family of models or 1 for every single model? I do not think this is made clear in the paper. 
- What is the reason to select only problems that require numbers? Could it not be easily added some problem on string manipulation with a very similar strategy?
- What is the takeaway for the gray line in Figure 9, is the goal to present how the OOD accuracy is reduced as the complexity increases? I think this is not highlighted in this case.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SCYLLA, a framework for evaluating LLMs' generalization abilities across different task complexities. The key findings include:

- Discovery of a "generalization valley" - a non-monotonic relationship between task complexity and the performance gap between in-distribution (ID) and out-of-distribution (OOD) data
- Observation that larger models can handle more complex tasks before reaching their critical complexity - the point where models rely most heavily on memorization
- Benchmark results comparing 28 LLMs, showing closed-source models generally perform better at generalization

### Strengths
1. Novel evaluation framework that considers both complexity scaling, generalization, and memorization.
2. Comprehensive evaluation across many modern LLMs
3. Clear methodology for testing generalization capabilities
4. Interesting findings about model scaling and complexity relationships

### Weaknesses
1. ID/OOD Definition Issues (Major Issue):


The paper's method of determining ID data by asking the model to generate examples could be problematic
By using model-generated examples to determine what constitutes ID data and simply using complementary numbers for OOD, the paper creates a potentially circular and oversimplified definition of distribution shifts. This approach lacks validation against actual training distributions and may not capture meaningful distribution shifts. 

2. Other Issues:


- Limited task types (mostly focused on numerical/algorithmic tasks)
- No ablation studies on different prompt formats or chain-of-thought methods

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
3
