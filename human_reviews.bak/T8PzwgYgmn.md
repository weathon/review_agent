# Not All LLM Reasoners Are Created Equal

- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
We study the depth of grade-school math (GSM) problem-solving capabilities of LLMs. To this end, we evaluate their performance on pairs of existing math word problems together so that the answer to the second problem depends on correctly answering the first problem. Our findings reveal a significant reasoning gap in most LLMs, that is performance difference between solving the compositional pairs and solving each question independently. This gap is more pronounced in smaller, more cost-efficient, and math-specialized models. Moreover, instruction-tuning recipes and code generation have varying effects across LLM sizes, while finetuning on GSM can lead to task overfitting. Our analysis indicates that large reasoning gaps are not because of test-set leakage, but due to distraction from additional context and poor second-hop reasoning. Overall, LLMs exhibit systematic differences in their reasoning abilities, despite what their performance on standard benchmarks indicates.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on compositional GSM, showing that most LLMs have not “mastered” grade-school math reasoning, despite high performance on common benchmarks. It introduces a two-hop test set where two linked questions must be solved simultaneously. Experiments reveal a significant performance drop compared to GSM8K.

### Strengths
The compositional evaluation demonstrates that high benchmark performance does not reflect flexible reasoning capabilities, which is vital for understanding LLM development. 

The findings on fine-tuning leading to task overfitting are very interesting. It illustrates how fine-tuning on a single type of data, even within similar domains, can cause models to overfit, highlighting the importance of balancing downstream task performance with generalization.

### Weaknesses
The authors suggest that reasoning gaps arise from distractions of additional context and poor second-hop reasoning. Have you formally summarized the common types of mistakes? These errors might reflect issues with length generalization, coherence between questions, or reasoning patterns.


In the setup, did you combine different Q1s with a fixed Q2? It seems that the authors used a one-to-one mapping for constructing compositional test questions.

Regarding section 3.1, it appears that cost-efficient models show greater performance declines than high-cost models. Could this suggest that the general capabilities of smaller models hinder their understanding and solving of more complex questions? Have you analyzed the mistake patterns and behaviors of small versus large models?

In lines 369-377, whether the experiment is to compare between the accuracy of independently answering Q1 and the accuracy of determining the value of X in the compositional setting? How is the value of X parsed?

The experiments described in lines 369-415 are intriguing. Including qualitative case analyses on page 10 could provide a more detailed understanding of model mistakes.

### Questions
Please see the weakness section

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This explores the reasoning capabilities of large language models (LLMs) in solving grade-school math (GSM) problems, specifically focusing on compositional reasoning—where the solution to one problem depends on correctly solving another. The study introduces "Compositional GSM," a benchmark to test LLMs on pairs of related math problems, revealing significant reasoning gaps in most models. The study suggests that prevalent benchmarks might overestimate LLMs’ reasoning abilities and proposes compositional reasoning tasks as a more rigorous measure for evaluating true understanding in LLMs.

### Strengths
It also highlights multiple interesting findings, such as "small and cost efficient LLMs, which are broadly accessible and
crucial for real-world applications (Wan et al., 2024), exhibit larger reasoning gaps".

### Weaknesses
This paper tries to cover too many aspects without studying each in a detailed and comprehensive manner. 

For example, it showed cost-efficient LLMs perform badly on compositional GSM8k, but does not try to probe into the reasons behind this with more detailed error analysis or case studies. 

The study focuses on GSM8k vs two GSM8k questions chained together. Although GSM8k is a very important math reasoning benchmark, it fails to discuss other popular math reasoning datasets such as MATH which involve more complex and diverse reasoning. The authors have mentioned this as a future work, but having not discussed MATH weakens the overall contributions of the paper.

### Questions
Can you perform more detailed error analysis

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper studies how well Large Language Models (LLMs) can handle compositional grade-school math problems by evaluating their performance on pairs of GSM8K questions where the answer to the first question is needed to solve the second. The main finding is that while many LLMs perform well on individual GSM8K problems, they show significant "reasoning gaps" when solving these compositional pairs. The paper positions itself not just as introducing another benchmark, but as a case study revealing systematic differences in LLM reasoning capabilities that aren't apparent from standard benchmark performance. The work suggests that current evaluations may overestimate LLMs' true mathematical reasoning abilities, particularly for smaller and more cost-efficient models that are important for practical applications.

### Strengths
1. Introduction of Compositional GSM, a new evaluation approach that chains two GSM8K test questions together, requiring models to correctly solve both in sequence

2. Comprehensive evaluation of various LLMs, including Gemini, Gemma2, LLAMA3, GPT, Phi, Qwen2.5, and Mistral families

3. Several important empirical findings:
- Most models show a clear performance gap between standard GSM8K and compositional problems
- Smaller, cost-efficient, and math-specialized models show larger reasoning gaps
- Instruction tuning affects models of different sizes differently
- GSM8K-specific finetuning can lead to task overfitting
- The reasoning gaps stem from distraction by additional context and poor second-hop reasoning, rather than test set leakage

4. Analysis showing that code generation helps improve performance on compositional problems, particularly for smaller models

### Weaknesses
- My biggest concern is the gap between the motivation of the study and the proposed evaluation method. The authors propose the Compositional GSM as a tool to evaluate LLM reasoning, but don't sufficiently validate that the task actually measures real compositional reasoning ability. While they show poor performance on the second question even when the first is solved correctly, they don't establish whether this definitively indicates a reasoning gap versus other potential issues like prompt sensitivity, instruction following ability, or other format issues. 
- Further clarification on the motivation of the study. Even though we see the dropping performance, what does it mean in reality? In what kind of real scenarios, the users would have such evaluation settings like that? How does the performance transform into the real challenge we would like to solve? 
- Lack of concrete solutions for the problem. Though the authors discuss several potential ways that lead to the phenomenon, they do not try to propose solutions to solve the issue. For example, it is still unclear whether the downgraded performance is caused by the sensitivity or the format of the new test setting. Would it be possible to quickly try some simple fine-tuning? For example, could you repurpose the GSM8K training set to something like how you constructed the test set? Would such a simple fine-tuning method solve the issue?
- The evaluation was only conducted on GSM8K dataset, which limited the conclusion's validity to other reasoning scenarios.
- The paper claims to examine math-specialized models but tests a relatively narrow set (Numina-7B, Mathstral-7B, Qwen2.5). Some more open-source and open-weight math models might be added for a more comprehensive analysis. e.g., deepseek-math, MetaMath, MAmmoTH, etc.

### Questions
See Weaknesses

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
4

### Summary
This paper introduces a two-hop version of GSM8K, called compositional GSM8K, where two questions from the original test set are combined into one after making minimal changes. The analysis reveals significant reasoning gaps in LLM performance, particularly in smaller, cost-efficient, and math-specialized models. Instruction tuning improves performance on compositional GSM, but to a lesser extent than on the standard split, highlighting a greater challenge.

### Strengths
1. The proposed compositional GSM8K is a straightforward and effective method for testing the two-hop math reasoning abilities of LLMs.

2. The paper is easy to follow. The experiments are comprehensive, covering various types of models. The analysis explores various potential factors that could lead to weaker performance, such as model sizes, instruction tuning, fine-tuning, and using code as a format. It sufficiently supports their central claim regarding the deficiencies of LLMs in compositional mathematical reasoning.

### Weaknesses
1. As noted by the authors in the related work section, there are already several benchmarks for assessing the robustness of LLMs in math reasoning. Although this paper includes extensive experiments, the conclusion that LLMs struggle with multi-hop reasoning is not particularly surprising to me.

2. The two-hop QA format appears to be a minimal approach for testing mathematical compositional reasoning abilities. I was hoping for a more intricately designed benchmark, similar to SCAN or CFQ.

### Questions
The authors argue in Line 477 that "This raises the question of whether small and cost-efficient models are fundamentally limited in their ability to achieve such generalizations" and suggest we should rethink how we assess “reasoning". Can you elaborate on your insights and how we can achieve this?

### Soundness
4

### Presentation
4

### Contribution
2
