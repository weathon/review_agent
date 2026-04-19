# Steering Large Language Models between Code Execution and Textual Reasoning

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
While a lot of recent research focuses on enhancing the textual reasoning capabilities of Large Language Models (LLMs) by optimizing the multi-agent framework or reasoning chains, several benchmark tasks can be solved with 100\% success through direct coding, which is more scalable and avoids the computational overhead associated with textual iterating and searching. Textual reasoning has inherent limitations in solving tasks with challenges in math, logics, optimization, and searching, which is unlikely to be solved by simply scaling up the model and data size. The recently released OpenAI GPT Code Interpreter and multi-agent frameworks such as AutoGen have demonstrated remarkable proficiency of integrating code generation and execution to solve complex tasks using LLMs. However, based on our experiments on 7 existing popular methods for steering code/text generation in both single- and multi-turn settings with 14 tasks and 6 types of LLMs (including the new O1-preview), currently there is no optimal method to correctly steer LLMs to write code when needed. We discover some interesting patterns on when models use code vs. textual reasoning with the evolution to task complexity and model sizes, which even result in an astonishingly inverse scaling behavior. We also discover that results from LLM written code are not always better than using textual reasoning, even if the task could be solved through code. To mitigate the above issues, we propose three methods to better steer LLM code/text generation and achieve a notable improvement. The costs of token lengths and runtime are thoroughly discussed for all the methods. We believe the problem of steering LLM code/text generation is critical for future research and has much space for further improvement. Project Page, Datasets, and Codes are available at https://yongchao98.github.io/CodeSteer/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores if querying LLMs to generate code can be more effective than textual reasoning for tasks involving math, logic, and optimization.
Experiments across 14 tasks with 6 different LLMs show that (1) there is no single best method to optimally decide when to use code or text reasoning, (2) forcing code reasoning all the time can deteriorate performance, (3) smaller-size models tend to use code more often and outperform larger-size models for some tasks, and (4) mixing code and textual reasoning together improves performance but is limited by the model capacity and is more expensive to run due to more predicted tokens.

### Strengths
This is a great case study on the usage of code to solve mathematical and logical tasks with LLMs. The experimental settings are clear and easy to follow. The paper shows a lot of experimental results with various models and tasks.
The experiments are well suited to answer the paper’s investigation.
The proposed prompting method performs better on average than all other methods tried, at the expense of generated token length. 
Experimental results should be of interest to the research community.

### Weaknesses
This paper is a technical analysis of various prompting models & methods on math and logical reasoning tasks. While the results are interesting and the experiments exhaustive, this work does not provide a lot of solutions. The only novel contribution of this paper (aside from the experimental results) is the 3 proposed prompting methods: CodeInterpreter+, Code+Text+Sum., Self-Estimate Score, among which only one (Code+Text+Sum.) seems to perform well across the evaluated tasks. This work could benefit from a more detailed investigation on particular tasks to help identify potential solutions. For instance, analyzing the probabilistic confidence of the model when generating a text answer rather than code.

The paper claims that all tasks tested in this paper can be solved with code (with varying difficulty), yet no code-generation model has been tried. I strongly suggest the authors to add an experiment with a code generation model. Although the only prompting method that would make sense here is the default “Only Question”, it is an important baseline to consider.

Minor: the fact gpt3.5 outperforms gpt4 in some tasks does not mean there is an “inverse scaling _law_”. This is a very specific case in which smaller models are less certain and thus use external tools (code interpreters), which is expected.

### Questions
In general, the approach of trying to find the prompting method to solve all these tasks could never end. Prompt engineering can be much more effective when targeted to single tasks as one can give more information about the task, in context examples, etc.
Did you try task-specific prompts? Do you think you could solve some of these tasks like this?

After finding that large models generate code that simulates text reasoning rather than actual code, did you try to further improve the prompt? Simple things to try would be to add phrases like "think of an algorithm to solve the task and implement it in python", "do not try to answer in one step",  "be careful this is hard, be less confident" etc...

### Soundness
3

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
2

### Summary
The article titled "LLM CodeSteer: Steering Large Language Models Between Code Execution and Textual Reasoning" explores the balance between code execution and textual reasoning capabilities in Large Language Models (LLMs). It argues that while textual reasoning has limitations, especially in tasks involving math, logic, and optimization, direct coding can often provide a more effective solution. The study conducts experiments with 14 diverse tasks and 6 types of LLMs, finding that there is no single optimal method for guiding LLMs to choose between code generation and textual reasoning. It also observes an inverse scaling law, where smaller models sometimes outperform larger ones when augmented with a Code Interpreter (CI). The paper proposes three methods to improve LLM decisions on code/text generation: Code Interpreter+, Code + Text + Sum., and Self-estimate Score. These methods aim to enhance performance by combining code and textual reasoning or by using multi-turn execution/refinement. The article concludes that guiding LLMs in code/text generation is crucial for developing more capable agents and that there is significant room for improvement in current methods.

### Strengths
1. Comprehensive Analysis: The study provides a thorough analysis of LLMs' performance across a wide range of tasks, offering valuable insights into their strengths and weaknesses in code execution versus textual reasoning.

2. Sound Method: Practical Approach: By focusing on real-world tasks that can be solved through coding, the research offers practical applications for enhancing LLMs' capabilities. The proposed methods, such as Code + Text + Sum. and Self-estimate Score, present innovative ways to improve LLMs' decision-making processes in choosing between code and text.

3. This paper attempts to address an important question and proposes an effective method that achieves better performance than the compared methods. 

4. The paper is well-organized and clearly written.

### Weaknesses
1. Dependence on Task Type: The effectiveness of code versus textual reasoning is highly dependent on the task at hand, which may limit the generalizability of the findings.

2. Overconfidence Issue: The study highlights that larger models tend to be overconfident in their textual reasoning abilities, which can lead to suboptimal performance when code execution is more effective.

### Questions
None

### Soundness
3

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
This paper benchmarks seven prompting methods that combine text and code to tackle 14 tasks using six strong LLMs. The results indicate no universally optimal prompting method across all tasks and models, leading the authors to propose three new prompting methods that yield consistent improvements.

### Strengths
- The paper effectively identifies and analyzes the challenges faced by state-of-the-art LLMs in determining when to leverage code and uncovers some interesting phenomena. These findings offer valuable guidance for balancing code and text in LLM prompting.
- The proposed prompting methods are simple yet demonstrate effectiveness across different models.
- The paper is well-structured and easy to follow, with charts and tables that greatly enhance readability.

### Weaknesses
The paper reads more as a technical report than a top-tier scientific contribution.
- From a scientific perspective, the problem, while useful as a testbed for prompt engineering with code and text, is not significant enough. If framed as a technical report, a broader workload and richer insights might be expected to better inform the community. While the prompting comparisons are comprehensive, they are relatively straightforward, meaning the work may not substantially save time for others. Insights largely remain at the level of observed phenomena, describing model behavior (e.g., line 274: "prompting LLMs to always respond with code (Fig 6b) performs just as poorly, or even worse"). Additional investigation into the reasons behind these behaviors might enhance understanding.
- Additionally, the three proposed methods appear to be ensembles of existing approaches, such as summarization and self-reflection. Though effective, these methods do not introduce significant new insights.

### Questions
- I am curious about how to delineate the boundary between text and code in prompting. For instance, if the model is prompted to generate "pure Python code with rich natural language-annotated comments," which category does this fall under? If it’s considered "All code," could we perhaps frame all tasks this way, where the simplest output might be `print(answer)`?

### Soundness
4

### Presentation
4

### Contribution
3
