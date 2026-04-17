# Complex Logical Instruction Generation

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Instruction following has catalyzed the recent era of Large Language Models (LLMs) and is the foundational skill underpinning more advanced capabilities such as reasoning and agentic behaviors. As tasks grow more challenging, the logic structures embedded in natural language instructions becomes increasingly intricate. However, how well LLMs perform on such logic-rich instructions remains under-explored. We propose LogicIFGen and LogicIFEval. LogicIFGen is a scalable, automated framework for generating verifiable instructions from code functions, which can naturally express rich logic such as conditionals, nesting, recursion, and function calls. We further curate a collection of complex code functions and use LogicIFGen to construct LogicIFEval, a benchmark comprising 426 verifiable logic-rich instructions. Our experiments demonstrate that current state-of-the-art LLMs still struggle to correctly follow the instructions in LogicIFEval. Most LLMs can only follow fewer than 60% of the instructions, revealing significant deficiencies in their capacity to handle instructions that involve complex logical structures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LogicIFGen, a framework which enables automatic generation of test instances for logic relations based on code. The framework is scalable and verifiable. Using this framework, the authors construct LogicIFEval, a benchmark curated to evaluate LLM capabilities in logic understanding and instruction following. Extensive empirical experiment results show that many of the models have poor performance on the benchmark, highlighting the issues in using LLMs to handle complex logic queries.

### Strengths
1. The proposed framework, LogicIFGen, is scalable and verifiable. The authors also conduct human studies to show that the generation is of high quality.
2. The research question itself is interesting and important: LLMs have to be able to follow various types of instructions, which potentially involve complex logic relations, to properly serve users.
3. Writing and presentation are very clear. It is very easy to understand the authors' points.

### Weaknesses
1. Related work section is not comprehensive enough. It reviews many works which are relevant but not closely relevant to the research question in the general LLM reasoning area. More strongly related works should be thoroughly discussed as I will elaborate below.

2. The novelty of the research question is not extremely clear given many existing works in highly similar areas. This is my biggest concern. I think there are many existing works in code execution which are highly relevant, but not fully discussed in the related work section. For example, [1,2,3,4, inter alia] have done comprehensive analysis on the limitations of LLMs in simulating code execution result, trace, and executing natural language instructions which can be solved/verified by code. The authors should more carefully review relevant literature and argue for their novelty.

[1] La Malfa, E., Weinhuber, C., Torre, O., Lin, F., Marro, S., Cohn, A., ... & Wooldridge, M. (2024). Code simulation challenges for large language models. arXiv preprint arXiv:2401.09074.

[2] La Malfa, E., Weinhuber, C., Torre, O., Lin, F., Huang, X. A., Marro, S., ... & Wooldridge, M. (2025). Code Simulation as a Proxy for High-order Tasks in Large Language Models. arXiv preprint arXiv:2502.03568.

[3] Sun, S., Hsieh, C. P., Ladhak, F., Arakelyan, E., Serano, S. A., & Ginsburg, B. (2025). L0-Reasoning Bench: Evaluating Procedural Correctness in Language Models via Simple Program Execution. arXiv preprint arXiv:2503.22832.

[4] Liu, C., Dylan Zhang, S., Ibrahimzada, A. R., & Jabbarvand, R. (2024). Codemind: A framework to challenge large language models for code reasoning. arXiv e-prints, arXiv-2402.

### Questions
1. It would be really helpful if the authors could kindly address concerns mentioned above.

2. The theme of this paper is testing complex logic. What is the great advantage of using code as backbone to generating the questions instead of using classic formal logic frameworks such as first-order logic? The formal logic is also verifiable and easily programmable.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces LogicIFGen, a method that uses an LLM to anonymize functional programs. This method further introduces state tracking variables and translates the programs into a natural language description of the function. The method is used to create a benchmark dataset, called LogicIFEval, for evaluating the ability of LLMs to follow instructions when asked to execute the provided natural language instruction step-by-step for several test cases. The complexity of the instructions is captured using a syntax-tree-based measure on the anonymized programs. Several frontier LLMs were tested on the benchmark, revealing declining performance with increasing instruction complexity. Further analysis of the models' different failure modes was conducted.

### Strengths
Overall, the paper is easy to follow. The experiments appear to be fully documented and reproducible, and the topic of assessing LLM abilities in relation to the logical complexity of the task is highly relevant.

### Weaknesses
1.  The paper's main weakness is the limited relevance and coherence of the analysis conducted. Although assessing the abilities of LLM in instruction-following tasks by investigating dependency on increasing levels of logical complexity is promising, the main results focus on comparing the performance of different models. The overall pattern of declining performance with increasing instruction complexity is briefly discussed. However, the subsequent analysis of different failure modes does not consider varying difficulty levels. Hence, there is no insight into whether or how these failure modes depend on instruction complexity. This does not align with the paper's motivation and potential novelty. To significantly improve the paper's quality, the benchmark and related analysis should focus on the connections between LLM behavior and the logical complexity of instructions. This would allow the paper to fulfill the promises made in the abstract and introduction. 
2.  A second weakness is the lack of discussion or theory on the motivation behind the introduced dataset. Why is it important to explicitly prohibit the model from using programming or tools in general? Does the benchmark provide evidence for or against systematic compositionality in LLM behavior?

In summary, the paper appears to be immature in its current state. It presents an interesting idea for using LLM to process a selection of functions into datasets for assessing LLM instruction-following abilities in relation to task complexity. However, it lacks a sufficient theory explaining why and how these abilities can be assessed with the proposed dataset. Additionally, the analysis is insufficient for meaningfully evaluating the LLM behavior related to task complexity.

### Questions
1.  Table 1 should specify whether performance is a percentage of test cases or questions. The number of questions in brackets after the difficulty levels seems to indicate a percentage of questions. However, it is unclear whether a question is counted if only some test cases are correct in terms of output, state, or both. The numbers appear to be a percentage of test cases, especially when compared to Figure 3. For example, GPT-5 has state tracking errors in 28.1% of questions but could not achieve a "state" performance of 89.44% correct answers; rather, it achieved this result for only a subset of test cases.
2.  Why is the fourth result group labeled "Average" when the subtitle says "Overall"? The most intuitive average would be the weighted and unweighted means of the three columns on the left, respectively. However, that is not the case. Is this about overall performance rather than an average?
3.  Figure 3 needs a subtitle and should specify if it is about at least one of the specific failure modes in some test case per question.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes LogicIFGen and LogicIFEval. LogicIFGen is a method framework of generating natural language instructions based on code, and LogicIFEval is a benchmark derived by using the framework on hard coding problems. Then the authors tested the current LLMs on the benchmarks and shows it is very challenging for the current models.

### Strengths
- The paper studies an interesting problem of building complex natural language instructions from code and test the model's instruction following ability by using these code generated instructions.
- The paper's main pipeline of building these instructions is interesting and solid. The paper also did a decent job in collecting the coding problems which could be a contribution to the community.

### Weaknesses
- The naming is very confusing. Fundamentally LogicIFGen is the framework and LogicIFEval is derived from using this. But the naming made this very misleading. 
- I think the analysis part is not solid enough. For example, some simple reasoning baselines such as Program-of-Thought should be tested and analyzed. Since the instruction is derived from code, I think in general the paper should consider the aspect of reasoning with code.

### Questions
- I would like to see how the current models w/ strong reasoning methods perform on this task.
- I'm curious how do the author view the difference between using natural language instructions vs directly using code as instructions and asking the model to follow the logic.

### Soundness
3

### Presentation
3

### Contribution
3
