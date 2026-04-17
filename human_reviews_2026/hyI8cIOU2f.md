# Are LLMs Better Formalizers than Solvers on Complex Problems?

- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
A trending line of recent work advocates for using large language models (LLMs) as formalizers instead of as end-to-end solvers for logical reasoning problems. Instead of generating the solution, the LLM generates a formal program that derives a solution via an external solver. While performance gain of the seemingly scalable LLM-as-formalizer over the seemingly unscalable LLM-as-solver has been widely reported, we show that this superiority does not hold on real-life constraint satisfaction problems. On 4 domains, we systematically evaluate 6 LLMs including 4 large reasoning models with inference-time scaling, paired with 5 pipelines including 2 types of formalism. We show that in few-shot settings, LLM-as-formalizer underperforms LLM-as-solver. While LLM-as-formalizer promises accuracy, robustness, faithfulness, and efficiency, we observe that the present LLMs do not yet deliver any of those, as their limited ability to generate formal programs leads to failure to scale with complexity, hard-coded solutions, and excessive reasoning tokens. We present our detailed analysis and actionable remedies to drive future research that improves LLM-as-formalizer.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper analyzes the model’s performance and behavior when serving as a solver or formalizer, aiming to better understand its real-world applicability. The results demonstrate that using the model as a solver yields the best performance, robustness, and efficiency. Further analysis identifies the failure modes of the LLM-as-formalizer paradigm, providing valuable insights for strengthening this approach in future research.

### Strengths
- The paper is highly interesting and presents insightful findings.

- The experiments are well-designed and intuitive.

- The presentation and writing are clear, structured, and easy to follow.

### Weaknesses
- The analysis could be expanded to provide deeper insights. For instance, solvers such as Isabelle offer detailed feedback on why an execution fails—this kind of fine-grained diagnostic information could be more informative than the primarily syntactic feedback from Z3. 

- Additionally, in Figure 2, Python appears to outperform SMT in many cases. This is a particularly intriguing finding, given that SMT solvers are specifically designed for constraint satisfaction tasks, whereas Python is a general-purpose programming language. Could Python potentially replace domain-specific languages in this context, considering that modern LLMs/LRMs are trained on large volumes of code? Expanding on this analysis could significantly strengthen the paper.

- Figure 4 appears overly complex and difficult to interpret. It might be clearer if presented in the same format as Figure 6.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a systematic evaluation of the "LLM-as-formalizaer" paradigm. This paper tries to answer an interesting question -- "is LLM-as-formalizer better or LLM-as-solver better?" With focusing on CSP domain, they found LLM-as-formalizer underperforms LLM-as-solver in many tasks; formalizers are not robust; and fomarl language easily prone to erros (parsing or missing constraints).

### Strengths
- Interesting research question 
- Comprehensive evaluation -- 6 LLMs, 4 domains, and 2 formal language
- FIne-grained analysis

### Weaknesses
- My biggest concern is the domain breath -- Im not sure if CSP is the best problem for comparing LLM-as-solver and LLM-as-formalizer. Some papers, for example [1], have studied the relationship between solver and formalizer by using a routing-based method. I feel like CSP are possibly not suitable to using symbolic/mathematical logic. Have you tried evaluation on logical / mathematical datasets?
- LLM-as-formalizer might be too simplified, have you tried more advanced LLM-as-formalizer methods? 

I'm happy to raise my score if you can answer my biggest concern well : )

[1] Han, S., Liu, T., Li, C., Xiong, X., & Cohan, A. (2024). HYBRIDMIND: Meta Selection of Natural Language and Symbolic Language for Enhanced LLM Reasoning. arXiv preprint arXiv:2409.19381.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the effectiveness of  large language models (LLMs) on planning tasks when they are used as formalizers (translating problem descriptions into formal programs) vs when they are used as solvers (directly generating answers).
The paper presents a series of empirical studies to investigate the research question. Specifically, the paper evaluates six models (including DeepSeek-R1, Qwen-3, and GPT-5) across four constraint satisfaction domains (such as trip planning). It analyzes accuracy, robustness to complexity, faithfulness, and efficiency. The paper draws the conclusion that current LLMs often underperform as formalizer, and do not deliver "accuracy, robustness, faithfulness, and efficiency”

### Strengths
The paper studies an important and interesting question to evaluate the advantage of LLMs as formalizers versus solvers. This has been a hot topic in reasoning research.

The paper is well-written and easy to understand, with well-organized research questions.

The experimental study covers multiple domains, models, and covers using both Python and SMT.

### Weaknesses
I appreciate the authors’ efforts in clearly organizing the research questions and results. However, I have some concerns regarding the interpretation of the findings.

In particular, for RQ3 (Section 5.3), the paper argues that LLMs-as-formalizers are unreliable (as also stated in the abstract) and provides a detailed analysis of their failure modes. However, it does not include an analysis of failures for LLMs-as-solvers, which makes the comparison incomplete.

While Section 5.3 touches faithfulness, it misses discussion of another important aspect of using LLMs-as-formalizers: LLM-as-formalizers may better refuse to give you an answer when it fails, whereas LLMs-as-solvers are more likely to produce an incorrect final answer. In this sense, LLMs-as-formalizers may be better calibrated or better in selective prediction behavior (it is better to abstain than give a wrong answer). Figure 4 suggests that SMT-based methods frequently fail to produce any solution (“no plan”). It would therefore be helpful to report the answer extraction rate for both paradigms and the accuracy normalized by the subset of cases where a final answer is produced.

The paper could also benefit from more discussion of problem complexity and scalability. As shown in Figure 3, many examples involve relatively few constraints, and Figure 5 indicates that the reasoning token counts are mostly within ~10K. This range may be insufficient to draw comprehensive conclusions about scalability. A more detailed analysis of performance versus reasoning tokens would strengthen the paper’s claims.

### Questions
Could the authors provide the answer extraction rate for the LLM-as-solver setting?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors rigourously explore whether LLMs as a formalizer where the LLM translates the NL input to language suitable to the solver is effective over LLM as a solver, where an LLM is directly used to solve the problem.

### Strengths
1. The paper is thorough with interesting experiments to show that LLMs/LRMs as a formalizer paradigm still suffers from major lacunae i.e, with increase in the problem complexity the performance of the LLMs becomes considerably low.
2. The paper is well presented with appendix consisting of all examples of prompts/generated code by the LLMs which is something I would really appreciate the authors for. This personally helped me go deeper in the paper and understand the work more throoughly.

### Weaknesses
I have the following concerns regarding the paper:
1. For LLM formalizer experiments, the solver being used for a given task also matters. Logic LM [1] and, [2] both use constraints library which is more natural for solving problems with planning/CSP related tasks i.e, Meeting Planning, Trip Planning and Calendar Planning. I think this experiments would be helpful. 
2. Both [1] and [2] which use LLM as a formalizer have been proven to be effective in logical reasoning tasks and planning tasks on Multiple datasets such as ProntoQA [3], FOLIO [4], Proofwriter [5] and these are in my opinion more suitable datasets for evaluating LLMs as a formalizer paradigm, Can you please explain the choice of the datasets that you have choosen and why do LLMs as a formalizer peform much better [3], [4], [5] than on the datasets that you have chosen? Is it because of the solver that has been chosen that i have mentioned earlier in 1.?
3. Z3 solver is more suitable for logical reasoning tasks and using LLM as a formalizer with Z3 would be also be more appropriate on Logical Reasoning datasets. Would you be able to confirm if the results are consistent on a logical reasoning dataset for example [6]







[1] LOGIC-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning

[2] FCoReBench: Can Large Language Models Solve Challenging First-Order Combinatorial Reasoning Problems?

[3]  Language models are greedy reasoners: A systematic formal analysis of chain-of-thought

[4] FOLIO: natural language reasoning with first-order logic.

[5] Proofwriter: Generating implications, proofs, and abductive statements over natural language.

[6] LogicBench: Towards Systematic Evaluation of Logical Reasoning Ability of Large Language Models

### Questions
Please address the weakness

### Soundness
2

### Presentation
3

### Contribution
2
