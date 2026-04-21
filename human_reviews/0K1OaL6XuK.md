# Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming

- Avg Score: 4.75
- Decision: Accept (Poster)
- Scores: 6, 6, 1, 6

## Abstract
While large language models (LLMs) have recently demonstrated strong potential in solving planning problems, there is a trade-off between flexibility and complexity. LLMs, as zero-shot planners themselves, are still not capable of directly generating valid plans for complex planning problems such as multi-constraint or long-horizon tasks. On the other hand, many frameworks aiming to solve complex planning problems often rely on task-specific preparatory efforts, such as task-specific in-context examples and pre-defined critics/verifiers, which limits their cross-task generalization capability. In this paper, we tackle these challenges by observing that the core of many planning problems lies in optimization problems: searching for the optimal solution (best plan) with goals subject to constraints (preconditions and effects of decisions). With LLMs' commonsense, reasoning, and programming capabilities, this opens up the possibilities of a universal LLM-based approach to planning problems. Inspired by this observation, we propose LLMFP, a general-purpose framework that leverages LLMs to capture key information from planning problems and formally formulate and solve them as optimization problems from scratch, with no task-specific examples needed. We apply LLMFP to 9 planning problems, ranging from multi-constraint decision making to multi-step planning problems, and demonstrate that LLMFP achieves on average 83.7\% and 86.8\% optimal rate across 9 tasks for GPT-4o and Claude 3.5 Sonnet, significantly outperforming the best baseline (direct planning with OpenAI o1-preview) with 37.6\% and 40.7\% improvements. We also validate components of LLMFP with ablation experiments and analyzed the underlying success and failure reasons.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides the LLM prompting-based framework for planning tasks. The main contribution lies in its use of prompt and pipeline templates, which can be used across various planning tasks. Here, planning problems from various domains were considered, and planning is treated as an optimization problem. In summary, LLMs are used as an optimizer. They used the formal planner to achieve the goal of planning as already shown in the previous works that LLM still lacks the coherent reasoning needed for planning. The main contribution is an end framework that deploys a zero-shot learning approach for both single and multi-stage planning tasks. Additionally, the author claims that their framework can handle self-critique to assess the problem in planning code to change and achieve the goal. Effectiveness of framework components is supported via the ablations.

### Strengths
1. LLMFP introduces a new perspective on using LLMs for formal optimization-based planning, a method that significantly expands the generalizability of planning tasks.
2. Experimental results are solid, with clear evidence of performance gains across diverse tasks and models. The ablation studies reinforce the utility of the framework components, which I really liked.
3. The ability to solve multi-step, multi-constraint problems without task-specific examples or extensive prior efforts is a major step forward in the area of LLM-based planning.

### Weaknesses
1. Complexity in FORMULATOR: Some parts, particularly the JSON representation and the code generation steps, could be simplified. While important, the handling of different variable types and constraints might be a bit dense for readers unfamiliar with optimization theory.
2. Regrading Multi-Step Planning Problem: The predicate, object, and update structure are not clear in multi-step planning. Also image shown for this is not utilized in conveying the idea. Overall, Figure. 2 examples are not clear and make things confusing. 
3. The author claims that their framework is a "general approach, which does not require task-specific examples or task-specific efforts"; however, in the paper, this statement is not supported in terms of explanations and prompt structure.
4. Some theoretical insights regarding performance would make this work more strong, right now its presented more like a experimental results.

### Questions
1. How LLMFP handles generalization across different planning tasks. Please correct me, but it seems that we need a very Elaborate prompt with a high level of detail for each task. 
2. In section 3.4 (code generator), readers can Benefit from prior work such as "CAPE: Corrective Actions from Precondition Errors using Large Language Models" and "CoT-TL: Temporal Knowledge Representation of Natural Language Planning Task for Autonomous Agents using Chain-Of-Thought." Or is LLMFP doing differently compared to the above works; if yes, explain or not, and make sure to provide proper background.

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
3

### Summary
The paper presents a framework that pairs LLMs with optimization tools to solve planning tasks without using task-specific knowledge. The authors define consecutive stages of reasoning that, generally speaking, consist of understanding, coding, and refining. For each stage, they discuss the prompting, formatting, and other relevant decisions. Through experimental validation, they show that LLMFP outperforms baselines in 9 domains.

### Strengths
I like the general idea and the presented approach. One could argue that it is simply a combination of prompt engineering and the incorporation of external tools. However, showing an effective way of doing this can be a significant contribution.

The baselines and ablations are well-chosen for evaluating the performance of LLMFP.

The paper is written very clearly, making it easy to read. The figures are well-chosen (particularly Figure 1), they are helpful in understanding the pipeline. I like the section structure and the focus on key takeaways when discussing experimental results. Most of my questions that arose while reading the text were addressed in later sections.

### Weaknesses
The goal stated in the introduction is "Can we build a universal LLM-based planning system that can solve complex planning problems without task-specific efforts?". However, my main concern is whether the tasks used for experiments are indeed complex planning problems. Specifically, the 5 multi-constraint problems resemble simply optimization problems rather than planning problems. Hence it's quite clear that adding an external optimizer to LLM would be much better than just using LLM. On the other hand, the multi-step problems seem to be rather simple and the main difficulty is to understand what we have to do rather than finding a good solution. Hence, I suggest adding at least one multi-step domain with high underlying complexity (e.g. Sokoban). If I missed something and some of your environments are actually NP-hard (or hard in any other reasonable sense), it should be remarked in the paper.

Since the method you propose is clearly subject to a tradeoff between performance and computation time, there should be a discussion of that. What's the wall time of LLMFP compared to the baselines? What's the cost of using SMT compared to querying LLM?

The description of baselines should be extended a bit. Are they prompted vanilla models plus the components described in lines 410-416, or do they also include other components, e.g. formatter? Also, the Code variant uses pure Python, but for a completely fair comparison you should also add variant which is forced to use SMT like LLMFP does. After reading the prompts used, it's also not clear to me whether they are explicitly instructed to provide optimal solutions, which is captured by the metrics. Also, I suggest discussing the failure modes of the baselines (in the Appendix).

### Questions
1. What are the most common failure modes of the baselines?

2. Are the baselines prompted vanilla models plus the components described in lines 410-416, or do they also include other components, e.g. formatter?

3. What are the success rates of the tested methods? Do they all achieve 100% and the question is only whether the solution is optimal, or do some methods fail to solve some instances at all?

4. What's the wall time of LLMFP compared to the baselines?

5. Are the methods explicitly instructed to provide optimal solutions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
LLMFP is proposed which leverages LLMs to tackle complex planning problems by formulating them as optimization tasks.

### Strengths
LLMFP's ability to handle a wide variety of planning problems without task-specific examples is a significant strength.

### Weaknesses
1. The baselines for comparison do not seem to be a fair comparison to LLMFP. See questions.
2. The related work does not cover relevant set of papers that should have been used a baseline to compare this work. Mentioning a few of them below - 
[1] Webb, T., Mondal, S. S., Wang, C., Krabach, B., & Momennejad, I. (2023). A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models. arXiv preprint arXiv:2310.00194.
[2] Fabiano, F., Pallagani, V., Ganapini, M. B., Horesh, L., Loreggia, A., Murugesan, K., ... & Srivastava, B. (2023, December). Plan-SOFAI: A Neuro-Symbolic Planning Architecture. In Neuro-Symbolic Learning and Reasoning in the era of Large Language Models.
[3] Katz, M., Kokel, H., Srinivas, K., & Sohrabi, S. (2024). Thought of Search: Planning with Language Models Through The Lens of Efficiency. In The First Workshop on System-2 Reasoning at Scale, NeurIPS'24.

### Questions
1. What is the definition of a planning problem in this paper?
2. Why are the baselines only LLMs when the proposed approach is a framework/architecture? LLM-PFC [1] approaches planning problems similarly and there are other baselines to consider like Plan-SOFAI [2].
3. LLMs when used with API's are found to hallucinate new API functions or overuse a specific API call. Is such behavior observed here?
4. When it is a planning problem, why not directly use a symbolic planner and why is this architecture beneficial?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of solving planning problems that are given in natural language. The proposed algorithm they propose – LLMFP -  is a workflow of multiple LLMs, including an LLM to extract variables and constraints from the text, an LLM to formulate the extracted variables and constraints as an SMT problem in a specific format,  an LLM to convert this format to code that can be run by an SMT solver, and an LLM to verify and correct mistakes by the other LLMs. This LLM workflow is evaluated against the other LLM-based methods to solve planning problems, including one that is similar to the LLMFP but creates PDDL instead of an SMT problem. The authors also examine how results can be better by adding some task-specific expertise. The results over a set of benchmark problems show that LLMFP is, in general, much better than the baselines.

### Strengths
Strength
- The paper is in general clear (even if it is sometimes hand-wavey)
- The problem is interesting and the related work seems to cover all bases
- The results are impressive, and much better than the baselines. 
- The proposed workflow makes sense and works well.

### Weaknesses
- I’m not sure if the novelty of the proposed work over the PDDL-based approach is sufficiently novel for a top conference. 
- The appendix is huge (~40 pages!). This seems to me not reasonable, as the main paper should be self contained. 
- The presentation is too much hand-wavy. It would be great to try to capture more of it in a more formal manner

### Questions
1.  As the authors noted, LLMs have been used to translate natural language to planning problems. Similarly, the mapping from planning to SMT is well known in the planning literature. So, is the novelty is limited to combining the two ideas together??
2. In page 3, just above the first paragraph, you seem to say that encoding to PDDL requires more human effort than encoding to SMT. Can you elaborate why?
3. How do you encode the length of the plan? When compiling planning to SAT or SMT, this is an issue because the solver (SAT/SMT) requires to set an upper bound while in PDDL it does not have too.

### Soundness
3

### Presentation
3

### Contribution
3
