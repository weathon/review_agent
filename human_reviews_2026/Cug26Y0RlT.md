# AI-for-Science Low-code Platform with Bayesian Adversarial Multi-Agent Framework

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Large Language Models (LLMs) demonstrate potentials for automating scientific code generation but face challenges in reliability, error propagation in multi-agent workflows, and evaluation in domains with ill-defined success metrics. We present a Bayesian adversarial multi-agent framework specifically designed for AI for Science (AI4S) tasks in the form of a Low-code Platform (LCP). Three LLM-based agents are coordinated under the Bayesian framework: a Task Manager that structures user inputs into actionable plans and adaptive test cases, a Code Generator that produces candidate solutions, and an Evaluator providing comprehensive feedback. The framework employs an adversarial loop where the Task Manager iteratively refines test cases to challenge the Code Generator, while prompt distributions are dynamically updated using Bayesian principles by integrating code quality metrics: functional correctness, structural alignment, and static analysis. This co-optimization of tests and code reduces dependence on LLM reliability and addresses evaluation uncertainty inherent to scientific tasks. LCP also streamlines human-AI collaboration by translating non-expert prompts into domain-specific requirements, bypassing the need for manual prompt engineering by practitioners without coding backgrounds. Benchmark evaluations demonstrate LCP’s effectiveness in generating robust code while minimizing error propagation. The proposed platform is also tested on an Earth Science cross-disciplinary task and demonstrates strong reliability, outperforming competing models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel AI4S (AI for Science) Low-Code Platform powered by a Bayesian Adversarial Framework. The goal is to empower domain experts by translating high-level natural language prompts into executable, domain-specific requirements, eliminating the need for intricate prompt engineering. The framework contains three parts, including a Task Manager, a Solution Generator, and an Evaluator. The framework iteratively refines the prompt through Bayesian update rule. Results show that the framework works well when the user prompt is written by non-experts and it also works well on general code generation questions.

### Strengths
1. The paper is clearly written and the ideas are easy to follow
2. The code generation framework is novel and achieves SOTA performance on both AI4S benchmarks and code generation benchmarks
3. Extensive experiments have been conducted to show more in-depth details about the framework

### Weaknesses
1. For the SciCode benchmark, the baseline is somewhat unclear, and comparing it to only a single baseline seems insufficient.

### Questions
1. What would the token budget–performance trade-off look like compared to other methods?
2. Could you provide more insights into the other modules? For example, how important is it to design the TM prompts as shown in the appendix? Which pieces of information are most important?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposed a multi-agent framework for robust code generation. The robustness in the generated code is ensured by assigning roles to each agent in the framework, one agent sets increasingly difficult tests for the code while one aims to generate code to pass them. A third permits preemptive evaluation of generated code by predicting performance (by generalizing from existing data) rather than having to execute the code, saving time and computation resources.

### Strengths
- The underlying idea of gradually increasing the difficulty of tests that generated output must pass, where the update is done on the basis of actual external evaluation and not on the basis of potentially untrustworthy LLM assessment, is a very good one. And has been implemented well: being resource-efficient by evaluating only a subset of code generations.
- Strong empirical performance of the proposed framework, across a diverse range of testbeds and use-cases.

### Weaknesses
- It is very unclear *where* the “bayesian update rule” is used. As in Algorithm 1, it suggests that it is used to re-weight existing test cases to present increasingly challenging ones as iterations proceed—that is, for the final prompt curation as each iteration. As per Appendix B, the rule is used for selecting a candidate code for evaluation (hence, by the evaluator agent).
- Unclear algorithmic steps: Line 12 says the $\lambda$'s are the test case weights, however they never get used in subsequent steps. The text suggests that they are updated based on $S_3^t$, however, the distribution over **existing** prompts (used interchangeably with test cases, seemingly) gets updated.
- Missing details: In Appendix D.1, a case study of the proposed method vs Cursor and Windsurf is presented. But it is unclear what underlying base models are being used, which as per the results in Table 2, has a strong bearing on the final performance. Further, the framework will have *three* (same or different) base models while Cursor and Windsurf presumably work with just one. Which models are used by all these methods add important insight on the downstream performance. It is difficult to interpret the results and determine the effectiveness of the framework without these details.
- Appendix B talking about “code embeddings” and a general setup for how GPs work. Details about how the code is embedded are lacking? What model is used to encode the code into embeddings?

### Questions
Does one of the agents iteratively generate test cases over iterations? If so, how are the weights over existing test cases, denoted by $\lambda$ in Algorithm 1, used?

### Soundness
3

### Presentation
1

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
The paper proposes a multi-agent framework for AI for science code generation, which is trainable via Bayesian adversarial learning. The multi-agent framework enables a logical breakdown with a task manager, solution generator and an evaluator, where each agent has dedicated capabilites. The key idea is that the task manager iteratively modifies the prompt and test cases to different difficulty levels conditioned on the previous attempt by the solution generator. This naturally fits the notion of adversarial learning where two different modules compete against each other to achieve an equilibrium.

### Strengths
1. Very neat idea that can be plugged into any existing agentic framework and LLMs.
2. By design, the framework allows for curriculum learning (although this is not explicitly mentioned by the authors) -- I think this is afFigure 5 strength where the difficulty level can be adjusted over iterations.

### Weaknesses
1. Line 276: "TM adapts its weights for future evaluation" -- what weights are being updated here. Make it clear that the Bayesian updates are over the choice of prompts/codes, and not over tokens. Also, sub-agents are not trained, so there are no weight updates to the LLMs. 
2. Are the test cases part of the plan that goes through loop 1 in Fig 2? Ideally, the domain expert should also provide some feedback on test cases during the planning phase. It is possible that the LLM generating test cases only generate dummy tests especially if the task is really difficult.
3. Based on the example in D.2.3, the extra advice provided by the domain experts can be very specific modeling design ideas. How to ensure the extra advice do not divulge the ground-truth best solution?

### Questions
Address weaknesses above.

Additional comments: 
1. What is S(j+) in equation 2? 
2. Provide some outputs of TM agent with sample test cases. 
1. Can you upload the entire code snippet in D.1.6? 
2. You use different icons for plan and codes. Actually, the legend in Fig 2 maybe wrong -- the icon used to denote N parallel codes is marked as plan.
3. The distinction between loop 2 and loop 3 in Fig 2 should be explicitly discussed somewhere in Sec 2. 

Minor nitpickings:
1. Typos in line 15 algorithm 1: "Equation equation 234", "Test sase"
2. Typo line 230: "in Loop 3 of fig:Diagram",

### Soundness
3

### Presentation
2

### Contribution
3
