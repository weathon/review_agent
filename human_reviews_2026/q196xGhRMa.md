# CodePDE: An Inference Framework for LLM-driven PDE Solver Generation

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Partial differential equations (PDEs) are fundamental to modeling physical systems, yet solving them remains a complex challenge. Traditional numerical solvers rely on expert knowledge to implement and are computationally expensive, while neural-network-based solvers require large training datasets and often lack interpretability. In this work, we frame PDE solving as a code generation task and introduce CodePDE, the first inference framework for generating PDE solvers using large language models (LLMs). With CodePDE, we present a thorough evaluation on critical capacities of LLM for PDE solving: reasoning, debugging, self-refinement, and test-time scaling. CodePDE shows that, with advanced inference-time algorithms and scaling strategies, LLMs can achieve strong performance across a range of representative PDE problems. We also identify novel insights into LLM-driven solver generation, such as trade-offs between solver reliability and sophistication, design principles for LLM-powered PDE solving agents, and failure modes for LLM on hard tasks. These insights offer guidance for building more capable and reliable LLM-based scientific engines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CodePDE, a inference framework for LLMs to generate code for solving PDEs given a problem described in natural language. This is done by treating PDE solving as a code generation task.
The paper incorporates LLM capabilities, chain-of-thought, closed-loop debugging and refinement for this task. The paper does analysis into the LLMs skill-sets and tradeoff between LLM reliability and code complexity. The paper demonstrates how various LLMs can be used as agents for scientific computing given the paper's inference strategies.

### Strengths
* LLM abilities are constantly improving. This paper presents an API to leverage LLMs by framing PDE solving as a code generation task, enabling LLMs to produce solver code directly from natural language.

* CodePDE integrates task specification, code generation, debugging, evaluation, and refinement in a structured pipeline.

* The evaluating is systematic. The paper analyses 16 LLMs across five PDE families using metrics like nRMSE, convergence rate, and execution time. The paper clearly presents experimental results and takeaways when answering their experimental questions.

* self-Refinement with coarse feedback (nRMSE) can improve LLM solvers

* Unlike numerical solvers, LLM generated code is human-readable facilitating error diagnosis and transparency. The paper identifies trade-offs between solver reliability and sophistication.

### Weaknesses
> Takeaways. LLMs can improve code for better accuracy using simple performance feedback.
Interestingly, the best models at generating code are not always the best at refining it, suggesting
these are two different skills.
* (L356 above): Perhaps some percentage improvements in table 1 between " CodePDE: Reasoning + Debugging (best of 32)" and "CodePDE: Reasoning + Debugging + Refinement (best of 12)" would better help justify this claim, where the reasoning is currently unclear.

* Some mixed results between CodePDE and the neural network and foundation model baselines.

> In general, solution quality generally improves with
increasing sample count n, with the most significant gains observed between n=4 and n=16.
Beyond this point, returns diminish, suggesting that moderate sampling budgets often suffice to reach
near-optimal performance.
* (L364 - 266 above): Is it possible to classify the types of errors that each model makes? Such analysis may explain the performance plateau and how to improve the nRMSE lower bound.

### Questions
* For each PDE evaluation, what is the behavior of the LLMs over dataset subsets based on difficulties, initial conditions, or characteristics of the PDEs? For a given PDE, is there a relationship between PDE characteristics and CodePDE LLM performance?

> Interestingly, while advanced reasoning models (e.g., o3 and DeepSeek-R1) typically lead to better
solvers in the “reasoning + debugging” stage, they are not necessarily better than standard ones
(e.g., GPT-4o and DeepSeek-V3) in the refinement stage.
* ( L352 - 355 above ): Are there additional signals that can be passed to the LLM for self-refinement for improved performance? How much information can be transmitted by nRMSE alone and what effects does this have on the LLM's solutions? 

> However, LLMs can occasionally introduce redundant operations or inefficient looping structures,
which leads to slower execution.
* (L842 above): Understanding the reasoning, debugging or refinement iterations / runtimes is an important step in analyzing the efficiency of an LLM driven framework for solving PDEs. What is the relationship between code structures (inefficient loops, readability, redundant operations) and the abilities of the LLMs to improve the quality of their solvers through iterative refinement / chain-of-thought ?

> Takeaways. A low failure rate alone is not a sufficient measure of a model’s capability, the ability
to generate diverse, high-order numerical methods is also critical. For challenging PDEs where
single-shot generation is prone to failure, test-time scaling is essential for obtaining
diverse, correct, and robust solvers.
* (L407 above): What is the relationship between an LLM's distribution of numerical method order and its PDE solving performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces CodePDE, an inference framework for solving partial differential equations (PDEs) by framing the problem as a code generation task driven by large language models (LLMs). The CodePDE framework consists of five steps: Task Specification, Code Generation, Debugging, Evaluation, and Solver Refinement. The authors conduct a comprehensive evaluation of 16 different LLMs across 5 representative PDE benchmarks.

The primary contribution is demonstrating that LLMs, when guided by this structured framework, can generate high-quality PDE solvers. These generated solvers are competitive with specialized neural solvers and outperform manually crafted reference solvers on 4 out of the 5 tasks evaluated. The paper emphasizes the critical importance of inference-time techniques such as automated debugging (which improved the bug-free rate from 41% to 84% ) and self-refinement based on nRMSE feedback. Finally, the work highlights the interpretability of this approach, using a failure case (the Reaction-Diffusion equation ) to show how the human-readable code reveals the LLM's algorithmic reasoning.

### Strengths
1. Comprehensive & Rigorous Experiments: A major strength is the experimental breadth. The authors test 16 LLMs on 5 PDE benchmarks and compare them comprehensively against numerical solvers, specialized software, multiple neural solvers (FNO, PINN, etc.), and other agentic workflows.
2. Deep Insight on Interpretability: The failure-mode analysis for the Reaction-Diffusion equation in Section 5.7 is excellent. It perfectly illustrates the core advantage of this method over "black-box" neural solvers: the generated code is human-readable. This allows researchers to precisely diagnose the model's reasoning failure (i.e., missing the analytical solution trick for the reaction term ), which is critical for high-stakes scientific applications.

### Weaknesses
1. Dimensionality Limitation: The evaluation is focused on 1D and 2D PDEs. The true challenge for PDE solvers (the "curse of dimensionality") lies in high-dimensional problems. It is unclear how this framework would scale, as the complexity of the solver code (e.g., 3D FDM stencils) would increase dramatically.
2. Practicality of Refinement Signal: Step 5 (Solver Refinement) relies on nRMSE as the feedback signal, which requires a ground-truth solution. In real-world scientific discovery, the entire purpose of solving the PDE is that the ground truth is unknown. This seems to be a major limitation for practical application.
3. High Inference Cost: The framework relies on extensive LLM sampling: "test-time scaling" (best-of-n, where n=32), iterative refinement (12 samples), and debugging (up to 4 rounds). The total number of LLM calls to produce one high-quality solver appears very high. The paper measures the final solver's execution time but not the generation cost, which is a key practical barrier.
4. Weakened Novelty of Contribution: The paper's claim to be the "first inference framework for generating PDE solvers" is significantly weakened by existing work. There are already a number of published LLM-based solvers for PDEs, like PINNsAgent[1], which also operate by leveraging LLMs to generate solver code for solving PDEs. The existence of such prior art diminishes the claimed novelty of the paper's primary contribution.
5. Insufficient Baseline Comparisons: The experimental evaluation lacks comparisons against several key baselines. Notably, foundational neural solvers such as DeepONet[2] are omitted from the comparison in Table 1. Furthermore, the paper does not benchmark its performance against other similar and highly relevant LLM-based PDE solvers, such as PINNsAgent[1].

[1] Wuwu, Qingpo, et al. "PINNsAgent: Automated PDE Surrogation with Large Language Models." arXiv preprint arXiv:2501.12053 (2025).

[2] Lu, Lu, Pengzhan Jin, and George Em Karniadakis. "Deeponet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators." arXiv preprint arXiv:1910.03193 (2019).

### Questions
1. The self-refinement step (Step 5) uses nRMSE, which requires a ground-truth solution. How is this step intended to work in a real-world scenario where no ground truth is available?
2. The evaluation focuses on 1D and 2D problems. How do the authors envision this framework scaling to 3D or higher-dimensional PDEs?
3. The experimental evaluation in Table 1 appears to be missing key comparisons, such as some traditional neural operators (e.g., DeepONet), and LLM-based automatic solvers for PDE (e.g., PINNsAgent).
4. The framework relies on extensive LLM sampling, including best-of-n scaling ($n=32$) 1, 12 refinement samples 2, and up to 4 rounds of debugging3. This implies a significant "generation cost" (e.g., total LLM calls or tokens) to produce a single high-quality solver. While the execution time of the final solver is measured4, could the authors provide an analysis of this generation cost? How does this inference overhead compare to the practical cost of implementing a reference solver or training a neural baseline like FNO?

### Soundness
3

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
This paper introduces CodePDE, an inference framework that uses large language models (LLMs) to generate executable numerical solvers for partial differential equations (PDEs). The authors frame PDE solving as a code generation task and evaluate LLM capabilities across reasoning, debugging, self-refinement, and test-time scaling. Through systematic experiments on five representative PDE families, they demonstrate that LLMs equipped with their framework can produce solvers competitive with hand-crafted numerical methods and specialized software. The paper also provides insights into trade-offs between solver reliability and sophistication, design principles for LLM-based scientific agents, and failure modes on challenging problems.

### Strengths
+ **Pioneering Exploration**: The work courageously explores a novel paradigm of using LLMs for numerical code generation, opening up new research directions in AI-enabled scientific computing.
+ **Comprehensive Evaluation**: Extensive benchmarking across 16 LLMs and 5 PDE families provides valuable data for the community.
+ **Practical Framework**: The debugging and refinement mechanisms demonstrate real utility for improving code generation reliability in scientific contexts.
+ **Insightful Analysis**: The findings about test-time scaling, numerical scheme diversity, and the generation-refinement skill dichotomy offer meaningful insights.
+ **Foundation for Future Work**: The framework and evaluation methodology establish a strong baseline for subsequent research in this direction.

### Weaknesses
+ **Limited Forward-Looking Insight**: While empirically thorough, the paper misses an opportunity to deeply discuss how this LLM-driven approach might evolve to complement rather than replace traditional numerical methods, and what unique advantages the fusion might bring.
+ **Practical Deployment Concerns**: The generated solvers, while accurate, lack the performance optimizations and battle-testing of established numerical libraries, limiting immediate practical utility.
+ **Motivation Gap**: The paper could better articulate scenarios where generating new solvers is preferable to intelligently configuring existing high-performance solvers.

### Questions
1. Looking forward, how do you envision the optimal division of labor between LLM-generated solvers and traditional numerical methods? What unique capabilities might emerge from their combination?
2. Could your framework be adapted to focus more on high-level solver selection and configuration, while leveraging established libraries for the core numerical computations?
3. What are the most promising research directions for improving the reliability and performance of LLM-generated scientific code to bridge the gap with hand-optimized implementations?
4. Beyond the metrics you evaluated, what other aspects should be considered when assessing the practical usefulness of AI-generated solvers in real scientific workflows?
5. How might the interpretability of LLM-generated solvers (as you note) be leveraged to create hybrid human-AI scientific computing systems?

### Soundness
3

### Presentation
3

### Contribution
3
