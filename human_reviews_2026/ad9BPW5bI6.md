# Automated Algorithm Design with LLMs: A Benchmark-Assisted Approach to Black-Box Optimization

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Large Language Models (LLMs) have already been widely adopted for automated algorithm design, demonstrating strong abilities in generating and evolving algorithms across various fields.
Existing work has largely focused on examining their effectiveness in solving specific problems, with search strategies primarily guided by adaptive prompt designs.
In this paper, through investigating the token-wise attribution of the prompts to LLM-generated algorithmic codes, we show that providing high-quality algorithmic code examples can substantially improve the performance of the LLM-driven optimization.
Building upon this insight, we propose leveraging prior benchmark algorithms to guide LLM-driven optimization and demonstrate superior performance on two black-box optimization benchmarks: the pseudo-Boolean optimization suite (pbo) and the black-box optimization suite (bbob). 
Our findings highlight the value of integrating benchmarking studies to enhance both efficiency and robustness of the LLM-driven black-box optimization methods. The source code and auxiliary materials are provided at https://anonymous.4open.science/r/ICLR2026-submissionID2452-D709 .

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the automated algorithm design based on LLMs for black box optimization. Using token-wise relevance attribution (AttnLRP), the authors find that example code in prompts has a dominant influence on LLM outputs. Building on this, they introduce a benchmark-assisted guided evolutionary approach (BAG), leveraging historical high-quality algorithm codes to guide LLM-driven search. Experiments on two BBO benchmarks (pbo and bdob) show that BAG improves performance and robustness against several LLM-driven methods.

### Strengths
* The use of AttnLRP to quantify prompt component impact on output code is interesting and new to optimization tasks.
* The proposed method yields better performance for the two studied benchmark problem sets.
* The appendix provides detailed breakdown results and illustrations of the example code.

### Weaknesses
* The proposed concept of "benchmark guided approach" appears to be a rebranding of in-context learning, which is well-established in LLM literature. The distinct contribution beyond standard in-context learning practices remains unclear. The analysis lacks an in-depth discussion.

* While the AttnLRP analysis provides interesting insights that seem to be applicable to broader LLM-driven algorithm design methods across various optimization tasks, the paper narrows its focus to black-box optimization only. The motivation for this restriction is not explained, and it remains unclear whether the proposed method could be applied to other optimization domains.

* Also, the analysis lacks an in-depth discussion. Section 4 stops at showing that "example code has the highest attribution." The robustness of this conclusion is not examined.

* I also have concerns that the approach's effectiveness appears to depend heavily on the quality of exemplar algorithms in the benchmark set. The paper lacks sufficient analysis and discussion regarding whether the proposed method may overfit to the provided benchmark algorithms and how exemplar quality and diversity impact overall performance.

* The evaluation lacks comparison with other in-context learning techniques from the broader LLM literature. Additionally, detailed ablation studies and sensitivity analyses are absent.

### Questions
Please address the above concerns.

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
This paper analyses the token-wise relevance of the prompt to the generated black-box optimization algorithm code and finds that the code-related content obtains a strong influence. The authors then propose a benchmark-guided algorithm code generation method, which uses a set of algorithm codes as examples and asks LLMs to refine the examples or generate new algorithms. Experimental results show that the proposed method achieves better performance than some LLM-based optimization code generation baselines.

### Strengths
1. This paper analyses the token-wise contributions in the prompts for optimization algorithm code generation, which could be helpful for futher prompt design.

2. The authors introduce more expert knowledge in black-box optimization to enhance the quality of the generated algorithms.

3. Experimental results validate the advantage of the proposed method on some of the existing methods.

### Weaknesses
1. Several related works on optimization code generation are not mentioned, such as LLaMoCo [1], which directly generates algorithm code using algorithm benchmarking during training, and LLMOPT [2], which generates operator code via prompting.

2. In the heatmap, the most relevant token appears to be ''numpy'' in the import statement. The LLM also assigns significant attention to unimportant tokens such as ''#'' and '')'', while the performance score of the code is not considered. This suggests the results may be biased, making the LLM-based code search appear almost random and unrelated to the target problem or algorithmic performance.

3. In the experiment, a comparison between the proposed method and the benchmark algorithms (e.g., CMA-ES, DE, PSO) is necessary to validate its effectiveness. Given the computational resources consumed for LLM inference and algorithm evaluation, the generated algorithm is expected to outperform the examples.

[1] Ma, Zeyuan, et al. "Llamoco: Instruction tuning of large language models for optimization code generation." arXiv preprint arXiv:2403.01131 (2024).

[2] Huang, Yuxiao, et al. "Autonomous multi-objective optimization using large language model." IEEE Transactions on Evolutionary Computation (2025).

### Questions
1. In the Initialization of the method, the authors use a selected "promising" example code. How is this preferred algorithm chosen? How can users determine which algorithm is "promising" for a target problem without prior knowledge?

2. In Algorithm 1, the method introduce a parameter, frequency factor q. However, the authors do not specify its value or the methodology for setting it.

3. The codes generated by LLMs may not always be correct and may contain syntax errors, runtime errors, or other logical issues. How are these error codes handled in your framework? Does their failure feedback influence the scoring mechanism or inform the LLMs during training?

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
This paper presents BAG, a framework for automated algorithm design using LLMs in the setting of black-box optimization. The authors first conduct a token-level attribution study using AttnLRP to identify which parts of prompts most influence algorithm generation, concluding that code examples dominate the contribution. Motivated by this finding, they propose to embed prior benchmark algorithms from the PBO and BBOB suites into the prompts to guide LLMs toward promising search regions. Experiments on 47×5 benchmark instances using three LLMs (Gemini, GPT, and Qwen) show that BAG achieves higher AUC performance than existing LLM-driven optimization frameworks such as EoH, LLaMEA, and ReEvo. The paper claims that integrating benchmark knowledge improves the efficiency, robustness, and interpretability of LLM-based algorithm design.

### Strengths
1. The paper conducts large-scale evaluation across two widely accepted benchmark suites (PBO, BBOB) and three different LLMs, ensuring reproducibility and empirical rigor. 
2. Incorporating AttnLRP to analyze prompt token relevance is a useful attempt to make LLM-driven algorithm generation more interpretable, identifying which parts of a prompt truly influence generated code.
3. The paper is well-organized and clearly written, with figures and tables that effectively communicate experimental results. The released code repository also enhances transparency.

### Weaknesses
1. The proposed BAG framework primarily reuses existing benchmark algorithms as prompt examples. While this improves performance, it effectively narrows the search to regions of known good solutions rather than discovering new algorithms. Thus, the method relies heavily on prior knowledge injection rather than genuine algorithmic innovation, which weakens the originality of the contribution.
2. The paper frames BAG as a mechanism for discovering new optimization algorithms, but since benchmark algorithms are directly embedded into prompts, the system is refining known strategies rather than inventing novel ones. The contribution fits better under knowledge-guided prompt design than under automated algorithm design.
3. The AttnLRP analysis merely confirms an intuitive result—that code tokens dominate influence in code generation. This offers limited new understanding of LLM behavior beyond what prior intuition already suggested.
4. No ablation is provided for the frequency factor q, the size of the benchmark set, or the quality of the prior examples. It remains unclear whether BAG still performs well if benchmark algorithms are noisy or suboptimal. Without this, claims about robustness are not substantiated.
5. The core idea—guiding LLMs via benchmark priors—feels incremental relative to recent literature (e.g., EoH, ReEvo, LLaMEA). The results, while positive, are expected given the strong prior information injected into the prompts."

### Questions
1. How does BAG perform if benchmark algorithms are replaced with suboptimal or randomly perturbed versions?
2. Can BAG generate genuinely new algorithmic structures beyond minor code variations of the provided benchmarks?
3. Why not formalize BAG as a knowledge-transfer or fine-tuning paradigm instead of positioning it as algorithm discovery?
4. Is there evidence that BAG generalizes beyond PBO/BBOB, or does it overfit to those benchmarks?"

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a framework named the "Benchmark-assisted Guided evolutionary Approach" (BAG) for automated algorithm design in black-box optimization  using Large Language Models (LLMs). The core idea is to leverage a pre-existing set of high-performance benchmark algorithms to guide the evolutionary process of the LLM. The authors first use attribution analysis (AttnLRP) to argue for the importance of code examples in the generated output of LLMs. They then validate the BAG framework on the pbo and bbob BBO benchmark suites, claiming superior performance over several other LLM-based methods like EoH and ReEvo. The paper's main conclusion is that integrating domain knowledge from benchmarks can effectively enhance the performance of LLM-driven algorithm design.

### Strengths
The paper provides experimental evidence for the common intuition that code examples are crucial in prompt engineering by quantitatively analyzing prompt components using AttnLRP.

### Weaknesses
1. The core idea of this paper has significant overlap with a recent series of LLM-driven evolutionary algorithms, particularly EoH and ReEvo. These prior works have already established the basic paradigm of using an LLM as an evolutionary operator (e.g., for mutation or crossover) to iteratively generate and improve algorithmic code. The BAG framework is essentially a minor adjustment to this paradigm, introducing a strategy to "sample from a benchmark pool and improve." This is not a fundamental innovation but rather a simple heuristic sampling strategy added to an existing framework. The "guidance" effect claimed by the paper has already been embodied in EoH's "evolution of thoughts" and ReEvo's "reflective evolution"; BAG merely replaces the guidance source from LLM-generated "thoughts" to external "high-performing algorithms."
2. This is arguably the most critical flaw of the study. The experimental evaluation is entirely confined to comparisons with other LLM-driven methods, ignoring the highly optimized non-LLM methods developed in the field of Automated Algorithm Design (AAD). To substantiate claims of superiority, the method could be benchmarked against established, powerful domain-specific baselines, such as advanced automated algorithm evolution frameworks (e.g., OpenEvolve) or efficient surrogate-based Bayesian Optimization methods. 
3. The empirical evaluation is restricted to the pbo and bbob benchmark suites. While these are classic benchmarks in BBO, they primarily represent unconstrained numerical optimization problems. The scope of AAD is far broader, encompassing more diverse and complex domains such as combinatorial optimization (e.g., TSP, scheduling) and constrained optimization. Confining the evaluation to two similar testbeds significantly limits the generalizability of the conclusions. The study fails to demonstrate whether BAG remains effective in more diverse and challenging problem domains.

### Questions
1. Your experimental evaluation only includes other LLM-driven methods. Why did you not compare BAG against established, state-of-the-art non-LLM frameworks from the AAD field (such as OpenEvolve) or powerful general-purpose BBO solvers (like Bayesian Optimization)?
2. The core contribution of the paper appears to be the demonstration that "using high-quality algorithms as examples can better guide an LLM." This conclusion is not surprising and could even be considered intuitive. Given the high similarity of the BAG's evolutionary loop to that of EoH and ReEvo, could you precisely articulate what the fundamental methodological differences are between BAG and these prior works, beyond the strategy of "selecting parents from an external benchmark pool"?
3. The success of the BAG framework seems highly dependent on a readily available, high-quality set of benchmark algorithms (Abench). For emerging problem domains that lack a mature benchmark community and a set of recognized high-performing algorithms, would the BAG method fail entirely? To what extent are the scalability and generalizability of this method limited by its strong reliance on prior knowledge?

### Soundness
2

### Presentation
2

### Contribution
2
