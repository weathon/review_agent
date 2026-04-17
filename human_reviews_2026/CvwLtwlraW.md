# OptiMind: Teaching LLMs to Think Like Optimization Experts

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Mathematical programming -- the task of expressing operations and decision-making problems in precise mathematical language -- is fundamental across domains, yet remains a skill-intensive process requiring operations research expertise. Recent advances in large language models for complex reasoning have spurred interest in automating this task, translating natural language into executable optimization models. Current approaches, however, achieve limited accuracy, hindered by scarce and noisy training data without leveraging domain knowledge. In this work, we systematically integrate optimization expertise to improve formulation accuracy for mixed-integer linear programming, a key family of mathematical programs. Our OptiMind framework leverages semi-automated, class-based error analysis to guide both training and inference, explicitly preventing common mistakes within each optimization class. Our resulting fine-tuned LLM significantly improves formulation accuracy by 21.4\% across multiple optimization benchmarks, with consistent gains under test-time scaling methods such as self-consistency and multi-turn feedback, enabling further progress toward robust LLM-assisted optimization formulation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper propose a training and inference pipline, specifically,

* **Training Data Cleaning**: A semi-automated pipeline to clean and enhance training datasets (OR-Instruct and OptMATH) by correcting errors, filling missing data, and regenerating high-quality solutions using LLMs and majority voting.

* **Multi-Turn Inference Pipeline**: An iterative inference process that incorporates:

    * Class-specific error hints.

    * Self-consistency via majority voting.

    * Solver feedback for iterative self-correction.

### Strengths
1. Comprehensive, End-to-End Methodology
The paper's major strength is its holistic approach, systematically improving the entire pipeline instead of just one component. It rigorously cleans both training and test data to fix widespread errors, then uses the insights from this cleaning (common error patterns) to power a sophisticated inference process. This creates a synergistic system where high-quality data and domain-informed inference work together for maximum effect.

2. Innovative and Transferable Use of Domain Knowledge
The "class-based error analysis" is a highly effective innovation. Instead of just using more data, the authors distill expert knowledge into specific "error-hint" pairs for different problem types (e.g., TSP, network flow). Injecting these targeted hints into prompts significantly boosts accuracy across various models, proving this method is a powerful and transferable blueprint for embedding expert knowledge into LLMs.

### Weaknesses
1.  **Limited Conceptual Novelty:** The paper's core components are incremental combinations of existing techniques rather than fundamentally new inventions. The approach of cleaning training data, using error-specific hints, and employing multi-turn inference with solver feedback is a logical and powerful integration of established ideas (like chain-of-thought, self-consistency, and tool-use) applied to a new domain. However, it does not introduce a novel algorithmic breakthrough, instead demonstrating that a meticulous, engineering-heavy application of current paradigms yields significant gains.

2.  **Significant Manual Effort and Scalability Concerns:** The framework relies heavily on manual, expert-led processes that do not scale easily. The class-based error analysis required experts to manually identify common mistakes, and the benchmark cleaning took "over a month of manual effort from a team of optimization experts." This makes the method difficult to adapt to new or broader problem domains (like nonlinear optimization) without a similar investment of expensive expert time, limiting its general applicability.

3.  **Narrow Problem Scope:** The work is exclusively focused on Mixed-Integer Linear Programs (MILPs), explicitly omitting more complex problem types like nonlinear or second-order cone programs.

### Questions
same as the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the OptiMind method to improve the accuracy of LLMs in modeling optimization problems.

### Strengths
`Hint` is a good idea.

### Weaknesses
- The novelty of this paper is questionable. Similar work already exists, such as LLMOPT, ORLM and SIRL, many of which utilize intermediate inference processes and mathematical modeling to guide the gradual generation of correct solution code. Furthermore, related works have investigated components like self-correction and self-debug, similar to the self-correction proposed in this paper. Therefore, the novelty of the "multi-turn inference pipeline" is questionable.
- The proposed "domain-informed error analysis framework" relies too heavily on manual annotation by domain experts, involving a large amount of manual annotation in both error summary and training data cleaning. This will significantly limit the application of OptiMind in general problem domains.
- Too many works have claimed to have cleaned the evaluation set and provided multiple different versions of ground truth for the same evaluation set, then claimed to have achieved state-of-the-art (SOTA) results on the cleaned evaluation task. I believe this is highly inappropriate. Authors should respect the contributions of related works. If problems do exist, they should provide detailed analysis and explanations of which instances in these benchmarks have what kind of problems, rather than arbitrarily proposing new ground truths, or even allowing multiple objective function values to be considered correct for a single instance. This is not a trivial matter. For example, according to my research, the OptMATH test set has 166 samples, but this paper only uses 128 of them. Where are the remaining problems? What modifications were made to these 128 samples? As far as I know, OptMATH generates problem descriptions from mathematical models, making the possibility of errors extremely low. In the results in Table 1, the reproduction results of the OptMATH dataset differ from other related works. I believe all of this is highly unrigorous.
- Unable to find the code to reproduce the article. Only the test set is included in the attachments. The reproducibility of this article is questionable.

### Questions
As described in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors propose a pipeline to improve the accuracy of formulating optimization problems by large language models (LLMs). The pipeline consists of (1) cleaning the training and testing data, (2) supervised fine-tuning, and (2) multi-turn inference with error-aware prompting and majority voting.

The supervised fine-tuning is standard. Hence, The main contributions are in the data cleaning and the inference, and can be summarized as follows.
* Cleaning of the three most challenging datasets (IndustryOR, MAMO Complex, and OptMATH).
* Summary of common errors made by LLMs in formulating optimization problems.
* Incorporating the error analysis above in the prompts during inference.

Overall, I appreciate the authors' efforts in having operations research (OR) experts clean the datasets and analyze the errors commonly made by LLMs. I think these efforts are of great value to the community in creating more credible datasets and benchmarks.

However, I personally think the technical depth may not be enough for a conference like ICLR. In addition, the experimental results do not strongly support some of the claims by the authors. These limitations result in my current rating.

BTW: I also think this paper might be more appropriate for the dataset and benchmark track, because the contributions in improving current datasets and benchmarks are important, but the technical novelty seems limited.

### Strengths
+ The authors have the three most challenging datasets cleaned by OR experts (PhDs and researchers, according to the authors). The OR experts also analyze the common errors made by LLMs for each type of problems. These efforts should be appreciated.
+ The authors submitted the cleaned test datasets in the supplementary materials.

### Weaknesses
- I would like to see more detailed breakdown and statistics of the errors in the three datasets of IndustryOR, MAMO Complex, and OptMATH. The authors have some vague claims that "30%-60% test instances being incorrect" `[Lines 187-188]`. However, in the OptMATH paper `[R1]`, the authors claimed that "manual analysis confirming 99.6% equivalence accuracy across all triplets". Which claim is true? Since the paper builds on the premise of high error rates of existing benchmarks, it is important to make this evidence clear.
- While cleaning datasets is important, it seems to be mostly manual. So there is limited technical novelty in cleaning datasets, analyzing errors, incorporating them into prompts, SFT, and multi-turn inference.
- The experimental results do not really support the claims by the authors.
  - The accuracy of the proposal methods is worse than GPT (`Figure 5 and Table 1`), which begs the question of how valuable the fine-tuned LLM is. Note that the comparison is done under the condition of *no* majority voting ($K=1$) for GPT models and *having* majority voting ($K=8$) for the proposed model.
  - Ablation results do not show big improvements from SFT (`Table 2 in Appendix A.1`).

Minor comments:
- In the analysis of error types in `Appendix A.10`, I think "infeasible problems" and "out-of-scope problems" (i.e., nonlinear optimization) should not be counted as errors. A problem *can* be infeasible, and the LLM should correctly formulate it and return the solution as "infeasible". Nonlinearity is also not an error in problem description.
- The authors limit the scope to mixed integer linear programs (MILP). Why is the limitation? I thought the methods could be generalized to nonlinear problems, which are more challenging and arguably more valuable.
- The author mentioned that OptMATH uses DeepSeek-V3 in `Section 4.3`, which is not true. They just used Llama.

[R1] Lu H, Xie Z, Wu Y, Ren C, Chen Y, Wen Z. "OptMATH: A Scalable Bidirectional Data Synthesis Framework for Optimization Modeling," *ICML 2025*.

### Questions
Please refer to the points raised in "Weaknesses".

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes OptiMind, a framework for improving large language models' ability to formulate mixed-integer linear programming (MILP) problems from natural language descriptions. The approach combines two main components: (1) systematic training data cleaning through class-based error analysis and automated data quality improvement, and (2) multi-turn inference with domain-informed prompting using class-specific hints and solver feedback. The authors evaluate their approach on three manually cleaned benchmarks (IndustryOR, Mamo-Complex, OptMATH), demonstrating an average improvement of 14 percentage points in formulation accuracy over base models.

### Strengths
- The extensive effort to clean both training and test data addresses a critical issue in the field. The authors' finding that 30-60% of benchmark instances contained errors is significant and their cleaned datasets will benefit the community.
- The class-based error analysis provides a structured approach to incorporating optimization expertise into LLM training and inference, moving beyond generic prompting strategies.
- The evaluation spans multiple challenging benchmarks with careful ablation studies, demonstrating consistent improvements across different base models.

### Weaknesses
- The paper primarily combines existing well-established techniques (SFT, self-consistency, multi-turn refinement), and while it achieves good results in domain-specific applications, the contribution in terms of algorithmic innovation or theoretical insights is relatively limited. This combinatorial improvement, though practical, may not constitute a significant methodological breakthrough.

- While the paper presents encouraging success cases, the systematic analysis of failure modes or situations where the method performs poorly is relatively limited. Additional discussion in this area would help readers gain a more comprehensive understanding of the method's applicable boundaries and potential limitations.

- The multi-turn inference approach requires generating multiple samples (K=8) and iterative corrections, which increases inference time and computational costs. While accuracy improvements are achieved, this computational overhead may affect the method's practical application value in time-sensitive or resource-constrained environments.

### Questions
- Given that the class-based error analysis relies on manual expert review, have you conducted inter-annotator agreement assessments? Understanding the consistency among different experts in identifying error patterns, and how such potential variations affect the method's reproducibility, would be valuable.

- How does the system handle problems that cannot be clearly categorized into the 50 predefined classes? For cross-category or boundary-ambiguous problems, what is the classification accuracy, and how does this uncertainty affect the final results?

- Could you provide more detailed analysis showing in what proportion of cases solver feedback actually helps improve solutions? Understanding the ratio of positive and negative impacts during the multi-turn correction process, as well as convergence characteristics, would help better understand the value of this component?

- How does your class-based prompting approach compare with other common prompt engineering techniques, such as few-shot examples or structured formulation like Five-Element Formulation?

### Soundness
2

### Presentation
2

### Contribution
2
