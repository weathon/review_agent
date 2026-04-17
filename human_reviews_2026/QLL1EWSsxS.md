# SSR: Socratic Self-Refine for Large Language Model Reasoning

- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
Large Language Models (LLMs) have demonstrated remarkable reasoning abilities, yet existing test-time frameworks often rely on coarse self-verification and self-correction, limiting their effectiveness on complex tasks. In this paper, we propose **S**ocratic **S**elf-**R**efine (**SSR**), a novel framework for fine-grained evaluation and precise refinement of LLM reasoning. Our proposed SSR decomposes model responses into verifiable (sub-question, sub-answer) pairs, enabling step-level confidence estimation through controlled re-solving and self-consistency checks. By pinpointing unreliable steps and iteratively refining them, SSR produces more accurate and interpretable reasoning chains. Empirical results across five reasoning benchmarks and three LLMs show that SSR consistently outperforms state-of-the-art iterative self-refinement baselines. Beyond performance gains, SSR provides a principled black-box approach for evaluating and understanding the internal reasoning processes of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose Socratic Self-Refine (SSR), a test-time reasoning framework that enhances the reliability and controllability of LLMs' CoT reasoning.
SSR decomposes a model’s reasoning trace into Socratic-style question-answer steps, estimates step-level confidence via multiple re-solvings, and selectively refines the weakest step through targeted feedback prompts. 
Three variants (SSR-Lin, SSR-Ada, SSR-Plan) are presented to balance reasoning completeness and computational cost. 
Experiments on mathematical and logical reasoning benchmarks show consistent accuracy improvements and better interpretability over baselines such as Self-Refine and CoT.

### Strengths
1. Clear conceptual novelty: LLM reasoning is formalized as step-level Socratic process.
2. Well-grounded probabilistic formulation.
3. Improved interpretability through explicit step-level confidence estimation.

### Weaknesses
1. Reliance on prompt-based decomposition introduces noise and instability (Q1).
2. Step-level confidence estimation remains noisy and less discriminative than holistic evaluators (LLM-as-a-Judge) (Q2-Q3).
3. Step-level sampling ($M\times T$ steps) leads to expensive inference (Q4-Q6).
4. Evaluation limited to structured mathematical reasoning, leaving generalization uncertain (Q1).
5. Some minor writing problems.

### Questions
1. Given that SSR relies on prompting to decompose the reasoning traces into sub-questions, while the prompts corresponding to the same meaning may have multiple variations, how sensitive is SSR to such variations? Will subtle change of phrases in prompting lead to performance fluctuation, especially for queries other than structured mathematical reasoning?
2. The condition for the `Self-Refine Gating` is represented as $C^{(K)}=C_{max}$ in line 4 of algorithm 1, but changed to $c^{(K)}<c_{max}$ in Block 2 and $c=c_{max}$ in the caption of Figure 2. Which one of them is correct?
3. Line 803 indicates that `if Self-Refine fails or is overconfident`, SSR will be activated, while Line 4 in Algorithm 1 indicates that SSR will be activated only when $C^{(K)}=C_{max}$ (overconfident? not sure). How to determine whether `Self-Refine fails` anyway?
4. When deploying SSR-Ada, how can correct reasoning trajectories avoid excessive verification? It appears that all reasoning must undergo SSR's verification and refinement. If that is the case, what is the necessity of an additional `Self-Refine` step? Moreover, in the `SSR-Ada` method, self-refine seems to be performed only once. Does SSR then proceed regardless of whether the confidence score $C^{(K)}$ meets the condition $C_{max}$?
5. During the `SSR Decompose stage` (Block 4 and Line 5 in Algorithm 1), SSR incurs significant cost by generating a response for each Socratic step. When the process moves to `Socratic Verification` (Block 5, Line 10), the sub-question with the lowest confidence score will be inevitably identified. However, if the least confident step occurs at an early step (e.g., at the $t$-th sub-question), the verification will restart from the $t$-th point (Line 12 in Algorithm 1), rendering all subsequent generations after step $t$ redundant. Is it possible to avoid such computational waste?
6. The inference time is correlated with the number of `Socratic steps` and verification iterations. For instance, since the baseline method `Self-Refine` is incorporated into both `SSR-Ada` and `SSR-Plan`, the computational cost of `SSR-Ada` and `SSR-Plan` is inherently higher. Could the authors provide a relevant analysis of the time cost, such as under best-case, worst-case, or average scenarios?
7. There are some writing and formatting errors in the paper, for example:
   - What does the subscript $ti$ in Equation (7) represent? 
   - Missing punctuation between "constrain LLM outputs" and "We therefore resort to" in Line 235.

### Soundness
2

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
This paper proposed SSR, a test-time method that decomposes a model’s CoT into (sub-question, sub-answer) steps, estimates step-level confidence by re-solving with self-consistency, and iteratively fixes the lowest-confidence step to improve the reasoning.

### Strengths
1. Consistent gains across five reasoning tasks and multiple backbones
2. the paper is overall well-written

### Weaknesses
1. the fine-grained verification increases compute cost and limits scalability to long chains or large datasets
2. step-level decomposition depends on prompting and can be noisy or inconsistent, especially for ambiguous or ill-posed sub-questions
3. the planning component assumes independence between planning and execution and uses only a single plan check, which may miss plan-level errors

### Questions
N/A

### Soundness
2

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
They propose a method to correct the errors in reasoning process of LLMs, aiming to overcome the limitations of existing test-time frameworks that rely on coarse self verification and self correction. To do this, they break the reasoning into sequence of verifiable Socratic steps (sub questions and sub answers) then validate the correctness of each step, and if it’s wrong, fix it.
Variants of the framework, such as SSR-Ada and SSR-Plan, were introduced to balance efficiency.

### Strengths
No need for human annotation
No need for fine tuning or training
breaking the reasoning into small steps and processing those small chunks instead of long paragraphs makes the method more accurate

### Weaknesses
The method section is unnecessarily complex, which makes it hard to understand, while they could have skipped some of the unnecessary mathematical notations and instead describe verbally 
Lack of definition of some required concepts: definition of Self-Refine method is missing in the paper (especially in the methods section)
Execution time overhead is not mentioned.
Considerable increase in execution time: first needs to go through the whole reasoning stage, then refine it. If one of the intermediate steps gets changed, all proceeding steps should be changed as well because they depend on the changed step.

### Questions
How do you break the reasoning text into steps? because LLMs do not guarantee the format of generation and if you expect them to generate their reasoning part in a specific format to be able to automatically break them into sub steps.

In the paper you claim that: "Reasoning as Socratic Process. In this paper, we posit that the reasoning process is implicitly modeled as a sequence of goal-setting and problem-solving steps". But how do you make this assumption? do you have a reference/evidence for this?

In the paper you mention that: "we encode all relevant information into the context and ask the LLM to solve each sub-question independently M times.". what is the execution time overhead of this operation?

"We therefore resort to LLM self-evaluation, producing confidence scores directly with a context-free confidence estimation prompt". But LLMs are not reliable on judgments yet. How do you make sure that they are reliable. Do you have any measurements or studies on their error ratio?

In Formula 10, $A_{t'}$ is not defined. 

I suppose that you mostly suggest to use SSR-Ada because it's more efficient than SSR-Plan. Now, looking at Table 1, when using GPT-4.1-nano, a method from the section with white background beats your method in most of the cases. But when using GPT-5-mini, your method beats other methods. Do you know why?

Definitions of metrics used in the paper are missing. 

Can you think of a better metric to assess quality of reasoning in LLMs? The ones you use only leverage the final answer, which is not a god representative for the quality of the reasoning part.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
SSR (Socratic Self-Refine) is an inference-time framework that turns a model’s chain-of-thought into a sequence of Socratic, i.e. steps (sub-question, sub-answer) pairs, so each step can be re-solved multiple times to estimate a step-level confidence via self-consistency. The lowest-confidence step is targeted for refinement (majority vote over its re-solutions) and the full reasoning is revised with this ‘Socratic feedback’. Three variants are evaluated: SSR-Lin (always step-level), SSR-Ada (gate: use standard Self-Refine unless it fails, then apply SSR), and SSR-Plan (one round of plan evaluation/refinement before step-level SSR). On math and logic with two backbones (GPT-4.1-nano, GPT-5-mini), SSR consistently beats baselines. Context-management ablations show that keeping the natural reasoning trace and applying reflection (rather than early intervention) works best. Scaling studies (more iterations, more parallel samples) indicate SSR keeps improving where baselines plateau. Limitations are positioned around some descriptive and critical scoping aspects.

### Strengths
- Well balanced scientific discourse and method, covering both clarity/rigour.
- Sensible experimental design and empirical analysis.
- Timely topic and good positioning wrt novelty.

### Weaknesses
- No proper systematic qualitative analysis reflected on the methodology. D.4. provides some example outputs.

- Lack of a more descriptive and formal description on the eligibility criteria (inclusion and exclusion) for the baselines, base LLMs and datasets.

### Questions
- How consistent are the extracted Socratic steps across prompts, seeds, and backbones? Do humans agree on step boundaries/goals?

- How well do step-level confidence scores correlate with ground truth? What happens when algebraic equivalence isn’t textual (e.g., simplifications)?

- Error analysis: When does plan-level evaluation help or hurt? Any failure cases where plan edits derail otherwise correct execution?

- Can you reflect on how the approach works (qualitative analysis)? What are the challenging cases? How does it vary across different task domains?.

- Lack of a more descriptive and formal description on the eligibility criteria (inclusion and exclusion) for the baselines, base LLMs and datasets.

- Lack of a description of the underlying assumptions behind the tasks - these should not be seen as proxies for ‘reasoning’ and should be qualified.

### Soundness
3

### Presentation
3

### Contribution
3
