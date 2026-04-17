# Nexus: Execution-Grounded Multi-Agent Test Oracle Synthesis

- Decision: Reject
- Scores: 4, 2, 2, 8

## Abstract
Test oracle generation in non-regression testing is a longstanding challenge in
software engineering, where the goal is to produce oracles that can accurately
determine whether a function under test (FUT) behaves as intended for a given
input. In this paper, we introduce Nexus, a novel multi-agent framework to address this challenge. Nexus generates test oracles by leveraging a diverse set of
specialized agents that synthesize test oracles through a structured process of deliberation, validation, and iterative self-refinement. During the deliberation phase,
a panel of four specialist agents, each embodying a distinct testing philosophy,
collaboratively critiques and refines an initial set of test oracles. Then, in the
validation phase, Nexus generates a plausible candidate implementation of the
FUT and executes the proposed oracles against it in a secure sandbox. For any
oracle that fails this execution-based check, Nexus activates an automated selfrefinement loop, using the specific runtime error to debug and correct the oracle before re-validation. Our extensive evaluation on seven diverse benchmarks
demonstrates that Nexus consistently and substantially outperforms state-of-theart baselines. For instance, Nexus improves the test-level oracle accuracy on the
LiveCodeBench from 46.30% to 57.73% for GPT-4.1-Mini. The improved accuracy also significantly enhances downstream tasks: the bug detection rate of GPT4.1-Mini generated test oracles on HumanEval increases from 90.91% to 95.45%
for Nexus compared to baselines, and the success rate of automated program repair improves from 35.23% to 69.32%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces Nexus, a multi-agent framework for automated test oracle generation. Nexus operates through three phases, namely, deliberation, validation, and iterative self-refinement. First, four specialized agents in the delibration phase collaboratively generate and critique candidate oracles, then the validation phase validates them by executing against an LLM-generated implementation in a secure environment. In the end, the self-refinement phase iteratively refines those oracles that fail. Nexus is evaluated on seven benchmarks and four LLMs (both open- and close-source), performing better in oracle accuracy, bug detection, and automated program repair success. For example, it improves the test-level accuracy on LiveCodeBench from 46.3% to 57.7% and doubling repair success rates.

### Strengths
- Multi-agent approach and splitting the problem into multiple sub-problems
- Effectiveness compared to existing approaches

### Weaknesses
- The validation phase relies on LLM-generated plausible implementation. Why someone should trust on this implementation as ground-truth?
- All benchmarks are crafted and the size of programs are very small. For example, HumanEval.
- This architecture can be very expensive in reality, both in terms of dollar and runtime cost. No discussions about this in the paper.

### Questions
- Why the LLM-generated implementation should be trusted in the validation phase?
- What is the cost of Nexus? What is the token usage, total time in seconds of each component, dollar amount?
- Why the authors did not evaluate on real-life projects, with large codebases?

### Soundness
3

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
The paper presents Nexus, a multi-agent framework for automatically generating test oracles for individual functions. The system operates in three stages: (1) multi-agent deliberation among specialized testers; (2) validation through sandboxed execution; and (3) self-refinement using runtime error feedback. Experiments show improvements on standard benchmarks—e.g., oracle accuracy on LiveCodeBench rising from 46.3% to 57.7% (GPT-4.1-Mini) and bug detection on HumanEval improving modestly.

### Strengths
- The multi-agent deliberation and self-refinement pipeline is well designed and demonstrates good engineering effort.
- Clear and consistent improvements across multiple benchmarks.
- The validation loop leveraging execution feedback is a reasonable and practical idea.

### Weaknesses
1. Fundamental Paradox: Solves Only the Easy Problems

Nexus performs well only on simple, well-specified functions where expected behavior is trivial. Ironically, these are exactly the cases where human developers can easily write tests themselves, while the harder, more ambiguous cases remain out of reach.

2. Severely Limited Applicability (Technical Scope)

The framework assumes isolated, stateless, pure functions with deterministic I/O. It cannot handle realistic challenges such as shared state, object interactions, asynchronous behavior, or dependencies on databases, APIs, or file systems. In real software, these assumptions rarely hold, making the current design applicable to only a small subset of cases.

3. Circular Reasoning and Validation Ambiguity

Both the oracle generation and validation phases rely on model-generated reasoning. If the candidate implementation and oracle share the same misunderstanding, the validation step can falsely confirm incorrect behavior. Without an external specification or reference, this becomes a self-reinforcing evaluation loop.

4. Questionable Real-World Scenarios (Practical Value)

Even if technically sound, it’s unclear when developers would realistically use Nexus.
- For new code, they already know the intended behavior and can write tests faster.
- For legacy code, there’s no reliable ground truth for Nexus to infer correct behavior.
- For integration or regression testing, the system’s single-function assumption breaks down.

Thus, while academically interesting, the method lacks a clear use case in everyday software development.

### Questions
N/A

### Soundness
3

### Presentation
3

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
This work presents a method (Nexus), which uses a multi-agent framework and execution feedback to generate accurate test oracles for functions specified in natural language. The framework first generates test oracles one-shot using an LLM, then prompts an LLM with 4 different focus topics to find possible flaws, and aggregates the findings using an LLM. Then an LLM generates a candidate solution to the given function, and the output of that function is compared to the the generated oracles. In a self-refinement loops, an LLM can now adjust the generated oracles based on the execution feedback of the candidate solution. The method is evaluated on 8 different single-function benchmarks and with Qwen 7B/32B and GPT-4.1-nano/mini and shows between 1 and 40% improvements over a zero-shot baseline.

### Strengths
I think the idea of using specialized agents to analyze method outputs is a good idea to improve upon CANDOR. The figures and tables are legible.

### Weaknesses
I think this paper has several critical weaknesses, listed below.


**Lack of evidence**
It is unclear if the method results in true gains as claimed, and if its performance gains really stem from the proposed setup.
- Table 6 only shows performance difference after a round of debugging information after direct generation/CANDOR/Nexus. Please also include the performance without debugging information as a true baseline so that it can be seen whether the oracles increase performance at all.
- The method only allows adapting the tentative oracles based on execution feedback of the candidate solution, but not to adapt the candidate solution based on the generated oracles. This seems to presume that the candidate solution is correct - which clearly should not be assumed for a general task, and especially not if this method were to be used in a TDD-like setting as implied by the results presented in Table 6. Moreover, this raises the question why the candidate solution is not used directly to obtain adequate oracles, without the various discussions before. Does the candidate generation use the generated test oracles? This should generally be clarified in the text (and prompt B.6, e.g. what does test_examples refer to?)
- After all, the execution feedback seems to have marginal effect on the quality of the obtained oracles (Table 5). Given all the negative side effects of this execution feedback setup, I would recommend to majorly overhaul it or to drop it entirely  for a subsequent revision.
- The authors do not provide the prompt for the direct generation baseline. The prompt for the tentative oracle baseline (App B.1) directly tells the model to "[b]e quick, but it may not be perfect". Why is this included? I would expect this to (artificially) decrease the quality of generated tentative oracles, thus making the effect of the discussion seem to have a strong positive effect. Is the same prompt used for direct generation and how would the results look if the same prompt for direct generation was used for the tentative oracle generator? 
- The ablation for different agents does just remove critics from the nexus setups (sec 5.2). I think it would be cleaner to replace them with a generically prompted agent, strengthening the claim that the improvements in Nexus stem from more specific/diverse prompts. Currently, it mainly shows that fewer critics/fewer inputs to aggregate over reduce the performance of nexus.
- Why exactly were the 4 agents chosen that nexus employs? Do they result from the preliminary study, or are they chosen based on a validation setting, or based on educated guesses by the authors on what might be useful? Were other critics tried?

**Lack of comparison to related work**
- The work only compares to a rudimentary baseline and only a single related work, and even there makes no strong case that exactly the proposed improvements result in the claimed superiority. The main comparison is to CANDOR, which is a single work that uses multi-agent debate, similarly to the presented work, but does not use different prompts to discuss tentative oracles. The authors claim they outperform CANDOR specifically due this difference (+ leveraging execution feedback) but don't make clear if their comparison implementation of CANDOR actually differs only in this regard (down to information flow, used prompts etc).
- Other baselines would help contextualize the results, e.g., comparison to any of the many methods outlined in Table 1 of [1] or any of the testing agents designed and evaluated for SWT-bench [2]
- In several places (e.g. Line 450) the authors highlight how prior work is all "single-paradigm" and "no-execution feedback". This is, from what I see, only based on a comparison to CANDOR. It is also not true, for example Otter [5] uses execution feedback and heterogenous prompts for test case generation on repository-level tasks.

**Lack of evaluation diversity (benchmarks, LLMs)**
- The method evaluates on only two different model families: GPT-4.1 (mini/nano) and DeepSeek-R1-Distill-Qwen (7B/32B), both reasoning trained LLMs. The evaluation lacks any other open-source and closed source models (Mistral, Llama, Claude, Gemini, to name a few from the top of my head), in particular SOTA models (e.g., R1, GPT-4.1) or any non-reasoning LLMs or LLMs trained specifically for coding and provides no justification to exclude them (e.g. they saturate the benchmark already)
- The method is only evaluated on single function benchmarks. This results in the method being much less relevant, as nowadays benchmarks on repository level [2,3] are already being adressed by more advanced systems.
- The execution feedback based on a candidate implementation is highly contingent on the capability of the model to generate a correct solution (especially as the solution can not be adapted anymore based on the oracle feedback! only the other way around). This strongly skews the results, especially as HumanEval and MBPP and similar benchmarks are known to be somewhat saturated by recent models [4], thus making it easier to use generated solutions as trustworthy execution feedback. Evaluating on other benchmarks where generating a correct candidate implementation is not as easily possible [2,3] would be able to resolve this concern.


**Further Nitpicks**

- The code submission is not documented. It should contain a clear documentation on how to set up an environment to execute the scripts (which python version, packages etc) and how to run the scripts to reproduce the results in the paper (including baselines and other works). In the current state, I would mark this research as "not reproducible"
- The authors could use \paragraph{...} to structure long sections, for example \paragraph{Related works lacks diversity} to start the paragraph in line 77, similarly line 86 and line 99 ("this work does ...", "evaluation shows ...")
- I could not find an explicit explanation of the difference between task-level and test-level scores (Table 2). Please explicitly describe what each metric measures in the evaluation metrics paragraph (e.g., a formula that unambiguously defines the metric)
- The writing could use additional passes to increase conciseness.


References  
[1] Molina et al, Test Oracle Automation in the Era of LLMs, 2025  
[2] Mündler et al, SWT-Bench: Testing and Validating Real-World Bug-Fixes with Code Agents, NeurIPS 2024  
[3] Jain et al, TestGenEval: A Real World Unit Test Generation and Test Completion Benchmark, ICLR 2025  
[4] Yu et al, HumanEval Pro and MBPP Pro: Evaluating Large Language Models on Self-invoking Code Generation, ACL 2025  
[5] Ahmed et al, Generating Tests from Issues to Validate SWE Patches, ICML 2025

### Questions
Please see the questions raised in the weaknesses section.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents Nexus, a novel multi-agent framework addressing the difficult challenge of test oracle generation: determining whether a function’s output matches its intended behavior. Nexus generates test oracles by leveraging multi-agent collaboration through a cycle of deliberation, validation, and self-refinement.
Nexus introduces several innovations:
- A multi-agent debate among specialized testing personas (Specification Expert, Edge Case Specialist, Functional Validator, Algorithmic Analyst), each offering a distinct point of view when assessing the quality of a candidate oracle.
- An execution-grounded feedback mechanism, where candidate oracles are validated through actual test execution within a sandbox.
- An iterative self-refinement process, in which failed oracles are automatically improved using LLM feedback.

The framework achieves consistent improvements across seven benchmarks on oracle accuracy; additionally, it helps improving on downstream tasks such as bug detection and automated program repair.

### Strengths
1. The paper is well written and easy to follow, with a good balance between qualitative examples and quantitative evidence. The problem is well motivated as well: Table 1 convincingly illustrates that while LLMs can already generate syntactically valid test inputs, they still struggle with semantic reasoning to produce correct outputs.
2. The experimental evaluation is comprehensive, involving multiple open and closed source LLMs. Results across several benchmarks are consistent and demonstrate substantial improvements.
3. This work shows not only improvement on test oracle generation, but also highlights how downstream tasks, such as bug detection and automated program repair, can benefit from this.
4. Novelty on top of prior work (e.g., CANDOR):
    - expanded deliberation phase with different testing personas
    - execution-based validation and iterative self-refinement, grounding reasoning in executable evidence.
5. Each component (deliberation, validation, self-refinement) is thoroughly analyzed, showing measurable contributions to the overall performance.

### Weaknesses
1. Nexus is more complicated than prior frameworks. While the performance improvements are clear, the paper lacks a detailed analysis of computational cost (e.g., API budget, runtime).
2. The validation phase (lines 213–216) relies on an LLM-generated "candidate implementation" of the function under test. If this implementation is incorrect, it could lead to false positives or negatives: more analysis is needed to better understand how the oracle would behave in such a case. 
3. The comparison focuses primarily on CANDOR, but Nexus is not evaluated on all benchmarks used by that baseline (e.g., HumanEvalJava, LeetCodeJava).
4. Incorporating additional baselines, such as those appearing in SWT-Bench or other relevant test-generation benchmarks, would improve the completeness and fairness of the experimental section.

### Questions
- Lines 216-217 mention that passing oracles are considered correct: more analysis on this would help understand how much this hypothesis is correct.
- It would be interesting to see more qualitative examples from the preliminary analysis in the appendix.
- Question related to weakness #3: are baselines missing for HumanEvalJava and LeetCodeJava because Nexus has been tested only on Python? What would be the generalizability of the framework?
- Question related to weakness #4: How is CANDOR established as the state-of-the-art baseline? It does not appear in public leaderboards such as LiveCodeBench or SWT-Bench. The paper might benefit from a comparison with other baselines.
- Some of the datasets are more academic (e.g., HumanEval, MBPP): in the past, models were achieving high scores on such benchmarks but very low ones when tested on more realistic ones (e.g., SWE-Bench). Have the authors considered evaluating Nexus on more real-world benchmarks (e.g., SWT-Bench, SWE-Bench)?
- Could the paper include qualitative success and failure cases of Nexus to illustrate its reasoning and limitations better?

### Soundness
3

### Presentation
4

### Contribution
3
