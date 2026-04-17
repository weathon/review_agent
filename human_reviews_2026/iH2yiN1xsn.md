# Think Twice: Branch-and-Rethink Reasoning Reward Model

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Large language models (LLMs) increasingly rely on thinking models that externalize intermediate steps and allocate extra test-time compute, with think-twice strategies showing that a deliberate second pass can elicit stronger reasoning. In contrast, most reward models (RMs) still compress many quality dimensions into a single scalar in one shot, a design that induces judgment diffusion: attention spreads across evaluation criteria, yielding diluted focus and shallow analysis. We introduce branch-and-rethink (BR-RM), a two-turn RM that transfers the think-twice principle to reward modeling. Turn 1 performs adaptive branching, selecting a small set of instance-critical dimensions (such as factuality and safety) and sketching concise, evidence-seeking hypotheses. Turn 2 executes branch-conditioned rethinking, a targeted reread that tests those hypotheses and scrutinizes only what matters most. We train with GRPO-style reinforcement learning over structured two-turn traces using a simple binary outcome reward with strict format checks, making the approach compatible with standard RLHF pipelines. By converting all-at-once scoring into focused, second-look reasoning, BR-RM reduces judgment diffusion and improves sensitivity to subtle yet consequential errors while remaining practical and scalable.  Experimental results demonstrate that our model achieves state-of-the-art performance on three challenging reward modeling benchmarks across diverse domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes BR-RM, a two turn reward model that first performs adaptive branching to select a small set of instance critical evaluation dimensions and sketch hypotheses, then performs a branch conditioned rethinking pass to verify those hypotheses and decide a preference. Training uses GRPO with a strict format check and a binary outcome reward. On RewardBench, RM Bench, and RMB, BR-RM reports strong average accuracy, with particularly large gains on RM Bench, and presents ablations on turning off the second pass, removing branching, and changing reward design.

### Strengths
*  The paper identifies judgment diffusion in reward models and motivates a focused second pass that aims to allocate test time compute where risk is highest. The concept and naming are crisp and intuitive. 
* The strict formatting penalty plus binary outcome reward is easy to implement and aligns with the evaluation objective. The paper also shows why finer grained scoring or extra branch rewards underperform. 
* BR-RM-Qwen-14B achieves 92.1 on RewardBench, 85.9 on RM Bench, and 74.7 on RMB, producing the best average among compared methods. The 8B model is competitive as well. 
* Removing the second pass or the adaptive focus yields consistent drops, supporting the core design claim that thinking twice with focus matters.  
*  The training data ablation clarifies contributions from HelpSteer, safety data, math, and code preferences.

### Weaknesses
* The paper highlights best averages, but several baselines appear very recent and some cells are missing. It would help to provide complete, reproducible comparison tables and lock evaluations with identical prompting and sampling across all methods. The current Table 1 summary is helpful but not fully auditable from the text alone. 
* The format penalty is large in magnitude, and the same terminal reward is assigned to both turns uniformly across tokens. This could incentivize shortest valid traces rather than best targeted analysis. A token or section level credit assignment analysis would strengthen the case.  
* The nine dimension space is only sketched. The paper would benefit from concrete definitions, coverage analysis, and failure cases where the right dimension is off the list. 
*  The method doubles passes by design. Training steps, batch sizes, number of traces per item, and inference budgets are not fully spelled out for a fair cost adjusted comparison versus strong one turn GenRMs or scalar RMs. 
* The reward design is explicitly matched to binary preference accuracy. Generalization to settings that require calibrated magnitudes or multi way choices is not evaluated, and the “scoring on a scale” negative result is explained post hoc. A held out task with different decision granularity would build confidence. 
* The training data ablation shows safety data matters, but the paper does not include targeted red teaming or bias analysis of the judge decisions under adversarial style.

### Questions
1.  How many branches are typically chosen per item. What is the token budget split between turn one and turn two. Please report distributions and correlate them with accuracy. 
2.  You assign the terminal reward uniformly to all tokens of both turns. Did you try turn specific weights or variance reduction by per section advantages. Any signs of mode collapse to minimal valid traces. 
3.  What exactly are the nine dimensions and how were they defined. Do any items leak benchmark rubric wording into the branch names. Evidence that the model can discover off rubric issues would be valuable. 
4.  Please provide per example wall clock for training and inference, and normalized accuracy per thousand generated tokens. This is key for practical deployment against larger scalar or generative baselines. 
5. Can BR-RM be extended to listwise judging or magnitude scoring without losing the benefits of the two turn design. The scale based attempt failed for alignment reasons, but could a pairwise consistent magnitude be learned.

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
3

### Summary
This paper implements a multi-turn mechanism for reasoning RMs. By forcing decouple of rubrics selection and rethinking in two rounds, the proposed RM exibits strong performance on various reward model benchmarks.

### Strengths
1. The observation of focus dilution and shallow analysis are sound.
2. The benchmark performances are strong.

### Weaknesses
1. Lack of insights. The paper lacks in-depth analysis and the ablations are not informative (only benchmark scores).
2. Too much inductive bias. Many important design choices are manually picked without much validation.

### Questions
1. Weakness 1. Why do focus dilution or shallow analysis happen? Why multi-round query could mitigate this? Is the mitigation due to specific prompt design (e.g. round 1 reduce the amount of rubrics to take into account), or due to the forced "rethinking" by multi-round query?
2. Weakness 2.
- The number of turns. If think in two turns performs better than a single turn, can increasing turns lead to even better performance?
- The design of subtasks. First turn selects a tiny subset of rubrics to consider, second turn analyze condition on them. An alternative is to 1) generate a weight given problem and rubric name, 2) independently generate a rationale and a score for each rubric, and 3) compute weighted score. The possibilities of designs are endless. Why choose to design this way?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper diagnoses "judgment diffusion" in existing Reward Models (RMs), where attention is spread thinly across many quality dimensions, leading to shallow analysis. It introduces Branch-and-Rethink (BR-RM), a two-turn framework that first performs Adaptive Branching to select a few instance-critical dimensions and sketch hypotheses, followed by Branch-Conditioned Rethinking for a targeted, second-look analysis based on the initial findings . The model is trained with GRPO-style RL using a simple binary outcome reward and achieves state-of-the-art (SOTA) performance on three challenging RM benchmarks: RewardBench, RM-Bench, and RMB.

### Strengths
1. The paper's core idea is reasonable. Diagnosing "judgment diffusion" and transferring the "think-twice" principle from solvers (LLMs) to judges (RMs) is a clever and logical contribution.

2. The paper is supported by strong SOTA results across three diverse benchmarks and is validated by exceptionally comprehensive ablation studies that justify each design component.

3. The paper is well-written and clearly structured. The core problem and the proposed solution are easy to understand.

4. This work addresses a critical bottleneck in LLM alignment by creating RMs that are more sensitive to subtle yet consequential errors, which is essential for developing more reliable AI systems.

### Weaknesses
1. Cost: The BR-RM is a two-stage generative model. Compared to a scalar RM, which requires a single forward pass, this approach introduces substantial latency and complexity, especially during RLHF training where the RM is called millions of times. The paper doesn't quantify this two-turn cost, making its practical viability for large-scale application questionable.

2.  The method relies heavily on a predefined "universal set of criteria" and "task-specific evaluation hierarchies". The performance seems contingent on these human-designed sets, making the approach potentially brittle. If a critical evaluation dimension for a new task is missing from the universal set of criteria, will the model be unable to "branch" to it and fail?

3. The paper attempts to differentiate BR-RM from existing Reasoning RMs, but the distinction feels minor. The core contribution appears to be a strong engineering improvement and a clever prompting strategy rather than a fundamental conceptual leap over prior works.

### Questions
1. See weakness 1

2. See weakness 2

### Soundness
3

### Presentation
4

### Contribution
3
