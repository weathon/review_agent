# Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, yet their ability to perform structured symbolic planning remains limited, particularly in domains requiring formal representations like Planning Domain Definition Language (PDDL). In this paper, we present a novel instruction tuning framework designed to enhance LLMs' symbolic planning capabilities through logical chain-of-thought reasoning. Our approach focuses on teaching models to rigorously reason about action applicability, state transitions, and plan validity using explicit logical inference steps. By developing instruction prompts that guide models through the precise logical reasoning required to determine when actions can be applied in a given state, we enable LLMs to self-correct their planning processes through structured reflection. The framework systematically builds verification skills by decomposing the planning process into explicit reasoning chains about precondition satisfaction, effect application, and invariant preservation. Experimental results on multiple planning domains show that our chain-of-thought reasoning based instruction-tuned models are significantly better at planning, achieving planning accuracy of up to 94% on standard benchmarks, representing a 66% absolute improvement over baseline models. This work bridges the gap between the general reasoning capabilities of LLMs and the logical precision required for automated planning, offering a promising direction for developing better AI planning systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose the PDDL-Instruct framework, a model finetuning approach, to improve Large Language Models abilities on sequential planning tasks. The finetuning is split into two phases. The first phase is an instruction tuning phase, where the model is trained over both correct and incorrect plans. The model output for each plan is an analysis of whether the given plan is valid and why it is or is not valid. The second phase ia a chain-of-though tuningand  phase. The model is prompted to generate a chain of planning tuples, which is checked for correctness by an external verification algorithm. The input to a model tuned by this framework is then a set of natural language instructions, a domain description in PDDL, and goal to be reached described in PDDL. The LLM output is a set of (State, Action, Next State) tuples that can be used to reach the goal.

### Strengths
1) The finetuning framework combines both existing strategies and new finetuning approaches
   2)  The writing is clear and the provided tables / figures are easy to understand
    3) Inclusion of error analysis and failure modes is useful for understanding how future frameworks can improve this framework and also for helping interperate the results in the main paper
   4)  Training on negatives is not a new concept, but it is not one I've seen applied in the context of LLM planning, and is an idea I feel is typically neglected. Clever of the athors to include such an approach.
   5)  Results show strong improvements over baseline models
   6) Moderate novelty and significance

### Weaknesses
1) The authors do not compare against any other planning frameworks. Even if the other frameworks were not made for this task and the the authors were sure that the other frameworks would fail, it is still useful to see the performance of other frameworks to put the results in context. Can these problems be easily solved by a general solver for AI planning problems?
2) This work has some similarity with the approach in this paper that also uses feedback: https://arxiv.org/abs/2309.16436 However, that paper does not use any fine-tuning.
3) I guess the overall excitement about the paper is rather limited as all the the applied techniques are well established. However, they may not have been applied together to this type of problem.
4) The authors include some figures and tables in the appendices. It would be beneficial if descriptions of the figures/tables and analysis of the results were included.
5) Minimal results were provides. See questions.

### Questions
1) Have the authors tried training on one dataset (Blocksworld) and then testing on another (Logistics)? How well does the model generalize to unseen data?
2) Have the authors trained on two of the datasets (Blocksworld, Logistics) and tested on the heldout dataset (Blocksworld Mystery)?

### Soundness
3

### Presentation
3

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
The paper proposes PDDL-INSTRUCT, an instruction-tuning framework to improve LLMs’ classical planning in PDDL domains. The central move is to turn plan generation into explicit state–action–state chains and supervise them with a formal validator (VAL) that checks preconditions, effects, and goal satisfaction. Training is staged: first general instruction-tuning with explanations (including negative examples), then a chain-of-thought phase with two objectives: a step-level “reasoning-chain” loss over ⟨si−1, ai, si⟩ triplets and a final plan-validity loss. On PlanBench-style tasks, reported accuracy gains over base models and a Phase-1-only ablation are large, with the strongest results when using detailed validator feedback and more feedback iterations.

The contribution is best viewed as an integrative advance rather than a new paradigm. Each major component has precedent namely: instruction tuning, chain-of-thought prompting, and external verification, but the paper’s value lies in packaging them into a coherent training loop that treats symbolic planning as verifiable, stepwise reasoning and shows strong empirical gains. The two-stage loss that first optimizes step-level reasoning chains and then end-task validity is a nice, concrete design choice; using detailed validator feedback to supervise CoT at the triplet level is what most clearly differentiates this from prior self-critique/reflection approaches.

### Strengths
1. Clear problem framing: LLMs often miss formal action applicability and state updates. The paper targets that gap directly.

2. Explicit verifiability: grounding CoT in VAL feedback forces faithfulness to domain dynamics and mitigates ungrounded story-like CoT. 

3. Detailed documentation: The appendix provides extensive implementation details, hyperparameters, prompts, and mathematical formulations, facilitating reproducibility.
4.  Empirical signal: consistent improvements across Blocksworld, Mystery Blocksworld, and Logistics; detailed vs binary feedback ablation is informative.

5. Clear idea & engineering pipeline. Combining structured CoT outputs (explicit ⟨s, a, s′⟩ steps) with an external symbolic verifier (VAL) is a natural and well-motivated approach to close the gap between natural-language reasoning and formal symbolic planning. The proposed two-stage optimization (reasoning loss then final task loss) is intuitively sensible.

### Weaknesses
1. Scope: restricted to classical PDDL without conditional effects, temporal/durative actions, or costs; focuses on satisficing rather than optimal planning.
2. Generality: results are on three domains; transfer to richer planning languages and real-world robotics pipelines is untested.
3. Data and compute loop: accuracy improves with more VAL-guided iterations (n), but the cost/benefit curve and stopping criteria are not fully characterized.
4. Comparison set: baselines focus on untuned or instruction-tuned models; a head-to-head against strong LLM-modulo planners or LLMs-as-modelers plus classical solvers (on identical tasks) would sharpen the contribution.
 5. Ablate the effect of modified VAL outputs in training (show effect when you never tamper with VAL).
6. Generalization claims need clearer definition. The paper uses “generalization” loosely; clarify whether this means cross-domain transfer (different domain file), larger problem instances, or semantic obfuscation (Mystery Blocksworld). Provide explicit transfer experiments (train on a set of domains, test on held-out domains).
7.  Test set contamination risk: The paper states "We remove the solution plans from datasets D₂ and D_test" but doesn't clarify if D₂ and D_test contain problems from the same domains with different configurations. If test problems are structurally similar to training problems within the same three domains, this raises concerns about memorization vs. generalization.

### Questions
1. You state you sometimes alter VAL outputs to create incorrect explanations for a few plans (Sec. 5.1). Why was this done? How many examples were altered and what safeguards ensure the model does not learn incorrect inference from corrupted feedback?

2. Can you show results for (a) more expressive PDDL features (conditional effects, derived predicates) and/or (b) an additional diverse suite of PlanBench domains? If not possible, explain limitations and expected failure modes.

3. How robust are the CoT traces? Provide examples of unfaithful CoT traces (CoT that looks plausible but the plan fails) and quantify how often your external verifier catches such unfaithful traces. Compare this to literature on CoT unfaithfulness.

4. In result section I see SD value is quite high, why paper have such a high SD value. I assume SD mean standard deviation, correct me If I am wrong. Is it due to high temperature setting of the LLM?

5. How does your approach differ fundamentally from existing work on iterative refinement with external verification (LEPA, STaR, code as symbolic[1], Cot-TL[2])? What is the specific technical contribution beyond applying these ideas to PDDL? Please clarify.

[1]Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation 

[2] CoT-TL: Low-Resource Temporal Knowledge Representation of Planning Instructions Using Chain-of-Thought Reasoning

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
This paper introduces PDDL-INSTRUCT, a new method for training LLMs to be much better at symbolic planning in PDDL. The main idea is to teach the model a logical chain-of-thought where it has to explicitly reason about why an action is valid (checking preconditions, applying effects) at every step of a plan. The training has two phases: a basic fine-tuning on PDDL problems and plans, followed by a more advanced phase where the LLM's chain-of-thought plan is checked by an external VAL verifier, and this feedback is used to tune the model even more. PDDL-INSTRUCT shows 94% accuracy on Blocksworld, a 66% jump over the baseline.

### Strengths
- The PDDL-INSTRUCT framework is a novel and well-structured approach to instruction tuning that directly targets an LLM's weakness in formal logical verification.
- The paper includes a detailed analysis comparing binary vs. detailed feedback and tests across multiple frontier/public models, which validates the design.

### Weaknesses
- It's a good trial, but the results are showing essentially LLMs can still only solve TC0 problems, aligning with the LLM state-tracking literatures. The selected tasks, e.g., BlocksWorld and Logistics, are pretty simple in the planning domain, and some neuro-symbolic methods can perform 100% on them.
- The paper's claim that this could be combined with frameworks like LLM-Modulo to reduce feedback loops is interesting, but where are the results? How is that claim supported?
- The paper is missing many literature references and discussion points from the neuro-symbolic and LLM-Modulo research areas, which are highly relevant.
- The framework's assumptions limit it to simple PDDL features, avoiding common but complex features like conditional effects or durative actions.

### Questions
- Could the author provide qualitative examples of failures after tuning? e.g., How does the simple blocksworld planning fail? Is it on long-horizon planning task?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose an instruction-following tuning framework: PDDL-Instruct to improve the symbolic planning capabilities of LLMs using chain-thought reasoning. The solution decomposes planning verification into atomic reasoning steps incorporating the resulting structure for instruction-tuning. PDDL-Instruct  integrates external verification feedback from a plan validator (i.e. VAL by Howey et al. (2004)) during training to guide and refine the model's planning and reasoning outputs. Experimental results on three different planning domains from PlanBench show that the tuning paradigm introduced by PDDL-Instruct results in more capable planning models, substantially outperforming naive instruction tuning baselines.

### Strengths
- The proposed framework results in significant improvements in the underlying LLMs' planning capabilities across different symbolic planning domains.
- The use of an external plan validator, VAL, is well-aligned with the challenges of the domain, avoiding over-reliance on LLMs' imperfect self-reflection.
- The paper responsibly acknowledges the limitations of the proposed framework, such as restricted PDDL feature coverage, and outlines promising avenues for improving self-verification capabilities and broader domain coverage.

### Weaknesses
- The paper primarily compares the proposed PDDL-Instruct framework against naive instruction tuning baselines, where the LLMs are fine-tuned without the logical CoT and verification feedback mechanisms. It would be useful to compare or at least relate the performance of PDDL-Instruct to other symbolic planners tested on the selected domains from PlanBench.
- While splitting training into initial and CoT instruction phases boosts the end performance, the paper would benefit from clearer explanation and explicit details regarding test set construction.
    - How are test problems selected and controlled for overlap or similarity with training domains and instances to ensure cross-domain generalisation evaluation?
- The paper would benefit from a more detailed analysis of employing the specialised loss functions used in the experiments against more conventional alternatives in this space (e.g., against the negative loglikelihood instead of the $\mathcal{L}_{\text{plan}}$, as described in Appendix B.2.2). Such comparison could provide insight into the impact and necessity of the designed loss components.

### Questions
- As far as I understand, a separate model was trained for each benchmark, including all relevant training phases. It would be interesting to explore how a model jointly trained across all three domains would perform.

### Soundness
3

### Presentation
3

### Contribution
3
