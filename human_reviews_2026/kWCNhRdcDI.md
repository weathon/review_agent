# Natural Language PDDL (NL-PDDL) for Open-world Goal-oriented Commonsense Regression Planning in Embodied AI

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Planning in open-world environments, where agents must act with partially observed states and incomplete knowledge, is a central challenge in embodied AI. Open-world planning involves not only sequencing actions but also determining what information the agent needs to sense to enable those actions. Existing approaches using Large Language Models (LLM) and Vision-Language Models (VLM) cannot reliably plan over long horizons and complex goals, where they often hallucinate and fail to reason causally over agent-environment interactions. Alternatively, classical PDDL planners offer correct and principled reasoning, but fail in open-world settings: they presuppose complete models and depend on exhaustive grounding over all objects, states, and actions; they cannot address misalignment between goal specifications (e.g., “heat the bread”) and action specifications (e.g., “toast the bread”); and they do not generalize across modalities (e.g., text, vision). To address these core challenges: (i) we extend symbolic PDDL into a flexible natural language representation that we term NL-PDDL, improving accessibility for non-expert users as well as generalization over modalities; (ii) we generalize regression-style planning to NL-PDDL with commonsense entailment reasoning to determine what needs to be observed for goal achievement in partially-observed environments with potential goal–action specification misalignment; and (iii) we leverage the lifted specification of NL-PDDL to facilitate open-world planning that avoids exhaustive grounding and yields a time and space complexity independent of the number of ground objects, states, and actions. Our experiments in three diverse domains — classical Blocksworld and the embodied ALFWorld environment with both textual and visual states — show that NL-PDDL substantially outperforms existing baselines, is more robust to longer horizons and more complex goals, and generalizes across modalities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To address the open-world embodied AI tasks with partial observability and incomplete knowledge, this paper proposes NL-PDDL, which combines the strictness of symbolic planning with the semantic flexibility of LLMs. NL-PDDL first extends PDDL into a natural language form for representation, enabling LLM-based commonsense entailment reasoning, and finally employs lifted regression planning. The method maintains robustness under goal-action specification misalignment and generalizes across both textual and visual modalities.

### Strengths
1. Maintain the verifiability of symbolic planning.
2. Handle semantic misalignment through LLM.
3. Lifted representation avoids exhaustive grounding.
4. Works across textual and visual modalities

### Weaknesses
1. Limited novelty compared to prior work. I believe NL-PDDL lacks sufficient innovation. LLM-Regress [1] already proposed combining lifted regression with LLM for open-world planning. What advantages does NL-PDDL offer over this approach? From both the methodology and experimental results, NL-PDDL's advantages are not prominent. LLM-Regress achieves higher success rates and lower token consumption on ALFWorld.

2. Scalability and applicability limitations. NL-PDDL is not a general, scalable approach for open-world planning, as PDDL cannot fully model all problems in open-world settings. In addition, with increasing problem complexity, a sharp performance degradation of NL-PDDL (Figure 7) is observed. This instability may limit the method's practical applicability.

3. Inability to learn. NL-PDDL lacks learning capability, relying entirely on predefined domain specifications and LLM commonsense without the ability to improve from experience or adapt to new domains automatically.

[1] Liu X, Pesaranghader A, Li H, et al. Open-world planning via lifted regression with LLM-inferred affordances for embodied agents

### Questions
1. Given that NL-PDDL relies on natural language representations, could semantic ambiguity or variability in NL descriptions cause the LLM-based entailment reasoning to produce incorrect unifications? For instance, if the entailment check between goal predicates and action effects yields false positives or false negatives due to ambiguous phrasing, would this result in invalid plans or missed valid action sequences during lifted regression?

2. Could NL-PDDL be integrated as a verifiable planning component within broader LLM-based planning frameworks? Would this hybrid architecture improve both the correctness guarantees and the adaptability to novel scenarios beyond the predefined PDDL domain specifications?

### Soundness
3

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
Paper proposes NL-PDDL, a hybrid framework that integrates natural-language representations with symbolic regression planning to enable open-world, goal-oriented reasoning for embodied AI. The authors extend classical PDDL by allowing goals, predicates, and actions to be expressed in natural language, while preserving logical structure for verifiable planning. A lifted regression algorithm combines symbolic reasoning with LLM-based commonsense entailment, enabling the planner to infer semantically aligned actions and to reason under partial observability without exhaustive grounding. Experiments demonstrate higher plan-success rates compared with baselines. The work aims to bridge the interpretability of symbolic planners with the flexibility of language models for open-world embodied reasoning.

### Strengths
1.Paper conducts experiments on 3 benchmarks demonstrate the effectiveness of the proposed approach.

2.By leveraging lifted regression, NL-PDDL maintains scalability with respect to the number of objects, a property desirable for open-world and partially observable environments.

### Weaknesses
1.Unclear LLM-call efficiency and runtime impact
The regression algorithm requires repeated LLM entailment checks for each predicate and clause. Although the paper reports token counts, it omits time latency and number of LLM calls per plan. It could difficult to assess whether the approach is practical for deployment or real-time settings without this.

2.No clear bridge from symbolic plan to robot execution
While the framework grounds entities in images via a VLM, it stops short of showing how regressed NL-PDDL actions map onto executable robot actions. The granularity of generated subgoals may not align with the level of abstraction supported by downstream policies.

3.Incomplete treatment of entailment aggregation
The “aggregate all entailing predicates” step depends on the finite predicate set and LLM entailment checks, but how to prove the sufficiency of this set.

4.No real-world or hardware validation
All experiments are performed in simulated domains and lack of real robot experiments.

### Questions
1.Efficiency and scalability
How many LLM entailment calls are typically made per plan? What is the average planning latency compared to other planners? 

2.Execution interface:
How do NL-PDDL actions interface with a robot's control stack or low-level skill library? Can the planner adapt its decomposition granularity based on the robot's available primitives?

3.Entailment completeness:
How is the predicate set constructed, and how large is it in your experiments?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes NL-PDDL, an extension of classical PDDL for open-world goal-oriented planning in embodied AI. NL-PDDL represents goals and actions through typed natural language constructs to integrate LLM commonsense reasoning during planning. The framework avoids exhaustive grounding and leverags entailment-based unification to tackle misalignment between natural language goal specifications and action models. Experimental evaluation is conducted on ALFWorld (text and vision variants) and Blocksworld, comparing against SOTA LLM, VLM, and symbolic planning baselines across modalities, goal complexities, and specification alignments.

### Strengths
1. The idea of extending PDDL into a natural-language variant and embedding LLM-based entailment checks into the planning loop is natural and intuitive. This combines sound symbolic reasoning with the flexibility of language-based commonsense, addressing issues where classical PDDL or direct LLM planners individually fail.

2. The results on well-known datasets show that this form of representation works relatively well for autoregressive models. If this paradigm is proven to be more suitable for training, it could become a general paradigm.

### Weaknesses
1. There are some very relevant works that should be cited and compared, e.g. Ada (Learning adaptive planning representations with natural language guidance), and other works that translate between natural language and PDDL, or have LLMs generate PDDL for planning purposes, such as:
a. NLtoPDDL: One-Shot Learning of PDDL Models from Natural Language Process Manuals
b. GPTPDDL (Translating Natural Language to Planning Goals with Large-Language Models), 
c. Generating consistent PDDL domains with Large Language Models, 
..... and many more

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
3
