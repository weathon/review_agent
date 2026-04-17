# Interpretable Reinforcement Learning with Self-Abstraction and Refinement

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 4

## Abstract
We propose ReLIC, a reinforcement learning method with interactivity for composite tasks. Traditional RL methods lack interpretability, so it is difficult to integrate expert knowledge and refine the trained model. ReLIC is composed of a high-level logical model, low-level action policies, and a self-abstraction and refinement module. At its high level, it takes in predicates as its input so that we can design a synthesis algorithm to illustrate our high-level model's logical structure as an automaton, demonstrating our model's interpretability. At its low level, deep reinforcement learning is utilized for detailed action control to maintain high performance. Furthermore, based on the structured information provided by the automaton, ReLIC leverages GPT-4o to generate expert predicates and refine the automaton by injecting expert predicates and performing joint training, thereby enhancing RELIC's performance. ReLIC outperforms state-of-the-art baselines in several benchmarks with continuous state and action spaces. Additionally, ReLIC does not require humans to hard-code logical structures, so it can solve logically uncertain tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ReLIC, a hierarchical reinforcement learning framework that achieves interpretability and interactivity. It uses a high-level logical planner and low-level control policies. The core contributions are a self-abstraction module that synthesizes the learned logic into an interpretable automaton (DFA), and an interactive refinement loop that uses an LLM (GPT-4o) to analyze this automaton and suggest expert predicates to fix policy flaws.

### Strengths
*   The framework produces an interpretable automaton that is directly used to debug and improve the agent's policy, which is a significant step beyond post-hoc explanations.
*   ReLIC demonstrates superior performance against a wide range of baselines on difficult continuous control tasks, proving its practical effectiveness.
*   The model learns the core policy end-to-end, only engaging an expert (via LLM) for targeted refinement, making it more scalable than methods requiring full upfront symbolic specification.

### Weaknesses
*   The framework's performance may depend heavily on the quality and completeness of the initial, predefined predicate set. The paper would benefit from a discussion on how the system scales with a large number of predicates.
*   For extremely long-horizon or complex tasks, the synthesized automaton could become too large to be interpretable by a human or LLM, potentially limiting the approach's applicability.
*   It is unclear how well the framework, particularly the specialized low-level policies, would generalize to new tasks not seen during training.

### Questions
How sensitive is ReLIC to the initial set of human-defined predicates? Could you discuss the trade-offs between providing a sparse vs. a rich set of initial predicates?

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
2

### Summary
The paper proposes **ReLIC**, a hierarchical RL method for composite tasks. ReLIC has (i) a high-level **Differentiable Logic Machine (DLM)** that reasons over predicates, (ii) **low-level continuous control policies**, and (iii) a **self-abstraction and refinement** pipeline. From training rollouts, ReLIC extracts key predicates and synthesizes a **deterministic finite automaton (DFA)** to expose high-level decisions; an **LLM (GPT-4o)** then inspects the reduced automaton plus failure trajectories, proposes new expert predicates, and these are fed back via **joint training** of the logic planner and action policies.

### Strengths
* **Clear objective.** Integrates interpretability (automaton from key predicates) with interactivity (LLM-driven knowledge injection) without pre-defining a logical schema. 
* **Technical design.** Presents joint training of the logical planner and low-level policies, plus a pipeline to synthesize and reduce a DFA from learned behavior.  
* **Experiments.** Evaluated on Highway and four Fetch tasks; Highway reports **crash rate, velocity, length**, and ReLIC outperforms baselines across several method families. 
* **Interpretability + interactivity.** Shows how LLM-added predicates refine automaton states and improve **PickHighPlace**, with the LLM generating concrete expert predicates that split states. 
* **Reproducibility hooks.** Includes an anonymous repo link and a prompt-template appendix describing LLM usage.

### Weaknesses
* **Automaton fidelity not quantified.** No measure of how well the synthesized/reduced DFA predicts high-level choices on held-out trajectories or under counterfactual transitions. (Paper explains construction but doesn’t report such diagnostics.) 
* **Stability of key-predicate extraction.** The pipeline highlights “key predicates,” but there’s no analysis of run-to-run stability or recovery of known ground-truth predicates in a controlled setting. 
* **LLM refinement details are light.** The paper names GPT-4o and outlines prompting steps, but does not report failure filtering, sensitivity to prompt/temperature/model version, or API cost; training schedule is given (**500 epochs, refinement every 50**) but cost/variance for intermediate refinements are not.  

### Minor issue

* The paper links GPT-4o docs and an anonymous repo; it’s unclear if the repo contains runnable LLM-call code or a stub to reproduce one full refinement round. Clarification would help reproducibility.

### Questions
1. **LLM usage and comparative controls.** You state that GPT-4o refines the automaton by proposing expert predicates, but the anonymous materials don’t clearly show LLM-call scripts, logs, or stubs; could you point to the exact scripts (including prompts and temperatures) or provide a minimal runnable stub for one refinement cycle, and—critically—add a **ReLIC-w/o-LLM** ablation (keep self-abstraction and joint training but remove LLM-generated predicates) alongside **other planning/editing baselines** that use the same low-level policies, reporting effect sizes with confidence intervals to isolate the performance gains attributable specifically to your LLM-based refinement?

2. **DFA faithfulness and DFA specificity.** Please report the held-out accuracy of the DFA in predicting high-level selections, how this accuracy degrades as key predicates are pruned, and causal tests where alternative transitions are enforced; additionally, include a **“key predicates only (no DFA)”** ablation on at least one representative task and compare against alternative abstraction structures (e.g., any tranditional methods ) using the same inputs and policies to demonstrate that the observed benefit is not specific to DFAs.

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
This paper combines refines learned automata with LLMs to control MDPs. I found this paper unclear because it lacks key related work and explanations of the proposed method. However the overall ideas and experiments seem good.

### Strengths
Interpretable RL is important but difficult. Using LLMs to mimic expert interaction is a promising research idea.

### Weaknesses
As said above I like the overall idea of this paper. However, there are still too many weaknesses that I detail here.
## Lack of rigor
- Interactive RL is not defined properly. What I mean is that, while it is acceptable to not have a formal definition for interactive RL, it is seems to be that interactivity is used in the submission as a direct consequence of transparency rather than an independent property of the cited references (INTERPRETER & ScoBots). In interpreter, interactivity is not mentioned and in ScoBots, interactivity is a consequence of transparency: because policies are readable a human can interact witht them. I consider this as counter intuitive since you have both interpretable RL baselines and interactive RL baselines while from your presentation of interactive RL there are essentially the same. Maybe you should look for human in the loop or RLHF baselines to compare to real interactive RL?
- A key missing related work already does automata synthesis for RL: DeepSynth: Automata Synthesis for Automatic Task, Hasanbeig 2021.
This paper is very similar to your submission and not comparing with it is problematic in my opinion.

## Experiments could be better
- There is no guarantee that the refinement step will not degrade the automata. Did you try to benchmark the number of times refinement actually degrades performances ?
- Too much baselines. Since the key novely compared to DeepSynth is LLM refinement I would prefer better empirical analyses of the refinement.

### Questions
Did you try to benchmark the number of times refinement actually degrades performances?
Could you try to refine INTERPRETER's decision tree policies or Scobots policies?
How does your approach differ from DeepSynth?

### Soundness
2

### Presentation
2

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
The paper introduces ReLIC, a hierarchical reinforcement learning (RL) method for composite tasks that aims to enhance interpretability and interactivity. It consists of a high-level logical model (using predicates for symbolic abstraction), low-level action policies trained via deep RL (e.g., SAC), and a self-abstraction and refinement module. The logical model is synthesized into an automaton for transparency, allowing injection of expert predicates during joint training. Object-centric representations ground the predicates, enabling the method to handle continuous state/action spaces without predefined logical structures.

### Strengths
The paper tackles a relevant problem in RL: balancing interpretability with performance in composite tasks, where black-box models hinder alignment and debugging. The self-refinement via GPT-4o is a practical innovation, automating predicate generation and integrating external knowledge on-the-fly, which could reduce manual effort compared to prior works like SCoBots. The automaton synthesis provides a clear visualization of high-level logic, aiding post-hoc analysis.

### Weaknesses
Despite its claims, ReLIC lacks substantial novelty, primarily recombining established ideas: hierarchical RL, neuro-symbolic abstraction, and LLM-assisted RL without theoretical advances or unique insights. The "self-abstraction" is essentially predicate learning via RL, but the LLM refinement feels tacked on—prompts are simplistic, with no ablation on LLM choice or error handling (e.g., hallucinated predicates). Continuous space handling is overstated; benchmarks are mostly discrete or modified MuJoCo variants, ignoring complex real-world dynamics like partial observability or multi-agent interactions. 

Experiments are narrow: only 3-4 environments, and baselines are cherry-picked—missing SOTA like DreamerV3. Robustness to "logical uncertainty" is contrived (e.g., injecting noisy predicates), but no evaluation on distribution shifts (e.g., sim-to-real). Interpretability claims are weak: the automaton requires expert validation, and low-level policies remain black-box, undermining end-to-end transparency. Scalability is unaddressed; predicate explosion in high-dimensional states could lead to combinatorial issues, untested beyond toy tasks.

While interpretability in RL is valuable, ReLIC offers incremental tweaks to existing neuro-symbolic hierarchies without groundbreaking contributions—e.g., LLM integration is superficial, lacking analysis of biases or failure modes, which are critical for safety-critical applications. The methodological flaws, such as underdeveloped continuous benchmarks and weak baselines, fail to convincingly demonstrate advantages over priors. Claims like "solving logically uncertain tasks" are exaggerated without broader validation. The work feels preliminary; stronger experiments (e.g., more environments, ablations on refinement) and deeper theoretical grounding could make it suitable, but in its current form, it lacks the impact for acceptance.

### Questions
The specification "In general, a b-ary logical predicate P is defined based on a b-ary real transformation function f" is vague. The function is defined in real space, while a predicate is defined based on true/false. Could authors make the definition of a predicate more clear?

### Soundness
3

### Presentation
2

### Contribution
2
