# Reasoning Up the Instruction Ladder for Controllable Language Models

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first “think” about the relationship
between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises ∼7K aligned and conflicting system–user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks, achieving roughly a 20% improvement on the IHEval conflict setup. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks, providing up to a 20% reduction in attack success rate (ASR). These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model
behavior.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses instruction hierarchy (IH) in large language models (LLMs)—the ability to follow higher-priority (e.g., system-level) instructions while rejecting conflicting lower-priority (e.g., user) directives. The authors argue that current LLMs fail to respect this hierarchy, especially under adversarial inputs (prompt injection, jailbreaks), because they lack explicit reasoning over instruction sources.

They propose Reasoning for Instruction Hierarchy (IH reasoning), which reframes the problem as a meta-reasoning task: the model first reasons about the relationship between system and user prompts, and then decides which to obey. To enable this, they construct VerIH, a synthetic dataset that extends the RLVR-IFEval dataset with both aligned and conflicting system–user prompt pairs whose outputs have verifiable correctness functions. Using Group Relative Policy Optimization (GRPO) and a small “SysHint” instruction encouraging explicit reasoning, they finetune reasoning-capable models (Qwen3-4B/8B, Phi-4-mini-reasoning) on VerIH.

### Strengths
* A simple and effective method: Lightweight dataset + RLVR yields measurable performance gains with only ~7K examples.
* Broad benchmark coverage: Includes both in-domain (IHEval, IFBench) and out-of-domain (safety, jailbreak) tests.

### Weaknesses
* Incremental novelty: The paper extends earlier instruction hierarchy and reasoning-for-safety works but doesn’t fundamentally rethink model architecture or training beyond RLVR on synthetic conflicts.
* Synthetic dataset limitations: VerIH conflicts are LLM-generated and may lack realism or linguistic diversity; unclear if models overfit to the structure of these synthetic conflicts.

* Evaluation limitations:
    * Heavy reliance on automated verification or guard scoring; limited human evaluation.
    * Safety generalization gains could arise from exposure to adversarial-style constraints, not genuine reasoning over roles.

### Questions
* How realistic are VerIH conflicts compared to real-world multi-role instructions (e.g., multi-turn, multi-agent conversations)?
* Could the improvements on safety benchmarks be explained by simple refusal pattern learning rather than hierarchical reasoning?
* How does the model perform when system and user instructions are both benign but subtly contradictory (e.g., stylistic or prioritization differences)?
* Will VerIH be publicly released with generation scripts to verify reproducibility claims?
* Can the approach extend to three-level hierarchies (system–tool–user), or does performance degrade?
* How does the approach compare to embedding separation methods (Wu et al., 2024b) when evaluated under the same GRPO setup?

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
This paper addresses the challenge of enforcing instruction hierarchies in large language models (LLMs), specifically ensuring that system-level directives reliably override user requests in cases of conflict. The authors present VerIH, a newly constructed dataset of system/user prompt pairs (both aligned and intentionally conflicting), with verifiable constraints to facilitate deterministic evaluation. Models are finetuned using reinforcement learning with variable reward (RLVR) on VerIH to enable explicit meta-reasoning over instruction priorities. Experiments across multiple LLMs show improved compliance with instruction hierarchies and enhanced robustness against adversarial inputs, such as prompt injection and jailbreak attacks, while maintaining general reasoning ability.

### Strengths
1. Concrete Reformulation of a Critical Problem: The paper offers a clear and well-motivated reformulation of instruction hierarchy resolution as a reasoning problem, supported by real-world motivating scenarios. This framing addresses a persistent weakness in LLM deployment around controllability and safety.
2. VerIH Dataset with Verifiable Constraints: Construction of the VerIH dataset enables systematic training and evaluation. By creating both aligned and conflicting system-user prompt pairs with verifiable, automatable scoring, the approach yields a genuinely measurable target for RL-based learning.
3. Methodological Rigor: The proposed RLVR finetuning with Chain-of-Thought (CoT) reasoning and explicit "SysHint" instructions is well operationalized and clearly described. The role of various system prompts and ablation variants is made explicit.
4. Empirically Grounded Claims with Extensive Evaluation: The experimental section covers a diverse set of benchmarks, including general instruction following (IFEval, IFBench), instruction hierarchy (IHEval), safety, and general reasoning. 
5. Robust Generalization: The paper demonstrates that training on general instruction hierarchy tasks in VerIH out-of-distribution generalizes to safety-critical applications without explicit safety data (Table 2). This supports the claim that explicit reasoning about instruction hierarchies yields a flexible and robust mechanism for model controllability.
6. Analytical Depth and Ablation Analyses: The analysis includes careful ablations isolating the effects of reasoning traces and conflicting prompts, plus an explicit study relating reasoning “coverage” to task performance.

### Weaknesses
1. Limited Theoretical Analysis and Justification of Meta-Reasoning Efficacy: While the intuition for meta-reasoning over instruction hierarchies is plausible, the paper lacks a formal or semi-formal analysis or even a taxonomy of potential failure cases for instruction prioritization. For instance, there is no attempt to systematically dissect why explicit reasoning works better than implicit mapping for instruction hierarchy, nor to quantify its limitations (Section 2 and analysis in Section 6 are largely empirical/descriptive).
2. Supervision Signal and Potential RL Pitfalls Underexplored: The reward function Freward used in RLVR training is described in general terms (“verifiable answer constraints”, Section 3), but deeper discussion of its capacity, potential brittleness, or the risk of reward hacking/overfitting is minimal. There is no analysis of reward function coverage, adversarial manipulation resistance, or sensitivity of resulting models to reward definition.
3. Generality Limited by Data Construction Process: The conflicting prompt pairs in VerIH are generated by rewriting user prompts with an LLM. While this approach provides scale, it limits the complexity and naturalness of conflicts. The“conflicting”instructions may be somewhat artificial or simplistic，see the conflict rewrites in Appendix A, where many are mere format or scope differences. It is unclear if the method will hold on truly complex, nuanced, or human-elicited hierarchical instruction conflicts.
4. Result Interpretation and Failure Analysis: While results in Table 1 and Table 2 are generally positive, the paper does not sufficiently analyze failed or ambiguous cases. Which specific types of hierarchical or adversarial conflicts still“slip through”after VerIH RLVR training? The increase in attack success rate for certain settings (noted for Phi-4-mini-reasoning, Table 2) is only superficially discussed.

### Questions
1. Can the authors provide deeper theoretical justification (beyond empirical results) for why explicit meta-reasoning on instruction hierarchies achieves more robust compliance than input-response mapping? What are the boundaries of this approach?
2. What specific strategies were used to ensure realistic and challenging “conflicting” instruction pairs in VerIH, beyond formatting/scope variations? Do the authors have examples where existing conflict generation methods fail to produce non-trivial, subtle conflicts?
3. How does the reward function Freward handle ambiguous responses, partial compliance, or compositional system-user conflicts? Could the RL process reward superficial but noncompliant outputs (reward hacking)?
4. Have the authors explored compositional or multi-level hierarchies (more than 2 layers/roles)? How does the approach generalize, and are there examples in VerIH or evaluation where deeper nesting reveals new limitations?
5. In Table 2, Phi-4-mini-reasoning shows higher attack success rates on certain safety benchmarks. Can the authors clarify the cause (overfit, model limitations, or tuning), and suggest remedies or further experiments?

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
This paper addresses the critical problem of instruction hierarchy (IH) in Large Language Models (LLMs), where models must reliably prioritize higher-level instructions (e.g., system prompts) over potentially conflicting lower-level instructions (e.g., user prompts). The authors argue that current models fail at this because they treat instructions as a simple input-response mapping problem, leading to vulnerabilities like jailbreaking and prompt injection.


The paper's core contribution is to reframe IH resolution as an explicit meta-reasoning task. Instead of just responding, the model is trained to first "think" about the relationship between the system and user instructions, identify conflicts, and reason about which instruction takes precedence before generating a final answer.

To achieve this, the authors introduce two key components:


* VerIH: A new dataset of 7K instruction-following tasks with verifiable answers. It is built by taking an existing dataset (RLVR-IFEval) and using an LLM (Claude-4-Sonnet) to rewrite half of the user prompts to create explicit conflicts with the system prompts.

* Training Methodology: The authors use lightweight reinforcement learning (specifically RLVR with GRPO) to fine-tune reasoning-enabled models (Qwen3 and Phi-4-mini). The models are trained to generate a chain-of-thought (CoT) reasoning trace within \<think\> tokens before their answer, with a reward signal based on the correctness of the final (verifiable) answer.


The paper claims that this approach significantly improves models' ability to follow instructions and resolve hierarchies, especially in conflicting scenarios. The most significant claim is that this learned reasoning ability generalizes out-of-distribution (OOD) to safety-critical settings. By simply adding safety policies as a high-priority system prompt at inference time, the trained models show enhanced robustness against jailbreak and prompt injection attacks, despite never having seen safety-related data during training.

### Strengths
* **Novel and Effective Problem Framing:** The key insight to treat instruction hierarchy as a meta-reasoning task rather than a standard alignment problem is a strong and novel contribution. This moves the field beyond implicit learning and toward explicit, scrutable conflict resolution.


* **High-Quality Dataset (VerIH):** The creation of the VerIH dataset is a valuable contribution to the community. The methodology of generating conflicts from an existing verifiable dataset (RLVR-IFEval) is clever, as it preserves the original verification functions and provides a clear reward signal for the RL process.


* **Strong Empirical Results:** The paper demonstrates clear and significant performance gains on instruction hierarchy benchmarks, particularly on the IHEval-conflict set (e.g., +22.87% for Qwen3-4B). This is achieved while maintaining or even slightly improving performance on general reasoning tasks like MMLU and MATH-500, showing the training is targeted and does not cause degradation.


* **OOD Generalization to Safety:** This is the most compelling result of the paper. The ability to train a model on general, non-safety-related constraint conflicts (e.g., "use 8 highlights" vs. "use no formatting") and have that skill transfer to rejecting harmful adversarial prompts (Table 2)  is a significant finding. It supports the hypothesis that safety is a special case of IH and provides a practical path toward building more controllable and dynamically configurable models (e.g., via GuardRules ).

### Weaknesses
* **Scalability of Hierarchy:** The paper simplifies the IH problem to two levels: system and user. While it claims the method is "inherently scalable", this is asserted without proof. Real-world applications involve more complex hierarchies (e.g., developer system prompts, user-level system prompts, tool instructions, user data) that may have more nuanced precedence rules. The experiments do not test this scalability.



* **Diversity of Conflicts:** The VerIH dataset's conflicts are generated entirely by Claude-4-Sonnet following a specific prompt template. This could introduce a lack of diversity or an unknown bias in the types of conflicts generated. It is unclear if the model is learning to resolve conflicts in general or just the style of conflicts produced by the generator model.

* **Reliance on Scaffolding:** The method's success seems tied to specific scaffolding: the <think> tokens and the SysHint prompt that explicitly tells the model to reason about conflicts. It is not entirely clear whether the model has acquired a general meta-reasoning skill or simply learned to follow the SysHint prompt effectively.


* **Justification for RL:** The paper uses RLVR/GRPO, which is inherited from the base dataset. However, it is not clearly justified why this is superior to simpler methods. For example, one could simply perform supervised fine-tuning (SFT) on (CoT, Answer) pairs that are known to receive high rewards from the verification function. This might achieve similar results with less complexity.

### Questions
1. You claim the two-level (system vs. user) method is "inherently scalable". Could you elaborate on how you envision this working for a more complex, multi-level hierarchy (e.g., System > Tool > User)? Would this require a more complex SysHint prompt, or a dataset with more complex, multi-level conflicts?



2. The use of RLVR/GRPO  is a core part of the methodology. Have you experimented with a simpler SFT approach, where you only fine-tune on high-reward (CoT, Answer) pairs generated from the VerIH dataset? This would help clarify if the full, complex RL loop is necessary or if SFT on curated "good" reasoning traces is sufficient.


3. The VerIH dataset's conflicts are generated by a single LLM (Claude-4-Sonnet). Have you analyzed the diversity of the generated conflicts? Is there a risk that the model has overfitted to the style of conflict that Claude generates, rather than a general concept of instruction conflict?


4. The w/o CoT_train ablation removes reasoning entirely. What happens in an ablation that keeps CoT generation but removes the specific SysHint prompt? This would help disentangle the value of explicit reasoning (CoT) from the value of the specific instructions within the SysHint prompt.

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
4

### Summary
The paper introduces a simple data augmentation strategy to improve the instruction hierarchy ability of LLMs. They assume an access to a dataset of system and user prompts such that the responses can be easily verified for alignment. Then, they rewrite the user prompt using a larger LLM to specifically conflict the system prompt. Finally, they train reasoning LLMs with verifiable rewards over the original and conflicting user prompts to enhance their instruction hierarchy and safety.

### Strengths
- The paper can be easily followed with coherent writing. 
- Experiments show improved performance over the base models and CoT after full fine-tuning. 
- Results also show generalization to other benchmarks, particularly to safety-related tasks.
- Ablation of not using the conflicting prompts is also shown.
- Reasoning traces are also qualitatively evaluated to assess the relationship between system and user prompt.

### Weaknesses
- Originality is limited since the key benefit and contribution of verifiability of instructions is established from Lambert et al., 2025. On the other hand, the idea of conflicting user prompts is also originally provided in Zhang et al., 2025. Thus, the only contribution is augmenting the RLVR-IFEval dataset with the basic scheme of conflicting user prompts.
- Simple Claude-based rewriting may introduce bias and may not generalize to new types of rewriting structures. More analysis to the diversity of conflicting types should be studied. The examples simply add an additional line which does not seem diverse enough.
- Experiments do not compare with other fine-tuning-based baselines and are limited to prompt-based baselines which is not fair. For example, what is the effect of training on IHEval or the effect of SFT training instead of GRPO?
- Training is limited to reasoning-based LLMs and does not show significant improvement for non-reasoning baselines. More discussion and limitations should be discussed here. 
- Verifiable rewards can be problematic due to reward hacking and incorrect reasoning. Reasoning should be qualitatively and quantitatively analyzed as well. 
- Ablation on removing SysHint is not provided. Does "+IFEval" include SysHint or not?
- Training is limited to verifiable instruction types (through constraints), and the simple augmentation cannot be easily extended to non-verifiable use cases. 
- Minor:
  - Figures 1 and 2 are quite rudimentary and should be updated for better space utilization. Figure 1 should present a true example from the constraint types.
  - Typos in Section 3: RVLR -> RLVR
  - The code is directly copied from verl repository with their metadata (setup.py), which leads a reader to believe that the authors' identities are revealed. This should be updated to avoid any confusion.

### Questions
See above weaknesses

### Soundness
2

### Presentation
2

### Contribution
1
