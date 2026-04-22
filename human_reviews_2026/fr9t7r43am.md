# Hidden Markov Modeling of Reasoning Dynamics in Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Reasoning in language models involves both explicit steps in the generated text and implicit structural shifts in hidden states, yet their joint dynamics remain largely underexplored. We introduce a Explicit–Implicit Reasoning Lens (EIRL) that jointly models these dimensions: at the explicit stage, EIRL captures transitions between reasoning roles, and at the implicit stage, it models latent depth regimes that reveal how computation is allocated across layers within each role. By linking what function a reasoning step serves to where it arises in the network, our approach provides a unified lens for both understanding reasoning dynamics and the underlying mechanisms. Once trained on reasoning trajectories, the EIRL learns probabilistic transition patterns through hidden Markov modeling that characterize how models typically move between reasoning roles and allocate computation across layers. Our analysis reveals a clear internal-to-external progression in reasoning. At the implicit stage, hidden states organize into distinct depth patterns that differ across reasoning categories, indicating that the model allocates its layers differently depending on the functional role of the step. These internal configurations then give rise to the explicit stage, where the model expresses its reasoning through semantic transitions. This progression diverges between trajectories that succeed and those that fail to reach the correct answer. Leveraging the explicit–implicit reasoning structure captured by EIRL, our framework supports both causal interventions that steer models toward targeted reasoning paths and interpretability analyses that reveal how different external intervention strategies reorganize the semantic flow of reasoning to produce their observed effects.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a Hierarchical Hidden Markov Model (HHMM) that unifies explicit reasoning in text and implicit reasoning in hidden states. It models reasoning as semantic transitions over depth regimes and finds that successful trajectories align semantic and structural dynamics, while failures show unstable loops. Derived steering vectors enable targeted interventions that correct reasoning without increasing output length.

### Strengths
The HHMM provides an elegant probabilistic formalism unifying textual reasoning steps with internal representational dynamics—bridging behavior-level and mechanism-level reasoning analysis.

### Weaknesses
Both the correct and incorrect trajectories show strong self-loops at the analysis step (0.382 and 0.396), yet the paper provides different explanations for these similar patterns. This suggests the need for additional experiments and statistical analysis to ensure the interpretation is reliable. Similarly, when comparing the correct and incorrect transition matrices, the relative ranks of most elements appear similar. It would be better to include statistical significance tests or clearer evidence to support the claim that the transition dynamics truly differ between the two cases.

The experimental criterion — “included in the consensus set if at least two out of four models agree” — is also unconvincing. As Table 3 suggests, different models exhibit distinct reasoning strategies; thus, using shared anchors across architectures may conflate semantically different depth regions. More justification is needed to claim cross-model consensus on reasoning anchors.

Some descriptions are unclear or imprecise. For example, in Section 5, “surface-level” should likely be “top-level.” Further clarifications are needed (see questions below).

### Questions
What models and datasets were used to compute the statistics in Table 1? Could the authors include more datasets and LLMs to better illustrate the transition dynamics? Also, what do the colors in Table 1 represent?

In Table 2, what exactly does “retrieval” mean in the term setup_and_retrieval?

The paper poses an important question — how explicit and implicit reasoning work together in solving problems — but the conclusion remains unclear. Could the authors provide a more explicit summary or interpretation of how these two dimensions interact?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper models LLM reasoning as a two-level process that couples what each step is doing with how hidden states evolve across layers. It introduces a hierarchical HMM: a top chain over step semantics (setup/retrieval, analysis, verification, final answer) and a bottom HMM over latent “depth regimes” that capture layer-wise computation during each step. Using this joint model, the authors show that successful solutions follow stable semantic paths anchored by consistent late-layer patterns, while failures loop in verification or show unstable depth transitions. They then derive step-aware steering vectors from the learned transitions to gently nudge hidden states at critical moments, correcting some failures without changing weights or lengthening outputs.

### Strengths
Strengths

- Unified lens: Couples step semantics with hidden-state dynamics, giving a clear “reasoning trajectory.”

- Actionable diagnostics: Pinpoints where runs stall (e.g., verification loops) and which layers fail to anchor.

- Lightweight steering: Step-aware vectors nudge failing traces without changing weights or lengthening outputs.

### Weaknesses
Weaknesses

- Small gains: Correction improvements over baselines are modest; may feel insignificant for production.

- Label reliance: Top-level step tags are self-annotated by the model, risking bias and propagation of errors.

- Simplifying assumptions: Bottom-level uses PCA + diagonal Gaussians; may miss richer, non-Gaussian structure.

### Questions
same as weakness

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
3

### Summary
The paper presents a Hierarchical Hidden Markov Model (HHMM) framework to jointly capture explicit semantic stages and implicit structural depth regimes underlying reasoning processes in large language models (LLMs). Experiments demonstrate that the approach can systematically characterize reasoning motifs. Based on HHMM, intervention can be applied to improve performance without increasing sequence length.

### Strengths
1. **Important Topic**: The interpretability of LLM reasoning is an important and under-explored research area.

2. **Novel and Interesting Modeling**: The introduction of HHMM to connect explicit semantic transitions with latent computational regimes is a non-trivial approach for interpreting and intervening in LLM reasoning. The analytical results offer interesting insights about reasoning behaviors of current LLMs.

3. **Actionable Steering Mechanism**: Beyond analytical experiments, this paper proposes intervention technique derived from transition matrices. The steering mechanism not only helps validate the analysis reliability, but offers a pragmatic way to rescue failing trajectories at some times.

### Weaknesses
1. **Limited Justification of HHMM Assumptions**: This paper lacks a formal discussion or ablation on the necessity of the hierarchical structure for capturing meaningful interaction between semantic roles and depth regimes. It is unclear the observed improvements stem from hierarchy or from simply more latent structure.

2. **Experimental Issues** Some experimental settings including models, $C$, $K$, boundary clusters, etc., seem arbitrary. Results in Table 1, 2 and the relative differences are not significant enough for the analytical claims about reasoning patterns in Section 4 Q1, Q2.

3. **Insufficient Baseline Comparison for Steering**: In Table 6, the proposed intervention technique is only compared with a basic edge-agnostic baseline, rather than established steering or latent intervention frameworks. Considering the modest intervention improvement (about 1pp), comparison with more related methods is important.

### Questions
1. **Necessity of Hierarchy (Weakness 1 Related)**: Could the authors experimentally or theoretically justify why the hierarchical HMM is preferable to a single-level HMM or other latent-state models? For example, would a flat HMM over step * depth suffice?

2. **Model Selection (Weakness 2 Related)** Why use a Qwen3-1.7B instead of its 4B or 8B variations, which are consistent in parameter size with other models?

3. **Robustness (Weakness 2 Related)**: How sensitive are the bottom-level HMMs to number of regimes ($K$), or the PCA preprocessing step? And other experimental variations like tasks and models? Are the results stable across runs?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Reasoning in LLMs can be viewed as explicit and implicit steps. By explicit reasoning, the authors refer to step by step generation of text and by implicit reasoning, they refer to the changes in hidden states. By modeling explicit steps as Markov chains and implicit steps as Hidden Markov Chains they propose a new framework, HHMM (Hierarchical Hidden Markov Model), to model the reasoning trajectories in both explicit and implicit dimensions. Finally, they use their framework to steer the reasoning trajectories by adding step-aware steering vectors to the hidden states of the final layer during generation in order to guide the reasoning toward correct reasoning trajectories. Overall, they improve the reasoning for incorrect reasoning trajectories and also provide an interpretable framework to study reasoning behavior in terms of layer concentration and different aspects such as verification, analysis, and outputting the final answer.

### Strengths
A new framework that can discover and explain the patterns in reasoning and highlight the differences between correct and incorrect reasoning trajectories. 

In addition to studying semantic reasoning and structural reasoning dynamics, they propose a steering method based on their framework that can correct the incorrect reasoning trajectories and help the generation to be stable and truthful.

While many methods for improving reasoning lead to an increase in the number of generated tokens, their approach maintains almost the same and sometimes even lower token count.

### Weaknesses
There are no results for the fraction of originally correct predictions becoming incorrect after steering (since in inference time we can't predict the model is going to do it wrong or correct and this experiment would help in observing the possible harmful effect of steering for already correct reasonings)

Lack of baselines in the steering section. A simple baseline like directly prompting the model to be more decisive during reasoning could suffice to show the superiority of using HHMM for steering as well.

### Questions
Could you use a more powerful LLM for classification (final_answer, setup_and_retrieval, …) for making sure there would be less error and mistakes in this phase?

### Soundness
3

### Presentation
3

### Contribution
3
