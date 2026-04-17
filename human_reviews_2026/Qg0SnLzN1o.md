# Agent-ScanKit: Unraveling Memory and Reasoning of Multimodal Agents via Sensitivity Perturbations

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Although numerous strategies have recently been proposed to enhance the autonomous interaction capabilities of multimodal agents in graphical user interface (GUI), their reliability remains limited when faced with complex or out-of-domain tasks. This raises a fundamental question: Are existing multimodal agents reasoning spuriously?  In this paper, we propose \textbf{Agent-ScanKit}, a systematic probing framework to unravel the memory and reasoning capabilities of multimodal agents under controlled perturbations. Specifically, we introduce three orthogonal probing paradigms: visual-guided, text-guided, and structure-guided, each designed to quantify the contributions of memorization and reasoning without requiring access to model internals. In five publicly available GUI benchmarks involving 18 multimodal agents, the results demonstrate that mechanical memorization often outweighs systematic reasoning. Most of the models function predominantly as retrievers of training-aligned knowledge, exhibiting limited generalization. Our findings underscore the necessity of robust reasoning modeling for multimodal agents in real-world scenarios, offering valuable insights toward the development of reliable multimodal agents. Our code is available at anonymous.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates a critical problem: the unreliability of modern multimodal GUI agents, questioning whether their apparent "reasoning" is merely "spurious reasoning" or sophisticated memorization. The authors propose Agent-ScanKit, a systematic probing framework designed to dissect agent behavior by introducing controlled perturbations. By applying this toolkit to 18 agents across 5 benchmarks, the paper's primary finding is that most agents are "memory-dominated." They rely heavily on memorized spatial priors and instruction patterns rather than generalizable reasoning, which explains their brittleness on out-of-domain tasks.

### Strengths
- The question of "reasoning vs. memorization" is fundamental to the progress of autonomous agents. This paper provides a valuable, critical perspective on the (lack of) reliability in state-of-the-art multimodal agents.
- The specific perturbations are well-conceived. The distinction between "masking" (to test spatial memory) and "zoom-in" (to break spatial memory and test local reasoning) is particularly insightful.

### Weaknesses
- The paper is difficult to read. The presentation of the complex methodology, especially the distinctions between various probing outcomes (e.g., high vs. low $\Delta P_{Type}$ vs. $\Delta P_{SR}$  and the interplay with VMC and RS), is dense and difficult to track. The numerous figures and tables contain highly specific metrics and model names, requiring very careful reading to connect the data back to the core memory/reasoning concepts.
- The paper defines reasoning as non-memorized behavior under perturbation (contextual reasoning, generalization). However, distinguishing rote textual memorization (e.g., instruction adherence) from simple reasoning mechanisms remains difficult, especially in the context of the token-level versus sentence-level probing results. For instance, if an agent adheres to an erroneous instruction (sentence-level perturbation), it is labeled as instruction adherence, but the distinction between this being a "memory error" versus a failure of semantic reasoning is not fully disambiguated in the context of the GUI task flow.

### Questions
- Could the authors please provide a concrete, step-by-step example of how the "visual shortcut" and "action shortcut" probes are implemented? 
- Can the authors provide a clearer explanation of how the quantitative metrics, particularly VMC and RS (Visual Memory Consistency and Reflection Score), directly translate into definitions of memory-dominated versus reasoning-driven behavior, especially regarding "over-reflection"? A formal definition of what constitutes "genuine reasoning" within the perturbed POMDP framework would enhance understanding.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Agent-ScanKit, a systematic probing framework designed to analyze and quantify the contributions of memorization versus genuine reasoning in multimodal agents interacting with Graphical User Interfaces (GUIs). Motivated by the observation that existing agents exhibit poor reliability on complex or out-of-domain tasks, the framework uses sensitivity perturbations across three orthogonal paradigms—visual-guided, text-guided, and structure-guided—to isolate and measure reliance on memory shortcuts. Evaluating 18 agents on five GUI benchmarks, the authors demonstrate that mechanical memorization often outweighs systematic reasoning. The findings suggest that many current models function primarily as sophisticated retrievers of training-aligned knowledge, leading to limited generalization and unreliable behavior, underscoring the urgent need for better reasoning mechanisms.

### Strengths
1. The introduction of three orthogonal probing paradigms (Visual-guided, Text-guided, and Structure-guided) offers a comprehensive way to dissect agent behavior along the entire input-to-output pipeline (perception, instruction, action selection).
2. The paper provides compelling quantitative proof of memory shortcuts. For instance, the high VMC (Visual Memory Consistency) and persistently low $\mathbf{\Delta P_{Type}}$ under visual perturbations strongly suggest reliance on memorized spatial priors (Table 1).
3. The structure-guided probing highlights concrete issues with state and reflection actions (e.g., poor performance on PRESSBACK/WAIT and reliance on Action Shortcuts in WAIT/COMPLETE), giving developers clear targets for improving the action space and reflection mechanisms.
4. Large-Scale Validation: Evaluating a large and diverse set of 18 state-of-the-art open-source agents across multiple benchmarks enhances the generalizability of the conclusions.

### Weaknesses
1. The paper uses high sensitivity ($\mathbf{\Delta P}$ is large) as evidence for memory dominance and low sensitivity ($\mathbf{\Delta P}$ is small) as evidence for reasoning/robustness. However, a small $\mathbf{\Delta P}$ under simple perturbations could also signify memorization (the model robustly retrieves the memorized item despite minor noise), while a large $\mathbf{\Delta P}$ under complex OOD perturbations (like zoom-in) could be attributed to a brittle policy that fails to generalize, rather than just "memory." The discussion could be strengthened by acknowledging this nuance and providing a clearer definition of where "memory" ends and "brittle policy" begins.

2. The visual-guided probing (Table 1) mixes visual input (screenshot) with optional text input (low-level instruction). A crucial ablation is missing: how does the dependency on visual memory (low $\mathbf{\Delta P_{Type}}$) change when the textual input (AX Tree or instruction) is removed or perturbed first? This would isolate the true contribution of visual memory versus memory formed by the visual-textual rule matching.

3. The paper notes RL agents exhibit stronger reflective capability but sometimes introduces side effects. A more focused discussion is needed on why RL—a paradigm often associated with generalization—still struggles with memory bias. Is the RL reward signal ($R$) insufficient to penalize memory retrieval, or is the environment ($T$) too visually/textually uniform to demand true OOD reasoning?

### Questions
1. Please clarify the role of the supplementary data (e.g., text representations like AX Tree) in the visual-guided probing (Table 1). Specifically, if an agent uses both the screenshot and the low-level instruction, what happens if the screenshot is masked but the instruction is left unperturbed? How does the model perform its action type prediction, and how does this performance compare to the visual-only results presented?

2. The paper suggests that RL agents' reflexivity introduces notable side effects. For models that rely on CoT (e.g., AgentCPM-GUI), can the authors provide a brief quantitative analysis of the cost (e.g., average token count or latency increase) associated with the reflexivity observed under perturbation, compared to normal execution?

3. In the zoom-in probing (Table 1), OS-Genesis-7B shows $\mathbf{\Delta P_{SR}}$ of $\mathbf{98.8\%}$ and $\mathbf{VMC}$ of $\mathbf{86.0\%}$ (high memory consistency), yet $\mathbf{\Delta P_{Type}}$ is low at $\mathbf{2.9\%}$. This suggests the model correctly predicts the action type (e.g., CLICK) but consistently fails the grounding/SR check. Is this because the spatial memory for the target location is disrupted by the crop/zoom, or because the zoom removes vital global context necessary for accurate localization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces Agent-ScanKit, a black-box probing framework for GUI agents that diagnoses whether success arises from memorization or genuine reasoning. It defines three orthogonal perturbation families—visual-guided, text-guided, and structure-guided—and evaluates 18 open-source multimodal agents across five public GUI benchmarks. Findings: many agents rely heavily on mechanical memorization.

### Strengths
1. Originality. Clear, orthogonalized probing design (visual / text / structure) that goes beyond final-answer or step-wise accuracy to characterize where agents lean on memory vs. reasoning. The structure-guided notion of visual vs. action shortcuts is a useful conceptual lens for reflection/state actions in GUI control. 

2. Quality. Concrete, interpretable perturbation deltas, plus VMC and RS for visual probes; detailed per-agent tables illustrate systematic patterns. 

3. Significance. Addresses a central question for GUI agents: are gains from true reasoning or overfit priors? The toolkit offers a portable, black-box diagnostic that practitioners can run today; results suggest research should re-balance away from spatial/instruction memorization toward robust reasoning.

### Weaknesses
1. Causal interpretation of “reasoning” vs. “memorization”.
The paper interprets small/large delta P under certain perturbations as evidence of memory/reasoning, which is the main assumption. But delta p can also be affected by nuisance factors (e.g., crop changing saliency distribution, off-screen context removal). Lack of ablation controls (random crops vs. target-preserving crops; mask non-targets; swap layout regions) and report causal contrast analyses to isolate the specific factor each probe intends to test. 

2. Metric grounding and thresholds.
VMC depends on a distance threshold (50 px); RS is introduced but its calibration isn’t deeply justified. Lack sensitivity sweeps for thresholds and show ranking stability under reasonable ranges.

3. Scope of structural probes.
“Visual/action shortcuts” are defined for a few reflection/state actions (WAIT, PRESS, COMPLETE, SCROLL); real deployments face richer failure modes (timeouts, latency spikes, theme/scale changes, language/locale, accessibility overlays). Would be good to extend structure-guided probes to system-level perturbations and tool/environment faults.

### Questions
1. Probe validity & confounds: For zoom-in, how do you ensure that the cropped view still preserves all necessary cues for localization/decision (e.g., loss of breadcrumb anchors)? Any human verification that tasks remain solvable under zoom-in? 

2. Instruction perturbations: In token-level tests, agents degrade mainly in vocabulary prediction while Type stays stable. Can you quantify which tokens (verbs vs. objects vs. app names) drive delta PSR most? 

3. Comparisons with alternative process evaluators: Could Agent-ScanKit be combined with trajectory metrics (e.g., grounded-step checks, evidence-bank style) to see whether memory-heavy agents also show more ungrounded reflections?

4. Practical guidance: Given your findings, what training interventions reduce shortcutting (e.g., layout-swap augmentation, coordinate jitter, instruction paraphrase curricula, reflection regularizers)—any preliminary results?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Agent‑ScanKit, a black‑box probing toolkit to disentangle memory vs reasoning in multimodal GUI agents via controlled visual, textual, and structural perturbations. It measures sensitivity by comparing performance with/without perturbations. Visual probes include object masking/editing (to test spatial memorization) and zoom‑in (to test local‑context reasoning); textual probes operate at token and sentence levels; structural probes diagnose visual and action shortcuts for reflection/state actions. Evaluated on 18 open models across 5 benchmarks, the study finds that models rely heavily on memorization and RL+CoT improves interpretability and some reasoning.

### Strengths
- The paper is well-motivated. The problem of distinguishing reasoning and memorization in GUI agents is important, and this work provides a timely response to it.

- The evaluation is comprehensive, as the study analyzes 18 open-source models across 5 benchmarks.

- The proposed approach of comparing performance with and without perturbations is innovative and reasonable, offering valuable insights into memorization behaviors in GUI agents.

### Weaknesses
- Although the analytical methods are innovative, the experimental design requires more rigor, and the current results appear to contain substantial noise.

  - In the visual-guided probing experiments, it is unclear whether the target elements are fully edited or masked. For example, in Figure 4, only the “Amazon App” icon is removed, while the text “Amazon” remains visible. Models could still take correct actions based on that textual cue, meaning this cannot be regarded as true “memorization.”

  - In the visual-guided probing case shown in Figure 4, a reasonable action would be to scroll left to find the “Amazon App” if the original app is masked. However, such actions are not included in the “RS”.

  - The analysis of “action shortcuts” seems less meaningful, as many actions (e.g., wait, complete, press back, press home) are naturally reasonable with only atomic guidance. For scroll, there are limited argument options. A high step-wise SR may not be very informative.

- Some assumptions and conclusions in the analysis appear overstated. For example, the sentence-level text probing results do not necessarily imply that the model relies on memorization. It could also indicate that the model is accustomed to following atomic commands or lacks sufficient planning ability. The authors should revisit their claims and provide more rigorous, evidence-backed interpretations in the whole paper.

- The paper needs thorough proofreading and improvements in presentation quality.

  - In Section 2, the phrase “This section reviews two lines of research that from the basis of this work” should use “form” instead of “from.”

  - The capitalization of subsection titles is inconsistent (e.g., Section 3.1 and Section 5.1).

  - The main paper should include brief explanations of the VMC and RS metrics, or at least reference the appendix where they are defined. Their absence makes it difficult for first-time readers to follow.

  - Tables 1 and 2 should explain the meaning of the arrows in $\Delta P$, VMC and RS. While it seems that the direction indicates better reasoning ability, this is not immediately clear to new readers.

  - The analysis sections are somewhat disorganized and would benefit from a clearer and more coherent logical structure.

- In paragraph 2 of Section 3.2, the discussion focuses on “weaknesses in training data hurting GUI agents,” but the connection to the concept of “infinite predictive space” is unclear and should be better articulated.

### Questions
Could the authors expand the analysis to closed-source models? It would be interesting to see how stronger proprietary models behave under similar perturbations.

### Soundness
2

### Presentation
2

### Contribution
3
