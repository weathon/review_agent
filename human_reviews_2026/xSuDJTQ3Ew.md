# Codified Finite-state Machines for Role-playing

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Modeling latent character states is crucial for consistent and engaging role-playing (RP) with large language models (LLMs).
Yet, existing prompting-based approaches mainly capture surface actions, often failing to track the latent states that drive interaction. 
We revisit finite-state machines (FSMs), long used in game design to model state transitions.
While effective in small, well-specified state spaces, traditional hand-crafted, rule-based FSMs struggle to adapt to the open-ended semantic space of RP.
To address this, we introduce Codified Finite-State Machines (CFSMs), a framework that automatically codifies textual character profiles into FSMs using LLM-based coding.
CFSMs extract key states and transitions directly from the profile, producing interpretable structures that enforce character consistency.
To further capture uncertainty and variability, we extend CFSMs into Codified Probabilistic Finite-State Machines (CPFSMs), where transitions are modeled as probability distributions over states.
Through both synthetic evaluations and real-world RP scenarios in established artifacts, we demonstrate that CFSM and CPFSM outperform generally applied baselines, verifying effectiveness not only in structured tasks but also in open-ended stochastic state exploration.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Codified Finite-State Machines (CFSMs), a framework that leverages large language models (LLMs) to automatically extract character states and generate executable state transition logic for role-playing (RP) agents, aiming to improve consistency and interpretability relative to prompt-based approaches. An extension, Codified Probabilistic FSMs (CPFSMs), models character states probabilistically, supporting nuanced transitions. Empirical validation includes synthetic (game-based) and real-world (Fandom Benchmark) RP tasks, demonstrating improvements in behavioral consistency, efficiency, and interpretability over established baselines.

### Strengths
1. Interpretability: The framework brings interpretability to state modeling in RP with executable, codified transitions derived directly from character profiles.
2. Probabilistic Extension: The CPFSM mechanism elegantly integrates stochasticity into state transitions, explicitly modeling uncertainty in RP.
3. Efficiency: CFSM delivers both accuracy and efficiency, as highlighted in Table 5.

### Weaknesses
1. Evaluation Scope (Generality): Empirical testing relies primarily on the Fandom Benchmark and three synthetic state machines. The real-world scenarios are derived from highly narrativized, structured data (Fandom plots) with limited diversity of state-space complexity and ambiguity. GPT-4.1 is both judge and model in several settings, and open-ended role-play evaluations rely heavily on LLM judgment. There is insufficient third-party or human evaluation of RP quality, which may limit claims of generality.
2. Limited Handling of Dynamic or Emergent States: The model assumes a fixed state set per episode. The limitations of this assumption are acknowledged in Appendix B but not addressed experimentally. Open-world RP often demands dynamic state growth or on-the-fly trait acquisition, which is not modeled or empirically probed in the present study.

### Questions
1. What is the meaning of "multimodal" in Line 053, and "multi-modal" in Lines 070, 080, and 092?
2. How would CFSM/CPFSM scale to open-world/large-scale RP where thousands of (possibly compositional) states, or dynamically constructed state sets, are needed? Any memory, efficiency, or codification tests on "harder" synthetic FSMs or real-world systems?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes CFSM and CPFSM for roleplaying tasks and games with LLMs to ensure behavioral coherence. This paper contributes by offering a framework that defines character transitions and extends it to a probabilistic version to offer multi-modal transitions. Authors have compared and presented the limitations of the LLMs in Section 4 while also showing that CFSM is superior to the baselines compared. The Authors have also discussed the setup and case studies considered for the real plot experiment. The number of models used for the experiment is good, considering the innovation is a technical advancement with LLM rather than the performance of the LLM itself.

### Strengths
1) The described methods work on various artifacts mentioned in the results, while
demonstrating the strong performance against the baselines.
2) The paper mentions the computational complexity for the both methods and shows
faster and efficient codification for the proposed methods.
3) This paper includes a very detailed analysis section mentioning synthetic and real plot
experiments, and is tested with multiple LLM models and techniques, and has various
kind of plots and scenes from various genres.

### Weaknesses
1) The “preliminary and denotation” introduces the necessary terminology but lacks
examples and a lucid explanation, which can be really helpful for the readers and the
general audience unaware of such methods.
2) The multi-modality and reactions of CPFSM lack depth and can be explained more
clearly.
3) The real plot experience can briefly explain one of the artifacts used in the work as a
running example. Not having this makes it lless intuitive for new readers.

### Questions
Detailed suggestions:
1) NLI full form could be better when referenced first.
2) Best@K can be explained.
3) Figure 4 Caption: There is no space between CFSM&amp;CPFSM.
4) Line 109 - evolve should be “evolves”.
5) Line 223 - w_i,j “is” then normalized.
6) Line 225 - in binary_questions, how the logits w_i,j are derived from the
“Yes/no/unknown” question. This can be explained in detail.
7) In Tables 2, 3, and 4, 6, the units of measurement are missing; it would be reader-
friendly to add them.
8) In the baseline, #Character is mentioned for each show, but lacks a reference to it. For
example, Haruhi mentions 5 Characters and AGOT 11, but a brief description of at least
one of the artifacts, such as JOJO, its characters, and the context of the scene, and
profiles would be intuitive for readers to analyze the results better.
9) Multi-modality and transitions of CPFSM can be explained in detail.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Codified Finite-State Machines (CFSM), a framework that enhances character consistency in LLM role-playing by automatically extracting and codifying character states from textual profiles. CFSMs transform character descriptions into explicit FSM structures using LLM-generated logic, grounding behavior in interpretable state transitions. A probabilistic extension, CPFSM, further models transition uncertainty by maintaining distributions over states. The paper evaluates the approach in both synthetic state modeling tasks and large-scale narrative role-play (via the Fandom Benchmark) and demonstrates improved consistency, interpretability, and transition traceability compared to prompt-based methods. Ablation and cost analyses show that CFSM/CPFSM are scalable and effective, offering a hybrid symbolic–neural approach to stateful role-play generation.

### Strengths
- The codification of character logic via FSMs, driven by LLMs, presents a novel mechanism to preserve behavioral coherence in long-form role-playing.

- Experimental results show a clear improvement in behavioral consistency after introducing CFSM. Whether in synthetic tasks (e.g., Mario state transitions) or real narrative scenarios, characters’ state transitions become more coherent and believable. CFSM and CPFSM effectively reduce the confusion and inconsistency commonly observed in prompt-based methods. Notably, CPFSM enhances the subtlety and realism of character responses by modeling weighted reactions across multiple plausible actions through probabilistic transitions.

- Unlike prompt-only state modeling, CFSMs generate explicit transition rules, enabling better control and debuggability in interactive settings.

### Weaknesses
The proposed framework heavily depends on the LLM to extract states and generate transition rules. If the LLM-produced code contains errors or omissions, it may compromise the correctness of the resulting finite-state machine. The paper provides limited discussion on how to validate or correct the logic generated by the LLM, leaving the reliability of the approach partially contingent on the quality of the LLM’s rule extraction process.

Another concern lies in the current evaluation, which primarily focuses on the Synthetic Validation setup and the Fandom Benchmark — both emphasizing narrative-driven scenarios and character-centric tasks. While these datasets are structurally sound and semantically rich, it would strengthen the work to include more conventional evaluation settings, such as open-domain human–AI dialogue, task-oriented dialogue systems, or social simulation environments. Extending the experiments to broader multi-turn dialogue contexts would better demonstrate the generality and transferability of the proposed CFSM/CPFSM framework.

In addition, incorporating more objective and independent evaluation metrics would provide a more comprehensive assessment of model performance. The selection of baselines also appears somewhat limited: although Codified Profile and PromptTrans offer partial validation of the proposed design, the absence of stronger or more up-to-date baselines weakens the comparative significance. Including results against more advanced or representative methods could substantially enhance the paper’s empirical rigor and impact.

### Questions
See in weakness.

### Soundness
3

### Presentation
3

### Contribution
3
