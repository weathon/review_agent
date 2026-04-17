# Human-Object Interaction via Automatically Designed VLM-Guided Motion Policy

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Human-object interaction (HOI) synthesis is crucial for applications in animation, simulation, and robotics. However, existing approaches either rely on expensive motion capture data or require manual reward engineering, limiting their scalability and generalizability. In this work, we introduce the first unified physics-based HOI framework that leverages Vision-Language Models (VLMs) to enable long-horizon interactions with diverse object types — including static, dynamic, and articulated objects. We introduce VLM-Guided Relative Movement Dynamics (RMD), a fine-grained spatio-temporal bipartite representation that automatically constructs goal states and reward functions for reinforcement learning. By encoding structured relationships between human and object parts, RMD enables VLMs to generate semantically grounded, interaction-aware motion guidance without manual reward tuning. To support our methodology, we present Interplay, a novel dataset with thousands of long-horizon static and dynamic interaction plans. Extensive experiments demonstrate that our framework outperforms existing methods in synthesizing natural, human-like motions across both simple single-task and complex multi-task scenarios. For more details, please refer to our project webpage: https://vlm-rmd.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
For HOI synthesis, this paper’s main contributions are twofold: (1) using a VLM as a high-level planner and (2) proposing RMD as a mid-level interface between planning and control. The method requires neither task-specific motion-capture demonstrations nor manual reward engineering, and supports long-horizon, multi-stage interactions with static objects, dynamic objects, and articulated objects. The authors also introduce InterPlay, a dataset containing thousands of long-horizon interaction plans for systematic evaluation. Experiments show consistent gains over strong baselines (e.g., AMP, InterPhys, UniHSI, TokenHSI) in completion rate and precision across single-task and multi-task settings. Ablations further demonstrate that VLM visual input and the fine-grained RMD representation are key to the performance.

### Strengths
* Comprehensive ablations and analysis. The paper isolates component contributions via ablations and alternative VLMs. Removing visual input or coarsening RMD degrades performance, underscoring the importance of VLM spatial grounding and fine-grained RMD guidance. A user study in the appendix aligns with quantitative metrics, indicating more natural and plausible motions.

* Thorough empirical validation. InterPlay spans diverse scenes and task types. Metrics (e.g., Completion Rate, Precision) are appropriate, and comparisons against representative baselines cover static, dynamic, and multi-task scenarios. Results show sizable improvements (often tens of percentage points in completion) and higher motion fidelity, supporting generalization and robustness.

### Weaknesses
* Planner reliability. While GPT-4V outputs are structured, the planner remains a black box. Erroneous sub-plans could hinder training or misguide rewards (e.g., incorrect reasoning about dynamic object trends). The paper does not quantify or correct such errors. Moreover, cross-phase physical causality is only implicitly encoded via RMD; the VLM may be pattern-matching rather than reasoning about outcomes, which could break in highly dynamic tasks (e.g., throwing/catching).

* RMD’s bipartite modeling misses self-contact and intra-object contact. By construction, RMD only encodes relations between human parts and object parts. Many everyday interactions require human–human self-contact and articulated object self-contact (e.g., folding a clamshell laptop). Such intra-agent and intra-object constraints are not representable with a strictly bipartite graph, which can lead to inaccurate guidance or reward shaping for these tasks.

* Task scope. Extremely dynamic, high-precision, or fast-response scenarios are not covered; it is unclear whether RMD + GPT-4V planning scales to these regimes.

* Writing. The related-work narrative around “using a planner” vs. “the representation that the planner outputs” is not sufficiently disentangled; these are distinct axes and should be contrasted more clearly.

### Questions
* Physical causal reasoning. The method relies on VLM-imagined motion trends. In highly dynamic settings (e.g., projectile interactions or multiple moving objects), can GPT-4V reason causally? Can RMD be extended to handle exogenous object motion (not initiated by the agent)? If objects move autonomously or unexpected events occur (e.g., a rolling ball), can the framework revise the plan online?

* Planner stability. Did you observe VLM hallucinations (e.g., incorrect part segmentation, anomalous RMD weights)? What filtering or correction mechanisms were used? How sensitive are plans to prompt phrasing? Please provide observations and mitigation strategies for planner stability.

* Open resources. Please confirm the release timeline for the InterPlay dataset and the codebase (including prompt templates and exemplar plans).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the limitations of existing Human-Object Interaction (HOI) synthesis methods—such as reliance on expensive motion capture data, manual reward engineering, and poor support for long-horizon/dynamic interactions—by proposing a unified physics-based framework. The core innovation is VLM-Guided Relative Movement Dynamics (RMD), a fine-grained spatio-temporal bipartite graph representation that encodes part-level relationships between humans and objects (e.g., stationary contact, approaching/separating motion). RMD enables Vision-Language Models (VLMs, e.g., GPT-4V) to automatically generate goal states and reward functions for reinforcement learning (RL), eliminating manual tuning. The framework supports interactions with static, dynamic, and articulated objects, and the authors introduce InterPlay, a novel dataset of thousands of long-horizon HOI plans. Experiments demonstrate that the method outperforms baselines (e.g., InterPhys, TokenHSI, UniHSI) in both single-task and long-horizon multi-task scenarios, producing more natural, task-aligned motions as validated by quantitative metrics and a user study.

### Strengths
1. Captures fine-grained spatio-temporal dynamics between human and object parts, addressing the limitations of coarse contact-based (e.g., chain-of-contacts) or kinematic-only representations. This enables modeling of continuous interaction dynamics (e.g., carrying an object) rather than discrete events.

2.  Leverages VLMs to translate high-level instructions and scene context into structured RMD plans, which are directly mapped to goal states and composite rewards. This eliminates labor-intensive manual reward engineering, a major bottleneck for scalable HOI.

3. Uniquely supports static, dynamic, and articulated objects, as well as long-horizon multi-task sequences (e.g., pick up → carry → place → sit). Most prior methods are limited to single-task or static-only interactions.

4.  Fills a critical gap in existing HOI datasets by providing long-horizon, context-rich interaction plans, enabling systematic evaluation of multi-task HOI synthesis.

3. Outperforms baselines across key metrics (completion rate, sub-step precision) in both single and multi-task settings. The user study further confirms superior motion naturalness and task alignment.

### Weaknesses
1. The framework only supports single-agent interactions, ignoring multi-agent collaboration (e.g., two people moving a sofa) or social dynamics—key for real-world applications like assistive robotics or collaborative environments.

2. The VLM planner may struggle with extremely complex long-horizon tasks requiring deep hierarchical planning (e.g., multi-step cooking with ingredient prep, cooking, and serving), as noted in the paper’s future work.

3. Object part decomposition depends on the VLM’s implicit judgment, with no explicit evaluation of decomposition accuracy or robustness to novel/unfamiliar objects (e.g., specialized tools).

### Questions
1. The edge weight categories (0=stationary, 1=approaching, 2=separating, 3=unstable) are heuristic—what empirical or theoretical basis supports this granularity? Could a learned weight space or more granular categories (e.g., varying speed of approach) improve interaction naturalness?

2. How does the VLM handle part decomposition for novel objects with ambiguous functional parts (e.g., a multi-purpose tool)? Is there a fallback mechanism, and how is decomposition quality evaluated?

3. For tasks more complex than those in InterPlay (e.g., "tidy the living room, wash dishes, and prepare coffee"), how does the framework scale? Would integrating chain-of-thought prompting (mentioned in future work) quantitatively improve plan coherence and task completion?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a unified physics-based HOI framework that uses a VLM (GPT-4V) to automatically construct both goal states and reward functions from a structured representation called Relative Movement Dynamics (RMD). RMD is a part-to-part bipartite graph over human and object components with discrete relative-motion labels (e.g., approach, separate, stationary ). The VLM planner outputs a sequence of RMD steps plus spatial anchors; the RL policy (PPO) then executes them with a composite task reward (human/object destination + RMD consistency) and a style prior. A new InterPlay dataset supports long-horizon static, dynamic, and articulated interactions. Experiments show large gains in completion, sub-step ratio, and precision, with ablations that isolate the value of VLMs, multi-part modeling, and the RMD features.

### Strengths
- The RMD abstraction gives the VLM a concrete “language” to plan in, and gives RL a direct mapping from that plan to goals and rewards. This avoids per-task reward tuning and makes long-horizon transitions feel principled instead of different parts being glued together. This is a meaningful step toward building more general-purpose simulated agents.
- I appreciate the completion definition (do the interaction and return to a neutral position). It addresses what usually breaks in HOI: recovery and chaining. The method’s advantage in hybrid / multi-task settings (Table 2) is meaningful.
- The Interplay dataset, including both static and dynamic objects and requiring multi-object sequences, is a welcome addition to the community.

### Weaknesses
- In the demo video opening door scene, it is actually very hard to see the agent using its hand/stump.
- The system leans heavily on GPT-4V with prompts.  What is the failure case for GPT-4V that may impact training and learning agents?
- There are two compute costs here: (i) planner inference (VLM) and (ii) training/execution. It would help to see end-to-end wall-clock and per-episode runtime (planning + control), and how that compares to baselines.
- The naturalness of the human motion ultimately leans on known motion-style priors (AMP-like discriminator), which means out-of-distribution motion may not be feasible.

### Questions
- What is the empirical error rate of the VLM planner on InterPlay, and how does task performance degrade?
- The stage transition uses a fixed and hand-picked 0.9 threshold. How sensitive are completion and precision to this value, and did you try progress-based or confidence-weighted switching that might be less brittle in cluttered or dynamic cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel framework for synthesizing physics-based human-object interactions (HOI) by leveraging Vision-Language Models (VLMs) to automatically generate goal states and reward functions. The core contribution is the Relative Movement Dynamics (RMD) representation—a fine-grained spatio-temporal bipartite graph that models part-level relationships between human and object components during interactions. This allows the VLM to reason about motion dynamics and generate semantically grounded, interaction-aware plans without manual reward engineering. The authors also contribute the InterPlay dataset, which includes long-horizon static and dynamic interaction tasks in diverse indoor scenes. Experiments in both single-task and multi-task settings demonstrate that the proposed method outperforms existing approaches in terms of completion rate, motion naturalness, and generalization.

### Strengths
1.	The introduction of Relative Movement Dynamics is a conceptually elegant and technically sound way to bridge high-level task instructions with low-level motion control. By modeling interactions as a bipartite graph of human and object parts with explicit motion trends, the method captures both spatial and temporal aspects of interaction in a unified manner.
2.	The method achieves state-of-the-art performance across multiple metrics (completion rate, sub-step precision) in both single-task and long-horizon multi-task scenarios. The improvements are especially notable in dynamic and hybrid interaction settings, where prior methods often fail.

### Weaknesses
1. The InterPlay dataset introduced in the paper includes articulated objects not found in current interaction datasets, but it does not specify what types of articulated objects are included or what kinds of interactions are involved.
2. The 6D pose of objects is crucial. How does the method prevent issues like unintended object rotation caused by inaccurate 6D pose estimation during interaction?
3. The construction of Graph B seems to be only briefly discussed in the paper, yet I believe the connectivity of this graph and the selection of edge weights would directly impact the generated motion outcomes.

### Questions
1. The paper claims the contribution of being the "first unified physics-based HOI synthesis framework leveraging the powerful world knowledge of VLMs," but it may not actually be the first work to use VLMs for physics-based HOI. This statement could be somewhat biased.

### Soundness
3

### Presentation
3

### Contribution
3
