# Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Text-to-motion generation has advanced with diffusion models, yet existing systems often collapse complex multi-action prompts into a single embedding, leading to omissions, reordering, or unnatural transitions. In this work, we shift perspective by introducing a principled definition of an event as the smallest semantically self-contained action or state change in a text prompt that can be temporally aligned with a motion segment. Building on this definition, we pro- pose Event-T2M, a diffusion-based framework that decomposes prompts into events, encodes each with a motion-aware retrieval model, and integrates them through event-based cross-attention in Conformer blocks. Existing benchmarks mix simple and multi-event prompts, making it unclear whether models that succeed on single actions generalize to multi-action cases. To address this, we con- struct HumanML3D-E, the first benchmark stratified by event count. Experiments on HumanML3D, KIT-ML, and HumanML3D-E show that Event-T2M matches state-of-the-art baselines on standard tests while outperforming them as event complexity increases. Human studies validate the plausibility of our event definition, the reliability of HumanML3D-E, and the superiority of Event-T2M in generating multi-event motions that preserve order and naturalness close to ground- truth. These results establish event-level conditioning as a generalizable principle for advancing text-to-motion generation beyond single-action prompts. Code and data are available at https://tjswodud.github.io/EventT2M.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Event-T2M—a diffusion framework that decomposes prompts into multi-events, encodes them with a TMR encoder, and integrates them via event-based cross-attention in Conformer blocks. The authors also build HumanML3D-E, a benchmark stratified by event count. Experiments show Event-T2M matches SOTA on standard tests (HumanML3D, KIT-ML) and outperforms baselines as event complexity rises.

### Strengths
The paper solves multi-action prompt mishandling via a principled "event" definition and Event-T2M (with event decomposition, TMR encoding, and ECA module), avoiding action issues like omissions. Authors also build HumanML3D-E, the first event-count-stratified benchmark, fixing existing benchmark gaps. This paper also provides solid experiments (matching SOTA on HumanML3D/KIT-ML, outperforming baselines on complex HumanML3D-E) and user studies validating event definition, benchmark reliability, and motion quality.

### Weaknesses
The ablation analysis in this paper is limited in scope. Although multiple new modules are proposed (e.g., LIMM, ATII, Conformer, ECA), the experiments focus solely on the ECA module and the text encoder, failing to assess the necessity and individual impact of the other introduced components.

### Questions
1. In Equations (3) and (7), a coefficient of 0.5 is applied to the residual term. It remains unclear why this specific coefficient was chosen instead of 1, and the rationale behind this design choice warrants further explanation.
2. The efficiency analysis appears to overlook the computational overhead introduced by the Large Language Model (LLM). While the baseline model (e.g., Momask) does not employ an LLM for text segmentation, the proposed Event-T2M model utilizes an LLM to partition the input text into events. The time cost associated with this LLM processing stage should be accounted for in the overall efficiency evaluation.

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
5

### Summary
This work introduces Event-T2M, a diffusion-based framework that decomposes complex text prompts into semantically self-contained events and generates motion through event-based cross-attention. It also builds HumanML3D-E, the first benchmark stratified by event count, and demonstrates that Event-T2M maintains state-of-the-art performance while significantly improving motion coherence and naturalness for multi-event prompts.

### Strengths
1.Proposes an event-based paradigm for motion generation.

2.Constructs the first event-level motion generation dataset.

### Weaknesses
1.Does event-driven motion generation offer advantages over action-driven or hybrid (action + event) methods?
2.Does the proposed method outperform approaches that enhance motion quality through motion retrieval?
3.In TMR, innovation based solely on input differences does not constitute true novelty.
4.LIMM, ATII, and ECA follow common module design patterns and lack sufficient originality.

### Questions
See weaknesses.

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
5

### Summary
This paper addresses a key challenge in complex Text-to-Motion generation: the difficulty of existing models in handling prompts with multiple sub-motions, which often leads to omissions, merging, or reordering of motions. To solve this, the authors propose Event-T2M, a diffusion-based framework. The core idea is to first introduce a definition of an "event" as the smallest, semantically self-contained action unit in a text. They then use a Large Language Model (LLM) to decompose the input text into a sequence of these "event" clauses. These clauses are subsequently encoded by a TMR encoder (trained for motion-text alignment) into "event tokens." Finally, these event tokens are injected into a Conformer-based diffusion model via a novel "Event-based Cross-attention" (ECA) module to guide the generation of the motion sequence. Furthermore, to specifically evaluate the model's ability to handle complex prompts, the authors construct a new benchmark, HumanML3D-E, which stratifies the HumanML3D test set by the number of events in the text. Experimental results show that Event-T2M achieves comparable performance to SOTA on standard benchmarks (HumanML3D, KIT-ML) but significantly outperforms baselines on the new HumanML3D-E benchmark, especially as event complexity increases.

### Strengths
- The problem significance is huge. Generating complex and consistent human motions is an unsolved challenge in the T2M field.
- This paper proposes a novel benchmark called HumanML3D-E. This is the first benchmark stratified by the "event complexity" of the prompts. It provides a very valuable evaluation tool for future research on long and complex T2M generation field.
- The idea of decompose the complex motions is very intuitive and logical.

### Weaknesses
- **Unfair Comparison**: The authors' new benchmark, HumanML3D-E, is constructed using an LLM and a specific "event-aware prompt." However, the proposed model, Event-T2M, **also relies on the exact same LLM and the exact same prompt** in its data preprocessing stage. Event-T2M is evaluated on a test set that is perfectly aligned with its own training and inference pipeline. In contrast, all baseline models are evaluated without using this LLM-based event decomposition preprocessing. This constitutes an extremely unfair comparison. The poor performance of the baselines on HumanML3D-E is likely just an artifact of their input representation (e.g., CLIP-based word tokens) being mismatched with the benchmark's construction (LLM-based clauses), not because their architectures inherently fail at complexity.

- **Limited Technical Novelty**: I must point out that there are already some methods trying to solve the generation of the long motions using LLM. For instance, the recent ATOM[1] framework uses GPT-4 to construct event-level prompts and GPT-4V as an AI reward model to fine-tune a generator, specifically targeting event-level alignment (integrity, temporal order, and frequency). Additionally, InstructMotion[2] explicitly uses an LLM to generate long prompts, subsequently using Reinforcement Learning (RL) to fine-tune an autoregressive motion generator. **Worryingly, these highly relevant prior works, which also tackle event-level or complex alignment using LLMs, are not cited or discussed in the paper's Related Works.**

[1] Han H, Wu X, Liao H, et al. Atom: Aligning text-to-motion model at event-level with gpt-4vision reward[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 22746-22755.
[2] Mao, Yunyao, et al. "Learning generalizable human motion generator with reinforcement learning." arXiv preprint arXiv:2405.15541 (2024).

### Questions
- Can you evaluate your model on a test set that was not constructed using your LLM pipeline? On a complex motion test set where events were manually segmented and temporally aligned by human annotators, would Event-T2M still show an advantage over baselines?
- For a fair comparison, if you were to replace the CLIP encoder in the baseline models (like AttT2M or MoMask) with the TMR encoder (and use TMR's word-level tokens as K/V), how much would their performance on HumanML3D-E improve? As I know, replacing the CLIP encoder with TMR encoder will significantly improve the performance.
- What are the differences between Event-T2M and other motion generators using LLM for decomposition?

**I will raise the score to a positive mark if you can address my concerns regarding the "unfair comparison"** (mainly concerning the experimental setup for the TMR encoder and the aspects mentioned in the Weaknesses section).

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an event-level text-to-motion generation benchmark and a stronger baseline for the task. To explicitly model the complexity of a target motion, the paper proposed to utilize an LLM to split the motion text prompts into event-level actions, where a larger number of events indicates harder cases. For a stronger baseline of the proposed task, an event-based cross-attention module is injected into the diffusion-based motion generation framework to improve the performance. Experimental results on the benchmark dataset validate the effectiveness of the proposed methods.

### Strengths
- The point of modeling the motion complexity by the number of events is straightforward and reasonable. The proposed HumanML3D-E benchmark will be beneficial to the community, which can evaluate motion generation frameworks on more detailed levels of complexity.
- The experimental analysis of different methods on different event counts supports the motivation of the proposed event-based benchmark. 
- The design of the event-based cross-attention module is reasonable and validated by ablation studies.
- The paper is well-written and easy to follow.

### Weaknesses
- The events of a motion are divided by an LLM with text input only. The label may contain errors. Manually validating the labels or sampling cases to check the accuracy rate of the LLM labels will be beneficial.
- The paper misses some comparisons with some recent stronger baselines, e.g., MoGenTS (NeurIPS 2024), MARDM (CVPR 2025), and LAMP (ICLR 2025). 
- The event-based benchmark only contains one dataset, HumanML3D. It's better to add more datasets, e.g., KIT-ML, Motion-X, to better validate the generalizability of different methods.
- Providing some failure cases of the proposed framework and previous work on complex scenarios, e.g., 4 events, will be beneficial.

### Questions
See the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
