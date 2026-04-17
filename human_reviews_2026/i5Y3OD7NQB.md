# UI2Code$^N$: A Visual Language Model for Test-Time Scalable Interactive UI-to-Code Generation

- Decision: Reject
- Scores: 2, 6, 2

## Abstract
User interface (UI) programming is a core yet highly complex part of modern software development. Recent advances in visual language models (VLMs) highlight the potential of automatic UI coding, but current approaches face two key limitations: multimodal coding capabilities remain underdeveloped, and single-turn paradigms make little use of iterative visual feedback. 
We address these challenges with an interactive UI-to-code paradigm that better reflects real-world workflows and raises the upper bound of achievable performance. Under this paradigm, we present UI2Code$^N$, a visual language foundation model trained through staged pretraining, fine-tuning, and reinforcement learning to achieve foundational improvements in multimodal coding. The model unifies three key capabilities: UI-to-code generation, UI editing, and UI polishing. 
We further explore test-time scaling for interactive generation, enabling systematic use of multi-turn feedback. Experiments on UI-to-code and UI polishing benchmarks show that UI2Code$^N$ establishes a new state of the art among open-source models and achieves performance comparable to leading closed-source models such as Claude-4-Sonnet and GPT-5. Both the model and code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes an interactive paradigm for UI-to-code which contains three capabilities: (1) UI-to-code draft generation, (2) UI polishing using target screenshot + draft code + rendered output, and (3) instruction-driven UI editing. The proposed model, UI2CodeN (9B), is trained via a three-stage pipeline: continual pretraining on large-scale crawled webpage pairs, supervised fine-tuning to seed the three tasks, and RL with a verifier and round-robin comparator reward. The authors argue this enables test-time scaling: iteratively polishing improves fidelity (they report ~12% gain with four rounds).

Empirically, UI2CodeN achieves strong results across Design2Code, Flame-React-Eval, Web2Code, and two new benchmarks (UI2Code-Real, UIPolish-bench), according to VLM evaluators.

### Strengths
- The paper introduces an end-to-end training recipe containing three distinct stages, continual pretraining + SFT + RL, to align VLMs with multimodal UI code generation.
- UI2CodeN achieves new state of the art among open VLMs on multiple benchmarks; competitive vs. top closed models according to VLM evaluators. Evaluation results exhibit test-time scaling capabilities.
- Contributes two new benchmarks for UI2Code generation: UIPolish-bench and UI2Code-Real.

### Weaknesses
- Heavy reliance on VLM-based judges for reward signals and evaluation, which can introduce biases and reward gaming risks. While the paper has shown efforts in calibrating rewards with comparator + round-robin, there's no evaluation on whether the GLM-4.5V based reward mechanism aligns with human judgments, or whether using different VLM evaluators provides consistent judgments. 

- All benchmarks are evaluated solely via VLM evaluators (o4-mini for UI-to-code and Gemini-2.5-Pro for UI polishing), without reporting any human evaluation or the original, non-VLM metrics for the cited benchmarks, making the validity and significance of the reported performance highly questionable.

- In reward design, GLM-4.5V is finetuned with SFT to improve robustness, but the paper does not disclose the training procedure, data curation, or any metric or evidence to support the claimed robustness.

- Missing citation for relevant literature “Sketch2code: Evaluating vision-language models for interactive web design prototyping”, which proposed a similar iterative UI-to-code generation framework.

- Data governance: ~10M crawled UI-code pairs seeded from Common Crawl; the paper does not detail licensing, PII filtering, or copyrighted asset handling.

- The comparator + round-robin scheme incurs O(N^2) VLM calls per rollout, casting doubt on the scalability of the proposed RL approach.

- The claim UI Editing capabilities are shown only through demos, without formal evaluations.


While the paper proposed an ambitious approach to train interactive UI-to-Code reasoning models at scale with test-time scaling, the authors need to revise the soundness and transparency of their method and evaluation protocol in order for this work to be accepted by the ICLR venue.

### Questions
- The paper reported using o4-mini for UI-to-code benchmarks but gemini-2.5-pro for UI polishing benchmarks. Any justification for why these two specific VLMs are chosen for the two evaluation tasks?
- How sensitive are results to evaluator choice and prompt wording? Do different VLM judges produce consistent/calibrated outputs? Please report results with at least one independent open judge and a human study (N≥50) on Design2Code-HARD.
- Did you test RL-trained models using a held-out verifier never seen during RL (e.g., DINOv3-based metric, IoU-based block/element matching, human preferences)? Any failure cases where visual similarity rises but DOM/typography regress?
- Can the authors perform ablation studies on the continual pretraining? Given a strong base model, would SFT + RL alone give competitive performance even without the pretraining stage?
- Do the authors observe any oscillations or quality regressions across the rounds?
- Can the authors report any quantitative or qualitative evaluations for UI editing capabilities of UI2CodeN against strong baselines?
- Please detail crawl sources, robots/ToS compliance, PII filtering, and license auditing; do the authors plan to release any of the curated datasets?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce UI2Code$^N$, a 9B-parameter open-source visual language model (VLM) tailored for interactive UI-to-code generation. Unlike traditional one-shot models, UI2Code$^N$ operates under a novel Interactive UI-to-Code paradigm encompassing three tasks: UI-to-code generation, UI polishing, and UI editing. The model is trained through a multi-stage process: large-scale pretraining on noisy web data, supervised fine-tuning on curated HTML datasets, and reinforcement learning (RL) with VLM-based reward functions. Experiments across benchmarks (Design2Code, Flame, Web2Code) and new curated datasets (UI2Code-Real, UIPolish-bench) demonstrate state-of-the-art performance for open-source models, with performance rivaling or surpassing leading closed-source models such as GPT-5 and Gemini-2.5-Pro. Notably, the model achieves up to 94% accuracy on synthetic UI polishing and demonstrates significant gains with test-time scaling (up to 12% improvement after 5 interaction rounds).

### Strengths
1. **Novel paradigm for interactive UI coding.** The proposed interactive paradigm realistically models the iterative design process of developers, introducing three interlinked capabilities (UI-to-code, polishing, editing) that improve practicality and performance.
2. **Comprehensive multi-stage training pipeline.** The model benefits from a carefully designed training strategy combining real-world pretraining, high-quality fine-tuning, and reinforcement learning with a sophisticated VLM-based reward design.
3. **Strong empirical results with new benchmarks.** The model outperforms prior open-source models across all tasks and matches or outperforms closed-source ones.

### Weaknesses
1. **Missing comparison with recent agent-based or modular systems.** While agent-style approaches (e.g., DECLARUI, DCGen, ScreenCoder) are mentioned in Sec 2. related work, direct comparisons with these systems are absent, which could provide insights into trade-offs between agent complexity and VLM-based simplicity.
2. **Lack of human evaluation.** All evaluations are based on VLM scoring or CLIP metrics. Human evaluations could validate whether the improvements translate into actual perceived quality and usability, especially in edge cases where automated scores may be misleading.
3. **Lack of UI editing evaluation.** Although UI editing is listed as one of the three core capabilities (UI-to-code generation / UI editing/ UI polishing). The paper lacks a detailed evaluation section or results table explicitly analyzing editing compared to other models, except for Appendix A.5 qualitative examples.

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces UI2CodeN for an interactive UI-to-code paradigm, which includes test-time scaling through multi-round polishing. Experiments show strong results over several VLMs. However, the main techniques are modest, such as standard prompt design and standard staged training, and the evaluation relies heavily on VLM-based scoring without human judgments or variance-aware reporting for this creative task.

### Strengths
1.	Clear problem framing that recasts UI-to-code as an interactive, multi-turn process with UI-to-code, polishing, and editing, which matches real workflows.
2.	Comprehensive training recipe with pretraining on web data, supervised fine-tuning with think/answer formatting, and RL with a comparator-style verifier.

### Weaknesses
1.	Missing crucial methodological details, including how the reward is computed and how the training loss function is designed. Detailed staged-training strategies are missing.
2.	The main techniques rely on standard pre-training, SFT, and RL process without designed training framework, a purpose-built loss or principled reward shaping objective for UI code fidelity, showing limited contributions.
3.	The evaluator’s reliability is not measured. Evaluation is heavily reliant on VLM-based scoring, and there is no human study for fidelity and any variance-aware reporting. The conclusions are not convincing, as this is a creative and subjective task.
4.	Limited comparisons of UI2Code relevant baselines. Experiments focus on vanilla VLMs without UI2Code advanced baselines, such as Uicoder [1] and ScreenAI [2], making it hard to attribute gains to the proposed paradigm.
5.	Limited ablation analysis. For the impact of each stage (pretraining vs SFT vs RL), the think/answer format, and the round-robin comparator, which components actually drive the improvements, and by how much?

[1] Uicoder: Finetuning large language models to generate user interface code through automated feedback. NAACL 2024.
[2] ScreenAI: A visual language model for UI and visually-situated. IJCAI 2024.

### Questions
1.	See above.
2.	It would be clearer to present the experimental settings in the Experiments section rather than in the Methods section.

### Soundness
2

### Presentation
2

### Contribution
2
