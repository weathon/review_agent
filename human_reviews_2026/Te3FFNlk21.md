# CoSMo-RL: Towards Trustworthy LMRMs via Joint Safety and Stability

- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Large Multimodal Reasoning Models (LMRMs) are moving into real applications, where they must be both useful and safe. Safety is especially challenging in multimodal settings: images and text can be combined to bypass guardrails, and single-objective training can cause policy drift that yields over-refusal on benign inputs or unsafe compliance on risky ones. We present CoSMo-RL, a mixed rein-
forcement learning framework that trains reasoning-oriented LMRMs under multimodal, multitask, and multiobjective signals, and we release the resulting model, CoSMo-R1. Our approach aims to let safety and capability grow together in one stable pipeline rather than competing during alignment. In experiments, CoSMo-R1 improves safety while maintaining—and often improving—multimodal reasoning and instruction following, shows stronger robustness to multimodal jailbreaks, and reduces unnecessary refusals. The framework also transfers across backbones with consistent gains. Ablations support the design choices, indicating a simple path to advancing safety and general capability together in LMRMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents CoSMo-RL, a reinforcement learning framework aimed at training Large Multimodal Reasoning Models (LMRMs) that are both capable and safe. The framework introduces a two-stage RL pipeline:

* Stage 1 builds general reasoning and instruction-following ability.

* Stage 2 jointly optimizes safety, value alignment, and general reasoning using a multi-objective reward.

Training stability is enforced via Clipped Policy Gradient with Policy Drift (CPGD), and robustness is enhanced through multimodal jailbreak data augmentation. The resulting model, CoSMo-R1, demonstrates improved safety and reasoning capabilities compared to strong baselines such as GPT-4.1, Claude Opus 4, and Qwen2.5-VL-72B. Evaluations across safety, moral reasoning, and multimodal reasoning benchmarks show consistent gains.

### Strengths
* Unified optimization of safety and reasoning: CoSMo-RL integrates safety, value, and reasoning into a single RL framework, moving beyond the traditional post-hoc safety fine-tuning used in prior works.

* Stable and practical training design: The two-stage RL process and Clipped Policy Gradient with Policy Drift (CPGD) maintain stability while preventing reward hacking and mode collapse.

* Exposure-based robustness: Training directly on multimodal jailbreak data (text + image) improves resistance to real-world attacks instead of relying on synthetic evaluations only.

### Weaknesses
* Utility drop: Instruction-following performance (IF-Eval) falls from 86.3 % → 74.9 %, indicating a significant decrease in helpfulness as safety increases.

* No baseline + defense experiments: The study only compares model-level results; it does not test baseline reasoning models combined with defense modules.

* Over-safety behavior persists: CoSMo-R1 occasionally refuses benign prompts, even with the Helpful reward added. Over-refusal and excessive caution remain unsolved and may hurt usability in real-world deployment

### Questions
See the Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents CoSMo-RL, a reinforcement learning framework for training LMRM that jointly optimizes safety and reasoning capability. The authors argue that safety should not be treated as an afterthought but co-evolved with general capabilities during training. They release CoSMo-R1, a model trained using this framework based on Qwen2.5-VL-72B.

### Strengths
Originality: Addresses timely safety degradation in multimodal reasoning models. Provides reasonable integration of staged RL, multiobjective rewards, and adversarial augmentation, though individual components are established techniques.
Quality: Evaluation across 10+ benchmarks and four model variants shows relatively consistent improvements competitive with GPT-4 and Claude. Ablations provide useful insights, but lack statistical rigor and reproducibility details.
Clarity: Well-structured with clear motivation and effective figures. Dense presentation sacrifices precision, with critical details deferred to appendix or omitted.
Significance:  Addresses real deployment needs and demonstrates safety-capability co-improvement is feasible. Incremental contribution with practical value if artifacts are released to enable community follow-up work.

### Weaknesses
1) Limited Baseline Comparisons
Absence of safety-specific baselines: The paper omits comparisons with recent safety-focused alignment methods, such as Safe RLHF-V [1] or LLaVA-RLHF [2], under the same evaluation protocol. Including these baselines would provide a clearer context for the contributions.
2）Overclaimed Contributions
The paper claims to present a "unified" framework for co-evolving safety and capability, but this framing overlooks prior work on multi-objective RLHF such as SafeRLHF-V [1], that similarly balances multiple objectives.
3) Cross-Model Results Lack Depth
Tables 6-8 show results on Qwen-7B, InternVL, and DeepSeek, but different benchmarks are used across models (e.g., HarmBench only for DeepSeek), making cross-model comparison impossible.

[1] Ji J, Chen X, Pan R, et al. Safe RLHF-V: Safe Reinforcement Learning from Multi-modal Human Feedback[J]. arXiv preprint arXiv:2503.17682, 2025.
[2] Sun Z, Shen S, Cao S, et al. Aligning large multimodal models with factually augmented rlhf[J]. arXiv preprint arXiv:2309.14525, 2023.

### Questions
1. Typo: "??" in Line 301 and 371
2. No explanation about "safety rate" in line 300
3. The paper states Stage 2 "jointly optimizes safety, value, and general capability" but doesn't specify. Are these trained on separate data batches or mixed? What's the sampling ratio across task types? How is catastrophic forgetting of Stage 1 capabilities prevented?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to address the prevalent "safety tax" problem in Large Multimodal Reasoning Models (LMRMs), where enhancing model safety (e.g., through alignment) often leads to a decline in reasoning capabilities or over-refusal on benign prompts. The authors propose a hybrid Reinforcement Learning (RL) framework named COSMO-RL. Its core idea is the co-evolution of safety and general capabilities, rather than treating safety alignment as a post-hoc step.

The main components of this framework include:
1.  **Two-stage RL Training:** Stage 1 focuses on building broad general reasoning capabilities; Stage 2 then jointly optimizes for safety and value alignment while maintaining general capabilities.
2.  **CPGD Optimization Algorithm:** Utilizes the CPGD (Clipped Policy Gradient Optimization with Policy Drift) algorithm to ensure the stability of RL training.
3.  **Multi-objective Reward Function:** A unified, four-part reward function is designed, including $R_{visual-focus}$ (visual focus), $R_{helpful}$ (helpfulness), $R_{format}$ (format), and $R_{task-aware}$ (task-awareness).
4.  **Specialized Outcome Reward Models (ORMs):** To implement $R_{task-aware}$, the framework relies on three independently trained reward models: a Safety ORM, a Value ORM, and a Knowledge ORM.

The model trained through this framework, CoSMo-R1 (based on Qwen2.5-VL-72B), demonstrates stronger safety, a lower unnecessary refusal rate, and enhanced multimodal reasoning capabilities in experiments.

### Strengths
1.  **Importance of the Problem:** This paper addresses a very important and timely challenge in the LMRM domain. As model capabilities increase, ensuring they remain safe and controllable without sacrificing their core reasoning abilities is a critical issue. Jailbreak risks specific to multimodality also make this problem more challenging.
2.  **Comprehensive Framework Design:** The authors propose an ambitious and systematic solution. COSMO-RL is a complex, end-to-end process that integrates CoT-SFT, two-stage RL, a stable optimizer, and a sophisticated multi-objective reward mechanism, demonstrating the authors' thorough consideration in tackling this problem.
3.  **Significant Improvements on Some Metrics:** The experimental results show that CoSMo-R1's safety rate on multiple safety benchmarks (e.g., MM-SafetyBench and SIUO) far exceeds that of the baseline models. Concurrently, on general reasoning benchmarks (e.g., Olympiad and GPQA Diamond), CoSMo-R1 also demonstrates stronger performance than the baseline Qwen2.5-VL-72B.

### Weaknesses
1.  **Core Claim Contradicts Data; "Safety Tax" Persists:** The paper's core claim is that it "resolves" the trade-off between safety and capability, asserting that CoSMo-R1 "maintains or even enhances" capability. However, the authors state at the end of Section 4.2.3 that CoSMo-R1 achieves only 74.9% on the instruction-following benchmark IF-Eval, whereas its baseline model (Qwen2.5-VL-72B) scores 86.3%. This is a **significant performance drop of 11.4%**. The authors describe this as "no significant drop," which is a severe misinterpretation of the data. This indicates the "safety tax" has not disappeared but has **merely been shifted from "reasoning capability" to "instruction-following capability,"** which is equally detrimental for building a useful assistant.
2.  **Method is Extremely Complex and Difficult to Reproduce:** The reproduction cost of this framework is extremely high. It not only requires a complex two-stage RL pipeline and the CPGD algorithm but also **relies on three independently trained, powerful reward models** (Safety ORM, Value ORM, Knowledge ORM). These ORMs are complex models in themselves (e.g., the Value ORM is built on Qwen2.5-VL-72B). This complexity makes the framework difficult to adopt widely and is far from the "a simple path" claimed in the abstract.
3.  **Lack of Transparency and Replication Details for Core Components (ORMs):** The success of the COSMO-RL framework relies heavily on three independent "Outcome Reward Models" (Safety ORM, Value ORM, Knowledge ORM), which collectively form the $R_{task-aware}$ reward signal. However, the training and deployment details for these three critical models are severely lacking:
    * **Opaque Datasets:** The paper briefly describes the data used to train the ORMs in Appendix A.1.1, such as "80k bilingual multimodal samples" for the Value ORM and "120k multimodal knowledge questions" for the Knowledge ORM. However, these datasets are not released, and their construction processes (e.g., "expert curation" and "closed-loop process") are vague. This makes it impossible for reviewers to assess potential biases within these ORMs.
    * **Missing Training Details:** The paper mentions the base models for the ORMs (e.g., Qwen2.5-VL-7B and Qwen2.5-VL-72B) but fails to provide sufficient details on training hyperparameters, architectural modifications, or validation procedures. For example, the base model for the Knowledge ORM is not even explicitly mentioned.

### Questions
1.  Given that the success of the COSMO-RL framework is highly dependent on three opaque ORMs, can the authors provide more detailed ablation studies to isolate the impact of the ORM quality? For example, if simpler, more transparent reward models (e.g., keyword detection or GPT-4-based API calls) were used for the $R_{task-aware}$ signal, would the COSMO-RL two-stage training framework still be effective?

2.  Given that the framework requires training three additional large reward models (e.g., the Value ORM is based on a 72B model), can the authors detail the total computational cost required to reproduce this framework (e.g., compared to training only the baseline SFT model)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The topic of this paper is the large multimodal reasoning model. The authors train CoSMo-R1 via CosMo-RL, a mixed RL framework that trains the models under multimodal, multitask, and multiobjective signals. The experiments show the performance and the safety the proposed model.

### Strengths
1. The motivation is clear, and the topic is practical. Training a better and safer LMRM is significant. 
2. The principles in the intro. part are interesting and can be used in other domains. 
3. The experiments and analyses, e.g., case studies, are comprehensive.

### Weaknesses
1. The four principles in the introduction part are interesting. Could the authors provide more insights and experimental evidence?
2. Some important details are missing in the main text. I recommend that the author move the detailed calculation of the different rewards to the main text. Besides, could the authors provide more details about the Safety ORM in Table 5?
3. In Table 1, the results of DeepSeek's and Kimi's models are missing. In Table 8, the results of OpenAI's, Google's, and Anthropic's models are missing.
4. The source code and training data are missing. Will the authors release them?
5. The performance of the proposed model under the jailbreak attacks is missing, e.g., AutoDAN, ReNeLLM, FlipAttack.  
6. Minor: missing discussion on [1] in the related work part. The notation table is missing. In Line 301, the number of the table is missing. In Line 370, the number of section is missing ('??'). 

[1] GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
