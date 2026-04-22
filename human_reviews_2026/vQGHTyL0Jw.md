# Doctor-R1: Mastering Clinical Inquiry with Experiential Agentic Reinforcement Learning

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6, 6

## Abstract
The professionalism of a human doctor in outpatient service depends on two core abilities: the ability to make accurate medical decisions and the medical consultation skill to conduct strategic, empathetic patient inquiry. Existing Large Language Models (LLMs) have achieved remarkable accuracy on medical decision-making benchmarks. However, they often lack the ability to conduct the strategic and empathetic consultation, which is essential for real-world clinical scenarios. To address this gap, we propose Doctor-R1, an AI doctor agent trained to master both of the capabilities by ask high-yield questions and conduct strategic multi-turn inquiry to guide decision-making. Our framework introduces three key components: a multi-agent interactive environment, a two-tiered reward architecture that separately optimizes clinical decision-making and communicative inquiry skills, and an experience repository to ground policy learning in high-quality prior trajectories. We evaluate Doctor-R1 on OpenAI's HealthBench and MAQuE, assessed across multi-facet metrics, such as communication quality, user experience, and task accuracy. Remarkably, Doctor-R1 surpasses state-of-the-art open-source specialized LLMs by a substantial margin with higher parameter efficiency and outperforms powerful proprietary models. Furthermore, the human expert evaluations show that Doctor-R1 achieves superior clinical capability and patient-centric performance, demonstrating the effectiveness of the framework.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes DOCTOR-R1, an AI agent framework designed to master both strategic multi-turn patient inquiry and medical decision-making through experiential reinforcement learning.

### Strengths
1. The paper identifies an important gap in medical AI research. The distinction between static medical knowledge assessment and dynamic clinical inquiry is clear.
2. The evaluation includes multiple dimensions: two dynamic benchmarks (HealthBench, MAQuE), static knowledge validation (MedQA, MMLU).

### Weaknesses
1. The most severe limitation is the pervasive use of LLM-as-judge throughout the system. The reward model is Qwen3-8B evaluating Qwen3-8B policy outputs. This creates a closed system where the system optimizes for LLM evaluator preferences rather than genuine clinical competence.
2. The experience retrieval mechanism offers minimal technical innovation. Stage 1 (embedding similarity + reward weighting) and Stage 2 (cross-encoder reranking) are standard retrieval techniques. Are retrieved experiences genuinely instructive, or does the system learn spurious correlations?
3. There are some Human Evaluation Limitations. No evaluation of diagnostic accuracy by medical professionals. Both primary benchmarks (HealthBench, MAQuE) use automated LLM evaluation, creating potential bias toward response styles favored by specific LLM families.
4. No comparison with supervised fine-tuning on high-quality human-authored dialogues. Given the complexity of the RL pipeline, demonstrating that RL provides advantages over simpler approaches is essential.
5. The author acknowledges the "key challenge" of patient agent adherence but provides insufficient evidence that the mitigation strategies are effective. What percentage of simulated dialogues are filtered out due to adherence failures?
6. The multi-objective reward structure, while helpful, does not eliminate the possibility of exploitation. Human spot-checking is mentioned but no statistics on frequency or findings are provided.

### Questions
1. Can you provide quantitative evidence that reward model judgments correlate with physician assessments? What is the inter-rater agreement between the LLM judge and medical experts?
2. What percentage of training dialogues are filtered due to patient agent adherence failures?
3. How does supervised fine-tuning on high-quality human dialogues compare to your RL approach?
4. Under what conditions does the system produce unsafe recommendations despite the veto mechanism?
5. What types of experiences does the system most frequently retrieve? Are there cases where novelty filtering excludes valuable high-quality examples?
6. How does performance degrade when evaluated on out-of-distribution cases (different patient populations, communication styles, or medical domains)?

### Soundness
2

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
This paper proposes an RL-based agent, namely Doctor-R1, for strategic inquiry and empathetic communication. Evaluating on HealthBench and MAQuE datasets, the proposed framework outperforms open-source and commercial general models. Human evaluation shows the Doctor-R1 has improvements over baselines.

### Strengths
(1) This paper aligns with the trend to extend the medical LLMs from static decision-making to dynamic inquiry through an agentic RL perspective.
(2) This paper explicitly models empathy and communication quality as trainable objectives.

### Weaknesses
(1) The technical details are missing. The authors need to provide more technical details. For example the authors need to define how vector rewards are aggregated or how gradient signals are propagated.
(2) The authors rely on the GRPO framework for the RL. Ablation studies need to be provided on how different RL frameworks will affect the performance.
(3) The authors use Qwen3-8B as the base model. Is this framework transferable to other base models?

### Questions
(1) Now the framework is offline and off-policy. Is it possible to change the framework to make it online?

### Soundness
4

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
5

### Summary
The paper proposes Doctor-R1, a clinical dialogue agent trained to conduct strategic, empathetic multi-turn inquiry while also making sound medical decisions. The system employs three key concepts: a multi-agent interactive environment, a two-tiered reward structure that distinguishes between outcome quality and process quality, and an experience repository that stores valuable trajectories for reuse. The model is evaluated on HealthBench and MAQuE with automatic metrics and human preference studies. The claim is that Doctor-R1 improves communication quality and user experience while maintaining competitiveness or surpassing it in task accuracy.

### Strengths
1. Most medical LLM work measures final answers. This paper addresses the gap in the skill of guided clinical inquiry and communication, which is crucial in practice.
2. The two-tiered reward isolates process signals from outcome signals, which is a sensible way to avoid reward dilution and to target conversational quality.
3. The experience module that learns from high-quality trajectories is a practical way to ground policy learning without heavy annotation.
4. The study reports multi-facet metrics and human judgments, not only accuracy.

### Weaknesses
1. The models used across experiments are not consistent. For example, the human evaluation uses GPT-5, but the main tables and ablations only report GPT-4.1. GPT-5 is also not mentioned in the Evaluation Settings section. Could you unify the model choices across experiments? If different models need to be used, please provide a brief explanation for the selection.

2. The authors state: “The annotators recruited were without a specialized medical background to ensure that metrics like Clarity and Empathy were evaluated from the perspective of a typical patient, for whom the agent is ultimately designed.” However, the paper sets out three research questions: (2) Empathetic Communication is only one, whereas (1) Strategic and Dynamic Inquiry and (3) Learning from Good Experience requires a medical background to assess. For instance, at each turn-level Doctor-R1 response, expert evaluation can better verify whether the questions asked are clinically appropriate; experts can also check whether the Experience Module’s stored items and retrieved content are indeed appropriate and helpful. I do not think these two research questions can be avoided in the human evaluation; focusing only on non-medical-background evaluation of (2) Empathetic Communication is insufficient.

3. An additional analysis would be to break down time and compute overhead by component. For example, after adding the “Learning from Good Experience” design, how much extra latency or token usage does the model incur?

4. The ablation study is missing analyses on how different reward designs affect results. A core claim of the paper is integrating a TWO-TIERED REWARD into RL training. Beyond the outcome reward, many additional rewards are proposed, and the method section includes empirically motivated statements such as “This (e.g.,  hierarchical penalty design) addresses the limitations of conventional weighted-sum models, which can fail to adequately penalize catastrophic errors due to an averaging-out effect.” It would be beneficial to include more recent related work, for example, how other studies attempt to incorporate additional process rewards on top of RLVR while maintaining training stability, or at least present supportive mini-experiments. Also, the paper only presents a coarse “w/o Process Reward” setting. Please consider ablating some of the eight process rewards to toggle them on or off, especially the key ones, to assess whether such designs can generalize to other medical scenarios.

### Questions
1. Why is Qwen3-8B also used as the LLM judge for the Process Reward?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose DOCTOR‑R1, an 8B-parameter doctor agent trained with an Experiential Agentic RL framework: (1) Interactive environment: a POMDP with a Doctor Agent (policy), an LLM‑based Patient Agent, and a Consultation Evaluator acting as the reward function; (2) Optimization: GRPO trains the policy to prefer higher‑reward responses within sampled groups.

### Strengths
1. Principled safety‑aware reward design that decouples soft‑skills and hard‑skills during training.  
2. Experience‑guided control via reward‑aware and novelty‑aware retrieval that measurably helps inquiry quality.
3. Strong qualitative demonstration on a truly high‑stakes case that aligns with clinical intuition (quantify bleeding, TB history, anticoagulants, urgent referral).

### Weaknesses
1. The policy, patient simulator, and reward judge are all Qwen3‑8B family members, raising systemic bias and overfitting risks to the evaluator’s rubric and the simulator’s discourse style 
2. the method contribution seems to be trivial, like previous rl works in huatuo-o1, FineMedLM-o1, m1 leverage rl and test-time scaling works like medprm and medadapter for medical reasoning. however in the related works, seems agentic rl for medical reasoning is not emphasized enough
3. The retrieval prepends prior “good experiences,” which could partly function as a prompting prior rather than learned policy alone. More careful controls (e.g., retrieval without actions vs. with actions; random but format‑matched snippets) would help isolate the effect. 
4. The final diagnostic reward is coarse (0/0.5/1), potentially limiting sensitivity to near‑misses or safe-but-incomplete triage decisions

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents DOCTOR-R1, an 8B parameter AI doctor agent. The authors argue that SOTA models fail at "dynamic clinical inquiry" (i.e., strategic, empathetic questioning) even if they ace static medical exams.

To fix this, they propose an "Experiential Agentic Reinforcement Learning" framework. The system is a multi-agent loop: a Qwen3-8B (Doctor Agent) interacts with a Qwen3-8B (Patient Agent). Its actions are scored by a third Qwen3-8B (Consultation Evaluator).

This evaluator (reward model) uses a "two-tiered" system. It gives a "Process Reward" (for soft skills like empathy, safety) at each turn and an "Outcome Reward" (for diagnostic accuracy) at the end. The agent also uses an "Experience Repository" to retrieve high-quality past interactions to guide its policy.

The authors evaluate DOCTOR-R1 on the HealthBench and MAQUE benchmarks. They report that their 8B model significantly outperforms its 8B base model, larger 70B specialized models, and even proprietary models like GPT-4.1.

### Strengths
The paper identifies a critical gap in medical AI evaluation. It moves beyond static knowledge benchmarks to address the challenge of dynamic, real-time clinical inquiry. The focus on training an agent for this process is a logical research direction. Combining RL with an experience-retrieval mechanism is a sound approach. It allows the agent to learn from a curated set of high-quality past trajectories.

The "two-tiered" reward model is a sensible design. The inclusion of a "hierarchical veto system" (Sec 3.2) to penalize critical safety failures is a necessary feature, preventing the agent from optimizing for "soft skills" like empathy at the cost of patient safety.

The core result—an 8B model beating 70B specialized models and GPT-4.1 on complex inquiry tasks—is a very strong claim. It suggests a path toward highly capable, parameter-efficient medical agents.

### Weaknesses
The evaluation methodology has a circularity concern - using the same Consultation Evaluator for both training rewards and test evaluation may inflate performance metrics.

All three components (agent, patient, evaluator) use the Qwen3-8B model family, creating a self-referential system that may optimize for model-specific patterns rather than generalizable clinical skills.

The Consultation Evaluator's ability to assess complex metrics like "Empathy" and "Reasoning Quality" lacks validation against medical expert judgment or inter-rater reliability studies.

The eight process reward criteria need better justification - prior work exists that could ground these choices (e.g., arXiv:2502.14860, 2502.07143). Including "Completeness" as a turn-level metric seems inconsistent since completeness requires full conversation context.
The 100k simulated patients lack validation for clinical realism, potentially creating a gap between benchmark performance and real medical consultations.

The experience retrieval mechanism yields modest improvements (2-3%) over simpler baselines, raising questions about cost-benefit tradeoffs.

Human evaluation would benefit from medical professional involvement to assess clinical appropriateness beyond lay judgments of empathy and clarity.

Key hyperparameters ($\pi$, $\lambda$, reward weights) are presented without ablation studies or sensitivity analysis to understand their impact.

### Questions
Minor comments:

Section 2 feels repetitive to read, for example, "the policy model is the doctor agent" appears twice in the section.

L211: typo "polify"

### Soundness
3

### Presentation
3

### Contribution
3
