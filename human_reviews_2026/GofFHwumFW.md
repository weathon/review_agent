# Unary Feedback as Observation: Incentivizing Self-Reflection in Large Language Models via Multi-Turn RL

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Large Language Models (LLMs) are increasingly deployed as agents that solve problems through multi-turn interaction, receiving feedback and refining their reasoning based on users' feedback. However, existing reinforcement learning with verifiable reward (RLVR) methods train them under a single-turn paradigm. As a result, we discovered that models often **fail to explore alternative reasoning paths or reflect on prior mistakes, producing repetitive and unadapted responses to feedback.**

To address this gap, we propose Unary Feedback as Observation (UFO), a framework that conditions policy updates on minimal unary feedback (e.g., “Let’s try again”) after incorrect answers. UFO is simple, compatible with existing single-turn RL setups, and incentivizes self-reflection. To further promote efficient and adaptive reasoning, we design reward structures that encourage _minimality_ (solving in fewer turns) and _diversity_ (exploring alternatives under failure). Experiments show that UFO preserves single-turn performance while improving multi-turn reasoning accuracy by about 14%. Crucially, UFO-trained models also **generalize beyond their training domain, transferring effectively to out-of-domain tasks** across mathematics, STEM, QA, and general knowledge, showing that **UFO teaches models self-reflective reasoning that carry over across domains**. Beyond these empirical gains, UFO points toward a broader paradigm for building adaptive reasoning agents: one that scales supervision from static datasets, reduces dependence on costly domain-specific feedback, and lays the foundation for more general, self-improving AI systems in open-ended real-world settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates Large Language Model (LLM) reasoning and Reinforcement Learning (RL) fine-tuning in a multi-turn interactive setting. It addresses a key limitation of traditional RL with Verifiable Reward (RLVR), which uses a single-turn paradigm. This traditional approach often causes models to fail to explore alternative reasoning paths or reflect on prior mistakes, leading to repetitive and unadapted responses to feedback.
To address this, the authors propose Unary Feedback as Observation (UFO), a framework that conditions policy updates on minimal unary feedback (e.g., Let’s try again) following incorrect answers.
Experiments demonstrate that UFO achieves superior performance and exhibits cross-domain generalization ability.

### Strengths
The work successfully identifies the importance of multi-turn interaction for LLM agents.
It shows that RL training under a multi-turn setting effectively incentivizes diversity in reasoning paths, thereby improving final performance.

### Weaknesses
1. The motivation for considering multi-turn interaction to encourage exploration and revision is sound. However, the paper's current setting seems limited to minimal unary feedback. It's unclear how this approach generalizes or applies to more natural multi-turn dense feedback scenarios (e.g., detailed human labels or code debugging feedback), as alluded to in paragraph 136. This limits the scope of the proposed method's applicability.
2. The algorithmic novelty appears limited. The approach essentially trains with multi-step environmental feedback without a clearly defined adaptive algorithm design. The distinction between the proposed method and previous single-step PPO/GRPO algorithms that also trained with repeated answer generation is not sufficiently clear
3. Regarding the mathematical reasoning problem, specifically:

	How is the end of a single generation step determined in this multi-turn setting?

	What is the fundamental difference between UFO and previous PPO/GRPO work, beyond the multiple interactions (turns) and the use of more data examples?

### Questions
1.	I am confused about Figure 1. What is the testing task used to generate this figure, and what is the specific definition of the effective answer ratio metric? A clearer explanation of why traditional RL training methods like PPO and GRPO show a performance drop compared to the original model is needed, perhaps with an earlier  and detailed explanation of the experiments would be better.

2.	In Equation 1, the formulation for the LLM generating repeated responses is introduced. To accurately model this repetition, shouldn't the LLM's input be modified to include its previous outputs? Specifically, should the formulation be adapted to something like $q(y|x) \times q(y|x, y_{prev})$, where $y_{prev}$ represents the previous response?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes UFO (Unary Feedback as Observation), a multi-turn reinforcement learning framework that enables LLMs to learn self-reflective reasoning from minimal feedback (e.g., “Try again”) on static single-turn datasets. UFO treats past attempts and unary feedback as part of the observation state, trains with PPO, and introduces reward decay and repetition penalties.

### Strengths
1. A simple and effective method to push LLMs learn self-reflective reasoning.

2. The reward decay and repetition penalties are introduced to encourage minimality (solving quickly) and diversity (avoiding repeated errors).

### Weaknesses
- The paper contains several errors that need correction. For example, Figure 7 is not cited in the paper, and the term “Multiturn” on line 462 should be hyphenated as “Multi-turn.”
- Since the authors provide the model with a prompt that includes prior attempts and feedback, it is difficult to disentangle whether performance gains come from true multi-turn interaction or simply from a richer prompt signal.
- **An Important question:** The experimental results suggest that models trained with multi-turn RL appear to require multi-turn evaluation to achieve improved performance. Does this imply that the model’s intrinsic reasoning capability—under single-turn evaluation—has not actually improved, and that richer prompt signals are still necessary to activate its parametric knowledge?
- Section 2.3 should introduce and cover the content of Figure 1.
- In Figure 2, what exactly is meant by "effective answer ratio"? Does it refer to answers that are both correct and derived via diverse reasoning paths? Which models were used in this analysis? How was the metric computed? Does the drop after RL imply that the model’s overall capability degrades?
- In Table 1, what is the "hotQA" dataset used for Qwen-3B? Why was this experiment conducted only on Qwen-3B and not on other models?
- Why did the authors choose PPO as the base RL algorithm instead of alternatives like GRPO or DAPO?
- As shown in Table 1, models trained on math datasets (e.g., MMQ-Math) generalize well to QA tasks, but models trained on HotpotQA perform poorly on math tasks. What explains this asymmetry?
- Why are hyperparameters (e.g., T_max, N) kept identical across models of different sizes? Shouldn’t larger models potentially benefit from different settings (e.g., more rollouts or longer interaction horizons)?

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

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
This paper identifies a critical limitation in current reinforcement learning with verifiable reward (RLVR) methods for large language models (LLMs): they are trained under a single-turn paradigm, which suppresses adaptive and self-reflective reasoning in multi-turn interactions. To address this, the authors propose Unary Feedback as Observation (UFO) — a simple yet effective framework that enables multi-turn reinforcement learning by conditioning policy updates on minimal unary feedback such as “Try again.” Experiments across multiple LLMs and nine benchmarks show that UFO improves multi-turn reasoning success

### Strengths
1.	Demonstrates that self-reflective reasoning transfers across domains and architectures.
2.	Introduces reward-shaping principles (minimality/diversity) that improve both efficiency and adaptability.

### Weaknesses
1.	Unary feedback (“Try again”) is idealized; real human feedback can be more ambiguous or inconsistent.
2.	The sensitivity to decay factor γ and repetition penalty λ could be analyzed more deeply.

### Questions
1.	Could this unary feedback mechanism be extended to graded feedback (e.g., “close,” “partially correct”)?
2.	What are the computational costs compared to single-turn RL — is training efficiency affected?

### Soundness
3

### Presentation
2

### Contribution
2
