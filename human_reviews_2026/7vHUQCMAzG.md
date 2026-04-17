# Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Clinical decision-making is a dynamic, interactive, and cyclic process where doctors have to repeatedly decide on which clinical action to perform and consider newly uncovered information for diagnosis and treatment. Large Language Models (LLMs) have the potential to support clinicians in this process, however, most applications of LLMs in clinical decision support suffer from one of two limitations: Either they assume the unrealistic scenario of immediate availability of all patient information and do not model the interactive and iterative investigation process, or they restrict themselves to the limited "out-of-the-box" capabilities of large pre-trained models without performing task-specific training. In contrast to this, we propose to model clinical decision-making for diagnosis with a hypothesis-driven uncertainty-aware language agent, LA-CDM, that converges towards a diagnosis via repeatedly requesting and interpreting relevant tests. Using a hybrid training paradigm combining supervised and reinforcement learning, we train LA-CDM with three objectives targeting critical aspects of clinical decision-making: accurate hypothesis generation, hypothesis uncertainty estimation, and efficient decision-making. We evaluate our methodology on MIMIC-CDM, a real-world dataset covering four abdominal diseases containing various clinical tests and show the benefit of explicitly training clinical decision-making for increasing diagnostic performance and efficiency. Our code is available at https://github.com/dharouni/LA-CDM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LA-CDM, a two-agent LLM framework for clinical diagnosis that mirrors clinicians’ workflow: a hypothesis agent proposes a diagnosis with an explicit confidence, and a decision agent either orders the next test or finalizes the diagnosis. The authors train three objectives in a cyclic schedule—(1) supervised hypothesis generation, (2) RL-based confidence calibration (a betting-style reward), and (3) RL for action selection with a test-cost penalty—so the system learns to raise hypothesis confidence while minimizing unnecessary testing. Evaluated on MIMIC-CDM (2,400 patients; four abdominal diseases; labs, notes, imaging reports), LA-CDM improves mean class accuracy to 81.3% with macro-F1 81.3 and reduces average test cost to USD1,295.61 versus a zero-shot ReAct baseline at USD1,480.32; an ablation shows the cost reward cuts cost from USD1,427.85 to USD1,295.61 with similar accuracy. Calibration also improves (ECE 0.069 → 0.037).

### Strengths
1. The paper's theme is based on real-world clinical problems. The design of the two agents reflects the actual diagnostic process: hypothesis formation → targeted testing → decision-making.
2.The paper demonstrates a strong focus on cost. The R_cost term explicitly penalizes expensive tests (e.g., MRI USD4,866 vs CBC USD71), driving parsimonious testing.
3.The paper has principles for handling uncertainty. Confidence calibration via RL (betting-style reward) trains the model to express calibrated confidences without ground-truth confidence labels.
4.The paper aligns with the guidelines for patient adaptive behaviors.The trained model preferentially orders ultrasound for cholecystitis (64.9%) and CT for appendicitis (85.1%), matching clinical practice.

### Weaknesses
1.This study's evaluation is limited to four abdominal diseases based on a single retrospective dataset; it lacks broader disease coverage or external validation. It lacks generalizability.
2.The data in the paper has limitations in retrospective constraints and test availability bias. Many requested tests are unavailable in logs, forcing the agent to try alternatives; this caps exploration to clinician-observed pathways and can bias learned policies.
3.The paper preprocesses patient information.Summarization of histories and imaging reports (e.g., keeping only findings) could remove cues clinicians use, but the impact isn’t quantified.
4.The paper did not analyze the sensitivity of the cost model.Test costs are tied to a single hospital’s 2025 charge schedule; robustness to alternative cost tables or scaling is not reported.

### Questions
1.How does LA-CDM generalize to other institutions/specialties or non-abdominal conditions? Any plans for external-site validation or leave-one-hospital evaluation?
2.Does summarizing histories and trimming imaging reports change diagnostic difficulty or shift feature reliance? Please provide with/without pre-processing results.
3.Do you quantify how often “test unavailable” occurs and its effect on learning/inference? Could synthetic imputation or simulation reduce bias from missing tests?
4.How sensitive are policies to the cost table (e.g., ±x% scaling, switching to a different hospital’s charges)? Please report accuracy/cost under several cost schedules.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes **LA-CDM (Language Agents for Clinical Decision Making)** — a two-agent LLM system for **iterative, hypothesis-driven diagnosis**.  
A **Hypothesis Agent** generates provisional diagnoses with calibrated confidence, while a **Decision Agent** chooses whether to request additional tests or finalize a diagnosis.  
Training uses a **hybrid pipeline**:
1. **SFT** on labeled cases for hypothesis quality.
2. **Reinforcement Learning (GRPO)** for calibrating confidence via a “betting” reward.
3. **RL** for cost-aware decision optimization, balancing diagnostic accuracy and test cost.

The environment is based on **MIMIC-CDM**, including text notes, imaging summaries, and lab data for 2,400 cases across four abdominal conditions.  
Results show LA-CDM achieves **higher accuracy and lower average testing cost** than ReAct or untrained baselines, with **better calibration (ECE ↓ from 0.069 to 0.037)**.

### Strengths
1. **Innovative design:** Two-agent division closely reflects clinical reasoning loops.  
2. **Calibration reward:** Improves reliability of verbal confidence estimates.  
3. **Cost-sensitive optimization:** Demonstrates efficiency improvements without sacrificing accuracy.  
4. **Transparent methodology:** Prompts, cost tables, and training configs are public.  
5. **Multimodal context:** Integrates notes, labs, and imaging text within a unified environment.  
6. **Clear ablations:** Each module’s contribution is empirically validated.

### Weaknesses
1. **Limited dataset scope:** Only four conditions; no cross-domain or cross-hospital validation.  
2. **Environment artifacts:** Some requested tests are unavailable, possibly biasing learning.  
3. **Weak baselines:** Comparison methods are not fully aligned in modality or setting.  
4. **Preprocessing bias:** Summarization of clinical notes may distort reasoning cues.  
5. **Safety unaddressed:** No human-in-the-loop validation or fail-safe mechanism.  
6. **Ablation granularity:** Lacks exploration of cyclic vs. joint training and reward sensitivity.

### Questions
1. How robust is LA-CDM when scaling to new diseases or hospital systems?  
2. What is the impact of unavailable tests on exploration and policy stability?  
3. Why does cyclic training outperform joint optimization—any gradient conflict analysis?  
4. Does better hypothesis calibration improve decision timing and stop behavior?  
5. How sensitive is the cost trade-off to regional price variations?  
6. Can hard safety constraints be added to the RL objective?  
7. Have clinicians evaluated qualitative reasoning traces for plausibility?

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
3

### Summary
This paper proposes LA-CDM, a two-agent system combining hypothesis generation and decision-making modules trained with hybrid supervised and reinforcement learning for iterative clinical diagnosis.

### Strengths
1. The paper addresses an important gap in clinical AI systems by modeling the iterative nature of diagnostic reasoning.
2. The paper provides thorough comparison against multiple baselines and includes cost-efficiency metrics.
3. The paper attempts to integrate uncertainty estimation with sequential decision-making in a clinical context.
4. The paper includes detailed prompts and implementation details that facilitate understanding of the approach.

### Weaknesses
1. The paper introduces separate hypothesis and decision agents without sufficiently justifying their architectural separation. Specifically:
- The hypothesis agent generates diagnostic hypotheses and confidence scores, while the decision agent uses this output to select tests or final diagnoses. However, both agents inherently engage in diagnostic reasoning, and the decision agent could potentially internalize hypothesis generation and confidence estimation.
- The confidence score produced by the hypothesis agent could be implicitly learned by the decision agent through its reasoning process, eliminating the need for explicit separation. This raises questions about whether the dual-agent design introduces unnecessary complexity.

2. The reinforcement learning component is trained on retrospective clinical data, which poses significant off-policy learning challenges:
- Historical data reflects actions taken by clinicians, which may not align with the optimal policy learned by the agent. This can lead to distributional shift and biased policy evaluation, as the agent may propose test sequences not represented in the data.
- The paper mentions handling unavailable tests by prompting the agent to choose alternatives, but this does not address the core issue of learning from a fixed dataset with limited action coverage. Off-policy correction methods (e.g., importance sampling, conservative Q-learning) or imitation learning techniques are not discussed.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
