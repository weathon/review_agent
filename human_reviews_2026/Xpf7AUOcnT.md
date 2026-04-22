# Answer, Refuse, or Guess? Investigating Risk-Aware Decision Making in Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Language models (LMs) are increasingly used to build agents that can act autonomously to achieve goals.
During this automatic process, agents need to take a series of actions, some of which might lead to severe consequences if incorrect actions are taken.
Therefore, such agents must sometimes defer—refusing to act when their confidence is insufficient—to avoid the potential cost of incorrect actions.
Because the severity of consequences varies across applications, the tendency to defer should also vary: in low-risk settings agents should answer more freely, while in high-risk settings their decisions should be more conservative.
We study this “answer-or-defer” problem with an evaluation framework that systematically varies human-specified risk structures—rewards and penalties for correct answers, incorrect answers, and refusals $(r_{\mathrm{cor}},r_{\mathrm{inc}}, r_{\mathrm{ref}})$—while keeping tasks fixed.
This design evaluates LMs’ risk-aware decision policies by measuring their ability to maximize expected reward. Across multiple datasets and models, we identify flaws in their decision policies: LMs tend to over-answer in high-risk settings and over-defer in low-risk settings.
After analyzing the potential cause of such flaws, we find that a simple skill-decomposition method, which isolates the independent skills required for answer-or-defer decision making, can consistently improve LMs’ decision policies.
Our results highlight the current limitations of LMs in risk-conditioned decision making and provide practical guidance for deploying more reliable LM-based agents across applications of varying risk levels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies risk-aware decision making in language models, focusing on whether LMs can adapt their “answer vs. refuse” policy under different risk structures.
The authors propose an evaluation framework that varies the reward for correct answers, penalties for errors, and payoff for refusal (r_{cor}, r_{inc}, r_{ref}) while keeping tasks constant.
Experiments on multiple-choice datasets (MMLU, MedQA, GPQA) reveal that models often over-answer in high-risk settings and over-refuse in low-risk settings.
To mitigate this, they decompose the task into three skills — task solving, confidence estimation, and expected-value reasoning — and test different prompting strategies.

### Strengths
- The evaluation setup is clearly defined and reproducible.
- Experiments covers several models and datasets.
- Clear and interpretable finding, may provide insight

### Weaknesses
- All experiments are limited to multiple-choice questions; the framework does not test open-ended, tool-using, or multi-step real-world tasks, which are more practical task and can provide practical insight.
- The main finding ("prompt chaining works better than single-pass prompting") is more of a prompt engineering heuristic than a scientific insight.
- The notion of "over-answer" or "over-refuse" is based on a theoretical optimal threshold that assumes perfect calibration. Many observed behaviors may instead reflect **calibration** or accuracy issues rather than genuine policy misalignment.

### Questions
- How would this framework generalize to free-form or multi-turn decision-making tasks?

### Soundness
2

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
3

### Summary
The paper investigates LLM's decision-making under a specific, quantifiable risk structure defined by rewards and penalties for correct answers, incorrect answers, and refusal. The authors identify a critical failure pattern: LLMs fail to apply expected-value reasoning to the task, even when the reward parameters are explicitly provided. The authors further confirm that LLMs possess this reasoning skill in isolation through a gambling experiment, concluding this demonstrates a 'knowing-doing gap'.

To address this gap, the authors propose a "skill-decomposition" method, primarily "prompt chaining," which explicitly instructs the LLM to first solve the problem, estimate its confidence, and then make a risk-aware decision. This modular prompting strategy is shown to significantly improve the LLM's ability to maximize expected rewards. Experiments demonstrate gains across MMLU, MedQA, and GPQA, particularly in high-risk settings.

### Strengths
The presentation is clear and follows a logical narrative. It first defines a risk framework, identifies a failure in applying expected-value reasoning, uses a "pure gambling" experiment to confirm this is a compositional 'knowing-doing gap', and then proposes a structured prompting method to fix the gap.

The paper provides thoughtful experiment design, including a multi-prompt evaluation using three prompt variations to rule out prompt sensitivity, ablations on reasoning vs. non-reasoning models, and an analysis of the impact of confidence calibration.

### Weaknesses
The risk structure framework is somewhat artificial and offers limited generalizability. The evaluation is confined to single-turn, multiple-choice questions using artificially created risk parameters $(r_{cor}, r_{inc}, r_{ref})$. This setup is disconnected from the risk-awareness problem many frontier LLMs are trained to handle (https://arxiv.org/html/2510.14276v1), which involves assessing risk based on the prompt's context and the potential real-world impact of its actions. It would be more interesting to evaluate decisions based on these more realistic, contextual risk implications.

The proposed "prompt chaining" method also lacks generalizability. The paper makes an insightful observation about the "knowing-doing gap" in an LLM's application of expected-value reasoning. However, the proposed solution feels very specific and applies only to this particular, quantitative risk structure for multiple-choice tasks. To improve the impact of this work, I suggest authors explore more generalizable (e.g. to more variety of risk-aware decision making problems) prompting or post-training methods to bridge this kind of knowing-doing gap during decision making

### Questions
The paper mentioned “The code is attached as a .zip file in the submission”, but does not seem to be available on OpenReview. Could the authors please provide the code?

Appendix H shows that better confidence estimation (calibration) improves final scores. This conclusion is based on an experiment comparing two specific prompt variants for the confidence-estimation step. I'm curious if authors have experimented with any other prompt variations for this step? I am wondering how robust this conclusion is and whether it holds across different ways of eliciting confidence.

### Soundness
2

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
This paper introduces a framework to evaluate language models (LMs) in risk-aware decision-making scenarios, where models can choose to answer or refuse based on a specified reward structure (correct, incorrect, refusal). Experiments across multiple datasets and LMs show that models often over-answer in high-risk settings and over-refuse in low-risk ones. The authors propose a skill-decomposition method (via prompt chaining) that guides LMs through problem solving, confidence estimation, and expected-value reasoning, improving outcomes.

### Strengths
* The paper tackles a critical aspect of LM reliability which is relevant for real-world deployment.

* The reward structure is formalized clearly and enables systematic testing of LM decision behavior.

* The empirical observation that LMs can perform expected-value reasoning in isolation but fail to apply it in mixed settings is important.

* Prompt chaining is a simple, effective strategy to improve reward maximization under risk. Results show meaningful gains in high-risk settings.

### Weaknesses
1. All tasks are four-option multiple-choice questions. The framework’s applicability to open-ended, real-world tasks is untested.

2. The refusal reward is always set to 0. In practice, refusal often carries non-zero cost (latency/missed opportunity), which may alter optimal strategies.

3. No statistical significance reporting: Results lack variance or confidence intervals, which reduces interpretability of small differences across methods.

4. Incremental novelty: The main techniques of prompt decomposition and risk prompting are adaptations of known strategies. The paper does not introduce new modeling or training methods.

### Questions
1. How would results change if refusal incurred a cost (e.g., r_ref = –1)? Would LMs still over-defer in low-risk settings?

2. Have you compared your decomposition method to simpler alternatives like using softmax confidence thresholds or calibrated logits?

3. How sensitive is performance to the wording of prompts or to different chain-of-thought styles? Could improved prompting alone close the gap?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the problem whether LLMs can make risk-aware decision based on a given risk structure (rewards for correct answers, penalties for errors, and payoff for refusal). The authors introduce an evaluation framework that systematically varies risk settings while keeping tasks fixed, allowing them to isolate LLMs’ decision-making policies from their raw task-solving ability. Several interesting findings are given from an evaluation of several LLM families.

### Strengths
This paper aims to address a critical problem in LLM application: how to make LLMs risk-aware in decision making. It proposes an evaluation framework to investigate the impact of varied risk structures for constant tasks. The study validates the effectiveness of this approach on multiple-choice question tasks, finding that LLMs tend to over-answer in high-risk scenarios while demonstrating excessive deferral in low-risk environments. A skill decomposition-based prompt optimization strategy is proposed to address these issues, which is simple, effective, and does not require model retraining.

### Weaknesses
**Limited Task**: The study is restricted to multiple-choice questions. It remains unclear whether the findings generalize to open-ended generation or real-world interactive tasks.

**Simplified Risk Structure**: The risk settings are static and pre-defined. Real-world risk can be dynamic and context-dependent. It is unclear whether the method generalize to more complex settings.

### Questions
**Generalization beyond multiple-choice**: How would you expect the results to transfer to free-form generation or real-world decision-making tasks?

**Dynamic risk adaptation**: Have the authors considered scenarios where the risk structure changes during interaction? Can LMs adapt to dynamic risk settings? Despite the settings given in the paper, further more complex risk structures are welcome for validation. 

**Human-in-the-Loop scenarios**: In high-stakes applications, should LMs always defer to humans when uncertain? How does the framework incorporate human oversight?

**Long-term decision behavior**:  The work mainly focuses on single-step decision task. How might these findings extend to multi-step planning agents scenarios that must reason about risk over a sequence of actions?

### Soundness
3

### Presentation
3

### Contribution
2
