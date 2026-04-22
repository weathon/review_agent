# Learning to Reason About Code Insecurity: Composite-Reinforcement Fine-Tuning for Cognitive Alignment

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Automated vulnerability analysis increasingly relies on language models, yet even strong LLMs exhibit unstable security reasoning: they either over-flag benign code or miss critical flaws, particularly under cross-language shifts. We present ARGO--Composite-Reinforcement Fine-Tuning for Cognitive Alignment--a label-efficient training framework that explicitly optimizes a composite reward combining (i) label-based decision scoring via a strictly proper scoring rule on predicted probabilities, (ii) explanation grounding and consistency through structure- and code-referencing heuristics that do not use CWE labels or definitions, and (iii) output-format coherence through a strict schema validator. This moves the objective from bare classification toward deliberative, auditable analysis while explicitly acknowledging and isolating the supervised component in the reward. We cast each example as a short two-phase episode: first, the policy produces an explanation; then it deterministically emits a calibrated probability through a regression head. The binary decision is deterministically derived from the probability at inference (thresholding) rather than being sampled as a separate action. Policy updates are stabilized via batch-level affinity-weighted neighborhood smoothing over deterministic encoding and a KL trust term to a reference policy. Across BIGVUL, DIVERSEVUL, and CLEANVUL, ARGO consistently improves macro-F1 over strong baselines (e.g., up to 0.71 in-distribution; substantial gains under cross-language transfer). Compared to standard supervised fine-tuning, ARGO reduces catastrophic bias toward predicting the vulnerable class and improves recognition of benign code without relying on CWE supervision.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes ARGO (Composite-Reinforcement Fine-Tuning for Cognitive Alignment), a training framework designed to make large language models (LLMs) both confidence-calibrated and explainable in the context of bug detection.

ARGO operates in two stages: first, the model analyzes the input code and generates an explanation describing the potential bug; then, it outputs a probability score indicating the confidence that a vulnerability exists.

Experimental results show that ARGO achieves higher F1 scores and better confidence calibration than baseline models on bug detection datasets such as BigVal.

### Strengths
- The problem addressed is meaningful. Bug detection is not a simple binary classification task—both the explanation and the model’s confidence are crucial. This paper aims to ensure that the model’s explanations are aligned with its detection confidence.
- The evaluation introduces detailed metrics for assessing confidence calibration and carefully considers randomness and potential data contamination.

### Weaknesses
- Although the paper aims to improve the quality of explanations in bug detection, the proposed reward only encourages the model to include *real code snippets* in its explanation. In many cases, even if the explanation refers to actual code entities, the underlying reasoning may still be incorrect.
- This paper is not well-presented. The introduction introduces many concepts without sufficient clarification, and the logical flow is fragmented, making it easy for readers to get lost.

### Questions
* Why does referencing real code entities in the explanation improve the model’s confidence?
* When the model’s reasoning in the first (explanation) stage is incorrect, how can the system correct or compensate for it?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces ARGO, a composite-reinforcement fine-tuning scheme for vulnerable-code detection that makes each sample a two-phase episode: the model first produces a structured rationale (Issue/Evidence/Mitigation) grounded in identifiers/spans, then outputs a single calibrated probability; the binary decision is a deterministic threshold on that probability. The training objective couples a proper-scoring term with label-free “grounding” and format signals (no CWE supervision or retrieval during training), aiming to reduce default-to-vulnerable bias and improve auditability and calibration. Evaluation on DIVERSEVUL, BIGVUL, and CLEANVUL compares ARGO primarily against zero-shot (±reasoning) and standard SFT under matched token/epoch budgets, reporting higher macro-F1, notably better benign recall, and improved calibration (ECE/Brier); the paper also highlights cross-language gains when training on C and testing on Java/Python/JavaScript.

### Strengths
- Composite objective with concrete, label-free signals. Proper-scoring on the probability plus grounding/format checks computed without CWE supervision or retrieval. 
- Consistent empirical improvements on the reported setups. Figures/tables show higher macro-F1 and better benign recall vs. SFT/zero-shot on BigVul/DiverseVul/CleanVul, and stronger cross-language transfer when training on C (e.g., 0.51 vs. 0.25 on Java; 0.64 vs. 0.47 on Python).

### Weaknesses
- Baselines are limited to zero-shot (±reasoning) and standard SFT under matched budgets; stronger alternatives (e.g., other calibrated training or RL-from-rationales) are not included.
- No in-depth analysis and ablation studies for designs.
- They evaluate on DIVERSEVUL, BIGVUL, and CLEANVUL at snippet/function granularity (CLEANVUL is explicitly function-level), and the paper provides no per-CWE coverage tables.
- Leveraging reasoning for vulnerable code is not very new.
- Many terms are abused in this paper, such as the "unsupervised" in format coherence reward.

### Questions
1. Can you provide experiment result of more baselines such as [1]?
2. Can you carry out ablation studies to justify each design in ARGO?

> [1] Weyssow, Martin, et al. "R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation." arXiv preprint arXiv:2504.04699 (2025).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims at using LLM to detect vulnerability. It proposes an RL training algorithm with a novel set of rewards.
While the paper is targeting at an important problem, I have significant concerns about its soundness and evaluation setups.

### Strengths
It targets an important problem with creative training objective design.

### Weaknesses
1. The experiment setup contradicts with consensus in the vulnerability detection domain. The experiment is conducted on individual functions. However, it is well established that the vulnerability of code snippet should consider coding contexts. It might not be meaningful to consider function separately when reasoning about its security.

2. The unsupervised reward design lacks rationale.
The paper proposes three reward signals for evaluating whether a natural language explanation about the give code is plausible. First, it evaluates how many variables in the code are mentioned in the explanation (more means better).
Second, it evaluates whether the sentence matches code statements. Actually, I cannot understand the reward clearly from the writing. At line 199, the function S(x) and function match(u, s) were not defined. I don't know what those functions mean.
Third, it evaluates whether the reasoning mentions "Issue", "Evidence", and "Mitigation" sections in the explanation.

From my understanding, the rationale behind those reward signals are unclear. For example, it is unclear why an explanation that mentions more variables is a better explanation about certain vulnerability. Moreover, the reward design seems very vulnerable to reward hacking. An explanation that simply repeats the code line by line will result a high reward. It is not related to the final task (i.e., vulnerability detection) at all.

The unclear rationale behind reward design significantly undermines the paper's technical contribution.

3. The RL algorithm lacks justification.
The paper proposes novel training losses to compute the advantages. It proposes a novel term named "surprise statistic". However, there is no ablation study to justify the effectiveness and improvement of the new algorithm. The contribution of such algorithm is unclear.

### Questions
Please see the above discussion.

### Soundness
1

### Presentation
1

### Contribution
1
