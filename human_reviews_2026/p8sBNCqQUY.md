# MedMMV: A Controllable Multimodal Multi-Agent Framework for Reliable and Verifiable Clinical Reasoning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Recent progress in multimodal large language models (MLLMs) has demonstrated promising performance on medical benchmarks and in preliminary trials as clinical assistants. Yet, our pilot audit of diagnostic cases uncovers a critical failure mode: instability in early evidence interpretation precedes hallucination, creating branching reasoning trajectories that cascade into globally inconsistent conclusions. This highlights the need for clinical reasoning agents that constrain stochasticity and hallucination while producing auditable decision flows. We introduce MedMMV, a controllable multimodal multi-agent framework for reliable and verifiable clinical reasoning. MedMMV stabilizes reasoning through diversified short rollouts, grounds intermediate steps in a structured evidence graph under the supervision of a Hallucination Detector, and aggregates candidate paths with a Combined Uncertainty scorer. On six medical benchmarks, MedMMV improves accuracy by up to 12.7% and, more critically, demonstrates superior reliability. Blind physician evaluations confirm that MedMMV substantially increases reasoning truthfulness without sacrificing informational content. By controlling instability through a verifiable, multi-agent process, our framework provides a robust path toward deploying trustworthy AI systems in high-stakes domains like clinical decision support.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present a controllable multimodal multi-agent framework for reliable and verifiable clinical reasoning called MedMMV. Their approach is based on empirical evidence from experiments, which they perform to demonstrate limitations of direct prompting chain-of-thought methods. Empirical results from their pilot studies demonstrate that high variability in direct prompting chain-of-thought approaches for various tasks, and hallucination from misinterpretation of facts at the beginning of trajectories, lead to unstable paths and unreliable results with majority voting approaches.

Following this analysis, their proposed approach consists of three main steps:

- Short initial rollouts, which are subsequently self-refined based on evidence graphs and hallucination detectors.
- Use of MLLM and web search to build evidence graphs using prior text and images, if available from the initial input.
- An uncertainty scorer which scores each final trajectory using a weighted sum of various metrics.

Their results demonstrate that MedMMV outperforms chain-of-thought and other medical agent approaches on 6 medical reasoning datasets benchmarks. Furthermore, various ablation studies demonstrate the importance of the various components of their design choices.

### Strengths
The paper proposes an interesting approach, notably the use of evidence graphs to self-refine reasoning paths and hallucination detectors to prevent hallucinations. The use of an uncertainty scorer is also interesting, as it can be applied in downstream approaches to validate the agent’s diagnosis only if the uncertainty is below a certain threshold.

### Weaknesses
- The presentation of the results in the preliminary analysis on the limitations of direct chain-of-thought approaches could be simplified or improved to make the message easier to understand (Figure 2). Furthermore, the description of the various metrics used there, notably RGM and CMHR, could be clarified to facilitate understanding.

- On most datasets, the improvement in accuracy compared to CoT is less than 5%, while the token usage of MedMMV and other agent-based methods is approximately nine times higher than that of CoT methods. This suggests that the proposed approach might be too computationally expensive for the modest performance gains achieved.

### Questions
- What motivated the choice of datasets? Among the baselines used, one of the most *reliable* approaches is **MDAgents** — I say reliable because it was published at NeurIPS. Why not use similar datasets to make comparisons simpler? Furthermore, I noticed that the accuracy for **MedQA**, which overlaps with the datasets used in the MDAgents paper, is lower than what was reported there. Is this due to the use of a smaller sample size?
- Were any measures taken to verify that the datasets used are not part of the training data for **GPT-5**, which serves as the base model for all the agents?
- Why do you only use a single rollout for the chain-of-thought methods as a comparison? Wouldn't a fairer comparison be to use the same number of trajectories as in **MedMMV**, with majority voting?

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
3

### Summary
The paper presents MedMMV, a framework designed to produce more truthful and stable responses from large language models in high-stakes domains such as healthcare prediction. Motivated by the challenge of hallucinations found in current LLMs, the authors propose a three-step approach: (i) generating multiple candidate solutions, (ii) iteratively refining and verifying them for truthfulness using web-based evidence while constructing an evidence graph, and (iii) evaluating all candidate solutions across several LLMs using multiple metrics, selecting the one with the highest aggregated score. Extensive experiments on multiple datasets demonstrate that MedMMV consistently outperforms several strong baselines.

### Strengths
The paper is well written and easy to follow. It addresses an important and timely problem that deserves further investigation. The motivation is well grounded, supported by both quantitative and qualitative analyses of current models in a specific high-stakes setting. The experimental evaluation is extensive and robust, covering multiple datasets and strong baselines. While the multi-step review process itself is not entirely novel, the proposed mechanism for selecting the best answer among candidates is original and well justified.

### Weaknesses
- Many figures are overly dense and difficult to interpret, which limits their readability and the clarity of the results.
- The framework relies heavily on LLMs, both in the evaluation process and in the preliminary study. While the inclusion of expert validation is appreciated, it would strengthen the paper to directly compare the expert assessments with the LLM-based evaluations to assess their alignment.
- Although the authors acknowledge the higher computational cost introduced by the proposed multi-step process, it would be valuable to include a quantitative analysis of the additional time or number of generated tokens required.

### Questions
In addition to the points discussed in the Weaknesses section, I have the following question:
- What would happen if the CU scoring mechanism were used to filter or rank the base LLM responses directly? Would this alone improve their performance, or is the full multi-step process necessary to achieve the reported gains?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes MedMMV, a multi-stage framework designed to enhance the reliability and verifiability of multimodal medical reasoning in large language models. By combining uncertainty-aware path generation, evidence-grounded refinement, and structured decision aggregation, MedMMV produces diagnostic outputs that are both logically coherent and empirically auditable. A key innovation lies in its construction of a multimodal evidence graph, synthesized by specialized agents across textual, visual, and external knowledge modalities.

### Strengths
1. Clear Motivation and Design: the paper addresses the challenge of hallucination and unverifiable reasoning in medical LLMs, motivating a shift from single-shot inference to multi-path verification.

2. Novel Multi-Stage Framework: MedMMV introduces a structured pipeline combining uncertainty-aware path generation, supervised refinement, and evidence-based scoring. The use of a multimodal evidence graph constructed by specialized agents is a key innovation.

3. Comprehensive Evaluation of Reasoning Faithfulness: the empirical study covers six medical QA benchmarks—three multimodal and three text-only—demonstrating consistent improvements in diagnostic accuracy, factual consistency, and evidence-groundedness over competitive baselines. Beyond accuracy, the evaluation incorporates hallucination rate and evidence coverage metrics, offering a more rigorous assessment of reasoning reliability.

### Weaknesses
1. Inference-Time Efficiency: while the multi-path framework improves reasoning robustness, the paper lacks analysis of inference-time efficiency, which may hinder its deployment in time-sensitive or resource-constrained clinical settings.

2. Limited Ablation Coverage: the ablation study should extend to additional multimodal and text-only benchmarks to better substantiate the generality and robustness of the proposed components across task types.

### Questions
Please see my weakness

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
MedMMV is a controllable multimodal multi-agent framework for clinical reasoning that targets the observed failure mode “instability → hallucination.” It generates diversified short rollouts, grounds and verifies each path against a structured evidence graph via a hallucination/consistency supervisor, and selects the final diagnosis using a Combined Uncertainty scorer. Across six benchmarks and multiple backbones, it improves accuracy (up to +12.7%) and boosts truthfulness without reducing informativeness; a blind physician study corroborates gains.

### Strengths
- Clear identification and quantification of instability preceding hallucination in multimodal clinical reasoning.
- Controllable, auditable pipeline: diversified hypotheses, evidence-graph grounding, iterative repair, and uncertainty-based aggregation.
- Consistent gains across datasets/backbones; physician study supports higher truthfulness.
- Solid ablations isolating main contributors (CU scorer, hallucination detector).

### Weaknesses
- Heavy reliance on LLM judges/evaluators for TRUE/INFO and CU sub-scores risks bias and circularity; limited human IRR details.
- Evidence graph quality is a potential single point of failure; robustness to noisy extraction/provenance underexplored.
- Higher compute/latency and API costs; scalability guidance limited.
- Heuristic CU weighting and limited sensitivity analyses.
- Evaluation on representative subsets; limited fairness analysis (context windows, image preprocessing) and contamination controls.

### Questions
- CU scorer: how were weights chosen; sensitivity to weights, rollout counts, and repair iterations?
- Evaluator independence: do results hold with different supervisor/evaluator backbones; add blinded human TRUE/INFO with inter-rater reliability?
- Evidence graph: how do you detect/resolve extraction conflicts and control web source quality (domain, recency, dedup)?
- Scalability/fairness: detailed compute/latency per stage; do gains persist under tighter context and lower image resolution; any contamination audits?

### Soundness
2

### Presentation
1

### Contribution
2
