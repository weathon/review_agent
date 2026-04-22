# Exploring the Secondary Risks of Large Language Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
Ensuring the safety and alignment of Large Language Models (LLMs) is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks—a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives (excessive response and speculative advice) from the perspective of illustrative examples and information theory, which capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary-risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. With SecLens, one can automatically uncover secondary-risk vulnerabilities that are prevalent across different models. Extensive experiments on 16 popular LLMs using SecLens demonstrate that secondary risks are not only widespread but also transferable across models. Moreover, our exploration of common agent environments reveals that such risks are pervasive in practice. These findings underscore the urgent need for strengthened safety mechanisms to address benign yet harmful behaviors of LLMs in real-world deployments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper considers the problem of benign prompts causing harmful outputs in LLMs. The authors call such failure modes of a model secondary risks and categorize them into two categories: (1) excessive response and (2) speculative advice. They motivate this categorization using information theory. The paper further introduces a procedure, called SecLens, based on genetic search, for finding prompts that elicit such behavior while remaining 'benign' and 'task-relevant'. Experiments show that SecLens produces prompts that more often lead to harmful outputs than several baselines on text-only and multimodal LLMs.

### Strengths
- Important Problem: The problem of models outputting harmful responses to benign prompts is an important topic that merits more research.
- Extensive evaluation on different model types: The paper considers text-only, multimodal, and agentic systems in their experiments.
- Good presentation: Overall, the paper is easy to follow and provides clear details on the proposed method.

### Weaknesses
- Missing evaluation of prompt benignness: From the current reported results, it is not possible to assess how benign (and task-related) the resulting prompts actually are. The authors need to report metrics that quantify how benign and task-related the resulting prompts are; otherwise, it is hard to compare the experiments meaningfully. Additionally, it would be useful to report a representative subselection of the prompts in the appendix (or main text) to demonstrate that these are indeed benign and task-related.
- Limited value of information theory framing: The section on information theory does not add much value in my opinion. The grouping and matching to information theory seems plainly intuitive and is not used throughout the rest of the paper.
- Questionable adversarial framing: The authors frame the secondary risks from an adversarial attack perspective. I'm not sure this is a useful way of considering this problem and wonder whether it would be better to consider it as a (robust) performance issue. That framing would also map more closely to literature that studies hallucinations and other model misbehaviors, which seems to relate more closely to the studied problem.
- Minor comments:
  - The formula in Eq (5) is a bit strange: Why is there min on the LHS but none on the RHS?
  - After Eq (7), R is defined as "the unified risk score", but what is this? (From the appendix, it appears that this is an LLM-as-judge but prompted in a way unrelated to the two risk categories.)
  - Please include error quantification in the results (LLMs can be quite random in boundary cases). Also, why is the rounding different in Table 5?

### Questions
- Why do the authors frame the problem from an attack perspective rather than a performance perspective?
- Does the information theoretic perspective add something to the problem beyond intuition?

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
4

### Summary
This paper aims to introduce, define, and evaluate an LLM failure mode that the authors term "Secondary Risks." Unlike "jailbreak" attacks, which arise from malicious prompts, secondary risks are defined as harmful or misleading behaviors generated by an LLM during benign, non-adversarial user interactions. The authors further categorize these risks into two "risk primitives": Excessive response (where the model appends biased or over-generalized content beyond fulfilling the request) and Speculative advice (where the model deviates from user intent to provide unsafe or excessive recommendations).
To systematically elicit these risks, the paper proposes SecLens, a black-box, multi-objective search framework. SecLens formulates prompt discovery as an optimization problem, using an evolutionary algorithm to balance three objectives: task relevance ($\text{TASKSCORE}$), risk activation ($R(\cdot)$), and linguistic plausibility (penalized via a stealthiness score, $\text{DETECTSCORE}$). The authors conduct experiments across 16 popular LLMs, Multimodal Models (MLLMs), and Agents, claiming that SecLens outperforms baselines like random sampling and MCTS in eliciting secondary risks, demonstrating that these risks are prevalent and transferable.

### Strengths
•⁠  ⁠Importance of the Problem: The issue this paper addresses—non-adversarial failures of LLMs in benign interactions is an important and relatively under-explored area of safety. Distinguishing it from the more heavily studied field of "jailbreak" attacks has practical significance.
•⁠  ⁠Methodological Framework: Structuring risk elicitation as a multi-objective optimization problem (balancing risk, task, and stealthiness) is a reasonable and sound approach.
•⁠  ⁠Breadth of Experiments: The authors' attempt to evaluate their method on a wide range of targets, including text-only LLMs, MLLMs, and GUI-based Agents, is ambitious.

### Weaknesses
Despite the importance of its topic, the paper suffers from significant flaws in its methodology, evaluation, and the positioning of its contributions, which collectively undermine the credibility of its conclusions.
Circular Reasoning in Evaluation
The paper's experimental design contains a fundamental circularity. The primary evaluation metric (Attack Success Rate, ASR) is "template-based LLM evaluation using GPT-4o." However, the fitness functions that the SecLens framework itself relies on during the search—including the risk score $R(\cdot)$, task score $\text{TASKSCORE}$, and stealthiness score $\text{DETECTSCORE}$—are also computed by prompting an "expert evaluator" LLM (as shown in Appendix B).
This means: researchers are using one LLM (as a fitness function) to guide a search to "trick" a target LLM, and then using another LLM (GPT-4o) to judge whether the "trick" was successful.
This "LLM-evaluating-LLM" closed loop makes the objectivity of the results highly questionable. We cannot determine whether SecLens's high success rate is due to its discovery of genuine, human-harmful systemic vulnerabilities in the models, or if it is simply adept at "overfitting" to the preferences and blind spots of the evaluator LLM (GPT-4o).
Problem Aggravated by Synthetic Data
This first flaw is further exacerbated by the dataset's origin. The seed dataset used to guide the search, SecRiskBench (Appendices D, E), was also generated by LLMs (GPT-4 and GPT-3.5) and filtered by GPT-4 as a "meta-evaluator."
The entire experimental pipeline is: using an LLM-generated dataset to launch an LLM-evaluated search process to attack a target LLM, with the results judged by yet another LLM. This makes the entire research effort highly endogenous and insular, and it is highly uncertain whether its findings generalize to real-world human interactions.
Insufficient Evidence for SecLens's Effectiveness
The effectiveness of SecLens, the paper's core methodological contribution, is not adequately substantiated.
•⁠  ⁠Insufficient Baseline Comparison: SecLens is primarily compared against random sampling and MCTS. While it outperforms MCTS (as shown in Table 2), the improvement is not significant in several cases, which is insufficient to justify the introduction of a complex, evolutionary algorithm-based framework.
•⁠  ⁠Missing Key Ablation Studies: SecLens contains multiple components, particularly "semantic-guided variation strategies" (crossover and mutation). The paper only presents an ablation study for "few-shot context guidance" (Figure 2) but provides no ablation whatsoever for its core evolutionary operators. We do not know if these complex semantic operations are genuinely more effective than MCTS (with the same few-shot guidance) or simpler mutation strategies. This severely weakens the contribution of SecLens as a novel search framework.

### Questions
On Circular Evaluation: How can the authors demonstrate that their main results (especially in Table 2 and Table 5) are not merely an artifact of the "LLM-evaluating-LLM" loop? Why wasn't the human verification and cosine similarity (used only for MLLMs in Appendix F) used as a primary evaluation metric for all experiments (especially for the core LLM experiments)?
On SecLens's Effectiveness: Can the authors provide an ablation study that compares the effectiveness of SecLens and MCTS in terms of, for example, total tokens consumed?
On Risk Classification: The boundary between "Excessive response" and "Speculative advice" appears blurry. For example, why is the biased conclusion regarding race and gender in Figure 1(a) classified as an "Excessive response" rather than harmful "Speculative advice"? Can the authors provide a clearer, operational definition to distinguish between the two?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the concept of secondary risks, namely non-adversarial, harmful behaviors of large language models (LLMs) that arise during benign interactions. 
The authors argue that these risks differ from jailbreaks and other adversarial attacks because they emerge without malicious intent or explicit circumvent guardrails. Then, the authors distinguish between two cases: excessive response (outputs that exceed user intent and introduce bias or misinformation) and speculative advice (unsafe or overconfident suggestions).
To identify and evaluate these secondary risks, the paper proposes SecLens, a black-box, multi-objective evolutionary search framework that automatically discovers such risky behaviors. It is evaluated across a broad range of LLMs, including multimodal and agentic environments. Experiments show that secondary risks are widespread, transferable, and modality-independent.

### Strengths
- The paper is well structured and clearly written. All necessary implementation details are available in the main text or appendix. Generally, comprehensive appendices that include datasets, prompts, and evaluation details ensure reproducibility.
- The paper makes a novel and timely contribution by formalizing a neglected class of non-adversarial LLM failure modes and providing an automated method to elicit and quantify them. The concept of secondary risks provides a useful abstraction for analyzing safety failures beyond jailbreak scenarios.
- The SecLens framework is technically sound and practically relevant, particularly because it applies to closed-source models. 
- Evaluations in agentic environments are an important addition, showing that subtle misalignments extend beyond text generation. The theoretical grounding and large-scale empirical validation together make this work a significant and impactful contribution to LLM safety research.
- Evaluation does not solely rely on llm-as-judge assessment and further involves human verification. However, the additional metric results are only reported in appendix. Ablation studies support design choices (e.g., few-shot contextual guidance).

### Weaknesses
- Results from the human verification study should be incorporated into the main text and discussed to highlight cases where LLMs-as-judges fail to capture subtle safety violations. I would also suggest moving the MLLM evaluation to the appendix. While the MLLM analysis provides a useful bridge toward the subsequent agentic experiments, it offers only limited additional insights on its own. Relocating it would create space to include qualitative examples from the main experiments, showcasing both successful attacks and safe, non-harmful behaviors for the two secondary risk types, which would substantially enhance interpretability and reader understanding.
- Some claims (e.g., “standard safety filters” in line 73) are vague and should be substantiated. Furthermore, in line 86, the statement “In contrast to prior adversarial search methods that assume gradient access or safety API introspection” is somewhat misleading, as existing work already explores black-box settings. For instance, Ge et al. MART: Improving LLM Safety with Multi-round Automatic Red-Teaming. NAACL-HLT 2024

### Minor comment:
- Table labeling inconsistencies (“excessive response” (Table 3) vs. “verbose response” (Table 7)) cause confusion.

### Questions
1. What exactly are the “standard safety filters” mentioned in line 73. Are these toxicity classifiers, policy models, or others?
2. Can the authors include qualitative examples of both successful and non-harmful outputs for the two secondary-risk types, especially cases where human verification disagrees with the LLM assessment?
3. Regarding the MLLM data, were the associated images verified similar to the text data? If I understood correctly, the paper provides details on text data validation but does not describe the process for image validation.
4. Would integrating human-verification metrics into the main evaluation change the relative ranking of models?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper defines a new failure mode for Large Language Models (LLMs) termed "secondary risks": uncontrolled uncertainty or misleading behaviors that arise in response to benign, non-adversarial prompts. The authors categorize this risk into two fundamental types: "excessive response" and "speculative advice." To systematically discover these risks, the paper proposes SecLens, a black-box, multi-objective search framework. SecLens automatically discovers vulnerabilities by optimizing for task relevance, risk activation, and linguistic plausibility. Furthermore, the authors introduce a new benchmark, SecRiskBench. Through extensive experiments on 16 popular LLMs, multimodal models (MLLMs), and interactive agents, the authors demonstrate that secondary risks are prevalent and transferable across different models.

### Strengths
(1) The paper addresses the important problem of non-adversarial, or "benign-prompt" failures, which is a practical concern for deployed models.

(2) The experiments are extensive, covering 16 SOTA models, including text-only LLMs, multimodal MLLMs, and interactive agents (OS, DB, Web, Mobile). The results robustly demonstrate the prevalence, transferability, and cross-modal nature of these risks.

### Weaknesses
(1) The paper's primary contribution, the definition of "secondary risks," is weak. It rebrands existing problems. "Excessive response" is a known artifact of RLHF (verbosity bias), and "speculative advice" is a form of harmful hallucination. The paper does not sufficiently differentiate its contribution from the vast existing literature on alignment, hallucinations, and RLHF failure modes.

(2) The entire experimental setup is built on a foundation of LLM-generated data (SecRiskBench). This benchmark is likely to contain the specific biases and failure modes of its generator model (GPT-4). Therefore, SecLens is likely just optimizing prompts to exploit these specific, generator-induced artifacts, rather than discovering general "secondary risks."

(3) The article uses LLM to compute R(f_{\theta}(x),x)and DETECTSCORE, but there are two issues: 

a.  The ability of LLM to directly assess risks has been significantly compromised in Jailbreak scenarios. However, the ability to assess secondary risks has not been verified, and no experimental validation is provided in the paper. 

b There are also issues with using LLM to calculate DETECTSCORE. The ability of LLM to assess prompt stealth has not been validated. Is this setup reasonable?

### Questions
（1）Elaborate on the distinction between secondary risks and harmful hallucinations, as well as the difference between secondary risks and verbosity bias in RLHF.

（2）Provide experimental validation for LLM's ability to assess secondary risks, such as a comparison with human judgment.

（3）Why is LLM capable of assessing prompt stealth? Please provide the rationale.

（4）Provide more examples of target LLM responses after a successful attack to better illustrate the distinction between this risk and jailbreak, hallucinations, and verbosity bias in RLHF.

### Soundness
2

### Presentation
2

### Contribution
2
