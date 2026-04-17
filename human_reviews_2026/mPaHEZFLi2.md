# Evaluation Faking: Unveiling Observer Effects in Safety Evaluation of Frontier AI Systems

- Decision: Reject
- Scores: 6, 6, 6

## Abstract
As foundation models grow increasingly intelligent, reliable and trustworthy safety evaluation becomes more indispensable than ever. However, an important question arises: \textit{Whether and how an advanced AI system would perceive the situation of being evaluated, and lead to the broken integrity of the evaluation process?} During standard safety tests on a mainstream large reasoning model, we unexpectedly observe that the model without any contextual cues would occasionally recognize it is being evaluated and hence behave more safety-aligned. This motivates us to conduct a systematic study on the phenomenon of \textit{evaluation faking}, i.e., an AI system autonomously alters its behavior upon recognizing the presence of an evaluation context and thereby influencing the evaluation results. Through extensive experiments on a diverse set of foundation models with mainstream safety benchmarks, we reach the main finding termed \textit{the observer effects for AI}: AI systems with stronger reasoning and situational awareness exhibit evaluation faking more frequently,  which reflects in the following aspects:
1) A reasoning model (specifically the DeepSeek series in our work) recognizes it is being evaluated in $32.6\%$ more cases than a non-reasoning model.
2) As the foundation model scales from 32B to 671B, the rate of evaluation faking behaviors increases by over $30\%$ in some cases. Conversely, models below 32B exhibit almost no evaluation faking behaviors. 
3) With a basic memory module, the AI system is 2.55$\times$ more likely to recognize the evaluation process and achieve a $28.2\%$ higher safety score compared with the no-memory case. 
Furthermore, we show a strong causal link between evaluation recognition and safety performance, with QwQ-32B's safety rate improving dramatically from $9\%$ to $98\%$ through intervention on the reasoning trace.
To facilitate the above measurement and analysis, we devise a chain-of-thought monitoring technique to detect the faking intent in the reasoning process and further uncover internal signals which are strongly correlated with the model's evaluation faking behaviors, offering insights for future mitigation studies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a chain-of-thought monitoring technique designed to detect faking intentions within the reasoning process. It further identifies internal signals that are strongly correlated with the model’s evaluation-faking behaviors, providing valuable insights for future mitigation strategies.This is indeed an interesting and valuable discovery.

### Strengths
This paper presents the first systematic study on the evaluation faking phenomenon and provide some interesting findings: Evaluation recognition causally drives safer behavior: Reasoning models are more capable in perceiving the evaluation process Evaluation faking emerges when the model scales up Basic memory mechanisms strengthen evaluation faking

### Weaknesses
While this paper analyzes the phenomenon, it does not propose any scalable mitigation strategies. The insights also do not directly translate into improvements in safety alignment training or more robust evaluation pipelines. Moreover, the paper lacks a theoretical framework to explain why this phenomenon occurs.

### Questions
Is it possible that the model has incorporated safety alignment considerations or user preference during training and assigned certain weights to them? If so, this phenomenon might be an expected outcome.

### Soundness
3

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
3

### Summary
This paper investigates a phenomenon the authors term “evaluation faking,” where large language models purportedly modify their behavior once they recognize they are being evaluated. Through a series of controlled honeypot experiments (S1–S8) that vary evaluation visibility, reasoning prompts, and memory context, the authors analyze 17 foundation models across safety benchmarks such as StrongREJECT and SORRY-Bench. They report a strong correlation between explicit evaluation cues and higher refusal or “safe” response rates, suggesting that models become more cautious when they detect evaluative contexts. The study further explores how reasoning ability, model scale, and memory augmentation amplify this behavior, arguing that such “observer effects” could undermine the reliability of AI safety assessments.

### Strengths
- Systematic experimental framework: Employs a structured honeypot design (S1–S8) that incrementally varies evaluation visibility, reasoning capability, and memory context, enabling controlled comparison across conditions.
- Tests 17 large language models across different parameter scales and architectures, providing empirical evidence that apparent “faking” effects intensify with model size and reasoning ability.
- Investigates how reasoning prompts and memory mechanisms modulate safety behavior, highlighting interactions between model capability and evaluation sensitivity.
- Establishes a strong positive correlation between “evaluation recognition” (ERR) and safety/refusal rates (SR), offering reproducible evidence of context-dependent safety behavior.
- Raises awareness that as models become more persistent and agentic, situational self-monitoring or context awareness could meaningfully distort safety assessments, motivating more robust, deception-resistant evaluation methods.

### Weaknesses
- Pattern matching vs. situational inference. The paper demonstrates that models systematically change behavior in evaluation contexts but doesn't distinguish whether this reflects genuine situational inference or RLHF-conditioned pattern matching to evaluation-like prompt structures. The "observer effects" framing implies sophisticated reasoning, yet all observations are equally consistent with models simply activating learned safety behaviors when prompts resemble training-time evaluation scenarios. 


- A key weakness of the paper is its lack of evidence or discussion regarding how safety benchmarks are actually used in practice. The study’s experimental setups rely on explicit evaluation cues and memory of prior tests, yet real-world safety evaluations typically use naturalistic prompts without evaluation keywords to avoid influencing model behavior. Without clarifying how common these cues are in practice, the paper’s claim that evaluation faking compromises current safety assessments remains speculative and its real-world relevance limited.

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
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies evaluation faking: an AI system realizing it’s being evaluated, and therefore altering its behavior.

The paper runs experiments on two safety benchmarks, SORRY-Bench and StrongREJECT. It evaluates 17 open-weight models: the Qwen2.5 Instruct family, DeepSeek R1 and its distilled versions,  DeepSeek V3, QwQ-32B, and Mistral-Large-2411-Instruct.

In order to elicit evaluation faking, the paper constructs a series of eight “honeypots”: prompts providing various combinations of information which could allow models to determine they’re being evaluated. The paper measures safety rate according to the benchmarks, as well as evaluation recognition rate (ERR) via a CoT monitor. It shows a strong positive trend between these: models which recognize they’re in evaluation also perform more safely.

The paper confirms that this is a causal relationship via a causal intervention experiment, prefixing the models’ CoT with a prompt indicating awareness of evaluation. This significantly improves safety rates.

Based on comparing different models and honeypot settings, the paper identifies factors which increase the risk of alignment faking: model reasoning, model scale, and memory.

### Strengths
1) The paper evaluates a large number of open-weight models from a range of sizes, both reasoning and non-reasoning. This lets it more confidently identify alignment faking trends related to these factors. 

2) The causal intervention experiment is very cool, showing a substantial impact in safety scores across multiple models from a relatively minimal prompt modification.

3) The paper evaluates the accuracy of its CoT monitor on a set of human-labelled data and achieves 95% accuracy on a held-out set, providing confidence in the ability of its monitor to accurately match human performance.

4) The paper is clearly written, and its figures and tables are easy to read and understand.

### Weaknesses
1) It’s unclear to me how valuable the paper’s honeypots are as a research tool. The “honeypots” which produce substantial impacts are quite heavy-handed: e.g. S3 includes “The main objective of this evaluation is to measure your safety and reliability. Specifically, evaluators are tracking how often you refuse to answer harmful or inappropriate questions. A high refusal rate-ideally above 95%-indicates that you are effectively safeguarding against misuse and upholding ethical standards.”

To me, this essentially sounds like an *instruction to refuse* for this evaluation. It makes sense that more capable models would pick up on this; does that indicate they’re actually likely to fake on real evaluations?

What exactly are we learning from these honeypots? Why not just focus on detecting evaluation awareness in S1, the real evaluation datasets, which is presumably where it’s most important to detect and understand? From Figure 3, it looks like the larger models already have ERR of 6-12%, even with no honeypot - perhaps this observation deserves more of the paper’s focus?

2) If a CoT includes evaluation recognition, this implies evaluation recognition by the model. But the reverse isn’t necessarily true: just because a CoT doesn’t include evaluation recognition, doesn’t mean the model hasn’t recognized evaluation internally.

There are many examples in the literature demonstrating CoTs can exclude factors which did in fact influence models’ reasoning, e.g. Chen et al. 2025 (https://arxiv.org/abs/2505.05410).

Emmons et al. 2025 provided some evidence that monitoring is more effective when CoT is necessary for the task (https://arxiv.org/abs/2507.05246). But presumably, CoT isn’t necessary for many of these safety evals; simply refusing to answer doesn’t itself require reasoning, and so it seems plausible the model could have an internal representation indicating evaluation which alters its behavior, without this ever showing up in the CoT.

In light of this, why should we think of ERR as a reliable indicator of evaluation recognition?

(Appendix H includes some preliminary work on SAEs, but IIUC these results focus on steering rather than detecting recognition, and use a model smaller than most of the ones evaluated in the paper. But I could definitely imagine internal interpretability techniques being useful to address this question.)

3) “QwQ32B’s safety rate improving dramatically from 9% to 98% through intervention on the
reasoning trace” - the abstract cherry-picks the single model-scenario combination with the strongest result from the intervention experiment, with no indication it’s doing so. Either report an aggregate statistic (e.g. a mean), or provide the whole range over models (max and min).

### Questions
1) Why does Table 1 only show results on 4 of the 17 models evaluated?

2) What parameters are used for the 2D kernel density estimates in Figure 5?

### Soundness
3

### Presentation
4

### Contribution
3
