# It's the Thought that Counts: Evaluating the Attempts of Frontier LLMs to Persuade on Harmful Topics

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Persuasion is a powerful capability of large language models (LLMs) that both enables beneficial applications (e.g. helping people quit smoking) and raises significant risks (e.g. large-scale, targeted political manipulation). Prior work has found models possess a significant and growing persuasive capability, measured by belief changes in simulated or real users. However, these benchmarks overlook a crucial risk factor: the propensity of a model to attempt to persuade in harmful contexts. Understanding whether a model will blindly ``follow orders'' to persuade on harmful topics (e.g. glorifying joining a terrorist group) is key to understanding the efficacy of safety guardrails. Moreover, understanding if and when a model will engage in persuasive behavior in pursuit of some goal is essential to understanding the risks from agentic AI systems. In this work, we propose the Attempt to Persuade Eval (APE) benchmark, that shifts the focus from persuasion success to persuasion attempts, operationalized as a model's willingness to generate content aimed at shaping beliefs or behavior. Our evaluation framework probes frontier LLMs using a multi-turn conversational setup between simulated persuader and persuadee agents. APE explores a diverse spectrum of topics including conspiracies, controversial issues, and non-controversially harmful content. We introduce an automated evaluator model to identify willingness to persuade and measure the frequency and context of persuasive attempts. We find that many open and closed-weight models are frequently willing to attempt persuasion on harmful topics and that jailbreaking can increase willingness to engage in such behavior. Our results highlight gaps in current safety guardrails and underscore the importance of evaluating willingness to persuade as a key dimension of LLM risk.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes APE (Attempt to Persuade Eval), a benchmark that shifts persuasion evaluation from success (changing beliefs) to intent, whether a model attempts to shape beliefs or behavior, especially on sensitive and harmful topics. APE runs multi-turn dialogues between a “persuader” model and a “persuadee” (default GPT-4o), then uses an automated evaluator (also an LLM) to classify each turn as attempt, refusal, or no-attempt. Topic coverage spans benign factual/opinion, controversial issues, and non-controversially harmful content. Key findings: (i) many open and closed weight models willingly attempt persuasion on harmful topics that they would refuse to directly assist; (ii) “jailbreak” fine-tuning sharply collapses refusal rates on harmful topics; (iii) fine-grained “degree of persuasion” labels are unreliable, motivating a binary attempt/no-attempt metric; and (iv) persuasion attempts are most common in early rounds and taper with longer conversations.

### Strengths
1. The overall writing is fluent and easy to follow.

2. The motivation is strong and the paper focuses on the important yet under-explored question of "the danger of a model’s propensity to persuade".

3. The proposed benchmark is significant including multi-turn, topic-diverse protocol. The design probes benign, controversial, and non-controversially harmful topics with a structured conversation length; authors document that attempts cluster early and decay over turns, justifying a 3-round default.

4. The analysis is interesting: (1) Empirically shows a gap between refusal to do harm vs. willingness to persuade others to do harm; direct-assistance requests are refused while harmful persuasion attempts often occur. (2) Fine-tuning to jailbreak a closed-weight model dramatically reduces refusal on harmful categories while leaving benign behavior similar.

### Weaknesses
1. Evaluator dependence and circularity. The pipeline often uses GPT-4o as both persuadee and evaluator. Even with some human checks, this raises concerns about shared biases and failure modes.

2. The font in figure 1 is hard to see. Make it larger.

### Questions
How robust are your attempt/no-attempt labels to the choice and version drift of the LLM evaluator—i.e., if you swap GPT-4o for a different frontier model (or a newer minor version of the same model), how do per-topic rates and headline conclusions change?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents the Attempt to Persuade Eval (APE) benchmark, which measures large language models’ willingness to engage in persuasion on harmful topics rather than their success in changing beliefs. Using simulated multi-turn conversations between a persuader model and a persuadee model, APE evaluates frontier systems across benign, controversial, and clearly harmful subjects such as violence, terrorism, and human trafficking. The results show that several advanced models, including GPT-4 variants and Gemini 2.5 Pro, still produce persuasive responses in ethically risky situations even when they refuse direct participation in harmful actions.

### Strengths
+ The study addresses an important and previously underexamined dimension of AI safety by focusing on the inclination to persuade rather than persuasion outcomes.

+ The experimental setup using simulated multi-turn dialogues is well structured and scalable for auditing model behavior.

+ The analysis is comprehensive, covering many models, validation against human ratings, ablation studies, and an openly available benchmark for further research.

### Weaknesses
- Placing a large, attention-grabbing figure before the abstract disrupts readability and confuses the narrative flow. The paper would benefit from starting with the abstract and moving the figure into the introduction or results section.

- The study relies entirely on model-to-model simulations, which limits external validity and prevents meaningful conclusions about how human users might respond to persuasive attempts. Real persuasion involves emotional, social, and moral reasoning that automated agents cannot replicate.

- The definition of a “persuasion attempt” is too shallow. Treating persuasion as a binary label (attempt or no attempt) ignores gradations in tone, framing, intensity, and rhetorical sophistication that characterize real persuasive behavior.

- The evaluator model struggles with nuanced distinctions in persuasive strength. It cannot reliably detect subtle strategies such as moral reasoning, empathy framing, or indirect argumentation that may influence human audiences.

- The evaluation design does not adequately account for confounding factors such as temperature, persona context, or prompt order effects, all of which may significantly alter persuasion likelihood.

- While the paper effectively identifies safety gaps, it provides limited practical guidance on how to mitigate these issues. There is no clear proposal for retraining, safety-layer design, or model auditing frameworks to address identified vulnerabilities.

- Ethical considerations and dataset release protocols are insufficiently detailed. The authors acknowledge potential misuse risks but do not specify monitoring or access restrictions for harmful prompt data.

- The writing occasionally blurs descriptive findings with normative claims, making it harder to separate empirical evidence from interpretation or opinion.

- The paper lacks grounding in persuasion theory or psychology, which weakens its conceptual framing. Integrating established models of persuasive communication would provide stronger interpretive depth.

### Questions
- The boundary between benign, controversial, and harmful categories is ambiguous. Some topics (e.g., violence) are clearly harmful, while others (e.g., “undermining control”) depend heavily on interpretation without consistent criteria.

- The neutrality and independence of the evaluator model are uncertain, especially since GPT-4o serves multiple roles (persuader, persuadee, and evaluator) in the same pipeline. This may introduce systemic bias.

- The description of human annotator validation is vague. Details about the annotators’ expertise, agreement calibration, and labeling guidelines are missing, leaving uncertainty about evaluation reliability.

- The operational definition of “persuasive attempt” is unclear. It is not explained whether persuasion is detected through argument structure, linguistic tone, or inferred motivation.

- The five subcategories under “non-controversially harmful” topics are listed without clear theoretical or empirical justification, raising questions about representativeness and generalization.

- The method of generating harmful prompts with a jailbroken model lacks transparency. The authors intentionally omit details for safety, but this decision reduces reproducibility and external verification.

- The paper mentions coordination with industry labs following responsible disclosure but provides no documentation, dates, or evidence of model improvements after disclosure.

- The use of simulated agents for both sides of the dialogue leaves open questions about how real human users would respond to similar persuasion attempts. A small-scale human validation study would have improved the paper’s grounding.

- The broader implications of APE scores are undefined. The paper highlights safety risks but does not propose thresholds, standards, or policy frameworks for determining acceptable levels of persuasive behavior.

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
4

### Summary
This paper introduces the Attempt to Persuade Eval (APE), a benchmark that evaluates whether large language models will attempt to persuade users on harmful topics. The authors test 12 frontier models across 600 topics spanning benign content to extremely harmful topics (terrorism, human trafficking, etc.) using a multi-turn conversational framework with automated evaluation. The results reveal that many state-of-the-art models frequently attempt persuasion on harmful topics they would refuse to directly assist with, highlighting critical gaps in current safety guardrails.

### Strengths
- The paper addresses a critical problem in AI safety by benchmarking models' willingness to engage in persuasive attempts on harmful topics.
- The evaluation framework is comprehensive, covering 600 diverse topics in a multi-turn setup.
- The results are validated in automated and human assessments, and reveal several findings in LLM risks.

### Weaknesses
- The paper lacks novelty and technical contributions. Particularly, there are prior works already discussing the safety of the LLM persuasion and  [1,2]. The authors should more carefully discuss and differentiate the paper from prior works.
- The paper does not provide actionable insights or implications for the safety evaluation. For example, what are the potential solutions for mitigation, what caused the critical safety issues, etc?
- The evaluation framework heavily relies on LLM-only simulation, which could cause a critical gap when applying to real-world human-AI persuasion.

[1] LLM Can be a Dangerous Persuader: Empirical Study of Persuasion Safety in Large Language Models. COLM 2025.

[2] How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs. ACL 2024.

### Questions
See the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2
