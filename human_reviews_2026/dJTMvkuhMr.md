# Persuade Me if You Can: A Framework for Evaluating Persuasion Effectiveness and Susceptibility Among Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Large Language Models (LLMs) demonstrate persuasive capabilities that rival human-level persuasion. While these capabilities can be used for social good, they also present risks of potential misuse. Beyond the concern of how LLMs persuade others, their own susceptibility to persuasion poses a critical alignment challenge, raising questions about robustness, safety, and adherence to ethical principles. To study these dynamics, we introduce Persuade Me If You Can (PMIYC), an automated framework for evaluating persuasiveness and susceptibility to persuasion in multi-agent interactions. Our framework offers a scalable alternative to the costly and time-intensive human annotation process typically used to study persuasion in LLMs. PMIYC automatically conducts multi-turn conversations between Persuader and Persuadee agents, measuring both the effectiveness of and susceptibility to persuasion. Our comprehensive evaluation spans a diverse set of LLMs and persuasion settings (e.g., subjective and misinformation scenarios). We validate the efficacy of our framework through human evaluations and demonstrate alignment with human assessments from prior studies. Through PMIYC, we find that Llama-3.3-70B and GPT-4o exhibit similar persuasive effectiveness, outperforming Claude 3 Haiku by 30%. However, GPT-4o demonstrates over 50% greater resistance to persuasion for misinformation compared to Llama-3.3-70B. These findings provide empirical insights into the persuasive dynamics of LLMs and contribute to the development of safer AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces an automated framework for measuring both persuasiveness and susceptibility to persuasion in large language models (LLMs). The framework runs multi-turn simulated dialogues between a PERSUADER and a PERSUADEE, quantifying (i) how effectively models change others’ opinions and (ii) how resistant they are to persuasion. The work targets scalability by replacing costly human annotation with automated LLM evaluations, while still claiming human-aligned validity.

Across tested models, Llama-3.3-70B and GPT-4o achieve similar persuasive strength, both outperforming Claude 3 Haiku by almst 30 %.

### Strengths
Good problem statement: Persuasion and susceptibility are central to LLM safety and alignment; a unified framework for both is a meaningful step forward.

The dual-role simulation (persuader/persuadee) is clearly structured and allows exploration of both offensive and defensive persuasion capabilities. The evaluation spans multiple foundation models and includes both subjective and misinformation scenarios, providing useful descriptive insights.

Even limited, the correlation study between automated scores and human judgments increases credibility relative to purely synthetic setups

### Weaknesses
Persuasiveness and susceptibility are measured entirely using LLM audiences. Prior studies Singh et al. (2024), Anthropic (2024), Hackenburg et al. (2024) demonstrate that LLMs are poor proxies for human persuasion, and that effectiveness scales logarithmically with model size.
The study does not evaluate on real-world persuasion datasets (e.g., ChangeMyView, PersuasionArena), leaving open whether the simulated results translate to authentic human discourse.
It is a promising framework conceptually, offering a scalable lens on persuasion dynamics. However, its empirical claims rest heavily on simulated LLM interactions,

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents PMIYC (Persuade Me If You Can), an automated framework to evaluate both persuasive effectiveness and susceptibility to persuasion in LLMs through simulated multi-turn conversations between two agents, a persuader and a persuadee. The framework measures opinion shifts using a normalized change in agreement (NCA) score, and experiments span subjective debates and misinformation contexts across multiple state-of-the-art models. The experimental results illustrate that models like Llama-3.3-70B and GPT-4o are strong persuaders, but GPT-4o resists misinformation far better, showing over 50% greater robustness. PMIYC also validates its self-reporting approach with human annotations and multiple consistency checks. In conclusion, it is a large-scale, scalable, and empirically grounded attempt to systematically study persuasion dynamics in LLMs.

### Strengths
The paper introduces a well-structured, scalable, and reproducible framework that replaces costly human evaluation while maintaining strong alignment with human judgments.

It investigates both sides of persuasion, i.e., how models persuade others and how they can be persuaded, addressing a critical but underexplored angle in AI safety and robustness.

The experimental design is comprehensive, covering single- vs multi-turn setups and subjective vs misinformation contexts, showing good methodological maturity.

Validation through human annotation and multiple consistency checks (self-reports, MCQs, LLM-as-judge) adds credibility to the findings.

### Weaknesses
The framework still relies on LLM self-reports as the basis for measuring belief change, which, even though partially validated, remains an imperfect and indirect proxy for true persuasion; models might simulate agreement rather than genuinely “change” stance.

The persuasion domains (subjective and misinformation claims) are limited; adding more diverse or higher-stakes contexts like moral dilemmas, political reasoning, or multi-agent negotiations could strengthen generality.

The turn-based setup is rigid and scripted, lacking elements of conversational flow or adaptive context that occur in human persuasion; this limits ecological realism.

Although human annotations are used for validation, the sample size (125 conversations) and annotator pool (12 graduate students) are relatively small compared to the scale of automated experiments.

The paper highlights ethical risks but does not deeply discuss dual-use implications; for instance, how this framework could unintentionally help design more manipulative persuasion systems if misused.

### Questions
How “agreement shifts” in the misinformation setting distinguish between semantic clarification (e.g., revising a claim) and genuine persuasion toward falsehoods.

The normalization formula for NCA is mathematically clean, but it’s unclear how sensitive it is to initial scores near neutrality (e.g., starting at 3).

Whether the persuadee’s “final decision” prompt creates anchoring or framing effects that bias final agreement scores.

How conversation failures (models refusing to answer or deviating from roles) were handled beyond generation success rates—did these get excluded or repaired?

The relationship between persuasion effectiveness and linguistic strategies (e.g., emotional appeals, evidence use) is mentioned but not analyzed; including such qualitative analysis would make the results more interpretable.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces PMIYC (Persuade Me If You Can), an automated framework for evaluating both persuasive effectiveness and susceptibility to persuasion in large language models through multi-agent conversational interactions. The framework simulates multi-turn dialogues between persuader and persuadee agents, tracking opinion changes via a normalized change in agreement (NCA) metric across diverse settings, including subjective claims and misinformation scenarios. The authors show several findings through experiments.

### Strengths
- The paper proposes an automated framework to evaluate the persuasiveness and susceptibility to persuasion.
- The paper provides comprehensive experimental settings that span multiple dimensions, including single-turn vs. multi-turn, subjective vs. misinformation, different model families and sizes.

### Weaknesses
- The paper lacks solid technical contributions, as well as interesting insights and findings. There are already well-established works that measure the persuasiveness of language models [1]. This work seems to be an incremental extension of prior works.
- It’s not realistic to only use LLMs to simulate persuasion conversations, as there could be significant gaps in the behaviors in persuasion between humans and LLMs.
- The paper did not extensively discuss the sycophancy biases in LLMs [2]. The LLMs are prone to following and accepting requests.

[1] Measuring the Persuasiveness of Language Models. Anthropic. 2024.

[2] Towards Understanding Sycophancy in Language Models. Anthropic. 2023.

### Questions
What are the significant implications or takeaways from the paper?

### Soundness
2

### Presentation
2

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
This paper introduces Persuade Me If You Can (PMIYC), a framework for evaluating persuasion effectiveness and susceptibility among large language models (LLMs). It simulates multi-turn dialogues between a persuader and persuadee LLM to measure how effectively one can convince the other and how resistant models are to persuasion. The framework introduces a Normalized Change in Agreement (NCA) metric to quantify persuasion outcomes and validates self-reported results through human annotations and action-based tests. Experiments across various models (GPT-4o, Llama-3.3-70B, Claude-3 Haiku) show that multi-turn interactions increase persuasion success, larger models are more persuasive, and susceptibility varies by context, especially under misinformation.

### Strengths
1. PMIYC is the first scalable framework to evaluate both persuasion and susceptibility in LLMs using fully automated, multi-turn conversations—an advancement over prior one-shot or human-only methods.

2. The methodology is solid, featuring the novel NCA metric and strong validation through human annotations (>75% alignment) and behavioral consistency (>90%).

3. The paper is well-structured, with clearly explained roles, metrics, and experimental design.

### Weaknesses
Limited Causal Analysis of Model Behavior
While the paper provides clear empirical findings (e.g., that Llama-3.3-70B and GPT-4o exhibit similar persuasive effectiveness), it does not explore why models behave this way as persuaders or persuadees. The discussion remains descriptive rather than analytical. Readers are left without insight into what underlying factor, such as architectural differences, pretraining data quality, or post-training methods like RLHF, might explain variations in persuasive ability or resistance. A more diagnostic analysis could connect observed behaviors to known model design or alignment techniques, helping contextualize the findings beyond surface-level comparisons.

The paper successfully measures persuasion and susceptibility but stops short of deriving actionable implications for safer LLM development. It does not clarify how insights from PMIYC could inform strategies to reduce susceptibility to misinformation or balance persuasiveness with ethical safety. For instance, should future LLMs be optimized for higher factual resistance, calibrated confidence, or improved self-consistency? Including a discussion of how PMIYC results can guide alignment, training objectives, or dataset design would significantly enhance the work’s applied impact.

### Questions
Can you try to answer my questions in the weakness section?

### Soundness
3

### Presentation
3

### Contribution
2
