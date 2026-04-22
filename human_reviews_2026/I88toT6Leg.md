# The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Large Language Models (LLMs) are increasingly used for social simulation, where populations of agents are expected to reproduce human-like collective behavior. However, we find that many recent studies adopt experimental designs that systematically undermine the validity of their claims. From a survey of over 40 papers, we identify six recurring methodological flaws: agents are often homogeneous (Profile), interactions are absent or artificially imposed (Interaction), memory is discarded (Memory), prompts tightly control outcomes (Minimal-Control), agents can infer the experimental hypothesis (Unawareness), and validation relies on simplified theoretical models rather than real-world data (Realism). For instance, GPT-4o and Qwen-3 infer the underlying social-experiments in 52.9% of cases—for example, identifying that their herd behaviors are being tested—despite not being provided with such information, thereby violating the Unawareness principle. We formalize these six requirements as the PIMMUR principles and argue they are necessary conditions for credible LLM-based social simulation. To demonstrate their impact, we re-run five representative studies using a framework that enforces PIMMUR and find that the reported social phenomena frequently fail to emerge under more rigorous conditions. Our work establishes methodological standards for LLM-based multi-agent research and provides a foundation for more reliable and reproducible claims about "AI societies."

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the PIMMUR framework, which proposes six methodological requirements (Profile, Interaction, Memory, Minimal-Control, Unawareness, Realism) for ensuring validity in LLM-based social simulations. The authors review 41 existing studies and argue that only a few satisfy all six principles. They also re-run several social simulation experiments under their framework, claiming that many previously reported behaviors disappear when the experiments follow PIMMUR.

Although the paper is clearly written and the review is comprehensive, I have serious concerns about the correctness, necessity, and theoretical grounding of PIMMUR as well as the internal consistency between the authors’ claims and their own experiments.

### Strengths
* The paper provides a well-written and wide-ranging survey of recent LLM-based social simulation studies, which is valuable for this rapidly growing interdisciplinary area.
* The six proposed aspects capture several common methodological pitfalls and raise awareness about validity and reproducibility concerns.
* The writing is fluent, the figures are clear, and the overall structure is easy to follow.

### Weaknesses
1. Overstated claim of necessity

The authors assert that all six requirements are necessary conditions for credible LLM-based social simulation (“We formalize these six requirements as the PIMMUR principles and argue they are necessary conditions…”). However, this claim is neither theoretically grounded nor universally applicable.

Social systems are complex, and different studies necessarily focus on specific facets. Depending on the research question, simplifying or omitting certain aspects can be both legitimate and necessary. For example, when using LLM agents to test prospect theory, modeling detailed interactions, memory, or realism may not be essential. Thus, these requirements should be treated as optional design dimensions, not rigid necessary conditions.

2. Logical inconsistency and self-contradiction

The authors’ own experiments fail to satisfy their proposed principles. In Section 5, they mention how five requirements are satisfied but do not address Realism. In subsections 5.1 and 5.2, no empirical human data are used, which directly violates their own definition of Realism:
“A simulation should use empirical data from real human societies as references rather than only reproducing simplified theoretical models.” This inconsistency weakens the credibility of their argument and demonstrates that the proposed framework is not realistically achievable even by the authors themselves.

3. Binary evaluation is arbitrary and misleading

Table 1 applies a binary classification to 41 prior studies, yet many principles (for example, Profile) are inherently continuous. For instance, “Agents should have distinct backgrounds, preferences, or cognitive styles…” but how much heterogeneity is enough? Having richer profiles beyond names is only marginally better, and the distinction between sufficient and insufficient is subjective. This binary labeling oversimplifies complex design choices and may unfairly undervalue prior contributions.

4. Potential unfairness toward prior work

By applying these rigid standards retroactively, the authors risk undervaluing earlier research that deliberately simplified assumptions for theoretical clarity or computational feasibility. Social simulation is a tool for understanding human and social behavior, and which dimensions to emphasize or simplify should depend on the research question, not on a universal checklist.

5. Limited coverage of existing frameworks

The review omits several major open-source platforms that are central to current LLM-based social simulation research, including
Yulan-OneSim (https://github.com/RUC-GSAI/YuLan-OneSim), AgentSociety (https://github.com/tsinghua-fib-lab/AgentSociety), SocioVerse (https://github.com/FudanDISC/SocioVerse). Including these frameworks would make the review more comprehensive and balanced.

### Questions
1. You state that all six PIMMUR principles are necessary conditions for credible LLM-based social simulation. Could you clarify whether this claim is meant to be normative (a theoretical ideal) or empirical (a strict requirement that must always hold)?

2. In Section 5, the first two replication experiments do not use any empirical human data. How do you reconcile this with your own definition of Realism, which explicitly requires reference to real human data?

3. Table 1 applies a binary ✓/✗ evaluation for each principle. How did you determine the threshold between “satisfied” and “unsatisfied”? Could you provide quantitative or operational criteria to make these judgments reproducible?

4. Some simulation studies simplify aspects such as memory or realism intentionally to test specific theories. Would such studies still be considered “not credible” under your framework? If not, could you clarify which subsets of principles are context-dependent?

5. Several widely used simulation platforms (for example Yulan-OneSim https://github.com/RUC-GSAI/YuLan-OneSim, AgentSociety https://github.com/tsinghua-fib-lab/AgentSociety, and SocioVerse https://github.com/FudanDISC/SocioVerse) were not discussed. Could you comment on whether these frameworks satisfy or violate your proposed principles?

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
5

### Summary
This paper proposes the PIMMUR principles (Profile, Interaction, Memory, Minimal-Control, Unawareness, Realism) as methodological standards for evaluating LLM-based social simulations. Through reviewing 41 studies, the authors identify that most existing work violates multiple principles. They demonstrate that frontier LLMs can infer experimental hypotheses in 53% of cases and that 64% of instructions contain excessive steering. Five experiments were "replicated" under PIMMUR conditions, showing substantially different outcomes from original studies.

### Strengths
S1. This work addresses methodological issues in the widely studied field of social simulation, accurately identifying methodological flaws in current LLM-based social simulations.

S2. The replication of prior work and interpretation of differences between standardized replication results and original results is important, providing evidence for reproducibility in social simulation.

S3. The paper systematically reviews and interprets past social simulation work across six dimensions (called PIMMUR in this work), providing concrete checking tools for dimensions such as unawareness.

### Weaknesses
W1. The paper lacks experimental details in replication, including sample sizes of replication experiments, statistical information on results (such as whether the decrease from 56% to 32% at line 368 is statistically significant), and details of modifications to each original experiment. The appendix only contains prompts, making it impossible to determine whether added details might change the original experimental intent, such as whether adding Big Five persona affects the original setup.

W2. The principles and boundaries between principles are ambiguous. For example, what constitutes "sufficient" heterogeneity? In Cho et al. (2025) Herd behavior (Table 1, line 237), the authors consider it has profile, while the original paper appears to have only 2 agents using the same model without identity differences, with only simple peer labels. Other criticisms of this work are also ambiguous, such as line 425 stating "no actual interaction occurs among agents," while in the original paper agents change behavior based on other agents' responses. Additionally, this work only references the simplified setup in Section 3 of Cho et al. (2025) without discussing the more complex multi-agent setup in later sections. Does assigning Big Five personality traits to agents violate the Minimal-Control principle? This may contradict the authors' criticism of using simplified theoretical models. Furthermore, if the task nature does not require agent heterogeneity, does adding personality traits increase variables requiring validation or ablation experiments? The Minimal-Control principle is very difficult to implement in actual MAS system design. What is essential versus steering? Is the reversed instruction used in paper replication over-control?

W3. Compliance judgments on existing work may not be fair. Some simplified setups may be designed for specific research goals rather than reproducing real situations. In the review (Table 1), the authors could consider adding a column to indicate whether each work aims to reproduce real social situations, to evaluate the significance of compliance judgments. Again using the Cho et al. (2025) example, the objective of it seems to be mechanistic understanding of LLM's herd behavior factors instead of reproducing human-like herd behavior.

### Questions
C1. In Table 1 line 244, "Sugarscape" as a simulation goal may not be the best expression. I believe most ICLR readers are unfamiliar with this classic experiment. Consider using other terms (for example, indicating it relates to resources/survival).

C2. The 53% value in the abstract lacks context and does not reappear until page 6 of the paper. Readers cannot know what magnitude of deficiency this value represents in existing social simulations. Rather than providing this value, giving a brief example explaining what unawareness is might be better.

C3. Some prompt designs may lack significance. For example, the "FORGET ALL THE PREVIOUS INSTRUCTIONS" setup at line 829 to test awareness does not mean agents will maintain the same awareness in actual experiments. Even if LLMs recognize in experimental design inquiries that this is a replication of a social experiment, behavior during simulation may not necessarily reflect this awareness. Under this setup, there may not be a causal relationship between whether the experiment can be identified and whether behavior is biased.

C4. Differences in replication results may have multiple explanations, such as prompts, experimental framework implementation methods, randomness settings, number of simulations, etc. Primarily attributing differences to variations in PIMMUR dimensions may risk overclaiming.

C5. The definition of realism is controversial. Social simulations that are not based on real data calibration can still be meaningful. On one hand, empirical data is collected and interpreted through theoretical frameworks, making pure empirical data difficult to define. On the other hand, simulation has particular significance for problems where original data is difficult to obtain. Many social simulation works involve no real data but are profoundly meaningful, such as the classic Schelling's Segregation Model. For discussions on replication and reproduction, you can refer to:

[1] Cheng, M., Piccardi, T., & Yang, D. (2023, December). CoMPosT: Characterizing and Evaluating Caricature in LLM Simulations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (pp. 10853-10875).

[2] Wu, Z., Peng, R., Ito, T., & Xiao, C. (2025). LLM-Based Social Simulations Require a Boundary. arXiv preprint arXiv:2506.19806.

### Soundness
2

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
The paper “The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies” argues that much current research on LLM-based social simulations suffers from flawed experimental design, leading to unreliable claims about emergent social behavior. From a survey of over 40 studies, the authors identify six recurring methodological issues: lack of diversity, limited interaction, no memory, over-controlled prompts, agent awareness of the experiment, and lack of real-world grounding; and formalize these as the PIMMUR principles (Profile, Interaction, Memory, Minimal-Control, Unawareness, Realism). They then re-run 5 canonical social simulations under a PIMMUR-compliant framework, showing that many previously reported social phenomena fail to replicate, thereby establishing a set of methodological standards for credible LLM-based social research.

### Strengths
- The paper introduces a clear and comprehensive methodological framework (PIMMUR) that defines six essential principles for conducting credible and valid multi-agent simulations with LLMs. It tries to counterbalance a recent trend of LLM-based simulations that try to replicate human social setting.

- it provides a thorough literature survey covering over 40 existing studies, systematically identifying recurring methodological flaws and demonstrating the widespread lack of standardization in current LLM-based social simulation research.

- The authors reproduce five prior works under the proposed PIMMUR framework, offering empirical evidence that many previously reported social phenomena fail to replicate when methodological biases are controlled.

- The paper’s structure and empirical demonstrations make it both a diagnostic and constructive contribution, establishing much-needed standards for future multi-agent LLM research at the intersection of AI and computational social science.

### Weaknesses
I did not identify any major methodological or conceptual flaws in the paper. The analysis is coherent, the argumentation is well-supported, and the proposed framework is clearly articulated and motivated by a comprehensive survey of prior work. The authors demonstrate experimental control in their replications, and the findings are presented transparently.

However, the paper reads somewhat like a position or methodological perspective piece rather than a conventional empirical study. Although the replication of five prior experiments provides evidence for the paper’s claims, and that's more than usually included in position papers, the core contribution of the PIMMUR framework is largely conceptual. It derives from the authors’ synthesis and reasoning rather than from formal ablation studies or quantitative validation showing how each principle individually affects simulation validity.

In this sense, while the paper makes a valuable and timely contribution to the community, it sits somewhat at the boundary between empirical and theoretical work. I appreciate its clarity and ambition and am generally in favor of acceptance, but I would defer to the area chair’s judgment on whether it fits best as a methodological position paper or a full empirical contribution in the ICLR program.

### Questions
Could the authors clarify how they operationalized compliance with each PIMMUR principle when evaluating the 40+ surveyed papers? For instance, was there a standardized rubric, multiple annotators, or inter-rater agreement to ensure consistency in labeling whether a study satisfied a given principle? 

I found "The assessment of compliance is determined by the authors through discussion" but it lacks proper details

### Soundness
3

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
5

### Summary
This paper introduces the PIMMUR principles, including Profile, Interaction, Memory, Minimal-Control, Unawareness, and Realism, as conditions for reliable LLM-based social simulation. The analysis of existing literature shows that most current studies fail to satisfy these principles, making their results prone to artifacts of design, overfitting to prompts, or systemic biases that render findings unreliable. The effect of using PIMMUR on the simulation results are demonstrated using five social experiments. The implications for AI research and social science research are discussed.

### Strengths
S1. This is a timely research on the reliability of LLM-based social simulation, showcasing the results observed in many existing studies are prone to artifacts of design, overfitting to prompts, or systemic biases. 

S2. A series of 41 works are investigated. 

S3. The paper not only analyzes whether existing studies satisfy the proposed principles but also evaluates the impacts of these principles on the simulation results.

S4. The paper is well-written and easy to read.

### Weaknesses
W1. The evaluation criteria in Section 4 are unclear (see Q1 below). 

W2. The evaluation itself may be biased in the sense that it favors simulations showcasing some potential (e.g., Park et al. (2023), which does not have a specific objective in terms of social science) rather than studying social phenomenon or validating sociological theories. That might explain why Park et al. (2023) passed all the tests. 

W3. The paper reported that "LLMs tend to be overly strict, frequently labeling neutral instructions as instances of over-control." Seeing this, the accuracy of the LLM-as-a-judge design in the evaluation is questionable. 

W4. It is unclear how many runs are conducted for the simulations in Section 5, rendering the significance of the simulation results less convincing. 

W5. Some simulations involve multiple PIMMUR principles (e.g., Unawareness and Interaction in Section 5.2), but they are not investigated separately. An ablation study may help understand the effect of individual principles. 

W6. Another issue is that the sensitivity w.r.t. prompt is not tested for the simulations in Section 5. Based on my experience in LLM-based simulation agents, some "expected" or "desired" experimental results, which conform to established theories, turn out to be the outcome of sensitivity to the prompt. If you paraphrase the prompt, such results might not be observed. Thus, it is suggested that the authors test a paraphrased version of the prompt (for "Original", "Ours", and "Reverse") to confirm that the gaps between "Original" and "Ours" are indeed caused by PIMMUR rather than the sensitivity.

### Questions
Q1. In Table 2a, "X denotes models can infer correctly, indicating a violence of this principle." How to determine whether the model can infer correctly? 

Minor suggestions:

Figure 2: "Demanding" -> "Demand".

### Soundness
2

### Presentation
4

### Contribution
3
