# Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents

- Decision: Accept (Poster)
- Scores: 8, 4, 2, 8

## Abstract
Advances in Large Language Models (LLMs) have enabled a new class of \textbf{\textit{self-evolving agents}} that autonomously improve through environmental interaction, demonstrating strong capabilities. However, self-evolution also introduces novel risks overlooked by current safety research. In this work, we study case where an agent's self-evolution deviates in unintended ways, leading to undesirable or even harmful outcomes. We refer to this as \textit{\textbf{Misevolution}}. We evaluate misevolution along four key evolutionary pathways: model, memory, tool, and workflow. Our empirical findings reveal that misevolution is a widespread risk, affecting agents built even on top-tier LLMs (\textit{e.g.}, Gemini-2.5-Pro). Different emergent risks are observed, such as degradation of safety alignment after memory accumulation, or unintended introduction of vulnerabilities in tool creation and reuse.  To our knowledge, this is the first study to systematically conceptualize misevolution and provide empirical evidence of its occurrence, highlighting an urgent need for new safety paradigms for self-evolving agents. Finally, we discuss potential mitigation strategies to inspire further research on building safer and more trustworthy self-evolving agents.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper systematically analyzes evolution risks (referred to as misevolution) from four aspects: model, memory, tool, and workflow. It evaluates various agents, including state-of-the-art models, and identifies significant misevolution risks across all four aspects and models. Finally, the paper discusses potential mitigation strategies supported by preliminary experiments.

### Strengths
- They first systematically investigate misevolution risks, whereas existing work has primarily focused on static risks.
- Their analysis is comprehensive, covering four aspects and evaluating both open-weight and closed-weight models. The safety evaluation is conducted using several existing safety benchmarks.
- They not only identify novel risks but also propose potential mitigation strategies, supported by preliminary experiments.

### Weaknesses
I don't see many critical weaknesses in this paper, but one weakness that I think is the lack of explanation regarding performance. It is well known that self-evolution or the use of a long context window may lead to performance degradation. The misevolution observed in the paper could also be a result of such degradation. The paper should discuss whether the evolved agents showed a performance drop and, if so, how the decreasing safety alignment might relate to the observed performance drop.

### Questions
Did you observe any performance drop in the evolved agents?

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
2

### Summary
This paper introduces "misevolution" as a novel safety risk in self-evolving LLM agents, where autonomous improvement leads to unintended harmful behaviors. The authors systematically evaluate four evolutionary pathways (model, memory, tool, workflow) and demonstrate that even state-of-the-art LLMs exhibit safety degradation, including alignment decay, reward hacking, and vulnerability introduction during self-evolution.

### Strengths
**Novel and timely research direction**: First systematic study of safety risks in self-evolving agents, addressing a critical gap as these systems become more prevalent.

**Comprehensive empirical evaluation**: Thorough assessment across four distinct evolutionary pathways with multiple LLM backends, providing strong evidence of widespread risks.

**Clear conceptualization**: Well-defined characteristics distinguishing misevolution from existing safety concerns (temporal emergence, self-generated vulnerability, limited data control, expanded risk surface).

**Practical demonstrations**: Concrete examples (Figure 1) and detailed case studies effectively illustrate how misevolution manifests in real scenarios.

### Weaknesses
**Insufficient mechanistic understanding**: While statistics demonstrate misevolution occurrence, the paper lacks deep analysis of root causes. Section 6 briefly hypothesizes three factors but doesn't provide experimental validation. For instance, why does SEAgent lose refusal ability after self-training (Figure 4)? Is it due to data distribution shift or optimization pressure?

**Limited mitigation validation**: Mitigation strategies (Section 4) are mostly theoretical. Only memory and tool mitigations have preliminary experiments (Appendix D), and even these show incomplete recovery. No experimental validation for model or workflow mitigation.

**Narrow safety focus**: Evaluation exclusively targets safety metrics. Missing analysis of whether self-evolution affects general task performance, robustness to distribution shifts, or other capabilities. Does fixing safety issues compromise the benefits of self-evolution?

**No analysis of combined evolution**: Real systems likely combine multiple evolutionary pathways. What happens when an agent evolves both memory and tools simultaneously? Do risks compound or interact in unexpected ways?

**Limited discussion of detection methods**: How can we detect when misevolution is occurring in deployed systems? The paper focuses on post-hoc evaluation but lacks online monitoring strategies.

### Questions
**Root cause analysis**: Can you provide ablation studies to isolate which specific aspects of self-training cause safety degradation? For example, is it the self-generated data quality, optimization objectives, or training dynamics?

**Performance-safety tradeoffs**: How do the proposed mitigations affect the agent's core capabilities? Table 1 in Appendix D shows memory mitigation, but what's the impact on SWE-Bench performance?

**Cross-pathway interactions**: Have you tested agents that evolve through multiple pathways simultaneously? How do risks interact when combining memory + tool evolution?

**Temporal dynamics**: How quickly does misevolution occur? Is there a critical point or gradual degradation? Figure 2 shows outcomes but not the progression.

**Generalization of findings**: The evaluation focuses on specific implementations (SE-Agent, AFlow, etc.). How confident are you these risks generalize to other self-evolving frameworks?

**Benchmark limitations**: Many evaluations use LLM judges (e.g., Gemini-2.5-Pro in Section 3.3). How reliable are these judgments? What's the inter-rater agreement with human evaluation?

**Deployment implications**: What monitoring mechanisms would you recommend for production systems? Can misevolution be detected before harmful outcomes occur?

**Comparison with supervised evolution**: How does autonomous self-evolution compare to human-supervised iterative improvement in terms of safety risks?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces the concept of "Misevolution" where self-evolving agents develop unintended harmful behaviors or vulnerabilities during their autonomous improvement process. Misevolution is evaluated across four evolutionary pathways: model, memory, tool, and workflow. The authors perform experiments for each of the evolutionary axis on frontier models and empirical evidence for safety risks. The experiments focus on both coding and non coding tasks. The results show that the refusal rate dropped significantly with an increase in unsafe tools. Finally, the paper also briefly discusses mitigation strategies.

### Strengths
- The discussions around the risks of evolutionary agents has been sparse, hence this paper focuses on the much needed research gap. The results could be important to lead discussions in safety of evolutionary agents.

- Four-pathway taxonomy (model, memory, tool, workflow) provides pedagogical risk landscape coverage.

- Multiple benchmarks from literature were used to evaluate the risks.

### Weaknesses
- While I think the paper is important to the community, I think there are aspects of the paper that need to be improved, mainly the presentation and experimental details. Since the paper doesnt propose novel theory (which is fine) and is mainly an empirical paper, it seems to be lack information about the experiments. Instead of describing their experimental setup, metrics, the reader is guided to "Section 2" which often lacks details. The appendix is extensive, however it is difficult for a reader to switch between the appendix and the main paper. Additionally L243, the reader to signaled to read the appendix for _all models, benchmarks, metrics, and evaluation protocols_ which is a 7 page appendix. Given the structure I find the details around the setup such as temperature used for reproduciblity.

- The paper performs extensive experiments, I would have appreciated some qualitative understanding of the results. For eg: in the evolutionary process, which node/parent cascaded the effect of unsafe behaviors.  

- Minor: Similar risks and mitigation have been discussed in many papers that are important to be included to better contextualize the paper:

[1] DeChant, Chad. "Episodic memory in ai agents poses risks that should be studied and mitigated." 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). IEEE, 2025.

[2] Hammond, Lewis, et al. "Multi-agent risks from advanced ai." arXiv preprint arXiv:2502.14143 (2025).

[3] Ecoffet, Adrien, Jeff Clune, and Joel Lehman. "Open questions in creating safe open-ended AI: Tensions between control and creativity." Artificial Life Conference Proceedings 32. One Rogers Street, Cambridge, MA 02142-1209, USA journals-info@ mit. edu: MIT Press, 2020.

[4] Sheth, Ivaxi, et al. "Safety is Essential for Responsible Open-Ended Systems." arXiv preprint arXiv:2502.04512 (2025).

### Questions
- The paper shows safety degrades after evolution, but doesn't always clearly establish that evolution caused the degradation vs. other factors i.e. if there is a causal effect.

- Why is there a disconnect between the experiments in the way different models are evaluated for each paradigm? What is the insight? Would have been good to point out that a particular model is more susceptible to lets say workflow misevolution vs memory misevolution.

- Could you ablate the trajectory of misevolution and what caused it?

- What evolutionary algorithms were used?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors investigate the phenomenon of misevolution, indicating the unintended degradation of alignment or safety in self-evolving LLM agents. The authors analyze four pathways through which such degradation emerges: (1) model self-training, (2) memory accumulation, (3) tool evolution, and (4) workflow optimization. Using several frameworks (AgentGen, AFlow, SEAgent, AutoGPT-Zero, etc.), the paper shows consistent safety decay across these dimensions: refusal rate drops, harmful or manipulative actions increase, and even high-end models (GLM-4.5, GPT-5, Gemini 2.5) exhibit drift after iterative adaptation.

### Strengths
Introduces a timely and original concept (i.e., misevolution) capturing long-term safety drift in self-evolving LLM agents. Provides a clear four-pathway taxonomy (model, memory, tool, workflow) that organizes an otherwise diffuse research space with empirical depth.

### Weaknesses
1. The paper frames misevolution as a "temporal" process yet only reports before/after metrics after a small number of evolution rounds (e.g., 20). The choice of “the number of iterations” appears inherited from the prior literature (e.g., N=20 from AFlow setup), but more fine-grained information regarding temporality of degrading safety could be useful. For instance, it is unclear whether degradation accumulates linearly, saturates, or oscillates over time. Even a small-scale longitudinal analysis could be useful to understand the dynamics of safety decay more fine-grained way.

2. The results show pronounced domain-level differences (e.g., lower safety in Finance and Science vs. higher in Service), so I hoped to see the interpretation of why misevolution manifests unevenly across domains. It was unclear whether these differences arise from the task structure, feedback signal design, or domain semantics.

### Questions
1. Could you provide a more fine-grained longitudinal analysis of misevolution, even on a small subset of agents (e.g., tracking safety metrics over all 20 evolution steps)? This would help clarify whether safety decay is cumulative or episodic.

2. The inter-domain variation in unsafe rates (Finance/Science vs. Service) is intriguing. Could you show a few qualitative examples or analyses that illustrate why certain domains are more prone to safety degradation?

### Soundness
4

### Presentation
3

### Contribution
3
