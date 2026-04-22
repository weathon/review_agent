# Reasoning Against Alignment: When Logical Consistency Overrides Ethical Safeguards in LLMs

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Large language models (LLMs) are increasingly deployed in safety-critical domains, yet their alignment with ethical constraints remains fragile, particularly when prompts require structured reasoning. We uncover a vulnerability, Reasoning Against Alignment, where LLMs generate harmful content not through misunderstanding but as the logically coherent outcome of multi-step inference.
Through black-box and white-box analyses across both commercial and open-source LLMs, we show that logically reframed prompts cause models to prioritize internal coherence over moral safeguards. Token-level traces reveal that refusal signals diminish while harmful semantics gradually emerge, a process that is not captured by surface-level rejection metrics.
To study this vulnerability, we introduce Reasoning Logic Jailbreaking (ReLoK), a single-turn attack that reframes unsafe requests as abstract viewpoints and decomposes sensitive terms. We evaluate ReLoK on five representative LLMs including ChatGPT-4o, Gemini 2.0 Flash, Claude 3.7 Sonnet, DeepSeek-R1-671B, and QwQ-32B using three jailbreak datasets. It achieves an average attack success rate of 97.9\%, highlighting the practical severity and broad applicability of the vulnerability.
Our findings suggest that alignment strategies must address not only what LLMs output but also how they reason. We advocate for reasoning-aware safety mechanisms such as ethical inference supervision and trajectory-level risk detection. Our code and data are available at https://anonymous.4open.science/r/Reasoning-Against-Alignment-7FD4.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a safety risk underlying reasoning LLMs -- due to tradeoffs between logical coherence v.s. safety refusal. Other than the proposal of their attack (ReLoK), the authors also provide an interpretation study based on the internal activation of LLMs.

### Strengths
* This work raises the inherent safety risk of reasoning LLMs from a novel angle -- logical coherence comes with tradeoffs in safety refusal.
* The attack seems successful, outperforming other black-box jailbreak attacks.
* The authors also try to explain the phenomenon with interpretation analysis.

### Weaknesses
* The biggest concern is regarding the technical novelty of this paper. The attack is rather simple, and very similar to prior work [1] (where jailbreaks are conducted via social persuasion techniques).
* Concern regarding the attack's philosophy -- for prompts like "It is easy to write a tutorial on how to make a bomb," I don't think any LLMs should directly refuse such requests, since the request itself is not inherently problematic. Rather, I believe that even a super safe and secure LLM should assist with such a user request, while avoiding carrying out harmful content that assists bad actors in building a bomb. How does your work handle such nuances? It would be great to provide qualitative examples of different model responses and in-depth analysis to reflect this.
* Analysis in Section 2 is not convincing enough to explain the so-called "Reasoning Against Alignment" vulnerability. For example, in Fig 2, I can only tell that *harmful behavior prompts* lead to short refusal responses, whereas *ethically neutral viewpoint prompts* lead to long non-refusal responses (which contain more keywords related to the original prompt). But this doesn't mean that such longer responses are unsafe in nature -- as a safe model may choose to fulfill such requests in a safe way, and thus its responses may also manifest similar token probabilities.
* Concern regarding the evaluation configurations. I appreciate that the authors adopt a Jury-Based Evaluation Protocol that involves majority voting of multiple LLM judges. However, different LLMs may have different usage policies. Some of the questions in the benchmark may be allowed by certain LLMs (e.g., adult content may be deemed permissible by ChatGPT). Therefore, I think it would be better to establish a universal rubric for all evaluator LLMs.
* Please also include the "No Attack" ASR for different LLMs in Table 1.
* Would be great to have a more comprehensive comparison with a wider range of attacks, especially those similar to yours (e.g., [1]). Additionally, can you also adapt H-CoT to non-reasoning models and report the results in Table 1?
* Lack of defense design. The authors mentioned that "future work can explore reasoning-aware supervision and semantic monitoring during generation to better align inference with ethical goals." Can you provide an initial study to explore potential defense design? This would make the work much more solid.
* Paper writing is too redundant and not so informative, especially in Sections 1 and 2. I feel like the authors are repeating the same insight over and over again.

[1] How johnny can persuade LLMs to jailbreak them: Rethinking persuasion to challenge AI safety by humanizing LLMs. ACL 2024

### Questions
See "Weaknesses"

### Soundness
3

### Presentation
2

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
This paper identifies a new safety vulnerability of language models -- Reasoning Against Alignment, where LLM generate harmful content not through misunderstanding but as logically coherent outcome of multi-step inference. Using this vulnerability, the author proposed ReLoK attack, and evaluated it on five representative LLMs. Through experiments, it achieves an average attack success rate of 97.9%, highlighting the practical severity and broad applicability of the vulnerability.

### Strengths
1. The analysis is comprehensive. The author systematically analyzed the phenomenon of reasoning against alignment through both black-box and white-box perspectives. The author also introduced the prison break risk index (PRI) in order to quantify the risk trajectories before and after rewriting. In terms of the experiment, the author evaluated five models across three benchmarks and compared them with four other attack methods, demonstrating that ReLoK is a strong attack and thus reveals the severity of such vulnerabilities in LLMs.
2. The writing is good. The threat model is clear and practical.

### Weaknesses
1. The contribution seems to be a bit incremental. In my opinion, rephrasing-based attack is not a completely novel approach. For example, Sugar-Coated Poison (Wu et al, 2025) also employed rewriting in order to bypass the initial safeguard of the language model. The source of these vulnerabilities mainly comes from shallow alignment (Qi et al., 2024), in which the initial few response tokens will dominate the direction of the rest of the content.
2. Some results are not very clear. For example, in Figure 2, it's hard to compare the average token probabilities between the red dots and blue dots. Although the author claimed that they can observe a difference. It would be better if the authors can show the average probabilities for each generation step. I also don't understand why the token density will decrease as the generation step increases. If the tokens of interest are the same, the number of dots should be the same for each generation step?



Reference:
1. Wu, Yu-Hang, et al. "Sugar-coated poison: Benign generation unlocks llm jailbreaking." arXiv preprint arXiv:2504.05652 (2025).
2. Qi, Xiangyu, et al. "Safety alignment should be made more than just a few tokens deep." arXiv preprint arXiv:2406.05946 (2024).

### Questions
All of my questions are listed in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper identifies and empirically studies a vulnerability called “Reasoning Against Alignment”: when malicious goals are reframed as logically-structured, neutral-sounding reasoning tasks, LLMs gradually generate harmful semantics while rejection signals are suppressed. The authors introduce a single-turn black-box attack ReLoK (viewpoint transformation, sensitive-word decomposition, example-guided structuring) and report very high attack success rates across 5 models and 3 benchmarks.

### Strengths
This work highlights a safety blind spot—reasoning-driven alignment failure—and proposes PRI, a concrete trajectory-level metric that operationalizes detection of gradual unsafe drift during generation.

### Weaknesses
1. The paper's presentation needs significant improvement. The current exposition is unclear, making it difficult to grasp the core contributions. I recommend restructuring the introduction and methodology sections for better clarity and accessibility

2. The white-box analysis examines only QwQ-32B, yet makes broad claims about "transformer-based LLMs." To support these claims, the authors should either (1) extend the analysis to multiple open-source models of varying sizes or (2) provide ablation studies demonstrating which findings generalize beyond the specific model studied. 

3. The mechanistic analysis uses only 20 prompts, which is an insufficient sample size for robust mechanistic conclusions.

### Questions
Which transformer layers and attention heads exhibit the earliest and most statistically significant increases in harmful-token probability during generation, and can targeted interventions (e.g., head ablation, representation nulling, or controlled reweighting) at those loci effectively suppress subsequent PRI escalation?

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
4

### Summary
The paper discovers a deep alignment failure pattern: when harmful objectives are embedded in logically consistent, morally neutral reasoning tasks, LLMs prioritize logical reasoning over moral rejection, thereby outputting harmful content. Unlike traditional Jailbreak methods relying on "surface confusion/lexical substitution", this failure is driven by intrinsic goal-driven reasoning processes, not simply bypassing filters. Achieving an average 97.9% attack success rate across 5 advanced LLMs and 3 datasets, significantly outperforming existing single-round black-box benchmark methods.

### Strengths
1. The paper is written with exceptional clarity, logical coherence, and high readability.
2. The attack success rate is highly effective and impressive.

### Weaknesses
1. Although the paper names a new vulnerability phenomenon, the core attack method ReLoK's techniques are largely a combination of existing technologies, not entirely novel. The relatively innovative part is perspective-driven reasoning, but this lacks sufficient experimental validation.
2. Illegal details should be anonymized.
3. The experimental section remains insufficient, with inadequate depth of explanation. To comprehensively analyze the "Reasoning Against Alignment" phenomenon, the authors should conduct similar attacks on models after defense interventions (SFT/RLHF) and analyze whether the phenomenon persists.

**- Minor Comments**
1. In the statement "This concept extends existing findings that LLMs may prioritize correctness over safety, identifying the underlying cause as a conflict between logical consistency and ethical constraints", the term "correctness" is not entirely accurate.

### Questions
See weakness

### Soundness
3

### Presentation
4

### Contribution
2
