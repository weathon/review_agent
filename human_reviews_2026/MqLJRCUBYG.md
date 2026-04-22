# Indirect Prompt Injections: Are Firewalls All You Need, or Stronger Benchmarks?

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
AI agents are vulnerable to indirect prompt injection attacks, where malicious instructions embedded in external content or tool outputs cause unintended or harmful behavior. Inspired by the well-established concept of firewalls, we show that a simple, modular and model-agnostic defense operating at the agent–tool interface achieves perfect security (0\% or the lowest possible attack success rate) with high utility (task success rate) across four public benchmarks: AgentDojo, Agent Security Bench, InjecAgent and $\tau$-Bench, while achieving a state-of-the-art security-utility tradeoff compared to prior results. Specifically, we employ a defense based on two firewalls: a Tool-Input Firewall (Minimizer) and a Tool-Output Firewall (Sanitizer). Unlike prior complex approaches, this firewall defense makes minimal assumptions on the agent and can be deployed out-of-the-box, while maintaining strong performance without compromising utility.  However, our analysis also reveals critical limitations in these existing benchmarks, including flawed success metrics, implementation bugs, and most importantly, weak attacks, hindering significant progress in the field. To foster more meaningful progress, we present targeted fixes to these issues for AgentDojo and Agent Security Bench while proposing best-practices for more robust benchmark design. Further, we demonstrate that although these firewalls push the state-of-the-art on existing benchmarks, it is still possible to bypass them in practice, underscoring the need to incorporate stronger attacks in security benchmarks. Overall, our work shows that existing agentic security benchmarks are easily saturated by a simple approach and highlights the need for stronger agentic security benchmarks with carefully chosen evaluation metrics and strong adaptive attacks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a firewall-based defense method against indirect prompt injection attacks. Specifically, the authors design two LLM-based firewalls: (1) a Tool-Input Firewall that minimizes unnecessary confidential information from tool input arguments, and (2) a Tool-Output Firewall that sanitizes returned tool observations from potentially malicious content. Both firewalls are implemented using LLMs with specifically designed prompts.

The authors evaluate their method on AgentDojo, Agent Security Bench, InjecAgent, and Tau-Bench. The results show that the approach effectively reduces attack success rates while preserving benign utility to some extent. Finally, the authors analyze the strengths and weaknesses of existing prompt injection benchmarks and propose suggestions for improvement.

### Strengths
1. The method is simple and effective, requiring no additional training to achieve defense capabilities.
2. The evaluation is comprehensive, covering commonly used datasets in the field.

### Weaknesses
1. The novelty is limited. The idea of using LLMs themselves to filter harmful information has been extensively explored in both jailbreak defenses and prompt injection defenses. The authors should clearly articulate what distinguishes their approach from prior work beyond the specific application to tool-input and tool-output filtering.
2. The method causes a degradation in benign utility. As shown in Table 1, benign utility drops dramatically from 83.02% without defense to only 58.41% with the firewall. This suggests the firewall is probably conservative and filters out valuable information necessary for legitimate task completion. The authors should provide a detailed analysis of what types of benign tasks are being incorrectly blocked and whether this trade-off is acceptable in real-world deployments.
3. The paper lacks important analysis on practical deployment considerations: (1) Computational overhead: Since both firewalls are LLM-based, what is the additional latency and cost introduced? This could be prohibitive for real-time applications.
(2) False positive/negative analysis: As mentioned in 2, the paper should provide detailed statistics on misclassification rates to better understand the firewall's reliability.

### Questions
Please refer to the weaknesses part.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a simple defense against indirect prompt injection (IPI) attacks in tool-using LLM agents. It introduces two lightweight modules — a Tool-Input Firewall (Minimizer) and a Tool-Output Firewall (Sanitizer) — that filter inputs and sanitize tool outputs. Experiments on several benchmarks (AgentDojo, ASB, InjecAgent, τ-Bench) show near-zero attack success rates. The paper also points out flaws in existing benchmarks and suggests improvements.

### Strengths
The paper is clearly written and easy to follow.

The idea is simple and practical — the defense can be easily applied without retraining.

The benchmark analysis is detailed, and the authors identify real issues in existing evaluation setups.

### Weaknesses
The core idea (“minimize & sanitize”) is too simple and incremental, offering little novelty beyond existing “firewall” or “guardrail” defenses.

Most results come from benchmarks that the authors themselves criticize as flawed, so the findings feel self-contradictory and less convincing.

The paper lacks deeper insight or analysis about why the approach works and how it generalizes.

The work doesn’t propose new attack strategies or theoretical understanding — it’s mainly an engineering evaluation rather than a research contribution.

Overall, the contribution feels minor; it’s a straightforward application rather than a new idea.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a two-fold contribution. First, it proposes a simple, model-agnostic defense against indirect prompt injection (IPI) attacks called the "minimize & sanitize" firewall. This defense consists of a "Tool-Input Firewall" (Minimizer) to filter sensitive data from tool inputs and a "Tool-Output Firewall" (Sanitizer) to remove malicious instructions from tool responses. The authors demonstrate that this defense, particularly the Sanitizer, achieves state-of-the-art (SOTA) results, reducing the attack success rate (ASR) to ≈0 on AgentDojo and $\tau$-Bench and to ≲0.3 on InjecAgent. The paper's second, and arguably more significant, contribution is a rigorous critique of these same benchmarks. The authors reveal that the SOTA results are largely an illusion, as the benchmarks suffer from critical flaws, such as "forced attack-tool injection" in ASB and brittle utility metrics in AgentDojo. This makes them poor evaluators of true security. To prove their point, the authors develop a stronger, obfuscated (Braille-based) attack that successfully bypasses their own SOTA firewall, thereby highlighting the urgent need for stronger, more realistic security benchmarks.

### Strengths
1. The paper's primary strength is its dual contribution. It not only proposes a simple, effective, and model-agnostic defense (the Minimizer-Sanitizer firewalls) but also provides a rigorous critique of the very benchmarks used to measure success. 
2. This paper uncovers flaws in ASB and AgentDojo that distort ASR and utility, and provide concrete fixes to make evaluations more trustworthy.
3. The proposed firewall defense is commendable for its simplicity and practicality. As a modular, model-agnostic approach, it serves as an excellent and easily replicable baseline.

### Weaknesses
1. Potential Data Contamination. While the reported results of the proposed defense are strong, this method relies primarily on frontier models (GPT-4o and Qwen3), and the paper does not analyze potential training–evaluation contamination (prior exposure to attack styles or benchmark artifacts). Could you replace the Minimizer/Sanitizer with an older model and report the performance? This would show whether the defense truly depends on frontier-model memorization.

Overall, this is an interesting and valuable paper that productively revisited progress on benchmarking prompt-injection attacks and defenses. It would be even better with a quantitative treatment of optimization-based adaptive attacks [1, 2]. Conceptually, these attacks should also serve as strong baselines, especially since many defenses in current benchmarks are largely static and plausibly vulnerable to adaptive optimization (e.g., tuning an adversarial suffix). 

[1] Liu, Xiaogeng, et al. "Automatic and universal prompt injection attacks against large language models." arXiv preprint arXiv:2403.04957 (2024). \
[2] Pasquini, Dario, Martin Strohmeier, and Carmela Troncoso. "Neural exec: Learning (and learning from) execution triggers for prompt injection attacks." Proceedings of the 2024 Workshop on Artificial Intelligence and Security. 2024.

### Questions
For rebuttal, please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper claims to provide a simple, effective, modular and model-agnostic defense for tool-calling agents based on: a Tool-Input Firewall (Minimizer) and a Tool-Output Firewall (Sanitizer). They demonstrate that their approach achieves 0% or the lowest attack success rate on four public benchmarks while maintaining high task success. They also found and fixed key flaws in widely popular benchmarks to enable more reliable evaluation in the agent attack community. Finally, they also provided case study about the failure of their own method to call for more stronger defenses.

### Strengths
1. The contribution of fixing existing benchmarks is very useful for future benchmarking in this field.

2. The method seems to work well on the 4 benchmarks, making them strong candidates for agent security defense.

3. this paper is easy to understand, and the demonstration is very good and illustrative.

### Weaknesses
1. The method is, despite its fancy names, imo, a pre-processor and a post-filter, which is not new. And I did not see any necessity to associate it with firewalls as inspiration.

2. The two filters (pre&post) seem only been built by a very short system prompt. thus, it's questionable why those system prompts serves for the claim of 'firewall is equipped with a robust system prompt'. How do you justify this? why that system prompt is robust? how did you choose those system prompt?

3. In 7 DISCUSSION, I understand stronger attacks can unsuperisingly bypass your firewalls. But how about other baselines? does the same attack also succeed on other baselines or it a flaw of your own methods? more discussion on this would be good.

### Questions
1. How about finetuning the two filters? Why is the system prompting enough?

2. What's the difference if the backbone model is not gpt4o? Any trade-off analysis?

### Soundness
2

### Presentation
3

### Contribution
2
