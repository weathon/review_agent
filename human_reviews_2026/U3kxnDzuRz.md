# Deep-Cover Agents: Long-Horizon Prompt Injections on Production LLM Systems

- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Instruction-following LLM assistants that read untrusted data are susceptible to prompt injection, wherein a malicious actor injects a harmful request that the assistant naively complies with, to the user's detriment. We analyze the structure of tool-using LLM agents to create a descriptive framework for prompt injection attacks. By examining this framework, we find that certain attack modalities are understudied, and observe important trends in attack performance as we vary how prompt injection attacks are introduced and their token budget with practical takeaways. Importantly, previous work does not significantly explore the dimension of time, and we make the key finding that after being prompt-injected, many agents can behave benignly for 50+ conversation turns before taking a malicious action. Finally, we validate our work by executing sandboxed attacks against deployment systems such as Claude Code and Gemini-CLI. Our attacks readily succeed, and additionally reveal as-yet undocumented emergent behavior in these models' responses to prompt injection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a framework for analyzing prompt injection attacks against tool-using LLM agents that process untrusted data. The authors identify understudied attack modalities and examine how attack performance varies based on injection method and token budget, yielding practical insights for both attackers and defenders. 

A key finding is the discovery of "delayed" prompt injection attacks, where compromised agents can behave normally for over 50 conversation turns before executing malicious actions—a temporal dimension largely unexplored in prior work. The research is validated through successful sandboxed attacks against real-world deployment systems including Claude Code and Gemini-CLI, which also reveal previously undocumented emergent behaviors in these models' responses to injection attempts.

### Strengths
1.The paper makes an important and understudied contribution by investigating how prompt-injected agents can exhibit delayed malicious behavior. This temporal dimension has been largely overlooked in prior work and has implications for detecting and mitigating prompt injection attacks in real-world deployments, as it suggests that simple immediate monitoring may be insufficient.
2. Prompt injection is a critical security concern for tool-integrated agents, and this paper identifies understudied attack modalities and examine how attack performance varies based on injection method and token budget, yielding practical insights for both attackers and defenders.

### Weaknesses
I think the main problem is the limited novelty in threat model and the contribution is limited as well.

1. The work appears to focus on direct prompt injection where attackers can modify system prompts, which is impractical. Real-world scenarios involve indirect prompt injections through untrusted data (emails, documents, web pages). The threat model may be too strong and less relevant than existing work on indirect attacks.
2. The delayed/time-based attacks (behaving benignly for 50+ turns) seem more like an interesting observation than a critical security concern. If we cannot effectively defend against immediate prompt injections, delayed attacks are a secondary issue. The practical significance is unclear.
3. The evaluation relies solely on ASR, while comprehensive benchmarks should also measure utility/benign task performance. This is critical for assessing the practical trade-offs of defenses and the true impact of attacks on system usability, for example agentdojo. 

4. Figure 1 is not referenced or explained in the text. Figure 3 (top left) lacks a legend making it difficult to interpret. Additionally, there are formatting issues with incorrect quotation mark usage (line 38 and 43).

### Questions
no

### Soundness
2

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
The paper proposes a framework for analyzing prompt injection attacks on tool-using LLM agents. It characterizes the relevant factors an adversary should consider while crafting such attacks including Target, Stealth, Vector, Budget and Timing. It also proposes timing as a new characteristic of prompt injection attacks. It provides reasonable amount of experimental results to observe some trends for each of the factors.

### Strengths
1. The paper takes an important step towards characterizing the relevant factors for the success of prompt injection attacks.

2. The paper attempts to provide experimental results for each of the proposed factors.

3. The paper presents attacks on real-world deployments including Claude Code and Gemini-CLI.

4. The writing is clean and easy to follow.

### Weaknesses
1. The experimental results have too much variation across models, it doesn't seem fair to draw any single conclusion from the average behaviors (refer Figure 3). Exploring the possible reasons for such huge fluctuations is an important task to claim anything about the transferability of the conclusions.

2. The paper doesn't discuss about the stealth (one of the factors in framework) concretely in the experiments.

3. The target for all the experiments is again fixed which might raise concerns about the transferability of conclusion (mentioned in the limitations as well).

4. All these factors are not independent, studying the correlation between factors would be important to actually gauge their importance. For example, the experimental results for the timing implies that it is optimal to provide the trigger context and the opportunity to corrupt the response at the same time, is there any benefit in keeping them separate in terms of stealth ?

### Questions
1. Is there any analysis/conclusions for each LLM that may provide adaptive attacks and defenses tailored to each model ?

### Soundness
2

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
4

### Summary
This paper studies long-horizon prompt injection attacks on LLM-based agents, focusing on cases where malicious instructions injected into the context may activate dozens of turns later. The authors propose a five-axis framework (Target, Stealth, Vector, Budget, Timing) to describe prompt injection attacks, conduct controlled experiments on synthetic multi-turn dialogues, and test zero-shot transfer to real deployed systems such as Claude Code and Gemini-CLI. The results suggest that some commercial systems remain vulnerable even after many turns, and that reasoning depth can affect attack success rate.

### Strengths
The problem setting of studying long-horizon prompt injections is novel and intuitively important, especially as LLM agents increasingly operate in multi-turn environments.

### Weaknesses
- Lack of formalization: The notion of “long-horizon prompt injection” is only loosely defined. The paper does not provide a clear formal threat model, trigger definition, or precise criteria for what constitutes a long-horizon attack versus a normal prompt injection. The attacks resemble backdoor-style conditional activation rather than classical prompt injection, but the paper does not clearly justify this distinction.

- Evaluation methodology is non-standard and underspecified with limited diversity of the evaluation set: Only one attack type (inserting remote code execution vulnerabilities) is studied. The dataset lacks diversity across tasks, modalities, or realistic user contexts.

- Writing and presentation issues: The paper’s structure and language are below top-conference standard.

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present empirical analyses of long-horizon prompt injections, those in which the trigger and the attack opportunity are separated by a large number of conversation steps, and find that simple attacks are successful in this context; for certain models, this is true across system prompt, user message, and tool response attack vectors.

### Strengths
- The authors present a large number of useful experimental results, which indicate that a large number of commonly deployed language models are vulnerable to long-horizon attacks.
- Results on real-world systems (e.g. Claude Code) are presented, and the examples of malicious behavior are useful.

### Weaknesses
- The conclusions drawn from the results by the author are not fully substantiated and generally speculative. While "testing a model's resistance to prompt injection at t=0 may not provide a strong indication.." [307] is intuitive, the results in Figure 3 do not clearly justify this conclusion, and generally show that long-run ASRs fluctuate about the short-run values for the majority of values. Similarly, while strong effects are visible for certain LLMs, the effect of reasoning effort in Figure 4 appears to be weak for the majority of models. It is unclear whether variance impacts these results; the experiments should be conducted over multiple runs and standard hypothesis tests should be performed.
- The results for ASR vs attack budget in Figure 3 actually measure the effects of summarization on the specific prompt injections used, not the adversary's budget, and should be described as such. Performance of a worst-case adversary should never decrease with budget.
- The results are highly variable by model, and hence it is unclear to what exent these results generalize.
- While the results presented here are a useful first analysis of long-horizon attacks, the notion of "long horizon" considered in this work is somewhat limited, and doesn't consider more subtle behavior, e.g. where the desired behavior is to delay defection and behave benignly initially.

### Questions
In Figure 4, ASR is, as might be expected, by far the highest when the attack opportunity is immediately following the trigger. How does this vary over a shorter time horizon (e.g. for additional timesteps in the 40-50 range)?

### Soundness
2

### Presentation
3

### Contribution
2
