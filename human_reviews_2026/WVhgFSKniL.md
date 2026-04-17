# ChatInject: Abusing Chat Templates for Prompt Injection in LLM Agents

- Decision: Accept (Poster)
- Scores: 6, 8, 2, 4

## Abstract
The growing deployment of large language model (LLM) based agents that interact with external environments has created new attack surfaces for adversarial manipulation. One major threat is indirect prompt injection, where attackers embed malicious instructions in external environment output, causing agents to interpret and execute them as if they were legitimate prompts. While previous research has focused primarily on plain-text injection attacks, we find a significant yet underexplored vulnerability: LLMs' dependence on structured chat templates and their susceptibility to contextual manipulation through persuasive multi-turn dialogues. To this end, we introduce ChatInject, an attack that formats malicious payloads to mimic native chat templates, thereby exploiting the model's inherent instruction-following tendencies. Building on this foundation, we develop a template-based Multi-turn variant that primes the agent across conversational turns to accept and execute otherwise suspicious actions. Through comprehensive experiments across frontier LLMs, we demonstrate three critical findings: (1) ChatInject achieves significantly higher average attack success rates than traditional prompt injection methods, improving from 5.18% to 32.05% on AgentDojo and from 15.13% to 45.90% on InjecAgent, with multi-turn dialogues showing particularly strong performance at average 52.33% success rate on InjecAgent, (2) chat-template-based payloads demonstrate strong transferability across models and remain effective even against closed-source LLMs, despite their unknown template structures, and (3) existing prompt-based defenses are largely ineffective against this attack approach, especially against Multi-turn variants. These findings highlight vulnerabilities in current agent systems. The code is available at https://github.com/hwanchang00/ChatInject.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes ChatInject, a family of indirect prompt-injection attacks against LLM agents that exploit chat-template role tags (e.g., system/user/assistant delimiters) and persuasive multi-turn dialogues to bypass instruction hierarchies. Compared to plain-text injections, ChatInject wraps attacker content to mimic native chat templates; a multi-turn variant ("Multi-turn + ChatInject") primes the model across turns. Across InjecAgent and AgentDojo benchmarks, the authors report large gains in attack success rate (ASR)  and show that attacks transfer across models, including closed-source ones, and evade common prompt-based defenses.

### Strengths
+ The paper frames how forged role tags inside tool outputs can be misinterpreted as higher-priority instructions, bypassing system, user, assistant hierarchies—distinct from prior plain-text injections.

+ The four variants—Default InjecPrompt, InjecPrompt+ChatInject, Default Multi-turn, and Multi-turn+ChatInject—are defined with a consistent template.

+ The transferable attack is an interesting and important angle, which is worth further study.

### Weaknesses
- While behaviorally persuasive, the paper lacks a deeper mechanistic explanation of why template tokens grant authority (e.g., attention patterns, representation shifts).

- Multi-turn dialogues are LLM-generated (GPT-4.1) and human-reviewed, but not drawn from real attacker corpora. This leaves ecological validity open: would organic, messy content sustain similar gains?

### Questions
1. If the same role tags are injected but tokenized differently (e.g., via Unicode homoglyphs or byte-level encodings), how do models react?

2. Have you validated multi-turn persuasion with human-written or naturally occurring sequences (e.g., scraped forum/HTML excerpts) to assess ASR/Utility differences vs. GPT-generated dialogues?

3. Table 2 shows higher transfer to certain closed models but less Utility degradation in some settings. Can you hypothesize which agent-stack choices (e.g., tool arbitration, post-filters) helped preserve Utility? Any evidence from ablations?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a novel type of indirect prompt injection attacks against language models (LMs) deployed in agentic settings. The attack exploits the prompt templates used by the models. More specifically, the proposed prompt injections are wrapped as single- or multi-turn conversations in the LM native prompt template using the special tokens from the template. This results in a "privilege escalation," where text that is supposed to be treated by the LM as a tool output gets interpreted as a higher-level user or system instruction.

The attack is demonstrated to be highly effective against several near-frontier LMs on two benchmarks. Injections also transfer non-trivially to LMs with different prompt templates. A method for making the attack universal against an unknown template is also demonstrated. Several existing defenses are tried against the new attack. Finally, a separate defense of detecting the malicious template in the payload is considered, and a counter-attack with randomizing injection characters is evaluated.

### Strengths
- The paper is clearly written, with well-defined contributions and a logical red team-blue team structure. The figures are also helpful and aesthetically pleasing.
- The attack vector is, to my knowledge, novel.
- Evaluation is comprehensive (6 open-source LLMs and 3 closed-source LLMs). Reasonable ablations are performed (single- vs multi-turn, tool call hooks, reasoning hooks).
- I appreciated the fine-grained analysis of ASR against prompt template similarity, which clearly shows that the attack success is driven by guessing the correct template, rather than some unknown mechanism related to OOD inputs.
- A method for making the attack independent of the template shows that even an attacker with a few attempts before being discovered and blocked by the provider might succeed with this attack.
- Reasonable blue team counteractions are considered, although I am less certain here. The counter-measure of detecting prompt templates  and counter-counter-measure of introducing noise are very logical.

### Weaknesses
I'm not sure if the discussion of the possible defenses in Section 6.1 is comprehensive. For example, the paper discusses the ProtectAI prompt injection detector and shows that, although ASR under this detector drops significantly, high FPR of the detector results in significant utility drops. Since the paper tests closed-source LMs for vulnerabilities, it might also be worth it to try proprietary defense mechanisms against prompt injections, like the Lakera Guard.

Generally, the paper does not always report error bars, and when it does, the bars seem rather wide, especially given that std is reported instead of 95% CI. For the ASR values in tables 1 & 2 and in Fig. 3, one simple way to report the CI would be to use the Wilson interval formula, which only depends on the ASR and the size N of the dataset. Given that rerunning the queries might be expensive, this could provide a cheap proxy for at least one source of the noise (due to dataset size but not due to LLM generation distribution). I understand that the tables could then become cluttered, so the confidence intervals for at least some of the values could be put in the appendix. Relatedly, one expects the pi detector to result in consistently lower ASR. This is true for all bars but one (multi-turn with Grok 3). Is this outlier due to noise? Having the CI would help here too.

Current related work discussion is too small. If the paper is accepted, I would recommend to spend some of the extra page allowance on expanding to include more work on indirect prompt injections, instruction hierarchy, and other relevant topics, eg [1,2,3]. Prompt injection literature is rather rich.

[1] Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz.
Not what you’ve signed up for: Compromising real-world llm-integrated applications with indirect
prompt injection. In Proceedings of the 16th ACM workshop on artificial intelligence and security,
pp. 79–90, 2023.

[2] Zverev, Egor, et al. "Can llms separate instructions from data? and what do we even mean by that?." arXiv preprint arXiv:2403.06833 (2024).

[3] Florian Tramer, Nicholas Carlini, Wieland Brendel, and Aleksander Madry. On adaptive attacks to
adversarial example defenses. Advances in neural information processing systems, 33:1633–1645,
2020.


Minor nit: L374 "remains strong transferability" -> "retains strong transferability"

### Questions
- What are the data sample sizes for the two benchmarks that the paper considers? This could be useful to report in Sec. 3.3 (this subsection would also fit better in the beginning of Section 4 in my opinion, although this is a minor point). 
- What is the reason for missing values in Table 1? My guess is that this is is due to the models not having the capabilities to think / perform tool calls. It would be good to report in the table caption.

### Soundness
4

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
4

### Summary
This paper presents ChatInject, a new attack that enhances indirect prompt injection (IPI) in LLM-based agents by exploiting chat template structures (e.g., \<system\>, \<user\>, \<assistant\> tags) and multi-turn persuasive dialogues.
The first component (ChatInject) formats malicious payloads to mimic the model’s native chat template, making the model misinterpret the injected content as higher-priority instructions.
The second component (Multi-turn variant) uses persuasion-driven multi-turn dialogues to gradually legitimize malicious requests.
Extensive experiments on InjecAgent and AgentDojo benchmarks across 9 LLMs (both open- and closed-source) show substantial improvements in attack success rate (ASR) and cross-model transferability. The paper also evaluates standard defenses, finding them largely ineffective.

### Strengths
- The paper is clear and well-organized.
- The proposed attack is effective, increasing the ASRs significantly across different datasets and backbone models.
- Comprehensive experiments across multiple models, datasets, and defenses provide strong empirical support. The analysis is rich, covering template similarity, cross-model transfer, and robustness to perturbations, making the study thorough from an empirical perspective.

### Weaknesses
- The technical novelty is limited: the work mainly enhances IPI by reformatting or rephrasing injected content, and the ideas of leveraging chat templates or persuasive multi-turn dialogues have been conceptually explored in prior works. 
- ChatInject and Multi-turn components appear somewhat loosely connected —one manipulates structure, the other context; they are combined mainly for empirical gain without a unified theoretical justification.

### Questions
- Could a simple defense that detects the presence of special chat template tokens in external content effectively identify such attacks without significantly reducing utility?

### Soundness
2

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
3

### Summary
This paper introduces ChatInject, a novel indirect prompt injection attack that exploits chat template structures used in large language model (LLM) agents. Unlike prior prompt injection work focusing on plain-text manipulation, ChatInject forges system/user/assistant role tags to bypass instruction hierarchies and embeds malicious commands in persuasive multi-turn dialogues. Through extensive experiments on nine LLMs (both open- and closed-source) across AgentDojo and InjecAgent benchmarks, the authors show that ChatInject achieves dramatically higher attack success rates and maintains strong transferability even without template knowledge. They also show existing defenses are largely ineffective, motivating the need for stronger mitigations.

### Strengths
- Novel threat model with multi-turns persuasive which is indeed a solid area of study and contribution. 
- Comprehensive evaluation and Strong empirical results
- Clear writing and strong figures: Figures and tables effectively demonstrate the mechanisms and results (e.g., Figs. 1–6).

### Weaknesses
- The main limitations are lack of novelty and lack of baselines. - The contributions in prompt creation seems incremental, limiting its novelty. For example a missing related work: xteaming: https://arxiv.org/pdf/2504.13203 or Foot-In-The-Door, which is referred in the paper but not considered in the exp. 

- The treat model needs more clarity: I did not undergard from which point the attacker gets access to change the prompts.  

- Contribution is only in the attack side, nothing in the defense side. 
- I am not if the utility gets lower then how they are not detected easily? Intuitively, this should be easier to detect. 
- Table 2 is hard to follow, i think adding a simple avg per row can make it more readable.  

- Synthetic dialogue generation: Multi-turn persuasion data are LLM-generated (GPT-4.1) and manually filtered; realism and diversity might 

- Interpretability gap: In Fig 1 I found the prompts with not readbale tokens but reading the contribution does not give clear idea when they are not readbale and when. Alos this makes it less interpretable and somewhat detectable.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
