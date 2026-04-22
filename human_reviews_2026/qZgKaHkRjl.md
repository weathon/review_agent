# Agentic reinforcement learning for search is unsafe

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 2, 8, 6

## Abstract
Agentic reinforcement learning (RL) trains large language models to autonomously call tools during reasoning, with search as the most common application. These models excel at multi-step reasoning tasks, but their safety properties are not well understood. In this study, we show that RL-trained search models inherit refusal from instruction tuning and often deflect harmful requests by turning them into safe queries. However, this safety is fragile. Two simple attacks, one that forces the model to begin response with search (Search attack), another that encourages models to repeatedly search (Multi-search attack), trigger cascades of harmful searches and answers. Across the two most widely used model families (Qwen, Llama), multiple scales (3B to 32B), popular RL algorithms (PPO and GRPO), and both local and web search, these attacks reduce refusal rates by up to 60.0%, answer safety by 82.5%, and search-query safety by 82.4%.
The attacks succeed by triggering models to generate harmful, request-mirroring search queries before they can generate the inherited refusal tokens. This exposes a core weakness of current RL training: it rewards continued generation of effective queries without accounting for their harmfulness. As a result, standard RL for search training can override instruction-tuned safety to generate unsafe searches. This leaves RL search models with vulnerabilities that users can easily exploit, making it urgent to develop safety-aware agentic RL pipelines optimising for safe search behaviour.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper researches the safety aspect of agentic reinforcement learning. The authors first train the LLM agents from Qwen2.5-7B models and Llama-3.2.3B models through RL and conduct extensive experiments. They conduct extensive studies on both search attacks and multi-search attacks. They find that both attacks degrade the performance, while the agents trained from the IT models are more resistant to attacks. They also find that one harmful search is enough to jailbreak, and iterative searches can lead to more harm.

### Strengths
1. The paper researches an interesting problem of whether the model trained from agentic RL is safe or not.
2. The authors conduct extensive experiments to research the problem from an empirical perspective and provide insights.
3. The paper is quite novel.

### Weaknesses
1. My main concern with this paper is whether the conclusion from the paper can be generalized.
- Firstly, only small-sized LLMs are approached in this paper. It is quite possible that larger models, even the pretrained checkpoint, are better at preventing jailbreaking and thus perform better with regard to attacks.
- Secondly, the experiments are only conducted on Qwen2.5 and Llama3.2 models. It is possible that these two types of models are not trained with a large amount of safety data during pretraining.

2. I feel that the conclusion drawn in the paper largely depends on what LLMs are used for experiments, and their behaviors are determined by the training data they saw during pretraining.

### Questions
1. Do you think the conclusions from the paper can be generalized to larger model sizes and different types of LLMs?
2. I am not sure if the research is valid in this paper since we can always alleviate the problem by adding more safety-related data during pretraining or after RL to alleviate?
3. Is it true that the research conclusions drawn from the paper are customized for Qwen2.5 and Llama3.2, which are determined by the training data for these two types of models, and are hard to generalize?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper examines safety in agentic RL for search: LLMs trained with PPO to alternate between reasoning and tool use (“think → search → answer”) over a local Wikipedia index or web search. Using Qwen-2.5-7B and Llama-3.2-3B (base and instruction-tuned), it optimizes exact-match on HotpotQA and then evaluates 299 harmful prompts. Safety is judged by Prometheus-7B-v2 on refusal, answer safety, and search safety (scaled 0–100). Two simple interventions—forcing an initial search or many searches—significantly erode refusal and safety compared with instruction-tuned baselines.

### Strengths
- Safety of RL-trained tool-use agents (especially search) is under-examined and relevant for deployment.

- The three-metric evaluation (refusal, answer safety, search safety) provides nuanced assessment. In addition, findings hold across two model families, both local and web search, demonstrating some generalizability. 

- Prefill/prompt tweaks are realistic (user-accessible) and isolate a when-to-search failure mode.

### Weaknesses
- Novelty & Impact. The core finding feels incremental and closely aligned with well-known RAG/jailbreak dynamics: if you can steer retrieval early, you can bias generation toward unsafe outcomes. 

- Usefulness of the Study: The attack surface here (forcing <search> / multi-search) is quite simple, and the experiments use relatively not strong, non-SOTA models. As a result, it’s unclear whether the phenomenon meaningfully persists, and to what degree, in production-grade systems (e.g., Claude, Gemini, ChatGPT) that already deploy safety scaffolds and gated tool use. Without evidence on stronger models or more realistic threat models, I’m not convinced the contribution is broadly useful.

- The RL setup (PPO on HotpotQA with tool tokens) is quite specific and not clearly representative of how real systems are trained/deployed today.

- RL reward optimizes exact match on HotpotQA, but safety is evaluated on a separate harmful-prompt distribution; this mostly diagnoses reward mis-specification rather than demonstrating a general vulnerability.

Without stronger evidence of why the findings in the study are useful even on SOTA models, I am less confident that the proposed attacks are useful.

### Questions
See above.

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
2

### Summary
This paper tests the safety of RL to train LLM-based agents with access to search. 

The hypothesis of the paper is that safety-inducing training helps limit unsafe responses, but does not prevent the agents from performing unsafe searches. 

The paper proposes two attacks, the search and the multisearch attack, based on encouraging the agent to do a search first (search attack) or several searches (the multisearch attack). In their empirical evaluation, the paper demonstrates that state-of-the-art agents are actually vulnerable to these type of attacks.

### Strengths
S1. The paper identifies a relevant and timely problem, and demonstrates its existence empirically.

S2. Strong empirical evaluation on state of the art agents. 

S3. The paper is clearly written and easy to follow.

S4. The paper provides enough details and code as supplementary material for it to be reproducible.

### Weaknesses
W1. While using an LLM to judge safety is a standard practice nowadays, it is a limitation and begs the question of "unsafe according to whom?".

W2. The paper is primarily diagnostic: it identifies a safety weakness but does not experimentally test mitigation strategies. While the discussion section outlines possible remedies, they remain unvalidated.

### Questions
Q1. (Related to W1) How robusts are your findings to different notions of safety? Have you considered using different LLMs as evaluators to test robustness?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Search models, despite being trained to refuse harmful requests, are vulnerable to attacks forcing early searches. These searches generate significantly more harmful queries, which then retrieve toxic content, escalating the model's harmful outputs by up to 40%.  This effect is amplified by multiple forced searches. Attacks succeed due to conflicts between RL training (rewarding task success, including harmful behavior) and safety instructions, biased retrieval content, and a lack of safety examples in training data.

Defenses proposed include:
*   Pre-retrieval filters for harmful search queries.
*   Safety-aware RL training incorporating safe trajectories.
*   Analyzing and intervening on harmful query representations.

The core issue is a critical safety gap in agentic LLMs, arising from RL pipelines prioritizing task success over safety, creating exploitable contradictions. Mitigation requires redesigning agent training with safety-grounded rewards and retrieval sanitization, especially as these agents become more widespread.

### Strengths
The paper demonstrates a compelling analysis of agentic LLM safety vulnerabilities, supported by several key strengths. Its claims about RL training prioritizing task success over safety are empirically validated through controlled experiments comparing instruction-tuned and RL-trained agents. 

The experimental part is good, featuring systematic ablation studies that isolate the impact of different attack vectors. Controls for prompt length and complexity, combined with cross-model validation, enhance the reliability and generalizability of the 
findings. Methodologically, the analysis is thorough, providing mechanistic interpretations of failure modes like retrieval bias and RL-safety conflicts, supported by qualitative case studies and rigorous statistical testing.

### Weaknesses
First, while it identifies RL-induced behaviors as a core issue, the *explanation* for how RL overrides safety tuning is insufficient. The mechanisms behind this conflict need further conceptual clarification, ideally through diagrams or more detailed theoretical discussion, 
to fully explain this key finding.

Second, the paper's framing of its contribution could be strengthened by a more thorough engagement with related prior work. Although it cites foundational agent research like  WebGPT/RAGAS, it overlooks directly relevant studies on *agent jailbreaks* or *retrieval-based poisoning attacks*. This gap limits the novelty justification, as some aspects of the vulnerability landscape might already be partially explored elsewhere.

Finally, the transition from identifying vulnerabilities to proposing mitigations feels somewhat abrupt and disconnected. The empirical findings highlighting specific weaknesses  (like retrieval bias) are not closely linked to the suggested solutions (like 
search filters or safety-aware RL). A table explicitly connecting each vulnerability to a corresponding proposed mitigation would significantly improve the paper's logical flow and strengthen the argument for these specific defenses.

### Questions
In your PPO implementation, did you observe cases where the  *value function* learned to penalize harmful searches? If so, why does search-triggering still dominate?

When testing with live APIs (e.g., Google Programmable Search), what % of adversarial queries actually returned harmful content? Does API filtering mitigate this vulnerability?

For the proposed search classifier—did you prototype it?

### Soundness
3

### Presentation
3

### Contribution
3
