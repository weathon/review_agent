# Enabling Agents to Communicate Entirely in Latent Space

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
While natural language is the de facto communication medium for LLM-based agents, it presents a fundamental constraint. The process of downsampling rich, internal latent states into discrete tokens inherently limits the depth and nuance of information that can be transmitted, thereby hindering collaborative problem-solving. Inspired by human ''mind-reading'', we propose Interlat (Inter-agent Latent Space Communication), a paradigm that leverages the last hidden states of an LLM as a representation of its mind for direct communication (termed ''latent communication''). An additional compression process further compresses latent communication via entirely latent space reasoning. Experiments demonstrate that Interlat outperforms both fine-tuned chain-of-thought (CoT) prompting and single-agent baselines, promoting more exploratory behavior and enabling genuine utilization of latent information. Further compression not only substantially accelerates inference but also maintains competitive performance through an efficient information-preserving mechanism. We position this work as a feasibility study of entirely latent space inter-agent communication, and our results highlight its potential, offering valuable insights for future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes to train LLM agents in a collaborative framework in which a sender communicates in the latent space and an actor completes the task based on received messages. The authors combine multiple learning objectives, curriculum learning, and information compression in the framework to train the reasoning model of the sender with frozen-actor supervision from the task environment. Evaluation results show that agents trained to communicate in latent space outperform baselines where agents are supervised fine-tuned with CoT or direct natural language communication. Ablation experiments were conducted to evaluate the impact of information compression and other modules in the proposed framework.

### Strengths
1. The general idea of allowing LLM agents to communicate in a latent space instead of the constrained natural language space is novel and intuitive. 

2. The proposed method outperforms the baseline and ablated conditions in the evaluation benchmark.

### Weaknesses
1. The framework of training agents to communicate in a high-dimensional space to solve the sender-receiver task has been extensively researched in the emergent communication and multi-agent reinforcement learning (MARL) community. I would recommend the authors include a thorough literature review and position this work among previous work, highlighting its contributions beyond simply replacing the policy model with a transformer-based LLM. Attached are a few representative papers to start with:

a. Multi-agent cooperation and the emergence of (natural) language

b. Emergence of linguistic communication from referential games with symbolic and pixel input

c. Emergent discrete communication in semantic spaces

d. Trading off Utility, Informativeness, and Complexity in Emergent Communication

2. The methods section is hard to follow. I would recommend providing a more specific description of the proposed framework for readers with a more general background.

a. The description of the training framework could be more specific for better reproducibility. For example, how is the permutation done in conditional mind separation? What is the language space plan in plan-aligned regulation? How does stochastic replacement happen in curriculum learning?

b. The motivation and description of information compression are too brief to be comprehensible. It is not clear how the compression is done in Section 3.3 of the main text. Even after reading Appendix B, the use of a fixed instruction-tuned model and the distillation process lacks a clear rationale. I would recommend moving the description of compression to the main text and emphasizing the motivation for this module.

c. It is not clear how the environment reward backpropagates through the sender-receiver setup. Is reinforcement learning used to train the agent in an interactive environment, or is this offline training with a fixed trajectory dataset?

d. In the main text, the actor is described as frozen during training. However, the pseudocode in Appendix F shows a two-stage training framework. Please clarify this discrepancy.

e. The "training-free" and "trained" settings are not explained before their use in Section 4.2.

3. Minor format issue: The font size in Figure 1 should be enlarged for better readability.

### Questions
1. How were the MHA and Projector trained?

2. There seems to be an assumption that the fixed actor can understand the latent communication of the reasoning model because they have the same backbone (e.g., Qwen2.5). Would the findings reported in the paper still hold if these two modules were based on different models?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Interlat, a framework that enables large language model (LLM)–based agents to communicate entirely in latent space instead of through discrete tokens. Interlat allows agents to transmit their final-layer hidden states as communication messages, processed through a lightweight adapter. The framework introduces additional training signals, including separation and alignment losses and employs curriculum learning to help the receiver agent gradually interpret latent messages. A second stage further compresses the latent communication to achieve more efficient reasoning while maintaining performance. Experiments on the ALFWorld environment compare Interlat against both no-communication and text-based baselines, as well as multiple ablations, demonstrating that latent communication improves task success rates and efficiency.

### Strengths
1. The idea of exchanging continuous latent representations rather than discrete tokens is conceptually novel and connects to how humans may communicate through visual or implicit signals rather than purely language. 

2. The framework integrates several thoughtful components (JS separation loss, plan-alignment loss, curriculum learning) to ensure interpretable and robust communication. The compression analysis is particularly interesting, showing that latent messages can be shortened with minimal performance degradation.

### Weaknesses
1. Motivation and significance not fully articulated: While the paper demonstrates performance gains, it remains unclear what new behaviors or communicative properties latent communication enables beyond efficiency. For example, could this approach lead to more human-like communicative phenomena (e.g., implicit alignment, compositionality, or emergence of shared latent codes)? What is their grounding for semantic information being transmitted?

2. The paper should more explicitly position itself relative to existing literature, such as Learning to Communicate with Deep Multi-Agent Reinforcement Learning (Foerster et al., 2016) and other differentiable message-passing or embedding-based communication methods. Clarifying how Interlat differs conceptually and empirically would strengthen the contribution.

3. Some key terms lack clear definitions:
- What constitutes a supervised position and how are these positions selected?
- What exactly is the language space plan in the alignment loss?

4. The compression mechanism is one of the paper’s most promising contributions but is described briefly. The main text should include the loss formulation (currently only in Appendix B) and discuss how it maintains semantic alignment during compression. Referring to Table 3 without explaining these losses makes the narrative incomplete.

5. The task setup (ALFWorld) and evaluation metrics should be explained more clearly.
- What does accuracy or success rate precisely measure in these sequential tasks?
- What does step refer to — environment steps or rounds of inter-agent communication?

6. Providing qualitative examples or visualizations of the latent communication (e.g., probing or dimensional reduction) would make the results more interpretable.

### Questions
Since the curriculum gradually replaces text embeddings with latent states, the results seem to suggest that the text representations are initially easier for the receiver to interpret. If that is the case, it is unclear why latent-space communication is ultimately preferable. Would a simpler alternative yield comparable performance such as training the sender to regress the language-space plan?

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
3

### Summary
The paper proposes Interlat, a method that lets LLM agents communicate via latent representations instead of text. By exchanging hidden states through a small adapter network, agents share richer, faster information. On the ALFWorld benchmark, Interlat outperforms text-based and chain-of-thought baselines, even with up to 24× compressed latent messages. Results show that latent-only communication enables efficient and expressive multi-agent collaboration.

### Strengths
Interlat introduces an interesting way for language model–based agents to communicate directly through latent representations instead of discrete text. This marks an interesting shift in how multi-agent systems can share information, avoiding some of the inefficiencies of text-based communication. The idea seems especially promising for multimodal scenarios, where agents work across different representational spaces.

The proposed approach shows efficiency gains, achieving large reductions in communication latency. This makes it a strong candidate for scaling multi-agent systems where bandwidth and response time really matter. That said, these benefits likely come at the cost of interpretability, since it’s much harder to understand or inspect what’s being passed around in latent space compared to language-based exchanges.

In addition, the paper’s clear visualizations and detailed methodological explanations greatly enhance its readability and accessibility.

### Weaknesses
**Overstated and Misaligned Anthropomorphism**: The paper repeatedly compares latent-space communication to human *mind-reading*, which is wrong. What’s actually happening here is closer to the exchange of high-dimensional neural activations, not cognitive inference (or mind reading) on nonverbal cues as in human communication. This framing risks false anthropomorphizing what is essentially a representational alignment problem. I’d strongly recommend removing this metaphor to keep the paper conceptually grounded.

**Lack of Clarity Around Baselines**: The setup and training details for the baseline models are not clearly described. It’s hard to tell whether Interlat’s improvements are valid. The paper mentions that more details are in the Appendix, but that section is a copy-paste of the main text instead of providing the necessary experimental specifics. Without a clearer explanation, it’s difficult to judge how strong the improvements really are.

**Confounding from Shared Model Family**: All agents in the experiments use the same model backbone (Qwen2.5), which means the sender and receiver already share an aligned latent space. The observed performance gains might therefore reflect this intra-family compatibility rather than true inter-agent latent understanding. Testing communication across different model families or scales (e.g., Qwen -> LLaMA) would provide much stronger evidence for the claimed generality.

**Ambiguity in the “Information Parallel Budget” Analysis**: The "information parallel budget" idea is interesting but not very clearly defined or empirically grounded. The analysis treats differences in output probability distributions as evidence of parallel reasoning, but that interpretation is speculative — lower probability mass could just indicate more uncertainty, not necessarily parallel thought processes. Also, the analysis only looks at the first six hidden states, which likely misses longer-horizon reasoning effects. As it stands, this part feels underdeveloped and could use a clearer theoretical motivation or empirical validation.

### Questions
- Why do you use the base models as the actor models? Have you tried using instruction-tuned models?
- What is the actor model used for the Text baseline?
- In Table 1, why does Interlat require more steps than the other methods?
- Information parallel budget: Why do you use only the first six hidden states?
- How do you plan to address the interpretability issue? (I know this is out of scope, but I'm curious.)

### Soundness
2

### Presentation
2

### Contribution
3
