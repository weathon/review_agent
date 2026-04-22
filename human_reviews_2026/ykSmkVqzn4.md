# Flipping the Dialogue: Training and Evaluating User Language Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 2, 8, 6

## Abstract
Conversations with LMs involve two participants: a human user leading the conversation, and an LM assistant responding to the user's request. To satisfy this specific role, LMs are post-trained to be helpful assistants -- optimized to produce exhaustive and well-structured responses, free of ambiguity and grammar errors. User utterances, on the other hand, are rarely perfected, with each user phrasing requests in unique ways, sometimes putting in partial effort at each turn and refining on the fly. To evaluate LM performance in realistic settings, prior work simulated users in multi-turn conversations, often by prompting an LM originally trained to be a helpful assistant to act as a user. However, we show that assistant LMs make for poor user simulators, with the surprising finding that better assistants yield worse simulators. Instead, we introduce purpose-built User Language Models (User LMs) - models post-trained to simulate human users in multi-turn conversations. Through various evaluations, we show how User LMs align better with human behavior and achieve better simulation robustness than existing simulation methods. When leveraging User LMs to simulate coding and math conversations, the performance of a strong assistant (GPT-4o) drops from 74.6% to 57.4%, confirming that more realistic simulation environments lead to assistant struggles as they fail to cope with the nuances of users in multi-turn setups.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes UserLM to evaluate LM performance in realistic multi-turn conversation settings. The authors train UserLMs on flipped dialogues between users and LMs to simulate users, given user intents. The experiments show that UserLMs' simulations better match actual human responses and remain within the user role. As a result, the performance of assistant LMs is lower against UserLMs than against prompt-based baselines.

### Strengths
1. This paper addresses an important problem of improving and making more realistic user simulations for chat-based LM evaluation.

2. The proposed approach, termed UserLM, provides better user simulation that matches human responses in various aspects, whereas previous prompt-based approaches fail.

### Weaknesses
1. The paper does not provide evidence that better user simulation leads to better assistant LM evaluation. It seems the assistant LM performs worse against UserLM. However, it is not clear whether this is because real user requests are more challenging or because UserLM has some unexpected traits that confuse the assistant. Moreover, this does not mean that assistant LM evaluation based on UserLM offers better representation of the assistant's quality.

2. The WildChat dataset seems to contain many almost-duplicates that are not necessarily from the same IP addresses. This may be due to popular prompts shared by many users or to a single user using multiple IP addresses. Without careful data splitting to account for this, there may be data leakage into the test sets.

### Questions
1. What are the findings that are not available in other user simulation approaches but that UserLM enables?

2. Does an assistant LM performing worse against UserLM suffice to conclude that UserLM-based simulation offers better evaluation? How can we rule out that UserLM is just confusing the assistant LM?

3. Were there any additional checks of data leakage into the test sets?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper argues that assistant LMs, which are post-trained to produce exhaustive, unambiguous replies, do not faithfully mimic messy, indirect human user behavior in multi-turn dialogues. It shows that using assistant LMs as user simulators is flawed; stronger assistants make worse simulators. This paper propose “User LMs,” post-trained specifically to simulate human users, and report that these models align better with real user behavior and yield more robust simulations.

### Strengths
- This casts doubt on current practice and suggests the need for a user simulator better aligned with real users.
- Extensive experiments show that the proposed post-trained user simulator better captures the properties targeted by the designed metrics.

### Weaknesses
Evaluation Metrics
* While mimicking user behavior can be useful for building a user simulator, it is not clear that the properties highlighted as important for user behavior are actually critical for evaluating an LM assistant.

 * The results in Table 1 may offer limited insight: a model trained on conversational data should have lower perplexity on test sets from the same distribution (WildChat) or similar data (PRISM).

 * An intent (e.g., ‘You are a user chatting with an assistant language model … medications on weight gain.’) provides strong topical context. Given such a prompt, the conversation will predictably be steered toward that topic, which in turn reduces perplexity.

* Some evaluation metrics need stronger justification, as they do not appear to faithfully capture what is required for realistic user simulation.
    * There are many potential, confounding reasons for lower assistant performance. Is it reasonable, therefore, to conclude that lower performance indicates the user simulator is better aligned with real users?
    * Why is the simulator’s ability to end the conversation considered so important for evaluating multi-turn QA with LLMs?
    * Since the user’s response depends on the assistant’s response, is it appropriate to compute turn variance without accounting for this dependency?
    * The claim in Intent Decomposition section that “a lower overlap is particularly desirable because it indicates that the model expresses its intent using varied language while introducing details progressively” is not substantiated.

    * The experimental setup feels somewhat artificial. The methods for evaluating User-role adherence and Intent adherence seem overly narrow, specific, and somewhat unlikely to occur in practice.

Real-User Alignment
* What exactly constitutes a “real user” in this work? Was any human study conducted to validate that definition?
* As users become more accustomed to LLM chatbots and adapt their behavior to use them effectively, is a new human study needed to compare the proposed simulators with up-to-date real user behavior?
* What is the practical value of building the user simulator? Do you have empirical evidence showing that it improves the evaluation of LLM performance or leads to better model outcomes?

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents 1B and 8B models fine-tuned on the WildChat dataset, where the user utterances were unmasked and the user intent was put as a system message. These models can be used for simulating user utterances in multi-turn conversations.

Other details:
- The intent was inferred with GPT-4o in a few-shot mode.
- The base models were  Llama3-8b-Base and Llama3.2-1b-Base. Starting from the instruction models is worse.
- It was full fine-tuning on 4xA6000.

The main evaluation was by perplexity on the test set (WildChat) and on the OOD test set (PRISM)

Other metrics: first-turn diversity, intent decomposition across turns, dialogue termination, naturalness (Pangram AI-detector), and robustness (role adherence and intent adherence).

User LMs outperform prompted assistant simulators (including GPT‑4o roleplay) and a fine-tuned baseline (USP‑8b), with large gaps in metrics.

Extrinsic evaluations simulate math and coding tasks where UserLM‑8b yields more realistic, varied user behavior and reduces downstream assistant (GPT‑4o) task success from 75% to 57%, suggesting current assistants struggle more under realistic multi-turn conditions.

The authors claim the model will be publicly available for research purposes.

### Strengths
The topic is very timely: LLM-based user simulators are needed for offline evaluation of chatbot applications everywhere.

The implementation is very straightforward, which is a good thing in that case, and it is easily reproducible. Other works do almost the same (USP), but this work seems to be both better and simpler in terms of the general design.

The paper presents strong empirical evidence that assistant LMs are poor user simulators; User LMs substantially reduce PPL on PRISM and WildChat and improve multi-turn metrics. Some metric choices are nice, for instance, using the Pangram for naturalness. Conclusions are helpful: base models are better than instruct models; 8B is better than 1B, so scaling works in that case.

The presentation is wonderful, all tables and figures are clean, readable, and well-thought-out.

### Weaknesses
1. Extrinsic simulations introduce guardrails applied only to UserLM‑8b (described in Appendix D.1), which gives unfair advantage relative to GPT-based simulators; a fairness ablation is missing. 

2. The task scope for extrinsic simulations is very narrow (math/coding) and English-only; generalization to other domains, modalities, and languages is not evaluated. In general, math and code seem like a strange choice.

### Questions
1. I am surprised to see USP-8b performing that badly, being worse than Llama3.2-1b-Instruct by PPL. It also seems that USP has more sophisticated training procedures. Why do you think it is that bad?

2. Do guardrails described in Appendix D.1 affect Table 1 results? Or are these guardrails only for the extrinsic simulations?

### Soundness
3

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
4

### Summary
The paper explores an alternative perspective in large language model (LLM) training by focusing on the user simulator rather than the dialogue model itself. To address the challenge of maintaining user intent across multi-turn conversations, the authors propose an intent extraction and training method. Experimental results on multiple datasets (such as WildChat and PRISM) show consistent improvements over baselines and better align with human behavior by measuring multiple metrics.

### Strengths
1. The authors focus on a critical yet underexplored task (user model construction) and propose effective methods that achieve promising results.
2. The authors provide comprehensive and fine-grained analyses, offering valuable insights for future user simulator research.
3. The authors introduce diverse evaluation metrics tailored to user simulators, which can serve as useful references for subsequent evaluation frameworks.

### Weaknesses
1. A major concern is that the “FLIPPING the Dialogue” paradigm has already been partially explored in prior works (e.g., USP), which weakens the originality claim. The paper should clearly articulate how it differs from existing approaches.
2. The correctness of extracted intents used for training has not been validated, potentially introducing bias in both model performance and evaluation (e.g., UserLM is trained and tested on WildChat).
3. The baseline comparison is limited since only USP is included, while prior role-playing and persona-based approaches are not discussed. As a result, it is unclear whether the observed gains stem from the proposed methodology itself or simply from incorporating user profiles (in whatever form). Moreover, several metrics (e.g., Table 3) are model-specific, lacking objective comparability.
4. The evaluation relies heavily on LLM-based automatic assessments. Some degree of human evaluation should be included to validate the reliability of the results.

### Questions
1.	How exactly is the <|endconversation|> mechanism implemented across models? As mentioned in the paper, is it merely introduced through prompts? For UserLM, was this mechanism explicitly annotated during training? The notable improvements in Table 2 suggest further clarification.
2.	The paper claims that abstract rather than specific intent extraction is designed to avoid simple memorization by LLMs, yet this abstraction could harm intent fidelity. How is the correctness of extracted intents measured or validated?
3.	Are the datasets used for evaluation in Section 3 the same as those introduced in Section 2?
4.	From a writing perspective, the introduction mentions that only two prior works exist in this area (two citations). However, the related work section shows that many studies are indeed relevant. These should be cited earlier, along with a clear positioning of this paper’s novelty. Furthermore, while the analysis is extensive, it lacks a clear structure and logical flow; reorganizing the sections would greatly improve readability.

### Soundness
2

### Presentation
3

### Contribution
2
