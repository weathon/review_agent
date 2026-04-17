# Memory-T1: Reinforcement Learning for Temporal Reasoning in Multi-session Agents

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
Temporal reasoning over long, multi-session dialogues is a critical capability for conversational agents. As dialogue histories grow in length and accumulate noise, existing long-context models struggle to accurately identify temporally pertinent information, significantly impairing reasoning performance. To address this, we introduce **Memory-T1**, a framework that learns a time-aware memory selection policy using reinforcement learning (RL). It employs a coarse-to-fine strategy, first pruning the dialogue history into a candidate set with temporal and retriever filters, followed by an RL agent that selects the precise evidence. The RL training is guided by a multi-level reward function optimizing (i) accuracy, (ii) evidence grounding, and (iii) temporal consistency. This temporal consistency reward provides a dense signal by evaluating alignment at both the session-level (range proximity) and the utterance-level (evidence density), enabling the agent to resolve subtle chronological ambiguities. On the Time-Dialog benchmark, Memory-T1 boosts a 7B model to an overall score of 67.0\%, establishing a new state-of-the-art performance for open-source models and outperforming a 14B baseline by 10.2\%. Ablation studies show temporal consistency and evidence grounding rewards jointly contributing to a 15.0\% performance gain.Moreover, Memory-T1 maintains robustness up to 128k tokens, where baseline models collapse, proving effectiveness against noise in extensive dialogue histories.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Memory-T1, a two-stage framework that improves the temporal reasoning ability of large language models (LLMs) over long, multi-session dialogues. During the first stage, an LLM predicts the temporal window and filters sessions outside the window. Then, a retriever reranks the resulting sessions, and only the top-ranked sessions are kept. During the second stage, an agent is trained with GRPO to select the most relevant sessions. The reward is a combination of accuracy, evidence grounding, and temporal consistency. The approach is applied to Qwen2.5-3B and Qwen2.5-7B-Instruct, and compared against diverse baselines.

### Strengths
- The paper addresses an important problem which is temporal reasoning on long, multi-session dialogues.
- The paper is clear and easy to read. The appendix contains additional details about the dataset and setup. 
- Memory-T1 outperforms existing baselines, generalizes out-of-domain and is robust to long-context scenarios. 
- The ablation study emphasizes the contribution of each reward.

### Weaknesses
- Unlike the base models, the performance when applying Memory-T1 does not seem to increase with model size (from 3B to 7B). 
- Training requires a heavy annotation of the dataset which can be very expensive. 
- There is no qualitative analysis showing the kind of mistakes of the base model that Memory-T1 solves.

### Questions
-  Memory-T1 does not seem to improve with model size. What could be the reason of this limitation? Is it more about the task than the approach?
- What is the additional latency introduced by the approach compared to the base model and the other baselines? 
- Have you observed any reward hacking that you had to circumvent? Was it associated to particular reward or resulted from the combination of the rewards?
- Did you have to perform extensive hyperparameter search to obtain the results in the paper?
- Is it possible to fully automate dataset annotation?

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
This paper presents Memory-T1, a reinforcement learning framework designed to improve temporal reasoning capabilities in multi-session conversational agents. The authors address a critical challenge: existing long-context models struggle to accurately identify temporally relevant information from noisy, extended dialogue histories. The framework employs a two-phase architecture: 1) Candidate Generation: LLM first predicts its target temporal window followed by relevance-based retrieval (BM25) to narrow the search space; 2) Fine-grained Selection: Employs an RL agent trained with GRPO to select precise evidence sessions. A multi-level reward function is designed to jointly optimize the evidence selection and answer prediction. The framework achieves 67.0% overall score on the Time-Dialog benchmark with Qwen-2.5-7B, outperforming a Qwen-2.5-14B baseline by 10.2%.

### Strengths
- The coarse-to-fine retrieval strategy combined with multi-level RL rewards is well-justified. The temporal consistency reward design enhanced the temporal reasoning and evidence selection of the model to better predict the answer.
- Ablation studies (Table 2) provide clear evidence for each reward component's contribution, demonstrating the importance of individual design choices.

### Weaknesses
- The primary evaluation is conducted on the in-domain Time-Dialog dataset, where Memory-T1 is trained in-domain while other baselines (Time-R1, MemAgent) are evaluated in a zero-shot setting. This makes direct comparison difficult. It remains unclear how Memory-T1 would perform against these baselines on unseen benchmarks such as LoCoMo, particularly in a comparable experimental setup.
- The proposed framework requires extensive temporal annotations for training, yet the sensitivity to annotation quality and potential noise has not been explored. This raises concerns about the framework's practical applicability.

### Questions
What happens when the LLM-based temporal filter incorrectly predicts the query time scope and excludes ground-truth evidence sessions? Does the framework include any mechanism to allow error recovery from such retrieval failures?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Memory-T1, a reinforcement learning framework designed to improve temporal reasoning in long, multi-session dialogue settings. The method introduces a two-stage process: a coarse retrieval stage that filters relevant sessions by predicting temporal scope and ranking past interactions, followed by a fine retrieval stage where a reinforcement learning agent selects evidence guided by multi-level rewards. The model uses three reward components related to answer accuracy, evidence grounding, and temporal consistency, aiming to enhance both factual correctness and chronological alignment. Experiments on Time-Dialog and LoCoMo show that Memory-T1 improves temporal reasoning accuracy and robustness under long-context conditions.

### Strengths
The paper targets an important and underexplored aspect of dialogue modeling, namely temporal reasoning across multiple sessions. The proposed coarse-to-fine retrieval framework is intuitive and improves retrieval efficiency under long histories. Integrating reinforcement learning provides a structured way to optimize multiple supervision signals jointly. The experimental results demonstrate strong improvements on established benchmarks, and the ablation studies highlight the contribution of each reward type. The paper is clearly written, and the motivation is easy to follow.

### Weaknesses
The methodological novelty is moderate. The framework largely combines existing reinforcement learning and retrieval techniques with additional temporal supervision. The temporal consistency reward is well-motivated but not particularly new, and its formulation appears heuristic. The paper does not provide enough justification for the chosen reward weights or a systematic analysis of their sensitivity. The ablation results are limited and do not clearly establish whether improvements stem from the reward design itself or from better retrieval heuristics. The method also assumes access to accurate timestamps and event annotations, which may limit applicability to real-world data where such information is noisy or missing. Finally, the evaluation focuses on QA-style reasoning without exploring whether the learned policy improves temporal coherence in free-form dialogue generation.

### Questions
- How sensitive is the model to the relative weights of the three reward components?
- Were the weights tuned manually or derived from validation metrics?
- Could the method work in domains without explicit timestamps or structured temporal annotations?
- How much of the observed improvement comes from the reward design itself versus better retrieval filtering?
- Would the learned policy transfer to different backbone models or datasets?

### Soundness
3

### Presentation
4

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
This paper proposes Memory-T1, a two-stage framework for temporal reasoning over long, multi-session dialogues. Stage-1 performs coarse-to-fine candidate generation: an LLM predicts the query’s temporal window to hard-filter sessions, then BM25 ranks by textual relevance to form a high-recall pool. Stage-2 fine-tunes a policy with GRPO to jointly pick evidence sessions and produce the answer, supervised by a multi-level reward that combines task accuracy (Ra), evidence grounding via session-set overlap (Rg), and a temporal-consistency reward (Rt) with session-level proximity (Rs) and utterance-level temporal fidelity (Rf).

On Time-Dialog, Memory-T1 improves a 3B model to 66.9% overall, and the 7B variant reaches 67.0%, outperforming larger baselines. The ablations attribute gains to the temporal-consistency and grounding rewards; removing them substantially degrades performance. The method shows strong OOD results on LoCoMo (37.7% overall without RAG), and robustness up to 128k tokens.

### Strengths
1. Timely and important: Tackles temporal reasoning over long multi-session dialogues—an increasingly central capability for agentic systems and memory architectures.

2. Well-motivated reward design for GRPO: Clear decomposition into Ra/Rg/Rt; Rt thoughtfully mixes proximity and fidelity with soft penalties and explicit hyper-parameters; weights and sensitivity are reported.

3. Coarse-to-fine retrieval is executed cleanly with a precise time filter then lexical ranking; the analysis of top-k and recall is helpful.

4. Clear writing and comprehensive experiments: SOTA on Time-Dialog, detailed ablations of reward components, OOD evaluation (LoCoMo), and long-context stress tests.

### Weaknesses
(W1) Table 2 interpretation of Rt/Rs/Rf effects

The paper states that removing Rs or Rf yields a trade-off aligned with task difficulty (Category-A “simpler” tasks improve, while B/C “complex” tasks degrade). However the full removal of Rt (both Rs and Rf) does not show a monotone extension of the same trend.
If Category-A improvements under −Rs/−Rf are due to “easier” temporal structure, why does removing the entire Rt lower Category-A below the full model instead of amplifying that benefit? Two possibilities worth probing: 
1. Reward-interaction / reweighting hypothesis. With wt=0, the optimizer may over-fit Ra+Rg. E.g. Ra for Category-A can reward surface-level answers
2. Credit-assignment : GRPO with a batch-average baseline can shift exploration when a dense component disappears. Without Rt, advantages may over-credit generations that incidentally match Ra/Rg


(W2) Table 3 (LoCoMo) and how RAG is integrated with Memory-T1

Table 3 shows Memory-T1 is best without RAG overall (37.7% vs. 36.7%), supporting the claim that the policy “manages memory” well. But on the Adversarial subset, RAG helps Memory-T1 quite noticeably (29.8 with RAG vs. 26.0 without). Why? Can the authors explain this part? 
How, precisely, is RAG layered over Memory-T1’s time filter + BM25 + RL policy (i.e., is RAG material merged before/after candidate generation, and does it change the action space for evidence selection)? Please detail. 

One possibility here: distribution-shift hypothesis -- the RL policy learns a specialized memory manager for in-dialogue, time-anchored evidence. Injecting external RAG text can introduce off-distribution segments with weak or missing temporal anchors, which lowers precision overall.

### Questions
See weakness session

### Soundness
3

### Presentation
4

### Contribution
4
