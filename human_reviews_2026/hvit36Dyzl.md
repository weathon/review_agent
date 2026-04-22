# Think in Parallel, Answer as One: Logit Averaging for Open-Ended Reasoning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8

## Abstract
Majority voting has proven effective for close-ended question answering by aggregating parallel reasoning traces. However, it is not directly applicable to open-ended reasoning, such as code generation and web-based deep research, where a “majority” over complete solutions is ill-defined. We introduce THINKMERGE, a training-free, plug-and-play decoding strategy that runs K parallel reasoning traces and averages their next-token logits at synchronization points to produce a single coherent output. THINKMERGE integrates seamlessly with vLLM/SGLang and remains compatible with standard decoding techniques such as Top-p/Top-k. Empirically, it matches or surpasses majority voting on AIME and GPQA, while delivering consistent gains on open-ended coding tasks: on LiveCodeBench (hard), pass@1 improves by +8.28% for DeepCoder-14B-Preview and +7.58% for Qwen3-8B. Beyond code, we further show that THINKMERGE improves web-based deep-research agents (e.g., WebSailor-7B/32B) across GAIA, BrowseComp-en/zh, and XbenchDeepSearch. These results demonstrate that parallel test-time scaling can benefit open-ended reasoning without relying on voting over complete outputs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a new way to deal with open-ended questions with LLM, named ThinkMerge, in which authors used majority voting, map-reduce, etc.

### Strengths
The paper is well structured and clearly written. I was able to easily understand the ideas the authors intended to convey.

### Weaknesses
I believe the primary issue lies in the lack of novelty. Although the paper claims to focus on open-ended questions, the datasets used, such as AIME and GPQA, are predominantly composed of closed-ended questions. This mismatch undermines the validity of the problem statement from the outset.

Secondly, the proposed method, described as "think merge," appears to be essentially a form of majority voting. The use of a map-and-reduce framework is also not new, as similar approaches can be found in many prior works, such as GraphRAG.

In terms of experiments, I don’t see any significant improvement on genuinely open-ended questions. Overall, in its current form, I don’t find the paper acceptable.

### Questions
I have no questions, as the writing is very clear.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes THINKMERGE, a training-free way to make parallel chain-of-thought work for open-ended tasks where majority voting can’t, by averaging token-level logits from K parallel traces after the reasoning delimiter and decoding a single shared answer. On AIME/GPQA it matches or slightly improves over standard parallel CoT with voting, and on LiveCodeBench it gives clear pass@1 gains where voting fails. I’m slightly reserved about the reliance on logit access / synchronized delimiters and the fact that some gains on closed-ended tasks are modest, but the core idea is clean and the evaluation is broad enough to justify it. 

I would have preferred if the code was supplied to have a hand-on experiment with the approach, but I have not had time to reimplement it, however, theoritically it is intact.

### Strengths
- Addresses a real gap: parallel CoT doesn’t work for open-ended outputs; this makes it usable there.
- token-level logit averaging after a shared delimiter, easy to plug into vLLM/SGLang.
- Evaluates on both closed-ended (AIME, GPQA) and open-ended (LiveCodeBench), so it’s not overfitted to math-only.
- Keeps contexts synchronized while decoding, which is a nice engineering detail for practicality.

### Weaknesses
- This looks to be delimiter-dependent. It assumes clean think/answer separation; models that ramble or reflect will hurt alignment.
- On AIME/GPQA it sometimes only matches or even loses to majority voting.
- I’m skeptical about pure logit averaging, since it can dilute a minority-but-correct trace. A simple reweighting or confidence-based scheme might help; at minimum I’d like to see an ablation showing how often correct traces get downvoted by the ensemble
- The method gives clear gains at small K (e.g. K=2), but increasing K further doesn’t reliably improve pass@1 and can even regress, likely because divergent traces interfere when logits are averaged. I think the approach is effective, but scaling k for more benefit is not working as expected.

### Questions
Please refer to weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ThinkMerge, an alternate to majority voting paradigm. Instead of doing full rollouts and then majority vote, the authors propose doing parallel rollouts only for the "thinking" tokens, and then decoding from there on by averaging the token logits at each auto-regressive step, and sampling the next token from the resulting distribution. The proposed method neatly fits into widely used inference engines such as vLLM and SGLang, and is compatible with other decoding choices such as top-p/top-k/penalty etc.

### Strengths
Strengths: 
- The proposed method is quite simple to implement, doesn't require any training, and practically deployable on popular inference engines. 
- The technique addresses a limitation of majority voting in open domains, where the response is free form text (code/reports etc) where "majority vote" is ill-defined. 
- The four ablations seem well-motivated, and grounded in practical considerations.

### Weaknesses
Weaknesses:
- My major concern is with the limited innovation/novelty. The proposed technique is a straight forward application of product of experts ensembling (See [1]). Within the context of application in LLMs, there is prior work in this area: 
[2] applies the exact same technique of token fusion in the context of having several prompt variants. They too generate logits by processing each prompt and then averaging the logits at each auto-regressive step. ThinkMerge can be thought of as a special case of this where each prompt is a reasoning trace generated before token fusion. A more general problem of fusion across LLMs with different vocabularies has also been explored by [3, 4]. So the novelty seems to be limited to applying this technique with parallel `<think>...</think>` generations. 

- The proposed fusion is naive, and can potentially fail in scenarios where the reasoning traces diverge too much. Suppose, that across parallel generations, the model explores entirely different strategies to solving the problem, the ensembling can collapse. This is not sufficiently explored in the paper. 

- Token probability ensembling literature needs to be covered more in Related section [2,3,4,5].

- While the paper emphasizes open domain tasks, it's experimentally validated only with a single benchmark, LiveCodeBench. 


[1]: Product of Experts - Geoff Hinton
[2]: M-Ped: Multi-Prompt Ensemble Decoding for Large Language Models - Guo et.al, CORR 2024
[3]: Bridging the Gap between Different Vocabularies for LLM Ensemble - NAACL 2024
[4]: Token-level Ensembling of Models with Different Vocabularies - Rachel Wicks et.al (arxiv)
[5]: Llm-blender: Ensembling large language models with pairwise ranking and generative fusion - Jiang et.al

### Questions
Suggestions:

- This work can be a good empirical contribution but would require some more work, especially around analysing which scenarios benefit from fusion approach and investigating it's limitations and failure modes. This is important for the ICLR communities oriented towards practicable solutions. 
- Since the emphasis is on open-domain reasoning, include more benchmarks such as Terminal Bench, other coding benchmarks etc.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Authors present ThinkMerge which averages logits over multiple parallel reasoning traces to enable majority voting over open-ended QA. They outperform baselines on AIME, GPQA, and even LiveCodeBench.

First, they demonstrate that Pass@N indeed goes up for all models on all tasks, both open- and close-ended. This serves as a basis for the assumption that majority voting should benefit performance. Further, they find that these gains do come from the model gaining “capability” on difficult questions as inference scale grows.

Their approach follows a map-reduce paradigm. First, they sample K CoTs and then left-pad them so that the end of the CoT (the `</think>` token) in each seq is aligned. To do this they autoregressively have each CoT effectively “vote” token-by-token by averaging the pre-softmax logits that come from each chain. 

They considered a few other alternatives before landing on this approach. Across most tested models, their method outperforms simple majority voting when the chains are allowed to fully expand, rather than only using the shortest K, for AIME and GPQA. Additionally, on the open-ended livecodebench, their method consistently outperforms the baseline, and this shortest K early stopping approach still delivers gains. This is useful, as it means that potentially the method can be made more efficient for open-ended tasks.

### Strengths
Clear implementation of a straightforward approach to improve reasoning model performance.

Method delivers a clear improvement

Well-described and easy to followN/A I think this is a solid paper.

### Weaknesses
No discussion of closed models here. I’m curious, how well does this close the gap between the open and closed models?

### Questions
I think this is a pretty solid paper. What do you think about the weakness suggested?

### Soundness
4

### Presentation
4

### Contribution
3
