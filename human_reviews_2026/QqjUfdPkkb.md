# SEAL-RAG: Loop-Adaptive RAG with On-the-Fly Entity Extraction and Fixed-k Gap Repair

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
We propose SEAL-RAG, a training-free, inference-time controller (no fine-tuning
of retriever, reranker, or generator) for retrieval-augmented generation that targets
multi-hop precision. SEAL executes a fixed retrieval depth k (k = number of
passages retrieved per search/micro-query) in a Search → Extract → Assess →
Loop cycle. A scope-aware sufficiency check aggregates coverage, typed bridging,
corroboration/contradiction, and answerability signals to decide stop vs. targeted
repair. At each loop, SEAL performs on-the-fly, entity-anchored (head, relation,
tail) extraction, maintains a live entity ledger, and builds a gap specification (miss-
ing entities/relations) that triggers one micro-query per repair under the same top-k;
new candidates are merged via entity-first ranking (prefers passages anchoring
those entities) before a single final generation step. On a 1,000-example HotpotQA
validation subset in a shared setup, SEAL improves LLM-judged answer correct-
ness by +10–22 pp (k=1) and +3–13 pp (k=3) vs. SELF-RAG across backbones,
and increases evidence precision@k (gold-title precision) by +12–18 pp at k=3.
These gains are statistically significant (chi-square for correctness; paired two-sided
t-tests for precision/recall/F1; p<0.05). By keeping k fixed and bounding repairs
by T (maximum repair iterations), SEAL yields a predictable, bounded cost profile
while replacing distractors rather than broadening context.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SEAL-RAG, a training-free, inference-time controller for RAG. It operates on a Search → Extract → Assess → Loop cycle with a fixed retrieval context size. The core idea is to use on-the-fly entity extraction to identify specific information gaps and then issue targeted micro-queries to fill them. Unlike methods that expand the context, SEAL-RAG replaces less relevant documents with newly retrieved ones, maintaining a fixed cost profile.

### Strengths
The paper tackles the important and practical problem of improving RAG performance under a constrained budget. Besides, the method demonstrates significant and consistent performance gains over SELF-RAG on the chosen task in terms of answer correctness and evidence precision.

### Weaknesses
1. The core contribution appears to be a procedural and technical refinement of the iterative, self-correcting RAG paradigm pioneered by models like SELF-RAG. The overall framework about retrieving, assessing, and re-retrieving is not new. The work feels more like an engineering improvement focused on explicit entity-based control, rather than a fundamental conceptual advance.
2. The evaluation is confined to a single dataset, HotpotQA. This raises questions about the method's generalizability.
3. The paper introduces several new modules (e.g., entity-anchored extraction, scope-aware sufficiency gate) as improvements over SELF-RAG's mechanisms. However, the experiments do not offer an ablation or comparative analysis to demonstrate the specific effectiveness of these modules.
4. The paper introduces a complex control loop with many new, specific terms (e.g., "gap specification," "scope-aware sufficiency", "entity-first ranking"). This density can make the methodology difficult to follow and obscures the core mechanics.

### Questions
Why use 1,000-example HotpotQA rather than the complete HotpotQA?

### Soundness
2

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
5

### Summary
This paper proposes a plug-and-play style RAG method, SEAL-RAG, which performs a Search, Extract, Assess, and then Loop cycle. The experiments show the proposed method outperforms a specific existing method on a QA dataset. Their ablation study shows an iterative repair loop improves the performance.

### Strengths
- This paper proposes SEAL-RAG that uses on-the-fly entity extraction to build an explicit gap specification. This allows it to issue focused micro-queries to find specific missing facts for next retrieval.
- The proposed method keeps the retrieval depth fixed and replaces distractors with newly retrieved passages that seem more important.
- Since SEAL-RAG is a training-free, inference-time controller, it is easy to be combined with any LLMs.

### Weaknesses
- The experiments rely solely on HotpotQA. To demonstrate robustness and generality for multi-hop QA, it would be better to include additional benchmarks such as MuSiQue [1] and 2WikiMultiHopQA[2].
- Several core experiments fix k=1, i.e., retrieving a single chunk, despite the paper’s focus on multi-hop questions. Given the nature of multi-hop reasoning, evaluating larger k, e.g., k \in [5, 7, 10], would be more natural and enable fairer comparisons with prior work.
- If I understand correctly, Self-RAG [3] provides released fine-tuned 7B and 13B variants. The manuscript, however, states that the backbone is interchangeable and not fine-tuned, and I checked the supplementary material and found the authors used the implementation from LangChain, which uses OpenAI API, i.e., not fine-tuned models. No explanation about this is provided in the paper.
- Since the paper’s scope excludes fine-tuning of retriever, reranker, or generator, plug-and-play approaches seem better matched than Self-RAG for primary comparison. Consider adding Adaptive-RAG [4], which shows better performance than Self-RAG, Adaptive-k [5], and LC-Boost [6]. All of these include HotpotQA, enabling direct comparisons.

[1] Trivedi, Harsh, et al. "♫ MuSiQue: Multihop Questions via Single-hop Question Composition." Transactions of the Association for Computational Linguistics 10 (2022): 539-554.

[2] Ho, Xanh, et al. "Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps." *Proceedings of the 28th International Conference on Computational Linguistics*. 2020.

[3] Asai, Akari, et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." The Twelfth International Conference on Learning Representations.

[4] Jeong, Soyeong, et al. "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity." Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2024.

[5] Taguchi, Chihiro, Seiji Maekawa, and Nikita Bhutani. "Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-$ k$." arXiv preprint arXiv:2506.08479 (2025).

[6] Qian, Hongjin, et al. "Are Long-LLMs A Necessity For Long-Context Tasks?." arXiv preprint arXiv:2405.15318 (2024).

### Questions
Could you answer all the bullet points I raised in the Weaknesses section?

### Soundness
2

### Presentation
1

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
This paper introduces SEAL-RAG, a clever, training-free controller designed to make RAG better at handling complex, multi-hop questions. Instead of just retrieving a bunch of documents and hoping for the best, SEAL-RAG uses a smart Search → Extract → Assess → Loop cycle. What's really neat is that it operates on a fixed budget of retrieved documents (k), identifying specific missing facts by extracting entities on the fly and then issuing targeted "micro-queries" to fill those gaps. This process replaces less useful documents with more relevant ones rather than just expanding the context. The authors show on a HotpotQA benchmark that this approach significantly improves answer correctness and evidence precision over a strong baseline like SELF-RAG, especially when the retrieval budget is tight.

### Strengths
1. Its "replace, don't expand" strategy is a very intuitive way to fight context dilution by actively swapping out distractors for crucial bridge facts.

2. Being a training-free controller that works with existing models makes the method highly practical and easy for others to adopt.

### Weaknesses
1. The whole approach feels very tailored to the entity-centric, factoid style of HotpotQA and might not generalize well to tasks requiring more abstract reasoning.

2. The system's performance seems to hinge on the entity extraction and sufficiency check prompts working perfectly, which could be quite brittle in practice.

3. The performance gains, while impressive, seem to shrink as the base models get stronger and the retrieval budget (k) increases.

### Questions
It looks like the sufficiency checks and micro-query generation rely heavily on well-crafted prompts. How sensitive is SEAL-RAG to the specific wording of these prompts, and was there a lot of manual prompt engineering involved to achieve these results?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a pipeline for multi-hop retrieval-augmented generation. Without training any components, it iterates retrieval, entity extraction, and a ranking of evidence. The method is compared under retrieving k=1 or k=3 passages per hop against some implementation of Self-RAG (unclear if that implementation uses training, like the original Self-RAG) and finds higher accuracy and retrieval quality in both cases.

### Strengths
The method proposed performs much better on HotPotQA than the authors' baseline. This is also tested for statistical significance.

### Weaknesses
It is difficult to identify a specific novelty in the proposed methods or to ascertain if it is empirically superior to prior work.

The method combines a number of very standard techniques and does not offer a compelling reason for several of them. The techniques range from fairly natural (e.g., the repair check, which is fundamentally just a self-stopping criteria, an extremely standard component in these types of pipelines) to fairly surprising but not fundamentally justified in a new informative way (e.g., explicit entity extraction with modern LLMs).

The comparison against some implementation of Self-RAG raises more questions than it answers. Why Self-RAG in particular? While it is a reasonable system it is neither a particularly state-of-the-art or particularly popular/fundamental baseline, and --- if I understand correctly --- it fundamentally centers on training the models, whereas my understanding is that the authors test both methods without any training.

The authors evaluate only on HotPotQA. While HotPotQA is a fairly foundational dataset in this space, there have been numerous (harder) benchmarks in the space of multi-hop retrieval and "deep research" over the past 7 or so years since the introduction of this task. HotPotQA has over 3600 citations. Many of the methods evaluated in the literature on HotPotQA are also training-free and achieve a range of higher scores. In what way does the proposed method differ from or compare with the vast literature on this topic?

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
