# Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Evaluating the abilities of large language models (LLMs) for tasks that require long-term memory and thus long-context reasoning, for example in conversational settings, is hampered by the existing benchmarks, which often lack narrative coherence, cover narrow domains, and only test simple recall-oriented tasks. This paper introduces a comprehensive solution to these challenges. First, we present a
novel framework for automatically generating long (up to 10M tokens), coherent, and topically diverse conversations, accompanied by probing questions targeting a wide range of memory abilities. From this, we construct BEAM, a new benchmark comprising 100 conversations and 2,000 validated questions. Second, to enhance model performance, we propose LIGHT–a framework inspired by human cognition that equips LLMs with three complementary memory systems: a long-term episodic memory, a short-term working memory, and a scratchpad for accumulating salient facts. Our experiments on BEAM reveal that even LLMs with 1M token context windows (with and without retrieval-augmentation) struggle as dialogues lengthen. In contrast, LIGHT consistently improves performance
across various models, achieving an average improvement of 3.5%–12.69% over the strongest baselines, depending on the backbone LLM. An ablation study further confirms the contribution of each memory component.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes BEAM, a benchmark consisting of 100 synthetic conversations that span up to 10M tokens along with corresponding probing questions. BEAM aims to evaluate the abilities of large language models (LLMs) to reason over long-context input. Unlike existing benchmarks, it is more diverse, more realistic and covers multiple memory subtasks. Moreover, the paper introduces LIGHT, a framework that mimics episodic, working, and scratchpad memories to improve the long-context performance of LLMs. Existing LLMs do not perform well on BEAM. Augmenting LLMs with LIGHT improves their performance compared to standard RAG.

### Strengths
- The benchmark contains more realistic conversations, is more diverse, covers many memory subtasks, and contains very long context scenarios. It is challenging even for the best existing LLMs. 
- The paper contains all the required details to understand and possibly replicate the dataset creation. The appendix provides even more details about the dataset creation. 
- The proposed method LIGHT outperforms standard RAG, which is a strong baseline.

### Weaknesses
- There is no qualitative analysis about the kind of errors that the LLMs make.  
- Regarding readability, some sections of the paper are very dense and harder to read. The level of details describing the dataset creation and the propose method is adequate, but the flow of some paragraphs could be improved using for example more bullet points. 
- All the figures are tiny. They cannot be properly read without zooming. The captions of all figures are very short and do not clearly describe them. Figure 1, which describes dataset creation, and Figure 2, which explains the proposed method, should be highlighted more. The texts and notations used in those figures should be as self-explanatory as possible to allow a quick overview of the contributions.
- The paper could present a few baselines besides RAG at least for the small models if it is too expensive. 
- Concerning related work, you might want to consider similar work https://arxiv.org/abs/2502.13791 that also relies on plans/templates to generate synthetic conversations in order to evaluate LLMs' ability to follow instructions in long-context. 
- Minor typo: the generated dialogues in the appendix contains typos ("Trunicated" in red) .

### Questions
- Could you clearly list all the human interventions during dataset creation?
- Have you manually reviewed the generated conversations (not the probing questions or plans) ? Do you have metrics that asses the quality of these conversations? Do you observe hallucinations (not following the plans) in the generated conversations? 
- What are the main limitations that prevent the LLMs from performing well on this task (even using LIGHT)? What are the specific issues that LIGHT can or cannot address?
- How does LIGHT compare to other recent long-context baselines other than RAG? This could be done only on the small models if too expensive to run.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles the challenge of evaluating and improving long-term conversational memory in LLMs by introducing BEAM, a benchmark of 100 coherent, topically diverse dialogues (100K–10M tokens) paired with 2,000 validated probing questions spanning ten memory abilities. It also proposes LIGHT, a cognition-inspired inference framework that combines (i) episodic memory via retrieval over the full conversation, (ii) short-term working memory of recent turns, and (iii) a maintained scratchpad of salient facts filtered for relevance at query time. Experiments across multiple backbones show consistent gains of LIGHT over long-context and RAG baselines; ablations indicate each memory component contributes to performance.

### Strengths
- BEAM spans multi-domain dialogues at four lengths (128K, 500K, 1M, 10M tokens) and evaluates 10 distinct memory abilities, filling gaps in prior single-domain, recall-heavy datasets.
- Component-wise ablations substantiate that each memory module contributes meaningfully to performance.
- Results across context scales (100K – 10M) and models show consistent improvements; the paper also includes component-wise ablations and sensitivity to retrieval budget K.

### Weaknesses
- LIGHT’s episodic memory relies on FAISS with BGE-small embeddings; the paper does not systematically explore robustness to alternative embedding models or indexing setups beyond a brief note.
- While a human validation step is described for probes and a separate human evaluation of conversation quality is reported, rater pool size/protocols are not clearly specified, making it hard to assess annotation reliability and potential residual biases over very long chats.
- LIGHT maintains an episodic index and a growing scratchpad over extremely long contexts. The paper varies K but does not report end-to-end latency/memory/GPU budgets for 1M–10M-token settings, limiting practical interpretability.

### Questions
- Can you clarify which model you are using for the ablation study? Since the results on Table 8 do not match the results on Table1. 
- Why was BAAI/bge-small-en-v1.5 selected for the episodic index? Was the choice driven by speed/latency, availability, or prior retrieval effectiveness on dialogue passages? Any comparisons versus larger BGE variants, E5, or modern instruct-tuned embedding models like openai text embedding 001.

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
4

### Summary
This paper introduces BEAM, a large-scale benchmark for evaluating long-term memory in conversational settings, featuring dialogues up to 10M tokens and 2,000 probing questions across ten distinct memory abilities—including three newly introduced dimensions such as instruction following, event ordering, and contradiction resolution.
The paper presents LIGHT, a cognitively inspired framework integrating episodic (retrieval), working (recent context), and scratchpad (salient fact accumulation) memory modules. 
Experiments on multiple open- and closed-source LLMs demonstrate that LIGHT improves accuracy over RAG and vanilla baselines.

### Strengths
1. Good scale and diversity: context length in conversation setting up to 10M across multiple domains. The samples are validated by human.

2. New complimentary metrics (instruction following, event ordering, and contradiction resolution.) to evaluate the behaviour of LLM in long context setting

3. Cognitive framework - LIGHT - shows convincing improvement in comparison to strong RAG baseline

### Weaknesses
1. Limited dataset size: only 100 conversations for benchmarking, this could lead to high variance in evaluation results.

2. The paper doesn't deeply analyze why models fail on specific memory abilities. For instance, why do all methods struggle with contradiction resolution? No error analysis or qualitative examination of failure modes beyond the single case study in Appendix F.

3. The combination of retrieval + working memory + external memory is not particularly novel. The main contribution is demonstrating this combination works better than RAG for long conversations. When retrieval is removed (Fig 3), it appears that under 1M context length where the models were trained, the retrieval module doesn’t benefit much but may harm the performance. 

4. Unfair claiming at 10M context length evaluation between LIGHT and Vanilla. Models anyways were not trained to perform at 10M context length. It expected that they perform much worse. 

5. It’s unclear whether the author will open-source the data and the framework code. Without these, it’s hard to enable the community to gauge the impact of the work.

### Questions
1. Can you provide error analysis showing common failure patterns for each memory ability and also for each evaluated metrics?

2. At 1M context length experiment, LIGHT doen’s seems to outperform RAG at Abstenation, Contradiction Resolution, and Event ordering. Since RAG is similar to the episodic memory module of LIGHT, any reason why this is the case?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper makes two main contributions: a BEAM benchmark for evaluating long-term memory in LLMs and LIGHT framework that augments LLMs with extra tools to better handle long-context tasks. The authors provide a number of empirical results, evaluating a number of models on the new benchmark and ablating the new LIGHT framework components.

### Strengths
The paper focuses on a highly relevant problem.
The data generation method is sufficiently original.
Experimental evaluation reasonably complements the data-generation effort and the ablation studies help better understand the contributions of different LIGHT framework components.

### Weaknesses
My biggest issues with the paper are:

- Insufficient coverage of related work + I'm unsure about how this work fits into the broader research context in this domain.
The paper is positioned as addressing the problem of long-term memory, but is focused solely on conversation-like context. The way the authors brush off existing work seems a bit dismissive, especially since there are many works that do not suffer from the limitations the authors quote as disqualifying.

For example, Narrative QA and Narrative XL datasets do not consist of concatenated unrelated conversations and do require engagement with the material beyond simple retrieval.

The data generation approach is reasonably original, but I do not think that the authors provide a thorough enough justification for why this approach should be taken over the alternatives.

If the paper were explicitly targeting conversation-like usage contexts, it might have been more defensible. But still, when there are many datasets that are far from being solved, a lot of attention needs to be devoted to justify introduction of another one, with thorough proof (both empirical and theoretical) that the new dataset catches LLM performance gaps that can't be investigated using already existing datasets.

I found it especially unfortunate that in the related work, the paper focuses on the efforts in developing long-context models, while the primary focus of the paper is on long-context LLM evaluation & benchmarking. The latter is covered very sparsely.

- Empirical ablation results (e.g. Figure 3) don't seem to have confidence intervals, and, more imporantly, the results are not very consistent or strong. I.e. it seems that the proposed LIGHT model does not consistently improve performance, and that the helpfulness of its specific components depends on context length (and likely on many other factors like the nature of the task that were not investigated in as much depth).

Overall, I feel that the paper proposes a number of very promising ideas, but requires a bit more depth to properly place them in the broader research context and to make the results more reliably interpretable.

### Questions
Is the focus of the paper intended to be on conversation-like tasks only, or is the claim broader (long-context LLM application/evaluation in general)?

Minor suggestion: I believe that including a table of illustrative examples of probing questions into the main paper would greatly aid readability. It will clarify the approach for the readers who want to understand the paper without having to read the appendix.

### Soundness
3

### Presentation
2

### Contribution
2
