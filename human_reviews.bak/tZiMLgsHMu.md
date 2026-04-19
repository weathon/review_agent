# JudgeRank: Leveraging Large Language Models for Reasoning-Intensive Reranking

- Decision: Reject
- Scores: 5, 5, 6, 5

## Abstract
Accurate document retrieval is crucial for the success of retrieval-augmented generation (RAG) applications, including open-domain question answering and code completion. While large language models (LLMs) have been employed as dense encoders or listwise rerankers in RAG systems, they often struggle with reasoning-intensive tasks because they lack nuanced analysis when judging document relevance. To address this limitation, we introduce JudgeRank, a novel agentic reranker that emulates human cognitive processes when assessing document relevance. Our approach consists of three key steps: (1) query analysis to identify the core problem, (2) document analysis to extract a query-aware summary, and (3) relevance judgment to provide a concise assessment of document relevance. We evaluate JudgeRank on the reasoning-intensive BRIGHT benchmark, demonstrating substantial performance improvements over first-stage retrieval methods and outperforming other popular reranking approaches. In addition, JudgeRank performs on par with fine-tuned state-of-the-art rerankers on the popular BEIR benchmark, validating its zero-shot generalization capability. Through comprehensive ablation studies, we demonstrate that JudgeRank's performance generalizes well across LLMs of various sizes while ensembling them yields even more accurate reranking than individual models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents JudgeRank, a LLM-based re-ranking framework for reasoning-extensive retrieval. It consists of three steps (components): query analysis, document analysis, and final judgement making. In query analysis, an LLM is prompted to generate the core problem or question of the query/question. In document analysis, an LLM is prompted to analyze every top-k retrieved document with the analysis of the query, and the definition of relevance. In final judgement making, an LLM is prompted to generate a Yes/No judgement given the query and document analysis, and the definition of relevance. Experiments show that JudgeRank works well in both reasoning-extensive and ordinary retrieval tasks.

### Strengths
The proposed method is simple and effective, and generalizes well on a series of retrieval tasks.

### Weaknesses
- The authors argue that “In contrast, the original BRIGHT paper shows that GPT-4 with listwise reranking improves on top of vanilla BM25 baseline by an average of 2.7 points on nDCG@10, a much smaller improvement than our approach despite using a much stronger LLM”. However, the experiments from the BRIGHT paper is conducted on vanilla BM25, without the use of CoT reasoning on the query, whereas JudgeRank in this paper reranks top documents from BM25 with query-side CoT reasoning. So the comparison here is not fair - “GPT-4 with listwise reranking” should be run in the same setting.
- The effectiveness of this three-step framework is not clear. What is the result of only removing the query or document analysis step? Do both analysis steps help? And, what about putting the document analysis and judgment into a single prompt, simplifying the process into two steps? An in-depth analysis of the design of JudgeRank is needed.
- See questions.

### Questions
- How do authors obtain the probability of the “Yes” judgment? This needs to be explained.
- The efficiency of JudgeRank is not examined in this paper. Including an analysis of its running time and a comparison of RankGPT and other LLM-based ranking methods in a fair setting would be better.
- What is “{definition of relevance}” set to in each task? How does it affect the re-ranking result?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces an LLM-based document ranking pipeline given a query. The authors use 3 main stages, query analysis, document analysis and judgment. They evaluate their pipeline on 2 main benchmarks, BRIGHT and BEIR, with 3 different sized Llama models. The authors claim that their methodology performs better than existing state-of-the-art reranking models/methods.

### Strengths
- What I appreciated most is the decoupling of analysis and judgement. Breaking down a complex task as so is an elevated approach, especially given the query-based document analysis.
- Inclusion of prompt templates.
- Experimentation across model sizes.

### Weaknesses
- The paper could touch base further on the limitations of context length, this is a very intuitive limitation than can be mitigated with a parsing pipeline that creates overlapping shingles for example that are evaluated as separate documents…etc. Even in the presence of 128k context windows, models’ reasoning capabilities tend to decline with really large windows, making it harder to capture a nuanced understanding of the full context, potentially missing some aspects or attending to portions of the context in a manner biased to position. 
- Results could provide more example outputs.
- The pipeline might have a scaling limitation with a large number of documents. One thing that can be added to the pipeline is a global topic prompt per doc, followed by a filtering step per query for relevant documents given the latter (possibly even in batches).

### Questions
- Have you concluded anything from your experiments that differentiates between reasoning and knowledge heavy reranking tasks?
- Could you elaborate further on why do you think ensembling results are higher? Could it simply be the effect of randomness (especially given the large context window) of multiple retries and the advantage of a majority vote? Have you performed any experimentation to disprove this? majority vote measures with the same model for example?
- Have you explored COT judgement? (beside decoupled analyses)
- Have you explored ensembling with embedding based ranking methods?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
JudgeRank represents an advancement in reasoning-intensive document retrieval, demonstrating that structured reasoning approaches can substantially improve reranking performance. The method's success in both specialized (BRIGHT) and general (BEIR) benchmarks validates its effectiveness and generalizability, while the proposed model ensembling techniques offer promising directions for future improvements.

### Strengths
1. JudgeRank performs well in BRIGHT and is compatible with other methods in BEIR.
2. JudgeRank is more suitable for (RAG) tasks that require complex reasoning.
3. The paper is well-written.

### Weaknesses
1. The computational cost of multiple LLM inferences may not be justified by the performance gains.
2. Sensitivity to prompt wording.
3. Maybe over-complex for simple retrieval tasks?
4. The inability to perform document analysis in advance may results in high costs for each query.

### Questions
1. How does your computational efficiency compare to traditional encoding-based frameworks? Could you provide more details such as throughput rate, inference speed, and memory usage?
2. What are your thoughts on using closed-source LLMs like OpenAI's models? There might be situations where you can't obtain "yes" and "no" logits in the top@k (k=20?) logits.
3. Could you provide insights on why the performance difference between Llama 70B and 405B models isn't significant?
4. Perhaps you could use different types of models (maybe same parameters) to create an ensemble?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces JudgeRank, a new method for LLM-based pointwise passage reranking. The method diverges from existing methods in that they apply LLMs to do passage-level relevance scoring, breaking this process into three LLM-based steps: 1) query decomposition to understand the core problems needing to be solved 2) query-focused summarization of each document and 3) combining these two together outputs together to assign a relevance score to each document. They test their method on two popular reranking benchmarks, finding their method outperforms vanilla first-stage retrieval, and also two baseline LLM methods fine-tuned for this task.

### Strengths
The paper is well-written and very easy to follow. The technique is intuitive, taking retrieved top-k documents, and performing passage reranking through the use of instruction-tuned LLMs (LLama-3.1 model family). They do some interesting analysis, such as exploring the similarities between model predictions at different sizes (8b vs 70b vs 405b), exploring ensembling and finding this can improve accuracy, and also doing component-level analysis of the JudgeRank pipeline (comparing different scoring metrics, and also investigating the benefits of this modular reasoning approach).

### Weaknesses
Overall, the novelty of this work is somewhat lacking -- the method is effectively comprised of a few intuitive prompting steps which, while effective, are fairly intuitive and don't introduce any fundamentally new way of tackling more challenging reasoning; the authors note this work was inspired by popular techniques such as Chain-of-Thought and LLM-as-a-Judge, and it seems this is more or less a straightforward application of existing ideas to a particular use case.

I think some additional steps/experiments could help to bolster the paper and add further takeaways; some concrete directions might be (1) to dig deeper on the base model selection, understanding if/how other model families (both open and propietary models, so long as they provide relevant confidence scores) compare on this task, (2) further exploring aggregation/ensembling methods with better cost:performance ratio (e.g., what happens if you ensemble a mixture of several smaller models from different families), and (3) investigating more 'adaptive' prompting configurations targeted to each use case (e.g. you note: "Leveraging LLMs in a zero-shot setting allows us to flexibly define the relation between the query and the document....this flexibility is important because the user may define either “document that supports a query” or “document that refutes a query...”).

It would also be interesting to understand what the current 'ceiling' is for this task. It seems there is a first-stage BM25 step applied, and I'm curious what impact this has on final performance -- ie, what are the implications if you vary k during the first-stage (from latency/scaling to performance); could your method potentially perform better without the first-stage retrieval step?

### Questions
**Base models:** have you explored base models other than Llama3.1?

**Prompt design and variants:** How much impact do the prompts have on JudgeRank’s performance? The paper mentions potential prompt variations -- when would different prompts be applicable, and how might they influence retrieval effectiveness?

**Ensembling** Maybe I missed this, but can you share the model ensemble performance on the BEIR benchmark?

**Listwise reranking:** I understand this is somewhat orthogonal to your approach, but do you have any thoughts on how you might adapt your technique + intuition to listwise reranking (instead of pointwise).

**Typos / Grammar / Presentation**
- l138 - mimicking
- l312 - performing
- l271 - best-first retrieval? best retrieval-first model?
- Table 2 - I might suggest presenting the models in the same order as the prior table. Also, it might be helpful if you explicitly re-state (maybe in the caption?) the details of this JudgeRank variant (that it's the 70B variant w/ ensemble)

### Soundness
3

### Presentation
4

### Contribution
2
