# SpotlightRAG: Enhancing Factual Accuracy with Position-Aware Span Selection

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2

## Abstract
Retrieval-Augmented Generation (RAG) enhances LLMs with external knowledge, but current methods face key limitations. Most solutions operate at a coarse passage or sentence level, indiscriminately concatenating retrieved text, which introduces noise, overlooks decisive sub-sentential phrases, and is susceptible to positional bias where evidence is lost in the middle of long contexts. To overcome these challenges, we propose SpotlightRAG, an inference-time framework that enhances factual accuracy through precise, span-level context selection and explicit relevance signaling. SpotlightRAG employs a position-aware scoring mechanism to identify and weight critical text spans, directly countering positional bias. It then uses novel retrieval-aware prefix tokens to explicitly annotate the relevance of each span for the generator, providing fine-grained, interpretable control without model retraining. Extensive experiments on four benchmarks—PopQA, TriviaQA, Natural Questions, and MultiHopQA—demonstrate that SpotlightRAG consistently outperforms state-of-the-art baselines, including InstructRAG, RankRAG, and In-Context RALM, improving accuracy over strong baselines by 2.1% on PopQA and 1.2% on the challenging MultiHopQA dataset. An anonymized implementation is available at https://anonymous.4open.science/r/SpotlightRAG-5F6A/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces an inference-time framework, SpotlightRAG, which uses a position-aware scoring mechanism to identify and weight important text spans. The framework uses retrieval-aware prefix tokens to signal the relevance of each selected span. The experiments demonstrate improved accuracy on multiple benchmarks.

### Strengths
- This paper proposes the identification of decisive sub-sentential phrases and entities, which is finer-grained than existing sentence-level integration of retrieved passages. The proposed approach allows us to filter retrieval noise.
- The framework requires no reader model retraining, making it lightweight and broadly applicable.
- Their experiments show that SpotlightRAG outperform state-of-the-art baselines, including InstructRAG and RankRAG, on four QA benchmarks such as PopQA and TriviaQA.

### Weaknesses
- Though I understand the concept of the proposed framework, I wonder if examples like one in Figure 2 truly happen, particularly for recent LLMs. It would be great to show case studies where the proposed framework helps LLMs.
- The authors use a single retriever and reader model through the experiments if I understand correctly. This limits the generalizability of the proposed framework, e.g., the performance across model sizes and model families.
- Some critical information is missing. The experimental setting does not explain which retriever and reader models are used, which must be explicitly mentioned in the paper. I also could not find any concrete model information about a lightweight relevance scorer. This is a minor point but Table 4 is not mentioned in the paper. It seems that the authors should mention it in Section 4.5.

### Questions
- Could you provide a couple of concrete examples (case studies) where the proposed framework helps a reader model?
- Which retriever and reader models were used in the experiments and why? I found that llama-3-8B is used in the supplementary material. Was it the model the author used?
- Related to the above, is there any performance trend about the model size? Since larger and newer models usually have better context comprehension, is the proposed method less helpful for such models, e.g., llama-3.3-70B, gemini-2.5-flash, and OpenAI’s recent models?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents SpotlightRAG, a novel method to score important passages within a document for more effective retrieval augmented generation. The authors do so by computing pairwise token similarities and propagate this information to the actual inference / generation part. The experimental results show improvements over recent baseline of up to 4pp. At last the author include hyperparameter experiments to investigate performance impact of different choices of hyperparameters on their approach.

### Strengths
- The empirical results show improvements in fine-tuning and zero-shot settings on four selected benchmarks. The improvements range from 1pp. to 4pp., which is good considering recent baselines from 2024/2025.
- The ablations on removing different parts of the approach show that all presented components of SpotlightRAG and indeed necessary to get the presented performance improvements.
- The paper is easy to follow and the presentation of the proposed approach is sufficient.

### Weaknesses
- While I support introductory graphs, the data source and experimental setup for Figure 1 are not present. Thus, it is hard to follow the argumentation the current RAG methods do need more fine-grained capabilities, when the majority of information lies in sentence-level texts.
- The core of the approach relies on token-level similarities, but many things are not specified here such as whether we are using 1 or 2 models to encode query and context. Transformers are not necessarily optimized for token-level similarity and rather yields high similarity between all token-pairs rather than a proper distribution that is suited for RAG. Some qualitative experiments to see the relevance scores per sentence would be good at this point.
- Several hyperparmeters are missing such as the entire training setup for the baselines and the presented approach. Further, there is no comment about repeated experiments and no standard deviation reported, making it difficult to see improvements of 1-2pp. as clear improvements.
- Overall, I think the idea is simple and intuitive but the formal presentation could be improved. For instance, Formula 5 is a simple softmax distribution of pairwise similarities or the previously mentioned unexplained Figure 1. Further, the ablations could be extended as they currently only show minimal variations of crucial hyperparameters such as k. Further, we see in the ablations that performance degrades as soon as we change some of the hyperparameters.

### Questions
- What categories of evidence are you annotating the sentences with?
- How do you determine the sentence boundaries in your approach, are you simply taking dots? Have you done experiments trying out different sentence boundaries?
- Another point here is that I have doubts about actually retrieving the correct passages based on the token similarities. While I think it is helpful to point a LLM to the sections in a document that are relevant to answer a question, I think token-level similarities do not help for actual retrieval. Could you specify why you think token-level similarities do help for the retrieval process?
- Do you think considering you hyperparameter experiments, that it is universal applicable or do we need to tune the parameters every time for new datasets or domains?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes "SpotlightRAG", an inference-time framework designed to improve the factual accuracy of Retrieval-Augmented Generation (RAG) systems. The author 1) use "position-aware scoring mechanism" to identify and weight critical text out of the whole passage; 2) use "retrieval-aware prefix tokens" to pass these relevance scores to a generator model without requiring retraining; and 3) test their method "SpotlightRAG" on four QA benchmarks and outperforming over some methods.

### Strengths
1. The paper identifies positional bias as a key problem.
2. The paper introduced SpotlightRAG as an inference-time framework. The use of retrieval-aware prefix tokens, provides fine-grained control to the generator without requiring model retraining.
3. The author compares SpotlightRAG against numerous methods on various benchmarks.

### Weaknesses
1. The major problem for this paper is the lack of novelty, The paper states that RankRAG[1] "Employs sentence-level re-ranking to filter retrieval noise". SpotlightRAG, in its "Phase 1," computes a relevance score $R(s_t)$ for each candidate sentence $s_t$ and then selects the "top-K sentences" based on this score. This is a sentence-level re-ranking and selection process, just as in RankRAG. Additionally, given the marginal gains over RankRAG on TriviaQA (+0.3% w/Train, +0.8% w/o Train), it is questionable whether this represents a meaningful advance against RankRAG.
2. The ablation study in Table 3 make the SpotlightRAG performance and experimentation questionable. Removing the "Fine-grained Scorer" drops performance to 65.9% , which is below the InstructRAG baseline (66.2%). This suggests the proposed scoring mechanism is brittle and only works in combination with the prefix tokens.
3. The paper claims its $O(N\cdot|q|\cdot|c|)$ time complexity is "lightweight" and "manageable". Calling this "lightweight" is a misleading, as it adds a quadratic complexity step (query length $\times$ context length) for each of the $N$ passages. 

[1] RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs

### Questions
1. Please Justify your method against RankRAG. Given that both perform inference-time sentence re-ranking to filter noisem, is the difference in novelty is just the use of a ColBERT-style late-interaction scorer instead of an alternative sentence-level scorer?
2. Since you said your method is "lightweight", can you provide the promised latency data (ms) comparing your method against baselines like InstructRAG and RankRAG on GPU time?

### Soundness
1

### Presentation
1

### Contribution
1
