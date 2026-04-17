# Revela: Dense Retriever Learning via Language Modeling

- Decision: Accept (Oral)
- Scores: 6, 6, 6, 8

## Abstract
Dense retrievers play a vital role in accessing external and specialized knowledge to augment language models (LMs). Training dense retrievers typically requires annotated query-document pairs, which are costly to create and scarce in specialized domains (e.g., code) or in complex settings (e.g., requiring reasoning). These practical challenges have sparked growing interest in self-supervised retriever learning.
Since LMs are trained to capture token-level dependencies through a self-supervised learning objective (i.e., next token prediction), we can analogously cast retrieval as learning dependencies among chunks of tokens. This analogy naturally leads to the question: How can we adapt self‑supervised learning objectives in the spirit of language modeling to train retrievers?

To answer this question, we introduce Revela, a unified and scalable training framework for self-supervised retriever learning via language modeling. Revela models semantic dependencies among documents by conditioning next token prediction on local and cross-document context through an in-batch attention mechanism. This attention is weighted by retriever-computed similarity scores, enabling the retriever to be optimized as part of language modeling. We evaluate Revela on domain-specific (CoIR), reasoning-intensive (BRIGHT), and general-domain (BEIR) benchmarks across various retriever backbones. Without annotated or synthetic query-document pairs, Revela surpasses larger supervised models and proprietary APIs on both CoIR and BRIGHT. It achieves BEIR's unsupervised SoTA with ~1000x less training data and 10x less compute. Performance increases with batch size and model size, highlighting Revela's scalability and its promise for self‑supervised retriever learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Revela, a self-supervised retrieval model training method that employs a language modeling task. Revela incorporates in-batch attention, allowing the model to consider both the current sequence and the contexts of other sequences within the same batch when predicting the next token, thereby modeling semantic dependencies between documents. This approach eliminates the need for manually labeled query-document pairs and directly trains the retriever using the modeling task, significantly improving the training efficiency and effectiveness of the model.

### Strengths
1. This paper demonstrates excellent performance across a variety of retrieval tasks with different types, showcasing its versatility.
2. The method requires a smaller batch size. During training, the batch size used in this paper was 8, which is much smaller than other methods that employ contrastive learning for training. This significantly reduces the forest traversal cost. Additionally, the experiments clearly reveal the scaling law between retrieval performance and batch size.
3. There is no need for labeled q-d pairs as training data, and the model can be directly trained in an unsupervised manner on large-scale text datasets.

### Weaknesses
1. This paper compares retrieval models that include both encoder-only and decoder-only models, and it also conducts comparisons based on different model sizes. However, there are still some fairness issues in the comparison.  The original RePlug paper uses query-document pairs for KL training in the RAG setting, while this paper only uses document alone in the reproduction and comparison. This might not be a fair comparison. Could training on the q-d pair data mentioned in RePlug, such as the Pile training data or other q-d datasets like MS MARCO, provide a fairer comparison? RePlug is this paper's main baseline.

### Questions
1. In this paper, it is not explicitly mentioned whether the representation used during inference is from the last token or if mean pooling over all token representations is applied.

2. Is the backbone model used in REPLUG also Llama 3.2?

3. In addition to NDCG@10, could you provide results for other metrics such as MRR and recall?

4. It seems that no ablation experiments are designed in this paper.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
**Motivation:** Dense retrievers are important, but training them usually needs labeled query–document pairs, which are costly and scarce in specialized domains. The paper asks if we can train a retriever in a self-supervised way by treating retrieval like language modeling, where dependencies among chunks of text play the role that token dependencies play in next token prediction.  

**Approach:** Revela trains a retriever together with a decoder-only LM by adding an in-batch attention path that lets each sequence attend to other sequences, with those cross-sequence attention weights set by the retriever’s similarity scores, and both parts optimized through next token prediction. Documents are split into chunks and batched so the model can learn cross-document links; similarities are temperature-normalized cosine scores that modulate the in-batch attention inside each transformer block.  

**Key Results:** Across CoIR, BRIGHT, and BEIR, Revela beats or matches strong baselines while using only raw text: on CoIR it surpasses E5-Mistral-7B-Instruct by 2.8 points and outperforms E5-PT by 9.7 points at smaller scales. On BRIGHT it is on par with large supervised and API models, and on BEIR it reaches the unsupervised state of the art with about 1000× less data and 10× less compute, with gains that grow with model and batch size.

### Strengths
**Unified objective without labels:** The method removes the need for query–document pairs by turning retrieval into language modeling. I like the idea of using NTP for training retriever as it can provide fine-grained supervision compared to InfoNCE. There is a related work of “REPLUG” which also uses supervision from language modeling to train retriever. But the approach from this paper: adding a simple in-batch attention path to a standard decoder-only stack and learning both the retriever and LM with next-token loss has its novelty. 

**Strong results across domains:** It beats a larger supervised E5-Mistral-7B model on CoIR, stays competitive with top supervised and API systems on BRIGHT, and matches E5-PT on BEIR while using about 1000× less data and 10× less compute. The wins hold across different topics and corpus types, which suggests good robustness and transfer.

**Good scaling and clear ablations:** Performance improves as you increase batch size, retriever size, and often LM size. Ablations show the cross-sequence attention and similarity weighting matter, and the model outperforms a matched Contriever baseline with the same backbone. Training on mixed domains keeps quality and can even help.

### Weaknesses
**Applicability**: Adoption may be harder than plug-and-play methods like REPLUG. Revela adds an extra in-batch attention path inside decoder blocks and uses retriever similarities as attention weights, which means touching the LM internals and maintaining custom masking. Even if the code changes are “minimal,” such modifications increase maintenance and may impact speed or memory in non-obvious ways.  

**Batch-composition dependence:** The method conditions next-token prediction on *other documents in the same batch*. This can make results sensitive to how batches are formed or interleaved, and the paper does not deeply analyze robustness to batch construction.

**Resources**: Compared to approaches using single retriever with InfoNCE loss, training will probably likely to require high resources compared to existing works, as it requires optimizing both the retrievers and LLMs. This can make the approach use less batch size given same compute budget.

### Questions
1. Can we keep the LM frozen and train only the retriever while keeping most of the gains?
2. How did you deal with the causal mask when using LLM as the retriever backbone?
3. How robust is performance to batch composition choices, such as mixing documents from different topics or changing the interleaving pattern across chunks?

### Soundness
4

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
3

### Summary
Revela proposes a novel self-supervised framework that jointly trains a dense retriever and a language model by embedding retriever similarity scores as in-batch attention weights within transformer blocks.
This allows the model to learn retrieval ability directly from next-token prediction without any labeled query-document pairs, achieving strong performance on multiple retrieval benchmarks with far less data and computation.

### Strengths
- Revela proposes a novel self-supervised framework that jointly trains a dense retriever and a language model by embedding retriever similarity scores as in-batch attention weights within transformer blocks.
This allows the model to learn retrieval ability directly from next-token prediction without any labeled query-document pairs, achieving strong performance on multiple retrieval benchmarks with far less data and computation.
- The use of retriever similarity as in-batch attention weights seamlessly embeds retrieval into standard Transformer computation with minimal architectural changes.

### Weaknesses
- The motivation for using next-token prediction is unclear. The authors need to provide a detailed explanation of why this training objective can enhance the retriever’s capability.
- The In-batch Attention section is somewhat confusing. It states that In-batch Attention consists of two parts — Standard Self-Attention and In-batch Attention — but within In-batch Attention itself, there is another self-attention output s. I suggest the authors restructure this description for greater clarity.
- Could the authors provide a detailed explanation of the computation steps for In-batch Attention? My understanding is that it involves three internal computations (outputs e, s, and b), yet Figure 2 shows only a single attention map (indicating the computation is fused?)

### Questions
The questions have been presented above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces a self-supervised framework that jointly trains a dense retrieval model along with a LLM using NTP loss and in-batch, cross document attention. The authors posit that NTP loss, when conditioned on in-batch chunks from different documents, weighted by the similarity scores of the retriever, provides strong supervision for measuring similarity among documents. 

Experiments demonstrate that a dense retriever trained with <400k batches (~1000x less than modern retriever training datasets) out-performs strong baselines and enterprise API based models for standard tasks (BEIR) and coding (CoIR) among other sets.

### Strengths
1. The approach of using NTP paradigm to "distill" similarity signals into a dense retriever along with he in-batch attention and weighting by similarity scores method, are interesting and novel ideas.
2. Improved scalability and calibration compared to quadratic baseline (pairwise distillation).
3. Great experimental design including selection of training data and baselines.
4. Thorough ablation study showcases generalization, effect of batch size, LLM base performance, mixing training corpora among others. These studies provide a strong baseline for future exploration.
5. Overall, a very well written paper and easy to understand.

### Weaknesses
1. Training data creation methodology and statistics are underspecified. It would be helpful to understand how the filtering was done and what the "handcrafted rules" L249 are. Similarly, while constructing a batch (following example in Appendix B.2), how were the topics chosen?

### Questions
1. Given the joint training, what are the training dynamics and stability of the system? Does the retriever or the LM converge significantly faster?

### Soundness
4

### Presentation
4

### Contribution
4
