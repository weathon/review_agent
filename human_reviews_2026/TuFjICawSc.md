# Learning Retrieval Models with Sparse Autoencoders

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Sparse autoencoders (SAEs) provide a powerful mechanism for decomposing the dense representations produced by Large Language Models (LLMs) into interpretable latent features. We posit that SAEs constitute a natural foundation for Learned Sparse Retrieval (LSR), whose objective is to encode queries and documents into  high-dimensional sparse representations optimized for efficient retrieval. In contrast to existing LSR approaches that project input sequences into the vocabulary space, SAE-based representations offer the potential to produce more semantically structured, expressive, and language-agnostic features. By leveraging recently released open-source SAEs, we show that their latent features can serve as effective indexing units for representing documents and queries for sparse retrieval. Our experiments demonstrate that SAE-based LSR models consistently outperform their vocabulary-based counterparts in multilingual and out-of-domain settings. Finally, we introduce SPLARE, a 7B-parameter multilingual retrieval model capable of producing generalizable sparse latent embeddings for a wide range of languages and domains, achieving top results on MMTEB’s multilingual and English retrieval tasks. We also release a more efficient 2B-parameter variant, offering strong performance with a significantly lighter footprint.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces SPLARE, a novel Learned Sparse Retrieval model that ingeniously integrates Sparse Autoencoders (SAEs) with existing LSR frameworks like SPLADE. SPLARE projects text representations into a "latent feature space" learned by a pre-trained SAE, which is designed to be more semantic and language-agnostic.Through extensive experiments, the authors demonstrate that SPLARE significantly outperforms its vocabulary-based counterparts on multilingual and out-of-domain retrieval tasks.

### Strengths
1.The paper originally combines the semantic feature decomposition of SAEs with learned sparse retrieval;
2.The experimental results are impressive. SPLARE consistently outperforms SPLADE-Llama in multilingual and out-of-domain settings and achieves a level of performance comparable to SOTA dense models on comprehensive benchmarks like MMTEB.
3.Excellent Generalization and Efficiency: SPLARE demonstrates strong multilingual generalization even when trained only on English data.

### Weaknesses
1. Lack of Detailed Cost Analysis: The introduction of the SAE module incurs additional computational and memory costs at inference time. While the paper mentions mitigating this by using intermediate LLM layers, it does not quantify the latency and memory overhead introduced by the SAE projection step itself. 
2. Insufficient Context for Sparsity: The paper controls document vectors to ~400 non-zero dimensions. While this is sparse in a >130k-dimensional space, its advantage is not immediately obvious when compared to dense vectors, which may only have a few hundred to a thousand dimensions in total. The paper lacks a direct, end-to-end comparison of latency, index size, and memory usage between SPLARE (with inverted indexes) and a top-tier dense retriever (with an ANN index like HNSW) at a similar effectiveness level.

### Questions
1. Can you provide more information on the analysis and comparison of deployment costs?
2. Regarding sparsity control, the paper notes that combining training-time regularization with inference-time Top-K pruning was superior to using Top-K alone. Could the authors elaborate on the distinct effects of these two methods on the final representations? For example, does the regularization loss encourage the model to learn a better distribution of important features, which Top-K then effectively truncates, whereas relying solely on Top-K might crudely discard features that are contextually important but have marginally lower activation scores?
3. The model training relies on knowledge distillation from a cross-encoder. Was a contrastive learning approach, which is dominant in dense retriever training, considered or attempted?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes SPLARE (SParse LAtent REtrieval), a novel Learned Sparse Retrieval (LSR) approach that leverages pre-trained Sparse Autoencoders (SAEs) to represent queries and documents as sparse vectors over a latent feature space rather than the traditional vocabulary space. By inserting SAEs into intermediate layers of large language models (LLMs), SPLARE produces semantically rich, multilingual, and domain-generalizable sparse embeddings. The authors demonstrate through extensive experiments on benchmarks like MMTEB, MIRACL, and XTREME-UP that SPLARE consistently outperforms vocabulary-based LSR models—especially in multilingual and out-of-domain settings—while maintaining high retrieval efficiency. A 7B-parameter multilingual variant of SPLARE achieves competitive results against state-of-the-art dense retrievers, establishing it as the top-performing LSR model on MMTEB at the time of submission.

### Strengths
1.	The paper is the first to systematically use the pre-trained sparse autoencoder as the "implicit vocabulary" of LSR, replacing the traditional methods based on the original tokenizer vocabulary.
2.	The proposed method demonstrates outstanding performance in multilingual and cross-domain scenarios.

### Weaknesses
1.	Although the authors find that the best performance was achieved at Layer 26 for different SAE widths, this conclusion is only drawn from the experiments conducted on the Llama-3.1-8B model. If readers wish to apply the SPLARE method to other models (such as Gemma), there is no guarantee that the selection of this layer will be successful. It is suggested that the author provide more experimental results on the models and the layer selection strategies to enhance the generalization ability of the method.
2.	Current state-of-the-art dense retrievers generally adopt contrastive learning, but SPLARE chose distillation. The paper merely refers to distillation as a "common toolbox", but does not provide an explanation as to why distillation is more suitable for the sparse latent space. This crucial design choice lacks an analysis of motivation.
3.	The paper clearly states that only residual SAEs were used, but no ablation experiments or reasons are provided. SAEs using MLP or Attention might capture richer or different types of semantic information. Ignoring them could lead to information loss and limit the model's expressive power.
4.	Table 1 shows that in both the English and Multilingual benchmarks, SPLARE significantly lags behind SPLADE-Llama in the Code domain. The paper only mentions "the advantage diminishes" without providing any analysis (such as feature visualization, error cases). This may indicate that the SPLARE method has flaws when dealing with highly structured and symbolic text (such as code), and this needs to be emphasized.
5.	The abstract and introduction have repeatedly emphasized that SPLARE is the first LSR model that can match the SOTA dense model on MMTEB. However, Table 3 shows that its score of 60.9 is much lower than that of top dense models such as Qwen-3-Embedding-8B (70.9), inf-retriever-v1 (66.5), etc. The discussion in the paper contains an exaggeration.

### Questions
1.	The paper claims to propose "Learned Sparse Retrieval", but its core component SAE, is frozen throughout the training process. This means that the model cannot learn or optimize the "latent vocabulary" itself for retrieval purposes and can only be adapted to a fixed and non-designated feature space for the retrieval task through LoRA fine-tuning of the LLM's intermediate representations. This is fundamentally different from LSR methods such as SPLADE (whose LM head is learnable), and it weakens the claim of "Learned".
2.	The paper not only employs DF-FLOPS type sparse regularization but also adds Top-K pooling during inference to forcibly control the number of activations. The authors also admit that training solely with Top-K would be worse, but still choose the "moderately sparse" + post-cropping approach during training. This inconsistency will introduce non-differentiable post-processing and distribution drift, potentially leading to instability near the critical threshold and sorting reversals.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies sparse auto encoders based latent features effectiveness in representing documents and queries. Authors claim that sparse autoencoders based learned sparse retrievers outperform their vocabulary projection based counterparts. The main contributions  of this paper are:
1. This paper introduces SPLARE a new sparse retrieval technique leveraging sparse auto encoders.
2. Detailed investigations of benefits of using latent vocabulary vs standard LLM vocabulary.
3. New 7B multilingual sparse retriever model.

### Strengths
The main strengths of the paper are as follows:
1. Idea of using latent features for document and query representation might have a lot of applications in text retrieval
2. The SPLARE learned sparse retrieval framework is interesting.
3. Detailed performance comparison with SOTA text retrieval models.

### Weaknesses
The main Weaknesses of the paper are as follows:
1.  Details about SPLARE working is missing. It might need some more explanation.
2. Terms like SAE width needs some more details.
3. Limited novels of the overall retrieval model when compared to sparse embed and other LSR techniques.
4. How SAEs are trained is not clear from the paper.
5. Paper writing have a lot of scope of improvement.
6. Limited performance improvement on MTEB retrieval benchmark when compared to SPLADE-LLama

### Questions
1. How does the proposed approach with sparse embed  paper https://research.google/pubs/sparseembed-learning-sparse-lexical-representations-with-contextual-embeddings-for-retrieval/?
2. Is availability of Pre-trained SAE a constraint of this model? 
3. Any comparison of latency and training time when compared to spade and other dense models?

### Soundness
3

### Presentation
2

### Contribution
2
