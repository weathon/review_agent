# Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
Context compression is an advanced technique that accelerates large language model (LLM) inference by converting long inputs into compact representations.
Existing methods primarily rely on autoencoding tasks to train special compression tokens to represent contextual semantics.
While autoencoding tasks enable compression tokens to acquire compression capabilities, we remark that such capabilities potentially conflict with actual downstream
task requirements, prevent the models from learning the features more beneficial for real-world usage.
Based on this observation, we propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. To ensure that anchors can effectively collect information, SAC introduces two key designs: (1) anchor embedding, a learnable embedding vector attached to the selected anchor tokens to mark compression carriers and (2) bidirectional attention modification, which enables anchor tokens to integrate information from the entire context. Experimental results show that SAC consistently outperforms existing context compression methods across different compression ratios and model sizes on question-answering and long-context summarization tasks. Our data, model and code have been released at \href{https://github.com/lx-Meteors/SAC}{https://github.com/lx-Meteors/SAC}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes Semantic-Anchor Compression (SAC), which directly selects anchor tokens from the original context and aggregates contextual information into KV representations. It consists of two key designs---anchor embeddings and bidirectional attention to ensure the compression performance.

### Strengths
1. The proposed solution is simple but effective. The author shows good results on many downstream tasks.
2. The paper is generally well-written and easy to follow.

### Weaknesses
1. The intuition behind the proposed method is confusing. The authors argue that "SAC is distinct to previous compression-based methods" because "previous methods compress contextual information into context-agnostic special tokens". However, in a auto-regressive setting, the compressed token is actually context-dependent since it can see all tokens behind in attention computation.
2. SAC add lora to Llama-3.2-1B with attention map modification, from causal to bidirection attention. This sounds quite confusing and seems not to be a normal routine. Why does not use encoder models directly?
3. The idea of context compression has been well-studied in many previous works, such as [1]-[5]. These works impair the novelty of proposed method.

[1] Yen, Howard. Long-context language modeling with parallel context encoding. MS thesis. Princeton University, 2024.

[2] Li, Yuhong, et al. "Snapkv: Llm knows what you are looking for before generation." Advances in Neural Information Processing Systems 37 (2024): 22947-22970.

[3] Zhang, Zhenyu, et al. "H2o: Heavy-hitter oracle for efficient generative inference of large language models." Advances in Neural Information Processing Systems 36 (2023): 34661-34710.

[4] Han, Wei, et al. "Two are better than one: Context window extension with multi-grained self-injection." arXiv preprint arXiv:2410.19318 (2024).

[5] Zhang, Peitian, et al. "Long context compression with activation beacon." arXiv preprint arXiv:2401.03462 (2024).

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces **Semantic-Anchor Compression (SAC)**, a new training-based method for LLM context compression. Its generation pipeline follows the prevailing setup:

- one language model acts as the compression model to produce compressed tokens from a given context, and
- a (frozen) generation model consumes those compressed tokens to generate outputs for downstream tasks.

Prior approaches----ICAE, 500xCompressor, and EPL ----first **pretrain with an autoencoding (AE) objective** to reconstruct the original context from compressed tokens, then fine-tune by passing the compressed tokens to a frozen generator, optimizing the LM objective and, finally, the downstream (e.g., QA) objective. The authors argue these methods require new, randomly initialized compression tokens*whose embeddings must be trained----on top of LoRA fine-tuning of the compression model----which "lack inherent semantic information" and thus demand extensive AE pretraining.  

SAC addresses this by selecting anchor tokens directly from the original context, removing the need for AE pretraining and retaining only LM and task-level fine-tuning. Concretely, the context is partitioned into equal-length chunks, and the **center token** of each chunk is chosen as the **anchor** (following EPL). A fixed **anchor embedding** is then added to these tokens to highlight their special role. Finally, **bidirectional attention** is enabled **only in the compression model**, allowing each anchor to aggregate information from the **entire** context; the resulting **compressed KV states**----one per anchor----serve as the compact representation passed to the generation model. In experiments, the authors use Llama-3.2-1B for both compression and generation; during training, the compression model is fine-tuned with LoRA (r=128) while the generation model remains **frozen**.

To recap, SAC's contributions are twofold:
1. **Eliminates AE pretraining**, relying only on **LM** and **downstream (QA) fine-tuning**.  
2. **Replaces learned compression tokens** with **semantically grounded anchor tokens** taken **from the original context**.

Two simple architectural tweaks make this effective: (i) add a dedicated **anchor embedding** to mark anchors, and (ii) apply **bidirectional attention** (in the compression model) so anchors can gather information from both preceding and succeeding tokens.

Empirically on MRQA, SAC consistently outperforms strong AE-based baselines (500xCompressor, EPL) on both in-domain and out-of-domain datasets, with the gap widening at higher compression ratios (e.g., 15× and 51×). The method also shows better training efficiency due to removing AE pretraining.

### Strengths
1. The critique of the AE's inefficiency is valid.

2. The core idea of using in-context "anchor tokens" is reasonable. It simplifies the compression training process and, as the t-SNE analysis (Fig. 5) suggests, produces compressed representations that are more aligned with the original token space. 

3. SAC consistently outperforms baselines on MRQA, as well as during training (Appendix C trade-off curves). The fact that its performance advantage *increases* at higher compression ratios (15x, 51x) makes it more compelling. The experimental setups and compression ratios are consistent with prior works.

4. Table 5 shows that adding an AE task to the SAC architecture (SAC w/ AE+LM) actually *hurts* performance compared to the LM-only (SAC) model. This provides powerful evidence for the "AE mismatch" theory that the authors argue for, which is quite different from what the literature used to believe.

5. By removing the costly AE pretraining step, the method is inherently simpler and faster to train (as shown in Appendix B.1, Table 7).

### Weaknesses
1. The paper's default strategy (uniform chunking) is simple and effective. However, the ablation in Table 4 shows it performs similarly to a more complex method (Lingua-2). This suggests the selection strategy is important, but the paper stops short of exploring it deeply. 

2. The baselines compared in the paper seem to be a bit outdated. The paper is missing an important baseline, DAST [1], which is also chunked compression but with a dynamic token budget, achieving decent performance at a high compression ratio. 

3. The evaluation is performed exclusively on MRQA, a question-answering benchmark. This is a good choice, as QA is a key application. However, the paper's central critique is that the AE task is misaligned with *downstream tasks*. For a task like long-document summarization, an AE-style reconstruction objective might be *more* aligned than a QA objective. It is unclear if SAC's superiority would hold on tasks that require a more holistic understanding of the full context, rather than just extracting specific facts.

4. The paper states it "modifies the standard causal attention to a bidirectional attention mechanism." For clarity, it would be beneficial to explicitly state whether this bidirectional mask is applied *only* to the anchor tokens (i.e., their queries can see all keys/values) or if the entire compressor LLM (with LoRA) is run in a fully non-causal (BERT-style) mode during the compression pass.

5. All experiments are conducted on a single model at a single parameter size. It does not guarantee that the performance is generalizable to other models or models with much larger scales.

**Reference**:
Shaoshen Chen, Yangning Li, Zishan Xu, Yongqin Zeng, Shunlong Wu, Xinshuo Hu, Zifei Shan, Xin Su, Jiwei Tang, Yinghui Li, and Hai-Tao Zheng. 2025. DAST: Context-Aware Compression in LLMs via Dynamic Allocation of Soft Tokens. In Findings of the Association for Computational Linguistics: ACL 2025, pages 20544–20552, Vienna, Austria. Association for Computational Linguistics.

### Questions
1. (Regarding Weakness 1) Have you experimented with or considered a *learned* selection mechanism? It seems plausible that a small, trained module could identify more semantically salient tokens to serve as anchors, potentially improving performance further, especially at very high compression ratios. In addition, attention/entropy-based KV selection (i.e., from the KV eviction literature) could also be worth a try.

2. (Regarding Weakness 2) Can you add DAST[1] for comparison? Better if PCC [2] can also be included. 

3. (Regarding Weakness 3) How do you anticipate SAC would perform on non-extractive, non-QA tasks like long-document summarization? The AE objective, while mismatched for QA, does force a holistic reconstruction. Does the LM-only pretraining capture enough global information for tasks that require a broader, generative understanding of the entire context? 

3. (Regarding Weakness 4) Could you please clarify the precise implementation of the bidirectional attention? Is the attention mask modified only for the anchor token queries, or is the entire encoder pass non-causal for all tokens?

4. It would be better if more compression ratios could be experimented with, producing a trade-off curve between performance and compression ratio to further validate the robustness of SAC.

5. Can you clarify how you initialize the anchor embeddings?

**Minor Issues (does not affect evaluation)**:

1. At line 198, "While using original tokens … limiting its representation power." should be 1 sentence instead of 2.
2. At line 429, there should be a space within "Analysis.In".

**Reference**:
1. Shaoshen Chen, Yangning Li, Zishan Xu, Yongqin Zeng, Shunlong Wu, Xinshuo Hu, Zifei Shan, Xin Su, Jiwei Tang, Yinghui Li, and Hai-Tao Zheng. 2025. DAST: Context-Aware Compression in LLMs via Dynamic Allocation of Soft Tokens. In Findings of the Association for Computational Linguistics: ACL 2025, pages 20544–20552, Vienna, Austria. Association for Computational Linguistics.

2. Liao. 2025. Pretraining Context Compressor for Large Language Models with Embedding-Based Memory. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 28715–28732, Vienna, Austria. Association for Computational Linguistics.

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
4

### Summary
This paper proposes Semantic-Anchor Compression, a novel framework designed to improve the efficiency and performance of Large Language Models on long-context tasks. The method introduces an autoencoding-free approach to context compression. The core novelty lies in three architectural designs: 1) It selects 'anchor tokens' directly from the original context to act as information carriers. 2) It introduces a trainable 'anchor embedding' to semantically distinguish these selected tokens. 3) It enables 'bidirectional attention' specifically for these anchor tokens, allowing them to effectively aggregate global context. This design allows the compression mechanism to be optimized directly for downstream tasks. Experimental results on a benchmark dataset demonstrate that the method achieves strong performance across various compression ratios and shows favorable generalization capabilities.

### Strengths
1. The paper introduces a novel and effective information compression method based on anchor tokens. This "autoencoding-free" paradigm successfully moves beyond traditional context-agnostic compression tokens.

2. The proposed SAC method is simple, architecturally concise, and elegant in its design.

3. The paper is exceptionally well-written and well-structured, clearly articulating the core problem, the proposed solution, and its advantages.

### Weaknesses
1. The paper's default strategy is to select the "middle token" of a chunk. However, the ablations (Table 4) are missing a critical and intuitive baseline: selecting the "last token" of each chunk as the anchor. In a causal model, the last token has naturally seen the entire chunk. This is a significant omission that prevents a full assessment of the "middle token" strategy and makes it difficult to judge if the performance gain comes from a truly better mechanism or just a better-positioned token.
2. All experiments are conducted only on QA tasks. As the authors state, the AE task deviates from "real-world usage," but using only QA to represent "real-world usage" is far from sufficient. SAC is a lossy compression (as it discards the AE objective), making it highly likely to have overfitted to the specific patterns of QA tasks. Its performance on other long-context tasks (e.g., summarization, story generation) is unknown.

3. The paper's core argument is that AE-free is better. However, the authors never test the SAC + AE combination. That is, using SAC's anchor architecture (with bidirectional attention) to simultaneously perform the downstream task and the AE reconstruction task. It is plausible that SAC + AE would have slightly lower QA performance but be more robust and general-purpose (by retaining reconstruction). The authors claim AE is harmful but do not prove this experimentally (i.e., by comparing SAC vs. SAC + AE).

### Questions
1. Can you please provide a baseline experiment using the "last token" of each chunk as the anchor, while still applying your anchor embedding and bidirectional attention? I am very curious how this intuitive baseline compares to your "middle token" strategy.

2. Do you have plans to validate SAC on other long-context tasks besides QA (e.g., summarization, multi-turn dialogue)? The current setup makes me highly concerned that the model is overfitting to QA while sacrificing general compression capabilities.

3. Did you consider testing a SAC + AE setup (i.e., using your anchor architecture but retaining the AE loss)? What are your thoughts on the trade-off between QA performance and general-purpose robustness that this combination might offer?

4. Could you provide a detailed analysis of the inference overhead? While SAC saves on AE training, what is the specific extra compute (FLOPs) or latency cost of enabling bidirectional attention for anchor tokens (either over the full sequence or within chunks) during inference, compared to standard KV caching or older compression methods?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a context compression method that does not relying on autoencoding training. The proposed method, SAC, selects anchor tokens directly from the original tokens and aggregates contextual information into their key-value tokens using bidrectional attention. Experiments show that SAC outperforms existing context compression methods.

### Strengths
1. Context compression is an important problem worthy of investigation. 
2. The proposed method is simple. It combines token selection, common in KV cache compression but probably not in context compression, and bidirectional attention. 
3. The proposed method outperfoms existing context compression methods.

### Weaknesses
1. The method is not clearly described and sometimes very confusing. 
2. The contexts in the experiments are in general short. It is of more interest to compress long contexts. 
3. The experiments use a single model, Llama-3.2-1B. It is unclear whether the proposed method generalizes well to larger models and other model families. 
4. The paper has an observation that challenges previous work, but does not provide any explanation, so it is hard to see if is a bug or something new.

### Questions
1. Can you give a clear description of the method? For example, what are the anchor embeddings? How does anchor embeddings enable the compressor to identify critical tokens? The anchor tokens are just sampled evenly, so why are they more important? Why do you choose the output of anchor tokens from the LLM's final layers or the KV pairs from each layer? Any justifications? 
2. Can you demonstrate the effectiveness and performance improvement of the proposed method for long contexts? 
3. Can you provide evidence for the generalizability of the proposed model to larger models and other model families? 
4. What do you think might have caused the discrepancy in the observed AE effect?

### Soundness
2

### Presentation
2

### Contribution
2
