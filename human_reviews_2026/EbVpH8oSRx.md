# Substituting from the Input: Distilling Sequential Computation in Transformer Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Transformer language models process input sequences token by token, resulting in significant computation even when adjacent tokens are semantically redundant or compressible. We introduce a method for distilling sequential computation by replacing spans of input tokens with collapsed representations, computed on the fly by a shared, lightweight merge module. This module generates a single surrogate embedding from static token embeddings that captures the functional role of multiple tokens—without relying on model internals or context—allowing pre-trained models to operate on compressed inputs without architectural changes or re-training. We apply this approach during inference to compress both prompts and intermediate decoding steps, using a rollback mechanism to substitute stored multi-token KV cache entries with their single-step surrogates. Experiments with GPT-2 XL, LLaMA 3.1 8B, LLaMA 3.2 1.5B, and DeepScaleR across language modeling and downstream tasks (question answering, summarization, math reasoning) show up to 40% reduction in effective sequence length, with minimal accuracy degradation. These results highlight that sequential token computation in Transformers can be effectively approximated through condensed surrogate representations that preserves functional input behavior without model updating.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces UMIM (Universal Multi-step Input Merging), a lightweight framework designed to distill sequential computation in Transformer language models by merging contiguous input tokens into single surrogate embeddings. While previous approaches to compression—such as token pruning, context distillation, or tokenizer redesign—require access to model internals or retraining, UMIM operates purely on static input embeddings, enabling frozen LLMs to process compressed inputs without architectural modification. Through a distillation objective aligning next-token distributions between merged and unmerged inputs, the method effectively preserves generation behavior while reducing both sequence length and KV cache footprint.

### Strengths
1. Model-agnostic token compression is interesting.

### Weaknesses
1. The proposed method heavily depends on the merging set R: if the span does not exist in the merging set, it would not be compressed. I would question the generalizability of this method in comparison with other model-agnostic token compression method like Lingua2.
2. Since the compressed span depends on the merging set R, and it only contains at longest 4-gram subsequence. This limits the compression rate to be at most 4x (theoretically), yet in practice it can only reduces 10-ish% number of tokens.
3. In Table 4, why does the experiment limit the token reduction to be only 10%? Lingua2 can compress up to 5x or 6x. I believe the authors should provide a fairer base of comparison here.
4. Why doesn't the work experiment on long-context benchmarks such as LongBench or Ruler like lingua2?

### Questions
Please see Weaknesses.

### Soundness
2

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
3

### Summary
This paper focuses on the problem that Transformer language models perform redundant sequential computation by processing every token individually, even when adjacent tokens carry overlapping meaning. They propose Universal Multi-step Input Merging (UMIM), a lightweight, model-agnostic module that replaces spans of tokens with single surrogate embeddings derived from static token embeddings, allowing pre-trained models to operate on compressed inputs without retraining. Experiments are conducted on WikiText-103, BookCorpus, OpenWebText, CNN/DailyMail, and reasoning benchmarks such as AIME 2024 and AMC 2023, using models including LLaMA 3.1 8B, LLaMA 3.2 1.5B, GPT-2 XL, and DeepScaleR-1.5B. On QA tasks, the method is compared with Select Context and LLMLingua 2, while on CNN/DailyMail summarization, it is compared with LLMLingua 1/2, Select Context, H2O, and StreamingLLM. The results show that UMIM achieves up to 40% sequence-length and KV-cache reduction with minimal performance degradation.

### Strengths
1. The proposed method is novel as a lightweight, model-agnostic module for LLMs, though token merging is not novel in itself.
2. The merge rule, architecture, and distillation objective are clearly introduced, and experiments are conducted on multiple LLMs.
3. The detailed experimental settings are well-included.

### Weaknesses
1. The method does not truly treat language models as black boxes. The merge module must have access to the static token embeddings to compute a surrogate embedding for a merged span. Moreover, during training, the module requires access to the model’s predicted next-token distribution (the probability over the vocabulary) so that it can align the predictions between the merged and unmerged inputs through a distillation loss, such as minimizing KL divergence. Therefore, although the approach does not modify or rely on the internal transformer layers of the base model, it is not entirely black-box: it depends on access to both the embedding layer and the model’s output probabilities (logits or softmax).
2. The experimental scope appears limited: it covers selected pretrained models and a few downstream tasks, but doesn’t thoroughly explore extreme variations (e.g., very long contexts, highly domain‐specific text, alternate tokenizers), which might challenge stability.
3. Some related works on token-merging should be compared and discussed, for example [1-4].

[1] Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., & Hoffman, J. (2022). Token Merging: Your ViT but faster.
[2] Cao, Q., Paranjape, B., & Hajishirzi, H. (2023). PuMer: Pruning and Merging Tokens for Efficient Vision-Language Models.
[3] Kallini, J., Murty, S., Manning, C. D., Potts, C., & Csordás, R. (2025). MrT5: Dynamic Token Merging for Efficient Byte-level Language Models.
[4] Saad, M., Li, H., Sharma, T., & Hassan, A. E. (2025). On the Effect of Token Merging on Pre-trained Models for Code. arXiv.

### Questions
- Could you please justify why each baseline was selected for each dataset?

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
This paper proposes Universal Multi-step Input Merging (UMIM), a lightweight external module that compresses multiple input tokens into a single surrogate embedding, with the goal of reducing sequential computation in pretrained Transformers. The module is trained via distillation, and it learns to produce merged embeddings whose predictive distributions match those of the original multi-token sequences, without modifying or fine-tuning the underlying LM. At inference time, the method replaces n-grams in prompts or intermediate decoding steps with their merged embeddings, also reducing the size of the KV cache. Experiments on several LMs show 30–40% effective sequence length reduction with minor performance loss on LM and QA tasks.

### Strengths
1. The idea of compressing sequential computation post-training, without touching model internals, is interesting and practically useful. Most prior compression or merging techniques either modifies tokenization or re-train (fine tune) the model. The merge module is small and simple, making it appealing for efficiency improvements.
2. Empirical study covers multiple backbones and task types, consistent results, and clear reporting of token reduction and perplexity trade-offs.

### Weaknesses
1. Novalty is moderate. This work shares many similarities with zip2zip (2025), both using merging tokens to improve efficiency. Yet this discussion and comparison is missing in this paper. From what I can tell, the main difference is that this current method is not adaptive to a specific domain when merging tokens, requires a pre-selected vocabulary, which might affect its compression efficiency; the advantage of this work is that during training the backbone model is frozen and only a lightweight merge module is trained. It would be better if we have more comparison with zip2zip in expirical study. And perhaps with fine-tuning, the proposed method might gain some accuracy?
2. The merging rule, based on fixed n-gram frequency on a given corpus, feels somewhat heuristic. There’s no adaptive control over which spans should merge at runtime, so compression–accuracy trade-offs are fixed.
3. Another limitation is that the merging mechanism is mainly for the input sequence, and the output is not compressed directly. And the rollback design will lead to extra computation.  
4. Token efficiency does not automatically mean faster inference. The reported Latency and Throughput in Table 12 shows moderate wall-clock time speed-up. The authors acknowledge this and attribute it to lack of fused GPU kernels. Hence, a rigorous empirical efficiency evidence is still missing.

### Questions
1. Can you comment on how UMIM interacts with positional encodings, since merged spans effectively skip several token positions?
2. The merge rule relies on frequent n-grams extracted from WikiText-103. How sensitive is UMIM’s performance to the corpus used for building the merge set? For example, would it degrade if trained on WikiText but applied to code data? How much can we gain if we train it on the corpus of the same domain?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
“Substituting from the Input” introduces UMIM, a plug-in merge module that distills multi-token spans into single surrogate embeddings at input level only (no model surgery). A lightweight 1-layer attention network is distillation-trained on frequent n-grams (2-4) so that the frozen LM’s next-token distribution is preserved. At inference the module can compress prompts and roll-back KV-cache entries on-the-fly, cutting effective sequence length up to 40 % with < 2 % accuracy drop on GPT-2-XL, Llama-3.1-8B, Llama-3.2-1.5B and DeepScaleR across language modelling, QA, summarisation and AIME/AMC math reasoning. The method is model-agnostic, training-free for the LM, and orthogonal to other efficiency techniques.

### Strengths
1. UMIM is a plug-and-play module that operates only on input embeddings, requiring no access to model internals, no re-training, and no architectural changes to the base LLM. This makes it highly practical for deployment across a wide range of pre-trained models (e.g., GPT-2XL, LLaMA 3.1/3.2, DeepScaleR), and avoids the costly retraining or fine-tuning required by prior tokenization or compression methods.
2.The paper demonstrates strong empirical results. This up to 40% reduction in effective sequence length with minimal degradation in perplexity, downstream task accuracy (QA, summarization), and even math reasoning (AIME/AMC). The distillation-based training ensures that the surrogate embeddings preserve the functional semantics of the original token spans, as validated by high alignment metrics (e.g., 79–92% Top-1 accuracy in distribution matching).
3. UMIM supports on-the-fly compression during autoregressive decoding using a rollback mechanism that replaces multi-token KV-cache entries with a single surrogate embedding. This reduces memory usage and attention cost quadratically over generation length, offering cumulative efficiency gains—a significant advantage over static compression methods that only operate on the prompt.

### Weaknesses
1. UMIM relies on a fixed, frequency-based merge set (n-grams with frequency ≥ 5) extracted from a general corpus. This static rule set cannot adapt to domain-specific or rare token combinations at inference time. As a result, long-tail or context-sensitive phrases are unlikely to be merged, limiting compression effectiveness in specialized or low-resource domains.
2. Although the base model is frozen, UMIM still requires task-agnostic distillation training on a large corpus (e.g., WikiText-103), which introduces non-trivial compute and storage overhead. The merge module must be retrained for each tokenizer and embedding dimension, and the paper does not explore scalability beyond 8B parameters. This limits true plug-and-play adoption, especially for larger models or multilingual settings.
3. It always merges high-frequency n-grams regardless of context or semantic importance. This can lead to over-merging (e.g., merging idioms or named entities that should remain separate) or under-merging (e.g., missing semantically redundant phrases). There is no mechanism to control compression ratio or fidelity at runtime, which could be problematic in safety-critical or high-precision applications.

### Questions
1. How does UMIM handle tokenization mismatches across models or vocabularies? Since UMIM is trained on n-grams from a specific tokenizer (e.g., LLaMA’s BPE), how would it generalize to a different tokenizer (e.g., SentencePiece or a custom one)? Would the merge module need to be retrained from scratch, or can it be transferred or aligned across tokenization schemes?
2. What is the impact of merge errors on long-context coherence or reasoning chains? In long-context tasks like math reasoning or document summarization, minor semantic drift can cascade into large errors. Have you evaluated per-span fidelity or error propagation across long generations? Are there failure cases where merging leads to logically inconsistent or factually incorrect outputs?
3. Can UMIM be extended to adaptive or learned merge policies? Instead of relying on static n-gram frequency, have you considered learning merge decisions conditioned on context (e.g., using a small policy network or reinforcement learning)? This could allow dynamic compression trade-offs based on task, domain, or user-defined constraints—how feasible is this extension?

### Soundness
2

### Presentation
3

### Contribution
3
