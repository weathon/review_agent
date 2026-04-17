# Sparse Attention Adaptation for Long Reasoning

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6, 2

## Abstract
We introduce SeerAttention-R, a sparse attention framework specifically tailored for the long decoding of reasoning models.
Extended from SeerAttention, SeerAttention-R retains the design of learning attention sparsity through a self-distilled gating mechanism, while removing query pooling to accommodate auto-regressive decoding. 
With a lightweight plug-in gating, SeerAttention-R is flexible and can be easily integrated into existing pretrained model without modifying the original parameters. 
We demonstrate that SeerAttention-R, trained on just 0.4B tokens, maintains near-lossless reasoning accuracy with 4K token budget in AIME benchmark under large sparse attention block sizes (64/128).
Using TileLang, we develop a highly optimized sparse decoding kernel that achieves near-theoretical speedups of up to 9x over FlashAttention-3 on H100 GPU at 90\% sparsity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a sparse attention framework designed to accelerate auto-regressive decoding in large language models. The method builds upon SeerAttention, which learns block-level attention sparsity through a lightweight gating network trained via self-distillation. Unlike the original SeerAttention, the proposed approach removes query pooling to make the mechanism compatible with incremental token-by-token decoding. The model further introduces group-wise sparsity sharing aligned with GQA, enabling more efficient decoding with reduced computational overhead. Experiments are conducted to evaluate the proposed method, demonstrating its effectiveness in maintaining model accuracy while improving inference efficiency.

### Strengths
- The paper aims to improve the efficiency of auto-regressive decoding in large language models through sparse attention mechanisms. Given the growing importance of long-context inference and low-latency generation, this topic is both relevant and valuable for the ICLR community.

- The main idea is clean and easy to follow. The motivation for each design choice is well explained. The figures and descriptions effectively convey the framework’s structure and its differences from previous work.

### Weaknesses
- The proposed approach makes minor modifications to SeerAttention, primarily by removing query pooling to enable auto-regressive decoding. This adaptation is technically reasonable and practically useful, but it does not introduce novel algorithmic ideas or theoretical insights. The contribution feels more like an engineering extension of prior work rather than a fundamentally new method.

- The experimental section could be better organized. Sections 4.3 and 4.4 are clear and easy to follow, but I found Section 4.2 somewhat hard to get. Maybe I missed something, but it is not entirely clear what this part aims to show, and I am also confused about how the “oracle block sparse selection” is defined and implemented. Could the authors provide more explanation or additional details here? My feeling is that this section aims to show that attention in reasoning models is inherently sparse during decoding. Is that correct? If so, this experiment might be better suited as an appendix study or as an initial motivation to support the use of sparse attention in reasoning.

### Questions
See the weakness above.

### Soundness
3

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
3

### Summary
This paper proposes SeerAttention-R, a sparse attention framework designed for long-sequence decoding in reasoning models. As an extension of SeerAttention, SeerAttention-R retains the core design of learning attention sparsity through a self-distilled gating mechanism while removing query pooling to accommodate auto-regressive decoding. Experiments show that SeerAttention-R, trained on 0.4B tokens, maintains comparable reasoning accuracy on AIME benchmarks, even with large sparse attention block sizes. The optimized sparse decoding kernel implemented using Triton achieves 9× speedup over FlashAttention.

### Strengths
1. The paper is well-structured and clearly demonstrates the differences between SeerAttention and SeerAttention-R. Figure 1 contrasts the two methods, allowing to better understand the key insights (removing query pooling, adding shared sparsity design for GQA, etc).

2. The paper provides comprehensive evaluation across multiple models (Qwen3-4B 8B and 14B, DeepSeek-R1-Distill-Qwen-14B) and benchmarks (AIME24/25, MATH-500, GPQA-Diamond).

3. The high training efficiency is interesting. SeerAttention-R requires only 0.4B tokens for lightweight distillation, demonstrating its practical applicability.

4. The paper provides Triton kernel implementation, which is valuable for practical deployment.

### Weaknesses
1. The entire paper suffers from citation formatting. Academic papers should properly distinguish between \citep and \citet. The author requires to carefully review every citation in the paper and apply the correct format consistently. 

2. The evaluation is restricted to mathematical reasoning tasks (AIME, MATH, GPQA) with training on 0.4B tokens. Generalization to other important domains remains unvalidated, particularly long-context probe tasks and scientific reasoning benchmarks, which are crucial for assessing whether the learned sparsity patterns transfer beyond mathematical reasoning.

3. The paper proposes using a composition of Max, Min, and Average pooling for K compression (Equation 1b) but provides no ablation study comparing this design against alternatives. It would be valuable to see experimental results comparing: (a) using only Max pooling, (b) using only Average pooling, (c) Max+Avg combination, and (d) the full Max+Min+Avg composition. Understanding which pooling components contribute most to performance would strengthen the technical contribution.

4. The experimental comparison is primarily limited to Quest, lacking comparisons with other recent sparse attention methods including NSA, MoBA, and alternative efficient attention mechanisms like linear attention and DeltaNet. Including comparisons with at least a subset of these methods would better position SeerAttention-R's contributions.

Nevertheless, I really like that the authors provide a Triton kernel implementation in addition to TileLang, which significantly enhances the reproducibility and practical utility of this work - this is a major reason for my Weak Accept rating.

### Questions
See the Weaknesses described above.

One more question regarding Section "Training Setup for SeerAttention-R": The paper uses the AdamW optimizer with a learning rate of 1e-3 and cosine decay schedule. Could you clarify what weight decay value was used? Additionally, why was 1e-3 chosen as the learning rate, which seems relatively large compared to typical fine-tuning settings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
SeerAttention-R is a sparse attention framework for long decoding in reasoning models. It is a modification of SeerAttention: removes query pooling, keeps key pooling and the rest of its gating mechanism. It is lightweight mechanism that integrates into existing models (plug-in gating), needs a fraction of a billion tokens to train (learnable gate). SeerAttention-R maintains reasoning accuracy for large sparse attention block sizes (64/128) and modest reasoning token budgets (e.g. 4K for AIME benchmark). Further, to accelerate decoding under block-sparse attention a special kernel is implemented in TileLang (resulting in close to theoretical speedup). Empirical results (i) on sparse attention demonstrate that SeerAttention-R performs favorably to the approach in Quest (only 4K token to reach the full-attention accuracy) and (ii) on decoding, implementation is faster than one based on Triton.

### Strengths
- Important and timely enhancement proposals of SeerAttention for reasoning that are implemented alongside a "matching" decoding kernel using state-of-the-art libraries/languages/tools.

- Empirical results (largely "summarized" in  Figure 6 for (i) and Figure 7 for (ii)) demonstrate clean improvements across all dimensions (token budgets, block sizes, baseline LLMs and benchmark datasets).

- Pragmatic plans for integration with popular inference frameworks.

### Weaknesses
- Comparison to only one another sparse attention framework (Quest) could be considered a weak point (on the other hand it keeps the presentation take-aways crisper and easier to consume).

### Questions
Quest is a training-free approach, so it does not incur the (small) gate learning (training) overhead, however suffering an accuracy loss for shorter token budgets.
- What about the accuracy/training overhead tradeoff for approaches like those mentioned in Lines 444-446 in Section 5.1?
- How would these compare to SelfAttention-R?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SeerAttention-R, a sparse attention framework designed to enhance the efficiency of long-context autoregressive decoding in reasoning models. Building upon the original SeerAttention, SeerAttention-R employs a self-distilled gating mechanism to learn attention sparsity, eliminates query pooling to support autoregressive decoding, and offers a lightweight plug-in gating system for seamless integration into existing pretrained models without altering their original parameters. The authors demonstrate that SeerAttention-R, trained on 0.4 billion tokens, maintains near-lossless reasoning accuracy with a 4,000-token budget in the AIME benchmark under large sparse attention block sizes (64/128). Additionally, they develop a highly optimized sparse decoding kernel using TileLang, achieving near-theoretical speedups of up to 9x over FlashAttention-3 on an H100 GPU at 90% sparsity.

### Strengths
1. The self-distilled gating mechanism effectively learns attention sparsity, enhancing computational efficiency without significant accuracy loss.
2. The approach of learning attention sparsity through self-distillation is innovative, addressing the challenge of long-context autoregressive decoding in reasoning models. 
3. Empirical results show up to 9x speedup over existing methods like FlashAttention-3 on H100 GPUs at 90% sparsity, validating the proposed optimizations.

### Weaknesses
1. While the method shows promising results on the three math reasoning benchmark, its performance on other large-scale datasets or tasks remains unaddressed, raising questions about generalizability. 
2. The reported speedups are based on H100 GPUs; performance on other hardware configurations is not discussed, which may limit the applicability of the optimization. 
3. The paper lacks a direct comparison with other state-of-the-art sparse attention mechanisms, making it difficult to contextualize the proposed method's advantages. 
4. There is a lack of detailed ablation studies to isolate the contributions of individual components, such as the self-distilled gating mechanism and the sparse decoding kernel.

### Questions
1. How does the self-distilled gating mechanism compare to other attention sparsity learning methods in terms of efficiency and accuracy beyond Quest?
2. What are the limitations of the sparse decoding kernel on hardware other than the TileLang with H100 GPU tensor core, and how does it perform on different infra configurations? 
3. How does it performance on other long-context benchmarks with larger token budget?
4. Are there any trade-offs between the level of sparsity and the model's reasoning accuracy, and how can these be balanced?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes SeerAttention-R extending SeerAttention which was originally designed to improve pre-fill efficiency by sparsely activating important attention blocks with a self distilled gating mechanism at post-training time. The proposed SeerAttention-R keeps the original design choices of SeerAttention but removes sequence-level query pooling to support autoregressive decoding and adopts a shared sparsity design for Grouped Query Attention (GQA). They experimented with different reasoning models and benchmarks and showed good speedups while maintaining the accuracy, by training the gating function on 0.4B tokens.

### Strengths
1)  This paper is based on an already proposed sparse attention mechanism. Their contribution is to accommodate the original idea to make it work for autoregressive decoding by removing sequence level query pooling and adapted for recently proposed Grouped Query Attention.

2) They also developed a sparse decoding kernel  using TileLang that achieves up to 9x speedups over FlashAttention-3 on H100 GPU at 90% sparsity.

### Weaknesses
1) The results and design is based on GQA shared sparsity. How well does SeerAttention-R transfer to non-GQA like MHA models without the group-sharing trick? Can you provide comparative results on that?

2)  It’s unclear how the reported gains translate in practical inference stacks—e.g., vLLM with FlashAttention—and in settings that use speculative decoding or KV-cache quantitation. Can you quantify the benefits under these configurations?

3)  The proposed method requires additional training, unlike training-free approaches such as TidalDecode [1] and Quest. The paper also didn't a compare with TidalDecode. Can you report results on TidalDecode?

[1] Yang, Lijie, et al. "Tidaldecode: Fast and accurate llm decoding with position persistent sparse attention." arXiv preprint arXiv:2410.05076 (2024).

### Questions
Check the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
