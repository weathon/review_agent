# Partition Generative Modeling: Masked Modeling Without Masks

- Avg Score: 7.00
- Decision: Accept (Oral)
- Scores: 8, 8, 6, 6

## Abstract
Masked generative models (MGMs) can generate tokens in parallel and in any order, unlike autoregressive models (ARMs), which decode one token at a time, left-to-right. However, MGMs process the full-length sequence at every sampling step, including \mask tokens that carry no information. In contrast, ARMs process only the previously generated tokens.
We introduce ``Partition Generative Models'' (PGMs), which replace masking with partitioning. Tokens are split into two groups that cannot attend to each other, and the model learns to predict each group conditioned on the other, eliminating mask tokens entirely. Because the groups do not interact, PGMs can process only the clean tokens during sampling, like ARMs, while retaining parallel, any-order generation, like MGMs.
On OpenWebText, PGMs achieve $5-5.5\times$ higher throughput than MDLM while producing samples with lower Generative Perplexity. On ImageNet, PGMs reach comparable FID to MaskGIT with a $7.5\times$ throughput improvement. With twice as many steps, the FID improves to 4.56 while remaining $3.9\times$ faster than MGMs. Finally, PGMs remain compatible with existing MGM samplers and distillation methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the PGM, a new framework with core architecture innovation that combines the strengths of AR and MGM. PGM partitions tokens into two disjoint groups and constrains attention such that each group predicts the other. This removes the need for explicit MASK tokens while preserving parallel decoding and arbitrary generation order and address the training inefficiency of MGM.
Experiments on LM1B, OpenWebText, and ImageNet show that PGMs achieve up to 5–7× faster inference throughput than MDLM and MaskGIT, with similar or better metrics including perplexity and FID. PGMs also support distillation for additional speedups.
This work is overall sound to me, but I am not an expert in architecture design.

### Strengths
1. Good novelty: replacing masking with partitioning is a simple yet powerful idea that effectively unifies the efficiency of AR models with the flexibility of MGMs.
2. Solid architectural design: The GroupSwap mechanism and partition-wise attention are well-motivated and carefully engineered to achieve the partition mechanism.
3. The experimental results are strong, with improved performance for both text and image generation.

### Weaknesses
1. While the PGM is motivated be the inefficiency of MDM training, the authors are encouraged to provide evidence to show faster learning/convergence of PGM than MDLM. This probably relates to the training stability.

### Questions
1. Does "PGM 8 / 8"  mean 8 layers of encoder and 8 layers of decoder?
2. The origin of the training instability. Do authors still observe this when PGM is trained without complementary masking.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes avoiding the repetitive computation of the `MASK` token in masked generative modeling (MGM), switching from the decoder-only MGM to an encoder-decoder architecture. The inference model is defined as:
1. self-attention **only within known indices**.
2. cross-attention swapping to unknown indices (with opposite-group masking to prevent leakage).
3. cross-attention **only within unknown indices** (projecting to the embeddings of stage one).

The complexity is still O(L^2) (L = sequence length) but with a significantly smaller coefficient (encoder $(L-k)^2$ + decoder $k(L−k)$ per step when remaining sequence length = k). 
The practical speedup is about 5x and is scalable, with comparable generation quality against MGM.
Distillation-accelerated models maintains the acceleration against MGM.

### Strengths
- The empirical benefit is strong: 5x faster than MGM (4.6x faster with nucleus sampling).

- Complementary masking is a smart and original trick to let one training step effectively count as two steps.

- Section 5.3: fair comparison against MDLM (MGM) by isolating the complementary masking trick.

- The down-stream tasks spreads across image and language, and the evaluation is solid. Distillation is also explored, which improves the practical significance of the paper.

### Weaknesses
- The fairness of Table 2's comparison is not immediately visible—I believe the fairness should outweigh matching performance. Since the paper switches from decoder-only to encoder-decoder architecture, controlling hyperparameters (width, head, depth and MLP width multipliers) seems crucial to get a fair comparison.
In LM1B, it is a good idea controlling parameter counts and comparing with PGM(6/6)\~170M, but in OWT, that model is missing in the main text (only the dim. 1024 model is shown). I don't understand why it only appears in the appendix.

- I don't understand the labels (5.3) (5.4) (5.5) in Figure 4 (right).

- Minor: "sparse attention" is used to describe the masking mechanism, but I believe it is an overuse of the term, as the mask is not actually sparse—perhaps group-wise attention is more suitable.

### Questions
- Except for the top-k/nucleus confident tokens, the computations are wasted. I wonder if it is possible to reuse these noisy states instead of re-initializing decoder queries at each denoising step?

- The current decoder architecture is cross-attention-only—which makes it easy to control parameter count c.f. MDLM, but lacks the standard self-attention component. Have you thought about this variant?

- The information exchange from known to unknown indices entirely relies on the swap xattention layer. I wonder if it is possible to do the exchange in each decoder layer instead? (Of course this will make the complimentary masking trick not possible.)

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a new generative modeling framework for language modeling, termed Partition Generative Models (PGM), aimed at improving inference efficiency. Unlike Masked Generative Models (MGM), PGM avoids applying the forward process to masked tokens, thereby reducing computational cost. The authors present tailored architectural modifications, along with corresponding training and inference strategies, to enable efficient generation within this framework. Experimental results indicate that PGM achieves faster inference than existing MGM approaches while maintaining comparable generation quality.

### Strengths
1. The core idea of avoiding computation on masked tokens during inference, along with the corresponding training strategy, is interesting and effectively targets a key inefficiency in existing masked generative models.

2. The empirical results demonstrate that PGM can significantly accelerate inference while maintaining generation quality comparable to other state-of-the-art generative models, supporting the practical value of the proposed approach.

3. The paper is clearly written, well-structured, and easy to follow, making the technical contributions accessible to the reader.

### Weaknesses
I did not identify any major weaknesses in this paper. I do, however, have one question for clarification:

The proposed training pipeline includes two prediction components that operate on the same batch of data, which suggests that training efficiency could potentially be better than MDLM. Could the authors provide quantitative results or analysis regarding training efficiency, such as training speed, computational cost, or resource usage compared to MDLM?

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors introduce Partition Generative Models (PGMs), based on the observation that masked generative models (MGMs) waste compute on masked tokens, which contain no information. 
Instead of masking tokens, PGMs partition the input tokens into two disjoint groups and train the model to predict one group from the other.
This approach allows the model to process only unmasked tokens which eliminating the need for explicit masking and leads to significantly faster sampling.

### Strengths
- The GroupSwap layer and partition-aware transformer structure are well-motivated
- Includes analyses of perplexity, latency, throughput, and ablations on masking vs. partitioning.
- Strong empirical results across both text and image generation tasks, PGMs deliver substantial inference speedups (up to 7×) with little to no degradation in output quality.

### Weaknesses
- The architectural details (e.g., data-dependent vs. data-independent queries) are dense and could be clarified or simplified, the paper is a bit difficult to follow.
- The largest experiments are modest in size (268M parameters). It remains unclear if PGMs scale favorably compared to state-of-the-art large AR or diffusion model
- No comparison against recent SOTA model non-autoregressive language models beyond MDLM.

### Questions
- How does the choice of partition ratio (t) affect convergence and quality? Is it dynamically sampled or fixed?
- why cant you use KVcache that would reduce the time complexity from sampling in MGM? Would PGM will be still faster?

### Soundness
3

### Presentation
3

### Contribution
3
