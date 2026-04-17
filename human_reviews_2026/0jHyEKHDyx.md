# Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention

- Decision: Accept (Oral)
- Scores: 8, 4, 6, 8

## Abstract
The pursuit of computational efficiency has driven the adoption of low-precision formats for training transformer models. However, this progress is often hindered by notorious training instabilities. This paper provides the first mechanistic explanation for a long-standing and unresolved failure case where training with flash attention in low-precision settings leads to catastrophic loss explosion. Our in-depth analysis reveals that the failure is not a random artifact but caused by two intertwined phenomena: the emergence of similar low-rank representations within the attention mechanism and the compounding effect of biased rounding errors inherent in low-precision arithmetic. We demonstrate how these factors create a vicious cycle of error accumulation that corrupts weight updates, ultimately derailing the training dynamics. To validate our findings, we introduce a minimal modification to the flash attention that mitigates the bias in rounding errors. This simple change stabilizes the training process, confirming our analysis and offering a practical solution to this persistent problem. Code is available at https://github.com/ucker/why-low-precision-training-fails.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates a long-standing training instability in transformer models when using flash attention with BF16 precision, which manifests as catastrophic loss explosions during training. It provides a mechanistic explanation for this failure as (1) the emergence of structurally similar low-rank representations across different tokens and training steps in the attention mechanism's intermediate computations, and (2) biased rounding errors inherent in BF16 arithmetic operations. Through systematic experimental analysis on a reproducible GPT-2 failure case, they trace the error propagation, demonstrating how biased gradient updates accumulate in a specific low-rank direction rather than canceling out. The bias originates from a specific numerical condition which is detailed. To validate their analysis, the authors make a small modification to the flash attention's softmax computation, showing this stabilizes training while remaining mathematically equivalent.

### Strengths
This is an excellent paper.

**Clarity and Exposition:** Remarkably clear writing, particularly given the intricate technical details included. The notation is well-defined, figures are informative and well-integrated, and mathematical derivations provide appropriate detail. It reads very smoothly.

**Methodical Investigative Structure:** Reads like a detective story, systematically tracing the failure from symptom to root cause, via targeted experiments. The step-by-step elimination builds a clear causal chain that is highly rigorous.

**Highly Convincing Analysis:** The multi-layered evidence is persuasive. The successful flash attention code modification based on insights gained supports the preceding theoretical analysis with a long-term solution for the community.

### Weaknesses
Nothing major to speak of.

The focus of the paper is (by design) very narrow.  

Detail is lacking at a couple of points. But these ommissions are minor, and given the extensive detail included, I assume this was done to meet the page limit.

### Questions
The analysis focuses on BF16 arithmetic. The deep learning community is increasingly interested in even lower precision formats like FP8 or INT8 for training. Are you aware of other outstanding (BF16, FP8, INT8) issues that would benefit from this type of analysis? If so, consider listing them in an Appendix to promote their prompt solution by early-career researchers.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper explains why training Transformers with Flash Attention in low-precision (BF16) settings often leads to loss explosions. The authors show that the problem comes from an interaction between low-rank representations and biased rounding errors in BF16 arithmetic. They locate the issue in the backward pass and propose a simple fix to prevent the rounding bias. The proposed method stabilizes training and confirms their analysis, offering both practical and theoretical values for improving numerical stability.

### Strengths
1. This paper provides a mechanistic explanation for the long-standing issue of low-precision training failures in Flash Attention.  
2. The authors identify the root cause as a vicious cycle between emergent low-rank representations and biased BF16 rounding errors.  
3. The paper proposes a fix through a dynamic softmax adjustment that stabilizes training.

### Weaknesses
1. The analysis and solution are validated only on GPT-2, leaving their effectiveness on larger models unverified. For example, if the failure is located at layer 2, head 8, how can we identify the corresponding head in larger models? 
2. I wonder whether the analysis can be generalized to other numerical precisions beyond BF16 and FP32. 
3. The dynamic maximum adjustment introduces additional operations into Flash Attention's critical path. Whether do these operations incur extra computational overhead? What is the additional computation complexity?   
4. The proposed fix is verified only on a single model. Can this strategy be applied to other scenarios as well?

### Questions
1. It would be more readable to include the definition of Flash Attention (Line 98) in a self-contained manner (e.g. in appendices). 
2. The notations, such as "$d\mathbf{Q}, d\mathbf{K}, d\mathbf{V}, d\mathbf{O}$", are somewhat confusing and require additional clarifications to help distinguish them from derivatives. 
3. The derivation presented in Line 246 appears a bit abrupt; providing more intermediate steps would help improve the logical flow. 
4. Claim 3 in Line 424 contains a typo ("significant" is misspelled as "significand").

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
This paper proposes both an explanation and a solution to the problem of training instability for models using bf16 flashattention. They isolate the problem to a biased rounding error in the computation of delta, which produces correlated (low-rank) updates to the weight matrices, eventually leading to divergence. The paper proposes a modification to the flashattention algorithm to resolve this issue and demonstrates empirically that this modification stabilizes training.

### Strengths
* The paper proposes a plausible explanation for the observed training instability.
* The paper proposes a sensible fix to this problem and demonstrates that it works.
* To the best of my knowledge, the paper's ideas and contributions are original.
* The ideas are mostly clearly explained in a step-by-step manner that allows the reader to follow along.
* The problem is very significant, as low-precision flash-attention is very popular. For example, I have personally encountered issues related to training stability, as have other researchers I know.

Overall, I quite like this paper.

### Weaknesses
I would separate the weaknesses into two categories.

Weaknesses in the explanation for bf16 attention instability:
* The identified cause is a small but persistent bias. However, the observed effect is a sharp, near-instantaneous spike. The paper does not explain why the sharp spike occurs or why the persistent bias causes it. This is the most significant weakness, in my opinion.
* Minor: I believe remark 1 is slightly imprecise. For example, if $\bar{\mathbf{P}}[T,t] = 1/2$, its product with $\bar{\mathbf{P}}[t,i]$ would have last 16 bits 0. However, the intended meaning of the remark, that with high probability the product has non-zero least 16 bits, is true.


Weaknesses in experimental results:
* The findings isolating the source of training instability could be validated on other models, sizes, and/or training scripts.
* The evidence that the stabilized flash attention is actually stable could be more robust. The authors could test more model sizes, train for longer, etc.

### Questions
Questions:
* Can you explain how existing stability fixes (e.g., QK norm) fit into your explanation? I think this is related to my question regarding how the persistent small bias leads to the large spike in loss.
* Related to my other questions, shouldn't we expect gradient dynamics to correct the small rounding drift if the drift leads the model to higher loss regimes?
* Is there a reason you set m' = 0 in the case that r_m < 0? why not set it to gamma*r_m, where 1>gamma>0?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work provides the first mechanistic explanation for why transformer training with FlashAttention in low precision (bf16) leads to unstable loss spikes. Through detailed analysis and ablation studies, the authors attribute the training instabilities to two related effects: 1. the emergence of similar low-rank matrix updates in the model's internal representations across tokens and steps, and 2. biased rounding errors in bf16 arithmetic when multiple attention scores achieve the exact same maximum. Together, these create a biased gradient direction that inflates spectral norms and thus destabilizes training. Based on their mechanistic analysis, the authors propose a minimal “dynamic softmax” patch to the FlashAttention algorithm and show that it restores stable training.

### Strengths
- The work investigates a very important problem, loss spikes during Transformer training due to low-precision using FlashAttention. Since attention and FlashAttention are the workhorse of modern AI systems, this problem is of great interest to the community at large.
- The work is very thorough, with systematic ablations into every step of the FlashAttention algorithm to isolate the source of the unstable training. The authors provide a thorough characterization of the failure modes and, using this insight, propose a simple fix that alleviates the loss spikes.

### Weaknesses
Overall, the analysis and ablation studies are thorough, and the proposed fix is convincing. However, one key piece of the story remains somewhat unclear: the origin of the shared low-rank structure $R$ in Section 3.3.1 (see questions). An investigation into where this structure comes from (e.g. feature collapse, attention sinks?) would make the characterization more complete.

### Questions
- In Section 3.3.1, the authors observe that the outer products $(PK)[T]^T X[T]$ exhibit strong structural similarity and approximate them with a common low-rank matrix $R$. Did the authors investigate why this shared structure emerges? For example, could it reflect commonly-occuring tokens, correlated attention patterns across tokens, or a form of mode collapse within the model's internal representations?
- In Section 3.3.2, the authors attribute the biased rounding error to rows where multiple elements of $S$ share the same maximum in bfloat16, yielding several $\bar{P}[T, t] = 1$. It would greatly bolster the proposed mechanisms to provide an experiment measuring how frequently this condition occurs during training, and showing whether its occurrence correlates with loss spikes (e.g. in Figure 8).
- Relatedly, in the discussion, the authors suggest that attention sinks may contribute to loss spikes by producing more attention probabilities of 1 (in bfloat16) and thus triggering the biased rounding error highlighted in Section 3.3.2. Did the authors conduct experiments testing this connection?

### Soundness
4

### Presentation
3

### Contribution
4
