# On quantizing the state of the Muon optimizer

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
The Muon optimizer, based on matrix orthogonalization, has recently shown faster convergence and up to 2× computational efficiency over AdamW in LLM pretraining. Like AdamW, Muon is stateful, requiring storage of both model weights and accumulated gradients. While 8-bit AdamW variants mitigate this overhead using blockwise quantization, they are typically stable only under dynamic quantization - which improves stability on linear quantization for extreme values. In this paper, we introduce the 8-bit Muon optimizer using blockwise quantization, supporting both linear and dynamic schemes. We demonstrate that 8-bit Muon maintains stability under both, while delivering $\sim$74\% reduction in memory footprint compared to full-precision Muon. In extensive experiments, 8-bit Muon with linear quantization outperforms AdamW and 8-bit AdamW in pre-training a 1.6B model on 4B FineWeb tokens, while achieving parity with Muon for Chinchilla-optimal training with 32B tokens for both validation loss and downstream benchmark tasks. It also shows competitive results when fine-tuning the Llama 3.2 3B model on post-training data. We also provide a theoretical perspective to help explain this robustness under quantization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work studies why linear quantization is unstable for Adam but works well for SGD and Muon from a theoretical perspective. The authors evalute 8-bit Muon with both linear and dynamic quantization on pretraining (1.6B model on 4B tokens) and finetuning task.

### Strengths
The writing is clear.

### Weaknesses
- The main weakness is that the core part of this paper focuses on why low-bit Adam becomes unstable under linear quantization, whereas SGD and Muon do not (Section 4), which is less relevant to the title. Furthermore, the authors do not propose any new technique for 8-bit Muon. It appears that 8-bit Muon faces no particular challenges, and existing techniques already work well.
- The experimental setup is limited. Although the 1.6 B scale is reasonable for an academic paper, the tokens-per-parameter ratio (4 / 1.6 = 2.5) is too small.
- The assumption in Theorem 1, gradient coordinates of moderate size occur with non-negligible probability, is not justified.
- Theorems 1, 2, and 3 lack empirical justification. It would be better to measure the quantization error under linear quantization and compare it with these theoretical bounds.

### Questions
See weakness above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates applying quantization to the Muon optimizer state, in a manner similar to the prior work quantizing AdamW state by Dettmers et al.  The paper argues that, in AdamW, the key source of error for simple linear quantization schemes is via error in the second-order velocity term.  Since Muon (like SGD+M) does not use such a term, it trains fine with simple linear quantization.  But it also trains fine with dynamic quantization.  Either way it saves a lot of space compared to vanilla Muon.

### Strengths
Paper was fairly clear, easy to read, well-organized.

The key idea, that Muon is easier to quantize because it doesn't share AdamW's velocity term, is interesting.

### Weaknesses
Overall:
- The main overall idea, to quantize the Muon optimizer states, was a fairly obvious thing to try, although someone has to do it and test it.  So the main value of a paper trying something obvious like that is to do a really thorough empirical study/investigation.
- I didn't find the experimental work that comprehensive or perfectly done.  I felt there was more empirical work that could have been done to firmly establish the main premise (that velocity is the culprit for AdamW not doing well with linear).  I mean, creative ways to assess this beyond what's in the paper now (and beyond what I can think of).  I felt the key theoretical aspects of this could have been in the main paper rather than the appendix.
- So I feel it's kind of a borderline paper.

In more detail:

“Several large-scale studies have confirmed Muon’s ability to achieve a 2x efficiency… This includes extremely large models up to a trillion parameters in size” 
- Really? Did they confirm at 1T size that Muon has 2x efficiency over AdamW (like, actually ablating the optimizer)?  Maybe this is a bit misleading.  Especially given my next point.

This recent paper argues: “the speedup of matrix-based optimizers is inversely proportional to model scale, decreasing from 1.4× over AdamW for 0.1B parameter models to merely 1.1× for 1.2B parameter models.”
- Wen - Fantastic PT optimizers and where to find them - 2509.02046v2
- I'm just saying, let's acknowledge that Muon > AdamW isn't established at scale.

It seems like the fact that the error in Theorem 1 arising primarily from error in the second moment is a key part of the paper, yet it’s in the appendix!

- Since there is quite different behaviour of optimizers between language and image tasks (e.g., https://arxiv.org/abs/2304.13960), I don’t understand why we don’t confirm the lack of divergence for SGD+M on language tasks?  Like, it seems like an unnecessary leap to say, “Since SGD works well with linear quantization,” with the implication this holds in LLMs.

I have some issues with the experimental setup:
- ROPE is known to have some precision issues, I wonder how that affects things versus using AliBi
- Only the smallest model is trained to compute-optimal (20 tokens-per-parameter), so it's not clear that many of these settings are practically relevant.
- Using 0.9 and 0.999 Betas for AdamW doesn’t seem quite standard to me.  It seems especially important to test different beta2, because I would think beta2 would affect the second-moment error, right?
- Using the same LR for 3 different model sizes doesn’t seem optimal
- Using the same batch size across all scales also doesn’t seem quite right

- “Since most of the peak learning rates have been tuned for AdamW” – I don’t really understand what this is referring to.  Tuned by whom?

Nitpicks:
- Equation (2), last line, λ should multiply by W^(t-1), not W^(t).
- Section 4.3 could benefit from a concise statement like, “because Muon, like SGD, doesn’t use a second-order momentum term, we hypothesize it will train stably with linear quantization.”

### Questions
- What’s the real significance of the finding that Muon works well with linear quantization?  Like, we say that AdamW needs “careful blockwise + dynamic quantization,” implying that it takes more work than a simple linear scheme.  Is that so?  Are there other drawbacks of dynamic?

- What are the drawbacks of quantization in general?  It saves memory, but are there not other tradeoffs?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper shows the theoretical analysis of one step error in low precision Adam update, SGD update and Muon update. By showing the theoretical bound, the authors claim that Muon has small error similar to SGD, much smaller than Adam. The experimental results show the 8-bit Muon can perform well in pretraining setting and finetuning setting.

### Strengths
1. The paper provides error analysis on one step update for Adam, SGD  and Muon, roughly showing the error of each algorithm.

2. The paper shows experimental results that 8-bit Muon perform well under 1.6B model with 2B tokens pretrain and 3B model SFT.

### Weaknesses
1.  All of the error analysis should go through the whole training process instead of one-step analysis.

2. The error analysis of Adam should related to $\beta_1$ and $\beta_2$.

3. The NS iteration will not exact give the $O$ matrix for Muon. But the analysis is mainly based on $O$ matrix.

4. It is not enough for training 1.6B model with 2B tokens.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces 8-bit variants of the Muon optimizer using blockwise quantization, supporting both linear and dynamic quantization schemes. The authors provide theoretical analysis explaining why Muon remains stable under linear quantization (unlike AdamW): Theorem 1 shows AdamW's instability stems from quantizing the second-moment denominator, while Theorems 2-3 provide uniform error bounds for SGD and Muon under quantization. Experimentally, the paper demonstrates that 8-bit Muon achieves ~74% memory reduction versus full-precision Muon while maintaining competitive performance on pretraining (GPT-style models up to 1.6B parameters on FineWeb) and finetuning (Llama 3.2 3B on Tulu-3). The work validates that Muon's orthogonalization-based updates are inherently more robust to quantization than adaptive methods.

### Strengths
-- Originality: The paper provides novel theoretical analysis (Theorems 1-3) explaining optimizer robustness to quantization. While the quantization technique itself isn't new, identifying the structural property that makes Muon robust (absence of second-moment division) is an interesting insight. The systematic comparison between linear and dynamic quantization schemes for Muon is also original.

-- Quality: The experimental work is thorough within its scope. Multiple model sizes (97M-1.6B), proper baselines (AdamW-32, AdamW-8D, Muon-32), ablation studies, and stability analyses demonstrate careful execution. The theoretical proofs are rigorous. The ImageNet experiment elegantly isolates the core phenomenon. Training details are comprehensive for reproducibility.

-- Clarity: The paper is well-structured with clear writing. The motivation (LLM memory constraints) is compelling and immediately understandable. The theoretical sections balance rigor with accessibility. Figures effectively communicate key results (Figure 1 for memory, Figure 4 for ImageNet).

### Weaknesses
-- Limited novelty and scope: The core technique (blockwise quantization) is from prior work; this paper applies it to Muon. While the theoretical analysis is valuable, the contribution feels incremental—demonstrating that an existing technique works for a new optimizer. The focus on a single optimizer (Muon) without broader comparative analysis limits the contribution's scope.

-- Insufficient scale validation: Evaluation stops at 3B parameters, well below modern LLM scales (7B-70B+). Given that memory constraints are most acute for larger models, the limited scale raises questions about practical applicability. Do the 1-2% performance gaps compound at larger scales? Does the quantization remain stable? These critical questions are unanswered.

-- Missing comparisons: The paper doesn't compare with other memory-efficient training methods (Adafactor, GaLore, LoRA, Adam-mini) that address the same memory constraints through different mechanisms. Understanding the tradeoffs between 8-bit Muon and these alternatives is essential for practitioners but absent from the paper.

-- Performance degradation not fully addressed: The consistent 1-2% performance drop on smaller models (XS/Small/Medium in Table 3) is acknowledged but not deeply analyzed. For practitioners, especially at scale, these gaps matter. When/why do they occur? Can they be mitigated? Are they acceptable tradeoffs? Deeper analysis would strengthen the work.

### Questions
-- Scalability to larger models: Have you conducted any preliminary experiments beyond 3B parameters? What specific challenges do you anticipate at 7B, 13B, or 70B scales? Given that memory constraints are most acute for large models, this seems critical to validate.


-- Theoretical quantities in practice: Can you provide empirical measurements of the minimum singular values (from Theorem 3) and momentum norms (from Theorem 2) during actual training runs? Do these quantities remain in regimes where your bounds are meaningful?


-- Performance gap analysis: For the 1-2% performance drops on XS/Small/Medium models (Table 3), can you identify when/why these occur? Have you attempted hyperparameter adjustment or longer training to close this gap? Do the gaps persist or widen at larger scales?

### Soundness
2

### Presentation
3

### Contribution
2
