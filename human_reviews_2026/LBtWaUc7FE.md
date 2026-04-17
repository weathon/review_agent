# ReFusion: A Diffusion Large Language Model with Parallel Autoregressive Decoding

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Autoregressive models (ARMs) are hindered by slow sequential inference. While masked diffusion models (MDMs) offer a parallel alternative, they suffer from critical drawbacks: high computational overhead from precluding Key-Value (KV) caching, and incoherent generation arising from learning dependencies over an intractable space of token combinations. To address these limitations, we introduce ReFusion, a novel masked diffusion model that integrates sequence reorganization into the causal attention framework. By elevating parallel decoding from the token level to a higher slot level, ReFusion interleaves inter-slot diffusion-based selection with intra-slot autoregressive infilling, while reordering newly generated slots ahead of the remaining masks after each iteration. Consequently, this design simultaneously unlocks full KV cache reuse and reduces learning complexity from an intractable token combination space to a manageable slot-level permutation space. Extensive experiments on seven diverse benchmarks show that ReFusion not only overwhelmingly surpasses prior MDMs with a 34\% performance gain and an over 18$\times$ speedup on average, but also bridges the performance gap to strong ARMs while maintaining a 2.33$\times$ average speedup.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose ReFusion, a hybrid generative model that integrates inter-slot parallel decoding with intra-slot autoregressive decoding. ReFusion is the first Masked Diffusion Model (MDM) to achieve full KV cache reuse for every decoded token, a significant efficiency gain, while maintaining global generation flexibility and a tractable training process. Based on their experiments, the authors assert that ReFusion dominates all MDM baselines and is competitive with autoregressive models (ARMs).

### Strengths
* The paper combines the strengths of diffusion and autoregressive generation, helping to maximize parallelism and quality.

* The idea of achieving full KV caching for every token within a parallel generation framework is significant and worthwhile.

* The paper provides an interesting analysis regarding the locality of inter-token dependency.

* The reported experimental results are strong, showing performance that Pareto-dominates existing baselines.

### Weaknesses
* The proposed method appears conceptually similar to Eso-LM [1]. At a high level, both models seem to perform diffusion and autoregressive infilling. The authors must do more to clearly and substantially differentiate ReFusion from this existing work.

* The paper lacks clarity regarding the training and architectural choices. It is not at all clear how the model achieves "full KV cache reuse of every decoded token" (lines 103-104). Diffusion LLMs typically require bidirectional attention to support flexible, any-order generation, which inherently prevents KV caching. However, Table 4 claims ReFusion supports any-order generation while using a causal mask. It is very difficult to understand how the model can simultaneously support parallel sampling and use a causal mask for KV caching. This mechanism would be made more clear with a detailed example of a training input and its corresponding attention mask.

* The illustrative example provided in Section 5.5 is very confusing and fails to clarify the generation process. It is not clear which text segments were generated via diffusion and which were generated autoregressively.

* The manuscript reads as heavily LLM influenced, but no disclosure is made. 

[1] Sahoo, Subham Sekhar, et al. "Esoteric Language Models." arXiv preprint arXiv:2506.01928 (2025).

### Questions
1. In your discussion of Eso-LM, you state: “dynamically reposition newly generated tokens ahead of masked ones at each step to facilitate caching. However, this strategy introduces an intractable learning objective at a token-level permutation space” (lines 130-132). Can you please clarify what this "intractable learning objective" is and why it is intractable? What makes ReFusion different?

2. For the experimental benchmarks, how many tokens are being generated for each task? Is it possible that ReFusion achieves better quality simply by generating more tokens than the baselines?

3. What are the measured latencies for ReFusion and the baseline models?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a training method to achieve inter-slot parallel decoding with intra-slot autoregressive decoding for dLLMs. The core idea is that conditional independence assumption is most prone to failure for nearby tokens, and thus the authors propose to perform serial decoding within a slot, in contrast to prior method, e.g., block diffusion. The method consists of two parts. For inference,  it would first decide which blocks to decode in parallel, and then reorder the slots to do parallel decoding while still have autoregressive decoding in each slot. For training, the authors propose to adopt two objectives in training, one MDM loss for masked tokens and another AR loss for clean tokens inside the slots.

### Strengths
1. The authors try to challenge one common belief in current dLLM literature, that we need to perform intra-block auto-regressive decoding  instead of intra-block parallel decoding. The analysis is interesting, and it's also very interesting to see the challenge of common belief.
2. The proposed method is soundness, both for the training and the inference side.

### Weaknesses
My primary concern lies in the unfair, insufficient, and potentially incorrect experimental evaluations, which lead to several overstated claims in the paper.

* **Missing Comparison with Block Diffusion**

One of the key hypotheses this paper wants to show is that intra-block autoregressive decoding + inter-block parallel decoding is superior to intra-block parallel decoding + inter-block autoregressive decoding. However, no experiments are provided to substantiate this claim. A direct comparison with Block Diffusion would be necessary to validate the proposed design.  Without empirical comparison, it is unclear whether the proposed method offers any real advantage over Block Diffusion in this aspect.

* **Unfair Comparison with Prior MDMs/dLLMs**

The authors claim that Refusion overwhelmingly surpasses all prior MDMs in both performance and speed. **However, this performance gain comes mostly from a better base model.** The comparisons with previous dLLMs are not fair, as the proposed model adapts from a much stronger base model, Qwen3. While I acknowledge that the proposed method achieves the best overall results and appreciate this contribution, fair evaluation requires experiments starting from the same base model as prior works. Training from scratch may be costly, but at least one experiment comparing the proposed approach with DREAM, using the same initialization checkpoints, would be essential for a fair comparison.

* **Significant Performance Drop Compared to Qwen3, the base model of Refusion**

The adapted model exhibits an 8.5% average performance drop relative to Qwen3. This is a substantial degradation. The paper claims that the model “challenges strong ARMs,” but given such a drop in accuracy alongside a 1.34× speedup, the claim seems overstated. Besides, have you conducted experiments to evaluate whether the acceleration ratio still holds when the batch size is larger than 1?

* **Confusing Results in Table 2**
 
(1) For Qwen3, training with the original objective (presumably the NTP loss) results in a 10.9% performance drop, which seems unusual and needs clarification.  
(2) For the GPQA benchmark (a 4-choice classification task), the reported performance is 0, which is implausible. Previous works using lm-eval-harness constrain predictions to 4 choices, meaning even random guessing should yield roughly 25% accuracy.  
(3) The reported TPS (tokens per second) for LLaDA is suspiciously low. While prediction accuracy may differ, TPS should mainly depend on the model structure and sequence length. The discrepancy between Table 1 and Table 2 needs more explanation.  
(4) The paper claims that the proposed model outperforms Qwen3-8B on HumanEval by ∼9 points while being 1.3× faster. However, Qwen3-8B without the authors’ training already achieves 87.8, which is actually **25.6%** higher than Refusion.

### Questions
See weaknesses

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
This paper proposes ReFusion, a masked-diffusion LLM that performs plan-and-infill decoding at a slot level: a diffusion planner selects weakly dependent fixed-length spans (“slots”), and an AR stage that fills multiple slots in parallel. By reordering into a causal sequence during infill, the method reuses exact KV cache, narrowing the cache-efficiency gap between diffusion LMs and AR models. 

Across various benchmarks and two dLLM architectures (LLaDA, Dream), ReFusion reports consistent >10× speedup over prior MDMs, and about 1.4x faster than SOTA AR models.

### Strengths
1. The slot abstraction plus causal infill provides an intuitive route to exact KV-cache reuse, reducing efficiency gap with AR decoding. 

2. The paper provides comparison with competitive baselines, and it shows consistent wins on most tasks with both LLaDA and Dream across many tasks, supporting generality. 

3. The paper probes slot thresholds and provides qualitative evidence that aligns with the design intuition.

### Weaknesses
1. ReFusion introduces extra data preparation and training cost, adding pipeline complexity to realize its gains.

2. While ReFusion is much faster than prior MDMs, its throughput versus AR models is not significant better, and ReFusion may not be orthogonal to existing tricks.

### Questions
1. The authors mention 3.7M samples; approximately how many tokens does this correspond to, and how long does training take on 16x8 H20 GPUs (total GPU-hours)?

2. How orthogonal is ReFusion to other dLLM acceleration techniques (e.g. confidence-aware parallelism)? Could the slot planner or infill step compose with the techniques, and if so, what additive speed/quality gains should we expect? 

3. Do the authors have preliminary results on how does scaling the training data beyond 3.7M samples shift the speed–quality frontier? 

4. In Table 1, why is D2F significantly worse on ARC-C than its performance on other tasks? any intuitions?

### Soundness
3

### Presentation
2

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
This paper focuses on two intrinsic drawbacks of masked diffusion models (MDMs): the infeasibility of KV caching and missing token dependencies.
To alleviate these issues, the authors propose to combine the AR and MDM paradigm, called the ReFusion Model.
Although this idea is not very novel, the authors discuss in detail the practical design of the ``plan-and-infilling'' process.
Extensive experiments demonstrate the generation speedup benefiting from the KV caching and the improved accuracy gain.

### Strengths
- The authors propose a practical ``plan-and-infilling'' process that works fairly well, although the basic idea of ReFusion might not be very novel.
- The pilot study in Section 4.1 is very interesting and clearly exhibits how the distance affects the correlation.
- The experiments, including the ablation studies, provide a very comprehensive understanding of how ReFusion works.

### Weaknesses
- Similar ideas have been discussed in previous works. E.g., BD3-LM (https://arxiv.org/pdf/2503.09573) utilizes the block diffusion, EDLM (https://arxiv.org/abs/2410.21357) utilize AR models to model the correlations. The authors should clearly state the differences and their unique contribution.
- I find the two-step inference method in Section 4.2 somewhat obscure. I suggest the authors reorganize the desciption into mathematical equations and add more details (e.g., how to perform positional encoding for $S_t^{clean}$ and $S_t^{mask}$).
- The authors should discuss more detailedly how why the use the slot's first position to compute the certainty score. For example, the mean of the probs on all positions in a slot seems to be a more appropriate method.
- As an intrinsic drawback of the ReFusion method, there are many critical hyper-parameters to be tuned. e.g, $k$ and $\tau$ and $\lambda$. These parameters may depends on the data sets, and can affect the broader application of ReFusion.
- Why is the weight $1/t$ missing from MDM objective in equation (2)?

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
