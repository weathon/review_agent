# Self-Speculative Masked Diffusions

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
We present self-speculative masked diffusions, a new class of masked diffusion generative models for discrete data that require significantly fewer function evaluations to generate samples. Standard masked diffusion models predict factorized logits over currently masked positions. A number of masked positions are then sampled, however, the factorization approximation means that sampling too many positions in one go leads to poor sample quality. As a result, many simulation steps and therefore neural network function evaluations are required to generate high-quality data. We reduce the computational burden by generating \emph{non-factorized} predictions over masked positions. This is achieved by modifying the final transformer attention mask from non-causal to causal, enabling draft token generation and parallel validation via a novel, model-integrated speculative sampling mechanism. This results in a non-factorized predictive distribution over masked positions in a single forward pass. We apply our method to GPT2 scale text modelling and protein sequence generation, finding that we can achieve a ~2x reduction in the required number of network forward passes relative to standard masked diffusion models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The work presents an architecture combining a masked diffusion model that drafts multiple tokens in parallel, and an any-order autoregressive model which evaluates their likelihood to perform speculative sampling. This is done by adding a light causal Transformer layer on top of the non-causal layers of a masked diffusion model (along with positional encodings as in $\sigma$-GPT). This allows for faster generation than standard any-order autoregressive models, while mitigating the bias that masked diffusion models incur from sampling multiple tokens in parallel (since the independently factorized distribution they sample from is replaced by an autoregressive one). The method is shown to achieve a better NFE/performance trade-off when training from scratch on the Text8, and OpenWebText datasets, as well as when the causal layer is added (and trained) to a frozen pretrained masked diffusion model for protein sequences.

### Strengths
1. The work is well-written and flows clearly
2. The paper is clearly motivated and tackles an important problem in the bias that masked diffusion models have when sampling multiple tokens in parallel.
3. Experiments clearly demonstrate the impact of the causal layer in better matching the target distribution with fewer NFEs than a masked diffusion model (Figure 2, Table 1, Figure 4)
4. The broad applicability of the method to pretrained masked diffusion models is demonstrated empirically and convincingly (Figure 4). In my opinion this is a critical use-case of the work.

### Weaknesses
1. Figure 3 evaluates adherence to the target distribution by examining the spelling accuracy (presence of words that also occur in the training set). I think also reporting the perplexity of a larger/better fit model (as in Table 1) would help make this more convincing 
2. The baselines used appear to not use common heuristics for masked diffusion sampling, such as confidence based unmasking. Is the method compatible with such techniques. And if so, what is the impact on the NFE/performance trade-off.
3. Minor grammatical/spelling mistakes:
    - Line 233 “oowhere”

I recommend an accept due to the comprehensive and clearly investigated advantages of the approach. The most important issue I have with the work is point 2 above.

### Questions
1. How does this compare to drafting and evaluating likelihood in two separate models? 
2. How does adding more causal layers impact performance?
3. In equation 7, why is the weighting justified for the autoregressive loss term?
4. Does training with the ELBO in equation (10) offer any benefits in the performance of trained models (ignoring computational expense) over the chosen loss?

### Soundness
4

### Presentation
3

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
The paper introduces "self-speculative masked diffusions," a novel method to accelerate the sampling process of masked diffusion models (MDMs) for discrete data like text and protein sequences. The core problem addressed is the slow generation speed of standard MDMs, which stems from their factorized output distribution. To overcome this, the authors propose a hybrid transformer architecture that integrates speculative sampling directly into the model. During sampling, a single forward pass through the non-causal part generates a full draft. This draft is then efficiently verified in parallel against the target distribution from the causal part. This allows the model to accept and reveal a large number of tokens per step, significantly reducing the total NFE. The authors demonstrate the effectiveness of their method on text (OpenWebText) and protein generation (UniRef50) tasks, achieving a ~2x reduction in NFE compared to standard MDMs while maintaining or improving sample quality.

### Strengths
- The paper presents a clever and significant contribution by combining the any-order generation flexibility of MDMs with the speed benefits of speculative sampling. Addressing the sampling efficiency of MDMs is a critical research problem, and the proposed solution is both novel and effective.
- The hybrid non-causal/causal architecture is elegant. Using the non-causal component as a drafter and the causal component as a verifier within a single, end-to-end trainable model is a clean design. The inclusion of a residual connection from the non-causal to the causal output ensures the target distribution is a strict improvement over the draft, which likely boosts the acceptance rate.
- The authors validate their method on multiple diverse and relevant tasks (small-scale text, large-scale text, and protein modeling). The results are consistently strong, showing a clear improvement in the sample quality vs. efficiency trade-off. The authors also ensure a comprehensive evaluation by measuring both GPT2 NLL and Entropy.

### Weaknesses
- Unclear Advantages to Fully Autoregressive Approaches. While native autoregressive models are memory-bound, they achieve better performance than MDMs and can be accelerated by speculative decoding. Meanwhile, the paper is strongly inspired by methods like Medusa, and the paper's 2x speedup is also similar to its speedup to LLMs. It is unclear whether true advantages exist by adapting speculative decoding to MDMs.
- Limited Discussion on Wall-Clock Time and Computational Overhead. The primary metric for efficiency is the Number of Function Evaluations (NFE). While NFE is a good hardware-agnostic proxy for speed, it doesn't capture the full picture. The proposed hybrid model is slightly larger and computationally more complex per forward pass than a baseline MDM of the same depth. A discussion on the actual wall-clock speed-up and the overhead per forward pass would strengthen the paper's practical claims.
- Sensitivity to New Hyperparameters. The method introduces several new hyperparameters, including the split of non-causal vs. causal layers (e.g., 11-1), the number of verification steps per draft, and the parameters of the cosine windowing schedule. While the authors present a working configuration, a more detailed analysis of the model's sensitivity to these choices would be beneficial for future practitioners wanting to apply this method.

### Questions
- Could you provide insight into the wall-clock time for a single forward pass of your hybrid model compared to the baseline MDM and autoregressive model?
- What is the practical advantage compared to autoregressive+Medusa, in aspects of implementation difficulty, generation speed and performance?
- How does the acceptance rate of speculative tokens evolve during a typical generation process?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Self-Speculative Masked Diffusions (SSMD), a masked-diffusion generator for discrete data that aims to reduce sampling cost. The key idea is a hybrid transformer: most layers are non-causal to produce a factorized draft, followed by a small causal head that verifies and (via a specialized speculative step) yields a non-factorized distribution over the masked positions in a single forward pass. A theoretical treatment derives a likelihood decomposition for the induced sampler and an ELBO. Empirically, the method reports roughly 2× fewer network forward evaluations (NFE) than a standard MDM at matched sample quality on text (Text8, OpenWebText) and protein sequence generation (UniRef50 + ESMFold pLDDT).

### Strengths
1. **Principled modeling & architecture.** The hybrid non-causal/causal stack with σ-aware causal masking is clearly specified and lets the causal head “strictly improve” the draft through a residual connection, aligning draft/target and plausibly raising acceptance rates. The training objective jointly optimizes both heads in a single pass.  
2. **Sound theoretical analysis.** The paper characterizes the sampler’s distribution under changing targets and derives a tractable decomposition for likelihood and an ELBO; this is rare for speculative schemes beyond left-to-right AR. 
3. **Clear empirical win in NFE.** On OpenWebText, the method matches or improves GPT-2 NLL at **~½ the NFE** relative to a standard MDM baseline, with similar token entropy; ablations support the importance of the residual and “11 non-causal + 1 causal” design. 
4. **Protein application with minimal intrusion.** The protein experiment keeps a frozen ESM2-based backbone and trains just the causal head, showcasing plug-in potential.

### Weaknesses
1. **Reported “2×” is an NFE metric, not end-to-end latency.** The paper motivates gains by memory-bound inference and single-pass verification, but practical wall-clock speed depends on kernel fusion, batch size, cache behavior, acceptance rates, and framework overheads. The text emphasizes that longer sequences do not necessarily mean worse latency because the cost is memory-bound, yet the experiments still operationalize efficiency largely as NFE. It’s unclear how much of the 2× NFE translates into **real time** speedups on GPUs for moderate batch sizes typical in protein generation.  
2. **Protein benchmark strength and comparability.** The protein study uses **average pLDDT from ESMFold** on sequences sampled from UniRef50. While pLDDT is a useful proxy, the field has converged on stronger metric foldability. Compared to "Path Planning for Masked Diffusion Model Sampling", which reports much stronger protein benchmarks with higher-quality samples and broader evaluation, the current results look considerably weaker. This makes it difficult to assess the real progress or competitiveness of this approach in protein generation.

### Questions
see weakness.

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
4

### Summary
This paper proposes self-speculative masked diffusion,  a new class of masked diffusion generative models for discrete data that require significantly fewer function evaluations to generate samples. In self-speculative masked diffusion, the neural network is decomposed into a large bidirectional diffusion model and a small any-order causal autoregressive decoding head. The large bidirectional model produces candidates, while the small decoding head serves as a verifier. Through self-speculative approaches, the authors are able to get a 2x sampling speed boost with barely any quality decrease.

### Strengths
* This paper is practically significant. I think overall this technique is very promising for real-world deployment. 
* The empirical results are impressive. With barely any loss of generation quality, self-speculative masked diffusions achieve a 2x acceleration, making this almost a free lunch.
* I think overall the paper is well-written, with a clear motivation, a solid method, real innovations, and abundant theoretical analysis.

### Weaknesses
* The sampling algorithm is not explained thoroughly enough, especially regarding the attention patterns. Please see Question 3 for more details.
* I think Section 3.4 is less accessible to readers. If the meaning of this proposition can be explained more intuitively, readers will better understand it.

### Questions
* In Figure 1, do the dots connecting input tensors mean addition? I am assuming this because the token embeddings and positional encodings are connected by dots, but I think this should be more explicitly indicated.
* For the input to the causal parts, are the hidden states from the non-causal model necessary? I think from a theoretical perspective, only token embeddings and positional embeddings are enough. Can the authors explain, theoretically or empirically, why these hidden states are needed, especially given the residual connections to the output of the causal part?
* My biggest question is, during sampling, how do the authors handle the causal attention? If it still remains a causal structure, if I understand correctly, then in the case of $k\geq 2$, i.e., parallel decoding, the mask tokens in the last are attending to mask tokens before them, leading to a different situation from the model's training scenario. Can the authors provide any justifications on why this does not ruin the generation quality?
* The biggest contribution of this work is to accelerate inference without loss of quality. Can the authors provide any comparison with other acceleration algorithms, such as Fast-dLLM [1]?
* The authors do not point out if any part of the model is frozen during training, so I assume no parts are actually frozen. However, in the protein experiments, the authors freeze the backbone and only trained a causal head. What will the results be if the backbone is also finetuned?

---
**References**

[1] Wu, C., Zhang, H., Xue, S., Liu, Z., Diao, S., Zhu, L., ... & Xie, E. (2025). Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618.

### Soundness
3

### Presentation
3

### Contribution
4
