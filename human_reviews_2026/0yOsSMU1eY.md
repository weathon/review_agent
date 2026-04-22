# DPad: Efficient Diffusion Language Models with Suffix Dropout

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Diffusion-based Large Language Models (dLLMs) parallelize text generation by framing decoding as a denoising process, but suffer from high computational overhead since they predict all future suffix tokens at each step while retaining only a small fraction.  We propose $\textbf{Diffusion Scratchpad} (\textbf{\textit{DPad}})$, a training-free method that restricts attention to a structured subset of suffix tokens, preserving fidelity while eliminating redundancy.  $\textit{DPad}$ integrates two strategies: (i) a $\textit{sliding window}$, which maintains a fixed-length suffix window, and (ii) $\textit{distance-decay dropout}$, which deterministically removes distant suffix tokens before attention computation.  This concise design is compatible with existing optimizations such as parallel decoding and prefix caching, and lends itself to a lightweight implementation.  Comprehensive evaluations across multiple benchmarks on $\texttt{LLaDA}$ and $\texttt{Dream}$ models demonstrate that $\textit{DPad}$ delivers up to $\mathbf{61.4\times}$ speedup over vanilla dLLMs while maintaining comparable accuracy, highlighting its potential for efficient and scalable long-sequence inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This article proposes a novel and efficient diffusion model inference method called DPad (Diffusion via Dynamic Programming). This method transforms the sampling optimization problem in the diffusion process into a Dynamic Programming (DP) problem, significantly reducing the number of sampling steps while maintaining the quality of model generation. The author analyzed the optimal trajectory problem of diffusion sampling mathematically and proposed an efficient approximate solution algorithm, which enables the model to achieve comparable generation performance to traditional sampling in small steps. The effectiveness of the method was validated on multiple benchmark datasets, including image generation and conditional generation tasks, and compared with mainstream fast diffusion algorithms such as DDIM and DPM Solver, demonstrating superior performance and computational efficiency.

### Strengths
1. Highly innovative in theory: It is the first systematically modeled as a dynamic programming problem for the optimization of the diffusion process;
2. Elegant algorithm design: DPad has approximately linear complexity and significantly improved inference efficiency;
3. Sufficient experimentation: Its effectiveness has been verified on various tasks and datasets;
4. Strong compatibility with existing algorithms: It can be embedded as a plug-in module into other diffusion frameworks;
5. Good interpretability: The DP framework clearly demonstrates the sampling path optimization logic.

### Weaknesses
1. Lack of ablation analysis: The independent contributions of each module (such as heuristic search and value function approximation) were not systematically explored; 
2. Limited experimental scope: Only image generation tasks were covered, without text or cross-modal diffusion models; 
3. The paper is slightly biased towards algorithmic engineering: The discussion of convergence and lower bounds of complexity in the theoretical part is still insufficient.

### Questions
1. Could the author further explain the differences in the mathematical mechanisms between DPad and DPM-Solver? What is the relationship between the two in terms of their approximate objective functions?
2. Can the DP approximation still maintain stability under high-dimensional non-Gaussian noise distributions?
3. Have you considered extending it to video diffusion or 3D generation scenarios?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DPad, a training-free inference acceleration method for diffusion language models (dLLMs). The authors first identify and formalize the *Scratchpad* mechanism, showing that suffix tokens function as a dynamic cross-layer reservoir guiding the denoising process. Through systematic analysis, they reveal three key properties of suffix tokens—inherent sparsity, distance-decay attention, and position insensitivity—that cause major inefficiencies in existing dLLMs. Building on these insights, the paper proposes DPad, which applies distance-decay dropout to prune redundant suffix tokens before attention computation, effectively reducing computation while maintaining accuracy. Moreover, DPad can be combined with existing acceleration methods such as parallel decoding and prefix caching.

### Strengths
1. The paper conducts a detailed analysis of suffix tokens in diffusion language models from a sequence-level perspective. It introduces a novel viewpoint for understanding suffix tokens, identifies three key properties.
2. The proposed **DPad** method is training-free; by applying distance-decay-style suffix dropout during inference, it significantly reduces computational complexity while maintaining model accuracy.

### Weaknesses
1. Section 3.1 formally introduces the *scratchpad mechanism* and explains, from the perspective of attention-based information flow, that *‘the suffix serves as temporary memory to assist the ongoing denoising process’.* While this section highlights the potential ‘*Attention Connection’* role of suffix tokens, it does not evaluate how critical this effect is to model inference. No intervention experiments (e.g., masking parts of the suffix and observing changes in hidden states) are provided to further verify whether the scratchpad truly functions as an essential memory component.
2. Section 3.2 then intuitively suggests that *‘the scratchpad intuition does not require populating all suffix tokens.* To support this, the paper presents entropy-based analyses claiming that suffix tokens *‘carry little new information’.* However, the logical transition from Section 3.1 to 3.2 feels abrupt — low entropy does not necessarily imply uselessness*.* It only reflects prediction certainty, not whether these deterministic tokens still play structural roles during generation (e.g., alignment, formatting, or mask boundary control). In addition, Figure 3 reports attention decay but only for the final layer; it would be more convincing to include layer-wise attention distributions to examine whether similar decay occurs in lower or middle layers. Overall, the claimed *inherent sparsity* of suffix tokens requires further clarification and empirical support.
3. The proposed “Diffusion Lottery Ticket” hypothesis is intriguing and provides a theoretical grounding for DPad’s dropout design. However, there is an apparent inconsistency: Figure 4 shows that *“the model shifts its attention to nearby suffix tokens”* when a spike is removed, whereas later (and also in Section 1) the paper states that *“information carried by distant suffix tokens appears largely position-insensitive.”* These two claims seem contradictory—if attention shifts locally, how can the information be position-insensitive globally? Moreover, the observational experiments in Sections 3.2 and 3.3 are only conducted on the LLaDA model series. It would strengthen the argument to verify whether similar phenomena hold across other diffusion-based LLM architectures.
4. In Table 2, the model shows an exceptionally large improvement on the MBPP benchmark after applying DPad—the accuracy increase substantially. Could the authors clarify why DPad benefits MBPP so much?

### Questions
1. Could you provide evidence (e.g., masking or ablation) to verify whether the scratchpad mechanism in section 3.1 truly functions as an essential memory component?
2. How does low prediction entropy in section 3.2 directly imply token redundancy, and are the attention decay patterns consistent across all layers?
3. How can the local *attention shifting* in Figure 4 be reconciled with the claim that suffix information is globally position-insensitive, and does this phenomenon generalize beyond LLaDA?
4. What explains the exceptionally large accuracy gain on MBPP in Table 2, and does this improvement remain under shorter generation lengths (e.g., 256 tokens)?

### Soundness
2

### Presentation
3

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
The paper targets the core inefficiency of diffusion LLMs (dLLMs): at each denoising step they compute attention over the entire suffix (future) region even though only a few tokens are unmasked/kept. The authors argue these suffix tokens mainly serve as a transient memory that aggregates context and is then read back by the current block, and show attention/entropy patterns that decay with distance. Building on this, they propose DPad, a training-free inference strategy that (1) caps the suffix via a sliding window, and (2) applies distance-decay dropout to progressively prune farther suffix tokens before attention is computed. DPad composes with existing accelerations (parallel decoding, prefix/KV caching) and, on LLaDA and Dream, reports up to 61.4× end-to-end speedup in long-sequence settings while maintaining accuracy (notably large strict-match gains on GSM8K for LLaDA).

### Strengths
1. The suffix window + decay dropout is lightweight, does not touch weights, and composes with parallel decoding and caching; the large compounding speedups in long sequences are compelling. 

2. Attention/entropy analyses show strong distance decay and near-zero entropy far in the suffix; ablations identify a 64–128 token “critical window.” 

3. The empirical studies across tasks with both reasoning and code are conducted, and tables separate latency vs TPS have discussed why TPS can look smaller even when latency drops (shorter, cleaner generations). 

4. Long-sequence results are a highlight (e.g., LLaDA-1.5 GSM8K 1-shot, 1024-token cap: 20.3× from DPad alone; 61.39× with Fast-dLLM-style combos). 

5. The overall writing is clear and easy to follow.

### Weaknesses
1. Recent training-free accelerations for dLLMs cover Block-dLLM, Fast-dLLM (dLLM cache), Sparse-dLLM, etc. The paper cites some of them but does not always compare directly, especially on matched long-sequence regimes and memory usage. 

2. The “distance-decay” sparsity concept parallels existing efforts in distance-biased or entropy-guided pruning methods. While DPad’s pre-attention pruning is a nice twist, the paper should sharpen how this differs theoretically/empirically from Sparse-dLLM (eviction guided by attention saliency over steps) and from Fast-dLLM/dLLM-Cache caches, ideally with head-to-head plots of FLOPs/latency/accuracy. I may have missed out or misunderstood some settings, please point out if the authors have done so.  

3. The Gaussian sampler uses window size W, decay k, and magnitude a tuned per task/domain. Please provide broader sweeps and robust defaults, and test out-of-domain transfer (e.g., from code→math without retuning).

4. Benefits are strongest on LLaDA while Dream shows smaller or mixed gains—this dependence deserves deeper analysis and maybe a diagnostic to decide when to enable DPad. However, the analysis regarding this point is not  elaborated in the paper. 

5. Effective tokens × layers attended and wall-time per decoded token may be helpful to include. Latency alone can hide GPU-utilization effects the paper itself notes.

### Questions
1. When does DPad hurt? Any tasks with long-range cross-suffix dependencies (e.g., long code synthesis, theorem proofs) where aggressive distance decay reduces correctness or format adherence?

2. Can W, k, a be selected online using low-overhead signals (e.g., per-step confidence/entropy) rather than fixed grids?

3. With Fast-dLLM and dLLM-Cache, in what order should DPad be applied? Does pruning suffix tokens change cache-hit rates or destabilize delayed-update caches? Please explain the  possible interaction.

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
The paper proposes an inference-strategy to reduce generation cost in MDLMs by using summaries of the suffix tokens (future tokens that haven't been decoded yet). The authors discuss how the suffix tokens's latent representations can only serve as a per-layer "memory", which can be thought of as a scratchpad, and empirically find that there is a lot of redundancy in the suffix tokens and the attention scores decay rapidly with distance from the current block index, with the occasional spikes. These findings motivate a summarization sampler that   keeps track of close tokens from the suffix. Further, the spikes are found to be position-insensitive where removing that token position moves the spike to a closer token. Thus motivated, the window-based gaussian sampling for suffix dropout then leads to significant gains in speed.

### Strengths
- Clean analysis of what happens with suffix tokens in block-wise autoregressive generation with MDLMs.
- Well-motivated method for reduce unnecessary computation that can be applied at inference-time without changes to the model.
- strong improvements in efficiency and good ablations showing the gaussian sampler which reflects the decay in attention scores actually helps.

### Weaknesses
See questions.

### Questions
- Why use a random selection for masking? What is the value of the "dropout" part of the suffix dropout? Why not use a simple deterministic procedure ?
- I'm not sure I buy the lottery ticket connection. In the original "lottery tickets" paper, one needed to prune to get to the ticket. Your procedure is not weight or input-dependent and that only tells me  that the "memory" in the suffix tokens is not position-dependent, which is supported by your experiment with the locations of the spikes. So any suffix positions you give the transformer get used for the same purpose. One could check this: what happens if you have other sampling strategies, such as removing some of the closer suffix tokens and deterministically keeping moderately distant ones? Would the performance drop significantly?
- Why not train with the proposed restriction or other sliding window restrictions ? That also should get you similar gains and should be possible with the flex attention library?
- Interestingly, the above check  could  test the "lottery ticket circuits" hypothesis. If the explanation for the mechanism is "any suffix tokens will get used", then training with a limited windows will recover the same performance or be better. If lottery tickets are the actual explanation, the artificial restriction would degrade performance. I understand that this may be expensive, but it seems like a direct test. So either a small experiment would suffice or the authors can convince me that this is not a good test or clarify what I'm missing.

### Soundness
3

### Presentation
3

### Contribution
3
