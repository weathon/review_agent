# Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Language models with recurrent depth, also referred to as universal or looped when considering transformers, are defined by the capacity to increase their computation through the repetition of layers. Recent efforts in pretraining have demonstrated that these architectures can scale to modern language modeling tasks while exhibiting advantages in reasoning tasks. In this work, we examine the relationship between recurrent-depth models and diffusion language models. Building on their similarities, we develop a new diffusion forcing sampler for these models to accelerate generation. The sampler advances by decoding new tokens at every forward pass of the model, while the latent states of these tokens can be further refined in parallel through recurrence. Theoretically, generation with our sampler is strictly more expressive than the baseline autoregressive generation using the same time budget on modern hardware. Moreover, this sampler, based on principles from diffusion literature, can be directly applied to existing 3.5B recurrent-depth transformers without any tuning, leading to up to a 5x speedup. Consequently, our findings not only provide an efficient mechanism for parallelizing the extra computation in recurrent-depth models at inference, but also suggest that such models can be naturally viewed as strong continuous, though causal, diffusion language models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies a sampler for a recurrent-depth Transformer language model by framing the recurrent-depth model as a diffusion model, motivated by a desire to increase sampling speed. 

The authors detail the prerequisites of input injection and robustness to dynamic recurrence steps, as well as the importance of a recurrence-independent KV cache. 

The sampler is proposed with either a fixed number of inner recurrences per-token, or with a simple adaptive exiting mechanism that emits final tokens when the normalized update distance is beneath a threshold.

The authors remark on the convergence of the proposed algorithm, as well as defend their sampler design with theoretical analysis of the costs of depth and width during prefilling and decoding respectively.

Finally, the authors demonstrate that the proposed sampler achieves similar accuracy on math and code generation to existing algorithms while processing tokens nearly 5 times faster.

### Strengths
- The proposed method elegantly decreases the time to sample from the studied recurrent-depth language model at apparently little-to-no decrease in performance
- The paper flows well and is cleanly presented
- This work introduces a potential connection between recurrent-depth language and diffusion models

### Weaknesses
- The evaluation is limited to one model family and solely evaluates math and coding reasoning, ignoring other recurrent-depth models and natural language tasks
- The evaluation excludes analysis of memory cost, as the model decreases wall-clock time by increasing compute parallelization on GPU
- Several remarks (Remark 3.1, Theorem 4.4, Conclusion), while thought provoking, may be misleading:
  - The assumption that the recurrent block is a contraction is difficult to believe
  - Theorem 4.4 states only the advantages (and not the disadvantages) of diffusion forcing sampling, and does not "prove that recurrent-depth models should use diffusion forcing samplers during decoding"
  - A diffusion-inspired sampling method does not indicate that the model being sampled from is a diffusion model

### Questions
- In some inference systems, computational resources may be limited at varying timepoints. Can you profile the memory usage over wall-clock time for the evaluated samplers?
- Figure 6 demonstrates that the noise coefficient strictly decreases performance, and is set to 0 in evaluations in Table 1 and Table 2. Is it actually beneficial? How unstable is the recurrence without the additive noise? Is sampling from the model quantifiably unstable?
- Most hyperparameter choices (momentum, noise coefficient, headway, maximum wavefront size, initialization scale, continuous compute) appear to have little-to-no trade-off, generally decreasing model performance. The connection to diffusion seems tenuous given this. Is the main motivation of the paper to introduce an efficient parallel sampler or to examine the relation between recurrent-depth models and diffusion language models?

### Soundness
2

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
4

### Summary
This paper proposes an efficient mechanism for parallelizing the extra computation in recurrent-depth language models to accelerate their slow inference speed. Specifically, it introduces a diffusion forcing sampler that, at every forward pass, decodes a new "draft" token at the end of the sequence while simultaneously refining the latent states of all previously drafted tokens in parallel. This sampler uses an adaptive exit criterion to "freeze" tokens once their latent states stabilize, allowing the generation to proceed in an efficient "wave". The experiments show that this sampler can be applied directly to existing 3.5B recurrent-depth models without any retraining, leading to a 5x speedup on reasoning and coding benchmarks with only minor trade-offs in accuracy.

### Strengths
1. The connection between recurrent latent update and diffusion sampling is relevant and interesting. 
2. The proposed sampler achieves significant speedup with only mild degradation of quality. 
3. The method is designed to work with KV cache sharing, which allows it to have a memory footprint no larger than a standard fixed-depth transformer, preventing the cache from growing with the number of recurrence steps. 
4. The authors explored some key factors that affects the stability of latent recurrence.

### Weaknesses
1. The theoretical analysis section seems not fully formalized. The concepts of "depth scaling" and "width scaling" seem to be created specifically for their argument, rather than established, well-understood principles. Furthermore, the paper's core mechanism relies on the recurrent-depth model's states converging. However, the authors admit in Remark 3.1 that they cannot formally prove this. Sec 4.2 seem disconnected from the other part of the paper. There could be simpler way to formalize the high level intuitions. 
2. The authors explicitly state that their experimental evaluation is limited to a batch size of 1. They acknowledge that extending the sampler to batched or continuously-batched inference is complex and "fall outside the scope of this study". This is a limitation but I look forward to future development of the proposed method.
3. In Appx A.2, the authors state that "We experiment with headways greater than one, but while interestingly stable, this accelerates the speed of the sampler only slightly, at a large cost to accuracy". So even though the paper is written in a way that strongly relates to diffusion models, the generation is still largely next-token prediction.

### Questions
Why does the noise injection to z (Eq. 2) only happen at the start of a new round of inner recurrence? Actually, is it helpful or not? It looks like in the reported results, $\beta_s$ is set to 0. And Fig. 6 shows that on the GSM8k benchmark is achieved at $\beta_s = 0.00$. What motivates the introduction of this $\beta_s$ apart from connecting it to Diffusion forcing? And how to understand its impact on the throughput?

### Soundness
3

### Presentation
2

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
This paper proposes parallelizing inference for recurrent-depth transformers by processing multiple token positions simultaneously at different recurrence depths, achieving significant speedup. The approach is claimed to connect recurrent-depth models to diffusion models and can be applied to existing models without retraining.

### Strengths
* This paper tried to address a genuine bottleneck in recurrent-depth model inference, and the acceleration is significant.

* This method makes a good approach that can be applied directly to existing models without further training.

### Weaknesses
* The base model that this paper used was not proved to be fundamental better than GPT/AR based method. The contribution of this paper is questioned given if the recurrent-depth model is a promising direction or not. 

* The method cannot guarantee producing the same output as sequential generation, which creating fundamentally different computational paths. For use cases requiring reproducibility, this is a non-starter. The paper should clearly state this limitation and specify when the method is/isn't appropriate. 

* The evaluation results are very limited. All the four benchmarks share a very particular property: they use extraction-based metrics that completely ignore generation quality. What completely missing are summarization, translation, long-form QA, dialogue, creative writing, and general language understanding tasks where fluent generation matters. More language quality metrics are reported (perplexity, BLEU, human evaluation, token-level accuracy), while there are 15+ benchmarks results in the base model. 

* I'm not an expert of diffusion, but I think the connection to diffusion models is superficial to the point of being misleading. The paper reveals noise was added post-hoc as a hack, not as part of a principled diffusion framework.

### Questions
* Why only four benchmarks when the base model likely tested on 10-15? Can you provide results on all benchmarks from Geiping et al. 2025, particularly summarization, translation, long-form generation, dialogue, and general language understanding tasks?

* Can you provide perplexity, BLEU/ROUGE scores, human evaluation, and token-level accuracy to assess actual generation quality rather than just final answer correctness?

* What specifically makes this "diffusion" versus standard iterative refinement? Was the model trained with any diffusion objective, or is the connection purely post-hoc?

* On which task types does the method degrade significantly? Are there examples where generation quality is poor despite correct final answers?

### Soundness
2

### Presentation
3

### Contribution
2
