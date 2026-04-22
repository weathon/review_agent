# FlexLinearAttention: Compiling a Unified Abstraction into Scalable Kernels for Linear Attention

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
The quadratic complexity of softmax attention poses a major bottleneck for long-context modeling, motivating a surge of linear attention variants with linear complexity. Unlike softmax attention, which benefits from optimized kernels, linear attention lacks general-purpose, hardware-efficient support and scalable distributed implementations. We introduce **Flex**ible **L**inear **A**ttention (FlexLA), a domain-specific compiler that automates the generation of high-performance, scalable kernels for a wide range of linear attention models directly from high-level PyTorch code. At its core, FlexLA employs an intuitive programming abstraction that decomposes any linear attention algorithm into three canonical phases: intra-chunk computation, inter-chunk state propagation, and output merging. This unified abstraction enables FlexLA to perform domain-specific optimizations, automatically generating kernels that fuse computation and communication at a fine-grained tile level and eliminating host synchronization. Our evaluation demonstrates that FlexLA combines programmability with performance: a wide range of linear attention variants can be implemented in just a few dozen lines of code, while the generated kernels deliver 1.01x-4.9x the performance of sate-of-the-art expert-optimized library and scale with near-linear efficiency on scalar gated linear attention to 16 million tokens on 128 GPUs, surpassing the state-of-the-art distributed baseline by up to 7.2x.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Forge, a domain-specific compiler designed specifically for linear attention mechanisms. Forge abstracts the computation into three canonical stages based on chunk-wise parallelization: intra-chunk computation, inter-chunk state propagation, and output merging. This unified abstraction both simplifies the development of new linear attention variants (such as Mamba, RetNet, RWKV, GLA, HGRN, and GDN) and enables systematic optimizations for both single-device and distributed scenarios. Forge leverages Triton-Distributed as its backend to generate high-performance kernels that fuse computation and communication, and incorporates an adaptive scheduling system for parallelism and a static dispatcher to reduce runtime overhead. Experimental results demonstrate that Forge matches or outperforms state-of-the-art hand-written kernels (e.g., Flash Linear Attention) in both performance and scalability, while also requiring much less manual effort to implement new variants.

### Strengths
1. Proposes a unified, three-phase abstraction for linear attention that captures a wide range of state-of-the-art variants, enabling easy mapping from high-level algorithms to efficient implementations.
2. Automates the generation of high-performance, hardware-aware kernels from simple PyTorch code, reducing the barrier for researchers to experiment with or deploy new linear attention models.
3. Integrates native support for fine-grained, tile-level compute-communication fusion using Triton-Distributed, improving distributed execution and network bandwidth utilization compared to existing methods.
4. Demonstrates strong empirical results, with Forge-generated kernels achieving up to 4.9x speedup over expert-tuned baselines and near-linear weak scaling up to 128 GPUs and 16 million tokens.
5. Includes practical system-level optimizations such as ahead-of-time compilation and static dispatching to eliminate runtime overheads, particularly beneficial for short to medium sequence lengths.
6. Provides evidence that the abstraction is expressive enough to cover a broad family of linear attention models with minimal code changes.

### Weaknesses
1. As a compiler-based framework, the proposed abstraction is tailored specifically for linear attention and lacks extensibility to other attention mechanisms, which limits its general applicability.
2. Although Forge demonstrates competitive performance compared to Flash Linear Attention, the paper does not sufficiently emphasize its advantages, particularly in terms of implementation flexibility and extensibility relative to Triton-based Flash Linear Attention.
3. While Forge automates optimization, the actual benefits in terms of code brevity or maintainability versus existing Triton-based manual approaches are not quantified in user studies or qualitative analysis.
4. The distributed scaling results may be somewhat overstated: for example, in Figure 1, the single-GPU latency is 9.2ms, while 4 GPUs achieve 2.7ms. The real scaling efficiency is about 85%, not precisely "near-linear" as asserted.
5. The distributed experiments (Section 4.2) use H20 GPUs (with slower communication) and BatchSize=4, whereas single-GPU experiments use H100 GPUs and BatchSize=1. This difference in hardware and batch size may exaggerate the benefits of compute-communication overlap and makes the experimental comparison less fair.

### Questions
1. According to the experimental results, Flash Linear Attention already supports multiple linear attention variants and delivers strong performance across them. While Forge’s three-phase abstraction enables users to easily implement different variants, Flash Linear Attention is also Triton-based, which inherently provides customization capabilities. How does Forge compare to Flash Linear Attention in terms of flexibility when supporting diverse linear attention variants?
2. How does the Parallelism Scheduler mentioned in Appendix B adapt to distributed environments? Are there differences in scheduling strategies between single-GPU, intra-node, and inter-node scenarios?
3. Are there any limitations or edge cases where the three-phase abstraction might restrict the expression of certain linear attention updates, particularly for future architectures that may involve more complex dependencies?
4. What is the overhead (if any) of using Forge's compilation pipeline compared to maintaining optimized hand-written kernels, especially as new hardware or Triton versions are released?

### Soundness
3

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
3

### Summary
Forge is a DSL + compiler for linear attention. It abstracts linear attention variants to three definitions, and uses a compilation backend with a distributed version of triton-lang to optimize linear attention across computation and communication resources. Forge showcases its efficiency by comparing against Flash Linear Attention, a library of hand-designed kernels for linear attention, and performs on par or better than it.

### Strengths
Thank you for submitting your work. This was a pleasant read and I am optimistic it will be valuable to the community. I found the presentation to be very well-done, and the explanations of how the DSL is translated to intra and inter-GPU parallelized code was interesting to read. The experiments are expansive and the results are promising.

### Weaknesses
There are two issues I am concerned about.
* The central issue of abstractions is being future-proof. This boils down to 1) whether this abstraction is expressive enough for a wide range of designs, and 2) whether maintaining the abstraction (translation and compilation) is sustainable with hardware changes. These questions are challenging to answer, since to some extent they involve making educated guesses. But I still would have liked to see some discussion about these points in a paper of this sort.
* I could not find an anonymized code repository in the text. It is odd not to share code for papers with DSL contributions. Please add an anonymized code repository link to the paper text.

### Questions
* Why did the authors find this particular abstraction more expressive than the rest?
* Are there linear attention variants this abstraction does not support? Is there an example of this from recent publications in the community?
* Why did the authors not include source code with the text?
* Can you include the chunk, decay and merge sections for the five variants studied in the paper? This would be ideally be in a code repository, but in absence of that it should be added to the appendix.
* Have the authors read AttentionEngine [1]? Seems like relevant work.

[1] Chen, Feiyang, et al. "Attentionengine: A versatile framework for efficient attention mechanisms on diverse hardware platforms." arXiv preprint arXiv:2502.15349 (2025).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
What about the following summary, am I missing anything?
- Forge introduces a domain-specific language (DSL) for linear attention kernels - this is the key novelty and the abstractions defined there
- The DSL is compiled to triton kernels. The compiler can do optimizations such as compute communication fusion, and a range of targeted optimization of system bottlenecks
- in single GPU, forge generated kernels can give an increase of 1.01-4.9x speedup compared to hand-tuned kernels
- Forge can also generate distributed kernels accross multiple GPUs, which is demonstrated up to 128

Overall, forge is a well thought through engineering solution that can help to develop linear attention kernels with good performance, without the effort of manually implementation

### Strengths
- well-engineered compiler system that automates generation of linear attention kernels, both for single GPU and distributed kernels
- integrates well into existing pipelines such as Triton, this will make it easier to adopt.
- important optimizations for compute- communication fusion, and ahead-of-time (AOT) compilation
- good empirical results, showing 1.01x-4.9x speedup over FlashLinearAttention on single GPUs and near-linear scaling up to 128 GPUs
- practical significance for research on models that leverage new linear attention mechanism, reducing manual kernel engineering effort which is one of the limitations to scale more researchy models

### Weaknesses
- limited novelty (the DSL/abstraction for linear attention), but good integration of existing work
- too narrow. linear attention is not widely used, and mainly for more researchy models. Not relevant for predominant model architectures. This limits the practical use of this framework to a very specific set of research explorations. Expanding this to more common forms of attention would be desirable.

### Questions
I would love to understand what would be required to expand this framework to softmax based attention. The authors discuss associativity at  the beginning of the paper, but associativity also has been exploited for optimizations with softmax attention (see e.g. LeanAttention)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The AI community has been rapidly innovating on AI architectures, but it it painstaking to obtain hardware-efficient implementations. The kernels need to respect the GPU memory hierarchy, support multi-GPU execution, and remain easy to implement. This work observes that there are a few rules underlying linear attentions and encodes these patterns into the Forge DSL.

### Strengths
- This is a very important problem space. Many sub-quadratic models have been fast in theory but not in practice, and the challenge of developing kernels has prevented the community from understanding how to obtain the best quality for a fixed wall clock time.
- The paper considers a relevant baseline framework – FLA – and shows consistent speed ups.

### Weaknesses
It would be useful to understand how Forge kernels compare to highly optimized kernels, since Triton kernels are routinely slower than CUDA kernels:
- How does Forge generalize to linear attentions that use very large state sizes (e.g., Taylor approximations like ReBased, Based, Learned feature maps with large feature dimension, Mamba-2 with large state size)?  - Does register management, careful use of wgmma/tcgen05, etc., become important? 
- How do Forge kernels compare to other DSLs (e.g., TileLang, ThunderKittens which have open-sourced efficient linear attention kernels) in the single-GPU setting? It would be useful to have head-to-head speed comparisons. 

The writing would benefit from additional clarity and explanation in certain places:
- Are the kernels for inference/forwards pass or backwards as well? Does Forge consider decoding kernels as well, or prefill-only? It would be useful to clarify and discuss the scope of Forge in the paper, and how the ideas could generalize. 
- Do the kernels remain numerically stable for training and inference, for instance as compared to the FLA kernels?
- How do the multi-GPU implementations compare to popular multi-GPU softmax attention implementations? Do we see a speed up from linear attention? 
- It would be useful to provide clearer explanation on where the experimental gains arise from: L370-377 attribute it to Forge’s “system level overheads” and “parallelism strategy” but this is a vague explanation

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
