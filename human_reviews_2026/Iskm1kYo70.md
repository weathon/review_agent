# Towards Compressive and Scalable Recurrent Memory

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Transformers face a quadratic bottleneck in attention when scaling to long contexts. Recent approaches introduce recurrent memory to extend context beyond the current window, yet these often face a fundamental trade-off between theoretical principles and practical scalability. To address this, we introduce **Elastic Memory**, a novel memory architecture grounded in the HiPPO framework for online function approximation. Elastic Memory treats historical sequence as samples from continuous signals, applying optimal online compression to encode them into a fixed-size memory state. For retrieval, we propose a flexible \textit{polynomial sampling} mechanism that reconstructs a history summary from this compressed state. Elastic Memory consistently outperformed baselines on long-context (32k+) datasets across three domains. With equal parameters, it beat Memorizing Transformer by 16x memory and outperformed Melodi at all memory sizes, even when Melodi had 30\% more parameters. When scaling model size, Elastic Memory stayed ahead of all baselines and was significantly faster than Melodi at 4x size. Furthermore, its decoupled design allows for injecting inductive biases at test-time to boost performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Elastic Memory, a theoretically grounded and scalable memory module for long-context Transformers. The method interprets historical tokens as samples from a continuous signal, compressing them into fixed-size memory states via optimal polynomial projection. The model retrieves contextual information using a precomputed reconstruction bank and a flexible polynomial sampling strategy, allowing retrieval biases to be injected at inference. Experiments on long-document benchmarks show improved perplexity and efficiency compared with baselines.

### Strengths
- The work elegantly unifies two previously competing paradigm: principled associative memory and scalable external memor, through the lens of online function approximation. Also, the paper provides concrete, closed-form derivations linking continuous HiPPO dynamics to discrete, parallelizable updates, offering strong conceptual clarity.

- The precomputation of HiPPO transition matrices transforms a theoretically sequential recurrence into an efficiently parallelizable block update, yielding strong training throughput without extra parameters.

- The experiments are comprehensive, demonstrating both performance and robustness.

- The ability to change polynomial sampling strategies at inference time without retraining is an interesting demonstration of decoupled memory representation and retrieval.

### Weaknesses
- The theoretical optimality of HiPPO relies on continuous, smooth signals under a uniform measure, which does not align with the discrete, non-stationary nature of token sequences in language modeling. Hence, the “optimal compression” claim is mathematically elegant but practically ungrounded.

- All evaluations focus on perplexity over long sequences. While lower PPL suggests improved context retention, it does not confirm enhanced semantic or logical long-term reasoning. No ablation or probing task demonstrates that Elastic Memory truly captures non-local dependencies beyond text continuity.

- Although the HiPPO framework provides a rigorous derivation, the system is trained end-to-end with gradient updates that can easily violate the theoretical assumptions. The authors present no evidence that the model actually maintains the intended HiPPO structure during optimization.

### Questions
- How sensitive is performance to the truncation order N of the polynomial basis? Does increasing N monotonically improve fidelity or cause overcompression/interference?

- What happens when the model processes sequences longer than the precomputed maximum context length? Is extrapolation possible without recomputation?

- Could the authors provide qualitative evidence showing that Elastic Memory indeed recalls semantically relevant distant context, rather than merely smoothing over text statistics?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce a technique called "elastic memory", which is based on the HiPPO framework.  The keys and values in a transformer are treated as samples from a continuous-time signal.  This signal is compressed into a finite memory C by fitting a polynomial approximation to the signal.  Crucially, updates to the memory are incremental and recurrent, so C can be updated with a new K,V pair on each time step.  

The authors also derive a parallelizable block-level update operation, operating on blocks of length L.  Unfortunately, the block-level update operation, and the subsequent retrieval operations, depend on the absolute position of the block, so the authors pre-calculate and cache state-transition matrices and retrieval matrices up to some maximum context length.  

To perform attention, the authors sample from the continuous signal at a pre-defined set of points; uniform sampling spreads these points uniformly across the input sequence, while exponential sampling concentrates the samples at recent positions.  The transformer then does dense attention over the samples.

### Strengths
The paper is well-written and clear, with well-defined algorithms and equations.  

Unfortunately, I am not an expert in HiPPO, or state-space models in general, so it is hard for me to assess novelty and contribution.

### Weaknesses
The biggest weakness, as far as I am concerned, pertains to the writing style.  The method section presents a mathematical progression from HiPPO to elastic memory, but has almost no citations within it.  Thus, I cannot tell which parts of the math are novel derivations done by the authors, and which parts are drawn from prior literature.  I personally am not an expert in HiPPO or state space models, so I cannot determine contribution without citations.  The format of the equations differs enough from the HiPPO paper(s) that it is non-trivial to figure this out.  

The second major weakness of the paper are the baselines.  There are two ways to measure memory against baselines.  The first is by the size of the memory (in number of parameters), the second is by context length that the memory covers.  

The benefit provided by memory is largely measured in context length -- more context equals lower perplexity.  Memorizing transformer is extremely inefficient in its use of memory, so if you compare models based on the size of the memory, then memorizing transformer will obviously be worse when compared against anything that does compression -- a compressed memory model will *always* have more context for the same size of memory.  Thus, when used as a baseline, memorizing transformer should be used to measure the *quality* of compression -- a good compression mechanism should show little degradation over the uncompressed (memorizing transformer) baseline.  

Given that the context length in this paper is 32k tokens, the memorizing transformer context length should be set to 32k.  Melodi similarly supports a maximum context length, although it uses a compressed representation.  Unfortunately, that is not what this paper does -- it uses a confusing 1x, 2x, 4x, 8x, 16x metric that I have no idea how to interpret.  

Moreover, since this paper is based on HiPPO, and seems related to state-space models, it should be compared against some of the many models in the vast literature on linear transformers -- e.g. Mamba, RWKV-7, or DeltaNet.

Finally, memory models should ideally be tested on something other than perplexity -- ideally a downstream QA task, or at least a needle-in-a-haystack task.

### Questions
Which parts of the math are unique to this paper, and which parts are from prior literature?  Please list by equation (1) through (8).  

The comparisons in Table 3 measure memory size as "1x, 2x, 16x," etc.  I have no idea what this metric means, and it makes no sense for Memorizing Transformer and Melodi -- both of which are models that measure the amount of "memory" in terms of context length.  What are the memory context lengths in Table 3?  

What is trapezoidal attention?  Do you have a citation?

The retrieved keys and values do not have position encodings, because they are "inherently time aware" (line 228).  However, if C is indeed a "memory", then retrieving the memorized token at position x_j does not mean that the retrieved token has a timestamp incorporated into it.  Please explain.  

Please double-check your references, e.g. I happened to notice that the authors on the block-recurrent transformers paper are incorrect.  

The strategy of this paper seems to be:
(1) Store a compressed representation of the entire context.
(2) Sample a fixed (Lmem) number of samples from the memory, and then do dense attention. 

Since attention is based solely on samples, couldn't you just sample directly from the original, uncompressed context, rather than compressing it first?  That would be much simpler and easier, e.g. uniform sampling would sample a few tokens from each block, and you could implement exponential sampling by gradually dropping tokens from more distant blocks.  I also would expect sampling without compression to perform poorly, which makes me wonder whether compression is doing something else here...

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Elastic Memory, a novel memory architecture for transformers designed to efficiently handle long-context sequences by leveraging the HiPPO framework for online function approximation. Elastic Memory compresses historical sequence information into a fixed-size memory state and retrieves it using a flexible polynomial sampling mechanism. The method is evaluated on several long-context language modeling benchmarks, consistently outperforming state-of-the-art baselines such as Memorizing Transformer, InfiniTransformer, and Melodi in terms of both memory efficiency and computational speed. The architecture is notable for its decoupled design, allowing for test-time injection of inductive biases, and achieves these results without introducing additional trainable parameters.

### Strengths
(1) The use of the HiPPO framework provides a solid mathematical basis for memory compression, moving beyond heuristic or ad hoc designs.   
(2) Elastic Memory demonstrates state-of-the-art performance across multiple long-context benchmarks, outperforming strong baselines in both accuracy and efficiency.   
(3) The architecture achieves its gains without adding extra trainable parameters, making it attractive for practical deployment.   
(4) The ability to inject inductive biases at test time via different sampling strategies is a unique and potentially impactful feature.

### Weaknesses
(1) While the experiments are comprehensive within the long-context language modeling domain, the method is not evaluated on other important tasks such as vision or multimodal settings (e.g., vision-language models), which typically need long context window due to large number of image tokens from high-resolution images.

(2) The method is evaluated on models trained from scratch using metrics such as loss and ppl; its effectiveness when integrated into specific down-stream tasks remains to be demonstrated. For example, it would be great if the authors can run experiments for tasks such as long document summarization, which is a good use case for memory compression.

### Questions
see my comments above

### Soundness
3

### Presentation
3

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
Elastic Memory frames recurrent memory as an online function-approximation problem using the HiPPO (Legendre) projection and introduces a polynomial sampling reconstruction to produce a history summary that plugs into attention. The paper reports strong long-context results (32k+ sequences), claiming large memory efficiency and speed advantages over Memorizing Transformer and Melodi while keeping parameter counts comparable.

### Strengths
- Casting memory as HiPPO-based online compression gives a clear mathematical objective (optimal incremental polynomial projection) rather than an ad-hoc summarization heuristic; this grounds architecture choices in prior theory.

- The paper reports state-of-the-art performance on multiple 32k+ datasets, beating the Memorizing Transformer by large memory-efficiency margins and outperforming Melodi across memory sizes (including when Melodi has more parameters). These are high-impact claims if robust.

- The architecture decouples memory size from model dims and reports substantial runtime advantages when scaling model size (e.g., being faster than Melodi at 4× scale), which addresses an important production constraint.

### Weaknesses
- The HiPPO-projection order/size (N), the polynomial sampling schedule, and the weighting/measure choices can materially change performance. The paper gives strong aggregate wins, but I want systematic ablations (sensitivity to N, sampling density, training stability, and memory reconstruction quality metrics) to show results are robust and not brittle.

- The paper makes speed and memory-efficiency claims (e.g., 16× memory advantage; 50% faster at 4× scale) — but details on hardware, batch sizes, wall-clock measurement methodology, and training cost are thin. 

- Show failure cases and limits: at what sequence lengths or signal types does the polynomial compression lose important information? How does it handle highly discontinuous or structured tokens (e.g., code, tables)?

### Questions
see weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
