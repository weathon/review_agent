# FutureFill: Fast Generation from Convolutional Sequence Models

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
We address the challenge of efficient auto-regressive generation in sequence prediction models by introducing FutureFill—a general-purpose fast generation method for any sequence prediction algorithm based on convolutional operators. FutureFill reduces generation time from quadratic to quasilinear in the context length. Moreover,  when generating from a prompt, it requires a prefill cache whose size grows only with the number of tokens to be generated—often much smaller than the caches required by standard convolutional or attention‐based models. We validate our theoretical claims with language modeling experiments and demonstrate substantial efficiency gains when generating from a deep convolutional sequence prediction model.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper continues the line of work aiming to speed up inference with respect to sequence length and to avoid the quadratic complexity of standard Transformers. The authors propose a method for fast generation using convolutional sequence models. They focus on exact auto-regressive generation from convolutional models, reducing both generation time and cache size, achieving O(N log N) complexity instead of O(L²).
Empirical results at small scale show that the proposed algorithms achieve sub-quadratic scaling compared to naive convolution implementations, and they report up to 1.7× speedup over the baseline.
Overall, this is a solid  paper that provides a practical speedup method, but the experimental validation feels limited.

### Strengths
The work is technically sound and contributes to the ongoing effort of making sequence models more efficient at inference time.

### Weaknesses
The experiments are limited to relatively small language models (below 1B parameters), which makes it hard to assess the impact at realistic scales. In addition, only inference speed results are presented — there is no evaluation of model quality (e.g., perplexity or downstream performance), which is important to verify that the gains do not come at the cost of degraded output quality.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces "FutureFill," a novel and efficient method for auto-regressive generation from sequence models that use convolutional operators. The primary contribution is an algorithmic technique that reduces the computational complexity of generating L tokens from scratch from a quadratic $O(L^2)$, which is typical for naive online convolution, to a quasilinear. The paper presents two concrete algorithms based on this idea:

1. Continuous-FutureFill: An algorithm that achieves the $O(L log^2 L)$ runtime with $O(L)$ memory.
2. Epoched-FutureFill: A more practical variant that offers a trade-off between runtime and memory.

Furthermore, the paper shows that when generating K tokens from a prompt of length L, FutureFill significantly reduces the required cache size from $O(L+K)$ to $O(K)$, a crucial improvement for long-context applications. The authors validate their theoretical claims with experiments on both synthetic data and large-scale (up to 826M parameters) convolutional language models, demonstrating empirical speedups of up to 2x over baseline methods on modern hardware.

### Strengths
1: The paper tackles a critical and well-known bottleneck in sequence modeling, i.e. the slow quadratic-time generation process for models based on convolutions. Making this efficient, especially in long-sequnece scenerios is a major practical contribution.

2. The paper is clearly written and the theoretical claims are well-supported by theorems and complexity. The experiments not only demonstrate asymptotic behavior but also wall-clock time improvements.

3. The idea of futurefill is intuitive and based on observations and intuition regarding the properties of convolution and FFT.

### Weaknesses
1. The paper points out that there has been independent and concurrent work that achieves the same runtime complexity, which slightly tempers the novelty.

2. The paper seems to focus only on FlashSTU-T model and would be nice to show how this is generalizable to other convolutional models (e.g. Hyena).

3. While the paper does demonstrate real speed improvement of 2x, it was not as dramatic as the difference from $O(L^2)$ to $O(L)$. More detailed analysis here on why the speedup was not significant would be interesting and informative.

### Questions
1. Compared to other works that achieve the same complexity, could the authors elaborate more on the qualitative or practical differences between FutureFill and other concurrent works?

2. How would the algorithm perform if the generation length is not known in advance, i.e. the epoch length 
 was set to the theoretical optimum but what should peopole use in real-world scenerios?

### Soundness
4

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
The paper positions itself with respect to recent work that uses convolutional operators as a way of mitigating complexity issues in autoregressive attention-based models.  Specifically, they propose a method FutureFill that reduces complexity in text generation below quadratic in context length; a core part of the idea is that there is a memory trade-off that permits the reduction in inference time complexity, which also allows a spectrum of algorithm variants in terms of that trade-off.  The paper contains some theoretical results on this complexity, and also some experimental results looking empirically at inference times as a function of context length, as well as checking the performance on several downstream tasks.

### Strengths
* Overall, I think this is quite a strong paper.  The use of convolutional approaches as in e.g. Hyena is an important direction to address the quadratic complexity issue, and this paper’s contribution looks like an important step in that, and could well be adopted quite widely as a source of performance improvements.

### Weaknesses
These are mostly relatively minor.

* The Abstract is fairly short and bare-bones; it’s not really until reading the paper that the actual importance of the work comes through.

* It’s fairly reasonable given space constraints to save most of the literature review of Sec 1.1 for the appendix.  However, something that I thought was missing in both Sec 1.1 and the appendix’s extended version was the discussion of differences wrt Oncescu et al. (2024).  This is presented as an independent work that achieves the same complexity result as the present paper, but there’s no argument made as to why FutureFill then is necessary.  Is there something more advantageous about the memory trade-off inherent in FutureFill?  Is there some limitation to Oncescu?  It’s not made clear why a new method with this complexity is a useful thing to have, and this is quite crucial for understanding the importance of the present paper's contribution.

* Fig 5 in App D.3 is presented a depiction of the FutureFill operation between an input sequence and a convolutional filter; it’s where the reader is supposed to get a concrete idea of how the method works, to supplement the mathematical definitions.  However, the diagram is very schematic and abstract, and not actually much of a help.  Some fleshing out of the diagram and explanation in D.3 would be helpful.

* For the proof of Propn 1, it would be helpful to link explicitly to App F.

### Questions
Please see above.

### Soundness
4

### Presentation
4

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
The paper introduces FutureFill, a fast inference algorithm for convolutional sequence models that reduces generation complexity by precomputing future contributions via FFT-based convolutions. The method maintains exact output equivalence with standard convolution while achieving notable latency improvements on FlashSTU-T models, all without retraining or architectural changes. It also presents two practical variants—Epoched and Continuous FutureFill—that balance memory and speed for different deployment settings.

### Strengths
- Addresses a less-studied yet important bottleneck in convolutional language models.
- Strong theoretical foundation with clear runtime and correctness guarantees.
- Training-free and exact—no compromise on model quality.
- Shows consistent practical gains and integrates seamlessly with existing architectures.
- Provides clear implementation details enabling reproducibility.

### Weaknesses
- Scalability to multi-billion parameter models not yet validated.
- Baselines limited; comparison with Hyena, RWKV, and S4 models would be useful.
- Reports only latency metrics; including FLOPs or energy-based analysis would make results more hardware-agnostic.
- Hardware dependency unclear—speedups may vary across GPUs/TPUs.
- Memory–latency trade-offs and cache behavior could be analyzed more deeply.

### Questions
- How does FutureFill scale with model size and longer sequence lengths (e.g., >100K tokens)?
- Are the latency gains consistent across different hardware backends?
- Could the authors report FLOPs per token to complement latency results?
- How does the method compare with other efficient convolutional architectures in total throughput and memory cost?

### Soundness
3

### Presentation
3

### Contribution
3
