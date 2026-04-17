# TransFourier: FFT Is All You Need

- Decision: Reject
- Scores: 2, 2, 4

## Abstract
The scalability of Large Language Models (LLMs) to handle extremely long sequences is hindered by two foundational challenges: the quadratic computational cost of self-attention and the generalization limitations of positional encodings when extrapolating to contexts far beyond their training regime. These factors create bottlenecks for both the efficiency and the effective context window of current models. This paper introduces TransFourier, a novel architecture designed to address these challenges. TransFourier completely replaces the masked self-attention module with a parameter-efficient, $O(L \log L)$ Multi-Head Fourier (MHF) module. Our core contributions are threefold: (1) We propose a model that leverages the Fast Fourier Transform (FFT) for sequence information mixing, inherently addressing the aforementioned computational and generalization bottlenecks of attention. (2) We introduce a novel frequency-domain causal masking technique, which elegantly enforces autoregressive capabilities through asymmetric padding and truncation, overcoming a critical barrier that has historically limited Fourier-based models in generative tasks. (3) Our design is built entirely on highly-optimized, standard deep learning operators (e.g., FFT and convolution), obviating the need for hardware-specific custom CUDA kernels, unlike architectures such as Mamba, thus ensuring broad accessibility and portability. Evaluations on established academic benchmarks show that TransFourier is competitive with mature Transformer and State Space Model (SSM) baselines of comparable size. Given its strong scaling and architectural simplicity, TransFourier presents a compelling and practical pathway toward developing the next generation of efficient long-sequence models. The code is available in the supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to replace the attention mechanism by a linear convolution operation, which is calculated efficiently by FFT, multiplication in frequency space, and then iFFT. This reduces the attention complexity from O(L^2) to O(L log L).

GPT-2, Mamba, Mamba-2 and the proposed TransFourier are compared in three different model size configurations (mini, small, medium, where the largest setting medium means 1024 model dim and 24 layers, resulting in less than 500M params), trained on FineWeb-10B, evaluated on Hellaswag, ARC-Easy, ARC-Challenge, and Winogrande.

In the mini and small settings, TransFourier underforms the others. In the medium setting, it very slightly outperforms the others.

The haystack test is also performed, but fails in all cases.

### Strengths
* An interesting new model is proposed.
* It should give better training and generation speed.
* It slightly outperforms other models (GPT2, Mamba, Mamba2) in the medium setting.
* Code is published.

### Weaknesses
* More direct comparisons to other linear attention variants is missing (both in terms of math, and in terms of actual experiments).
* Explicit positional encoding variants should be tested.
* The model definition can be improved, and made more clear, by providing both the FFT-based definition, and then also the mathematical equivalent naive definition.
* Model sizes across model types (GPT, Mamba, TransFourier) might not be comparable. Better would be to compare training speed instead of model size. Or both.
* Training speed, nor generation speed is actually measured, even though this is the main motivation here.
* GPT-2 is not the best representative configuration for a standard Transformer model. You should use some more modern variant, like Transformer++ / Llama-style.
* Model sizes are overall too small.
* Scaling law studies are too limited. There should be more variations in model size, and also some bigger models, and that should be plotted in to a graph, to better see the actual trends. Again, better also using training speed instead of actual model size.

### Questions
There are many other alternatives how to replace the O(L^2) self-attention by O(L log L) or even O(L) variants. Related work section does a good work in giving an overview. But e.g. it also misses to discuss some more linear attention variants, like https://arxiv.org/abs/2102.11174.


The same argument about implicit positional information can be made about the decoder-only standard Transformer. (https://arxiv.org/abs/2501.00659) Still, one could argue, maybe having some sort of more explicit pos encoding might perform better. I think this should be tested.

It would be helpful to write down the formula for x_causal (e.g. Algo 1, or Sec 3.2), but not in terms of FFT/iFFT, but instead in terms of convolution, but carried out the full some, how you would calculate that naively/inefficiently. I think this would make it more clear, how x_g and x_v are being combined here.

I also wonder, how does this compare to linear attention. Here you also don't seem to have any softmax to calculate attention weights. So that seems to make it quite similar to linear attention. It would be good to compare that (not just in terms of experiments, where you have SSMs, but mathematically).

The setting that all model types (GPT, Mamba, TransFourier) use the same model size (mini, small, medium) is questionable. Different model types might use the layers, model dimensions, etc in different ways. A better way to compare these is by running many different settings for each, and plotting training time vs best performance at this point (collected from all the different settings that were run). I.e. basically this also covers scaling laws then.

Do not use the GPT2 here. First of all, as you point out, you cannot do the haystack test on longer seqs. But also, it is known that Transformer++ / Llama-like configurations (RMSNorm, RoPE, gated FF, no biases), perform better, and are much more standard nowadays.

You argue about speed. But where do I see the actual comparisons of e.g. training speed, and/or generation speed?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to replace the attention mechanism in Transformers with sequence mixing components based on Fourier transforms.
Specifically, the proposed multi-head Fourier layer consists of a 1D convolution and a “pad-FFT-multiply-inverseFFT-truncate” pipeline, which multiplies to projections of the input sequence in the spectral domain (which is equivalent to a convolution in the time domain).
In their language modeling experiments with 10B tokens and models with <500M parameters, TransFourier performed comparably to Mamba, Mamba-2 and softmax attention Transformers.

### Strengths
- The motivation based on efficiency & positional encoding is clear.
- The paper is well written.
- Code available.

### Weaknesses
The main motivation for replacing attention is the quadratic compute cost (efficiency) and the lack of length extrapolation capabilities (long-context). Therefore, I would expect experiments showcasing superiority in terms of runtime or long-context capability. 
While the paper makes attempts for long-context capabilities, the section is merely an explanation for why there are no more extensive experiments. I’d have at least expected experiments on synthetic tasks if the model size is too small, e.g. https://arxiv.org/abs/2312.04927 ). There are no runtime comparisons.

In general the experimental study is very small scale (10B tokens, <500M parameters) and metrics are only 4 tasks from lm-eval harness, which seem to yield very noisy results. Evaluating PPL on a validation set might give a bit cleaner signal. 


In my opinion the paper lacks a motivation for why the Fourier transform should be used for discrete sequences like language tokens. Previous applications of FFT were mainly in continuous signal domains like audio or vision.
The analysis in Section 4.1 is not convincing: E.g. in equation (1) the weights stem from interactions between tokens from different time steps, while in eq. (2) the weights come from standard basis functions only depend on the time and not on the inputs themselves.

As TransFourier is introduced as a language model, I would have expected a discussion on how it can be used for generative tasks or in general for decoding. The evals as well as the implementation in the code is purely log-likelihood based.

The ablation on the choice of the feedforward layers seems misplaced. The main intervention of the paper is at the sequence mixing component so, I would expect ablations on hyperparameters regarding TransFourier.

Finally, the overall method and design is very similar to long convolutions introduced in Hyena (https://arxiv.org/abs/2302.10866) and H3 (https://arxiv.org/abs/2212.14052). I am missing a relation to these works. In addition the causal implementation of long convolutions has already been shown in Hyena (and potentially even before).

### Questions
See weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The main contribution of this work is a novel convolution-based sequence mixer algorithm which aims at replacing Transformer module and operates with reduced $O(L \log L)$ computational complexity. It includes a sequence-length-long convolution kernel with the weight being a function of inputs, rather than just some learnable parameter. That direct dependency on the input is an innovation I have not seen elsewhere in the literature. 

The work is promising but there are some problems I’ve described in *weaknesses*, notably scarcity of ablations and incomplete validation. If my concerns are resolved, I’d be happy to increase my rating.

### Strengths
1)This work is highly relevant as it aims to address a pressing problem of algorithmic inefficiency currently dominating Transformer architecture on long sequences.

2) The method is novel and sufficiently distinct from previous attempts to adopt Fourier Transform as a sequence mixer, which I am aware of.

3) The code is provided, its implementation is correct and aligns with the explanations from the paper. A suggestion, by the way – you could include the code of `MultiHeadFourier` class as a listing in the Appendix for completeness and self-containedness.

4) A major advantage is that the architecture doesn’t require hardware-specific custom CUDA kernels and is formulated on several dozens of lines of code in PyTorch.

### Weaknesses
**Ablations**

The core sequence mixing mechanism can be described with the following equations:

$g(t) = f_1(x_t), v(t) = f_2(x_t), out(t)=\sum_{k=0}^{t} v(k) g(t-k)$

It means that for each subsequence ending with timestamp t, the corresponding subsequence $\{g(0), …, g(t)\}$ is reversed before element-wise multiplication with $\{v(0), …, v(t)\}$, and element $g(t)$ always interacts with $v(0)$, $g(t-1)$ with $v(1)$ and so on. Basically, the opposite elements of the subsequences $v$ and $g$ interact multiplicatively while elements close in time (like $v(t)$ and $g(t-1)$) don’t. It’s surprising and hard to interpret, especially given that well-established alternatives, such as Sliding Window Attention and SSMs (such as Mamba) have recency bias. 

I believe this trait of the architecture requires further exploration and analysis of why it works competitively this way. Perhaps, an ablation of what would happen if you replace convolution by cross-correlation ($[g(0), g(1), …, g(t)] \cdot [v(0), v(1), …, v(t)]$)  or shifted cross-correlation ($[0, g(0), …, g(t-1)] \cdot [v(0), v(1), …, v(t)]$) could benefit the work and event produce even better empirical results.

The method uses short convolutions on two separate occasions. This brings additional complexity to the TransFourier module, and it would be informative to ablate their impact.

**Scope of validation**

GPT-2 is an old-gen Transformer architecture with fixed positional embeddings. Llama (or Transformer++ as it’s called in many papers, including https://arxiv.org/pdf/2405.21060) is a much stronger baseline. It has Rotary Positional Embeddings (RoPE) which allow it to extrapolate to any sequence length. All recent sub-quadratic architectures, including Mamba 1 and 2, GLA, Gated Delta Net, XLSTM, use this specific Transformer baseline to compare with. 

Using GPT-2 is also not entirely fair because it uses standard MLP as FFN, as opposed to SwiGLU in TransFourier and Llama. As you observed in Table 3, SwiGLU brings tangible performance gains.

There is no comparison with another strong baseline – GatedDeltaNet (https://openreview.net/forum?id=r8H7xhYPwz). 

Table 2 provides only a subset of benchmarks usually used for testing autoregressive language models of moderate size. Such tests as Lambada (perplexity and accuracy), PIQA, SIQA, and BoolQ are omitted.

As the main advantage of sub-quadratic runtime architectures is their speed for processing  long contexts, it needs to be confirmed that performance doesn’t degrade on such contexts. But the TransFourier model was pre-trained on maximum context size 1024, according to the paper. It needs validation on longer contexts. 

Specifically, I would suggest pre-training a TransFourier model with 340M parameters on 4K context size with 15B tokens from FineWeb-edu. Having accomplished this, you could compare the model pre-trained on a longer context size with a wider variety of architectures from https://arxiv.org/pdf/2504.13173 including GatedDeltaNet and Transformer++, without the need to also pre-train those architectures.

No numerical results were provided in section **5.3 LONG-CONTEXT EVALUATION: NEEDLE IN A HAYSTACK TEST**. I suggest demonstrating these results even if they are incomprehensible, and testing the models on recall intensive benchmarks, used in the GLA paper (https://arxiv.org/pdf/2312.06635), such as MQAR, FDA, SWDE, and SQUAD. Please also consider length extrapolation tests, as in Figure 5 of the GLA paper.

There are no speed comparisons for inference/ training with other tested models on different sequence lengths, despite that reduced computational complexity of the new architecture should be put forward and showcased as its primary advantage. If the TransFourier model written in high-level PyTorch is indeed faster than even some of low-level CUDA implementations (Flash Attention, Mamba, Flash Linear Attention for GLA and GatedDelta Net), then it spells a great advantage of your architecture. 

In conclusion, I encourage you to conduct additional experiments and ablations I mentioned, at least partially, and to report the results even if they are negative. Negative results are as important and valuable as new SOTAs. 


**Miscellaneous:**

* Table 1  – it would be helpful to also include parameter count for each of the configurations.
* The work doesn’t discuss quite relevant algorithms – Long Convolutions (https://arxiv.org/abs/2302.06646) and Monarch Mixer (https://arxiv.org/abs/2310.12109). 
* It would be helpful to describe the architecture as a series of math equations in addition to a lengthy textual description 214-246.
* Line 222 – Step 2: Preparing Gated Signals. Presented architecture doesn’t involve gating. SiLU activation itself is not a part of a gating mechanism unless it’s elementwise-mupliplied with

### Questions
See weaknesses. Also, a couple of questions:

1) Is it possible to incorporate FlashFFTConv kernel into your architecture (https://arxiv.org/abs/2311.05908) to further accelerate it on CUDA devices?
2) This work focuses on autoregressive language modeling. But is it possible to adapt the architecture to bidirectional modeling?
3) Which numerical format did you use for pre-training (fp32, bfloat16, etc.)?

### Soundness
3

### Presentation
2

### Contribution
3
