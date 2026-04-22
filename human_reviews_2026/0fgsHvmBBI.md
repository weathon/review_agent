# AutoSP: Unlocking Long-Context LLM Training Via Compiler-Based Sequence Parallelism

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 2

## Abstract
Large-language-models (LLMs) demonstrate enormous utility in long-context tasks which require processing prompts that consist of tens to hundreds of thousands of tokens. However, existing LLM training libraries do not provide easy to use abstractions to optimize for long-context training, instead focusing on optimizations for models with large parameter counts through ZeRO-3/FSDP, Tensor and Pipeline parallelism. This forces users to rewrite LLM training libraries to incorporate compositions of various complex long-context optimizations, such as sequence-parallelism, to training pipelines; a process that requires in-depth expertise, reducing developer productivity. To tackle these challenges, we introduce AutoSP: the first automated solution to automatically optimize LLM training for longer-contexts. AutoSP compiles models and applies a targeted set of optimizations: automated sequence parallelism, and long-context aware activation-checkpointing, to drastically enhance LLM trainability at negligible cost to throughput. Our evaluation demonstrates AutoSP's capability on both NVIDIA and AMD hardware, increasing training contexts by upto 2.7$\times$ and 2.5$\times$ respectively over competitive hand-written baseline at negligible cost to runtime performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The compilation of a PyTorch 2.0 model goes through multiple stages. Dynamo, the stage responsible for running the model for the first time and recording the computation graph, can be split into two passes: Torch-IR (a higher-level graph that still resembles model layers) and Aten-IR (a lower-level graph of primitive tensor ops, e.g. matrix multiplies, permutes, convolutions, data moves). The authors present a PyTorch compiler patch integrated into the DeepSeek library that implements two optimizations:

1. Takes the implementation of sequence parallelism from DeepSpeed-Ulysses into the Torch-IR pass, so that the attention layers can be automatically distributed across GPUs for parallel processing of long context sequences. The compilerized version works with arbitrary PyTorch models with minimal code changes.

2. Adds a layer of optimization to the Aten-IR pass, optimizing activation checkpointing (AC). They show that PyTorch’s stock AC is too conservative for long-sequence training because it forbids rematerializing classically compute-heavy ops (e.g., matmuls, convs). They show that as sequence length grows, MLP/linear matmuls constitute a vanishing fraction of total FLOPs, so these “heavy” ops can be rematerialized cheaply. By removing this restriction, they enable a longer context to fit into the memory.

They conduct an extensive empirical study, demonstrating that their implementation 
- is compatible with different types of hardware (NVIDEA, AMD), 
- allows long sequences to fit into memory when compared to ZeRO-3 and hand-written DS-Ulysses, 
- Scales well with SP group size

### Strengths
The greatest value of this work is that it automates the implementation of SP for practitioners, requiring minimal code changes. This has the potential to accelerate research and development of language model architectures with a wide impact. The authors present an original contribution of an optimized activation checkpointing strategy and provide empirical results that demonstrate the effectiveness of the proposed changes. The paper is clearly written and provides sufficient documentation on the empirical experiments.

### Weaknesses
- The paper lacks low-level technical details about how AutoSP manipulates Torch-IR in complex models beyond toy examples. The authors should make sure this is addressed during the release of the source code.
- The paper overemphasizes the benefits of AutoSP without detailing potential downsides, such as the scenarios where the overhead of recomputation in AC becomes substantial.

### Questions
- Comparing AutoSP to other context-length extension strategies (e.g., HuggingFace Accelerate with Context Parallelism) would make the results more trustworthy

### Soundness
4

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
2

### Summary
This paper introduces AutoSP, a novel compiler-based system for PyTorch-2.0 that automates optimizations for training Large Language Models (LLMs) with very long contexts.

### Strengths
The article's writing is commendable, particularly for its very clear and accessible explanations of technical details.

### Weaknesses
1.  I am skeptical of the core claim that context parallelism is difficult to implement. An API-based approach, inspired by Flash Attention (e.g., `flash_attn_func(query, key, value)`), seems more suitable than integrating it to `torch.compile()`. While also a one-line modification, this API provides users with greater transparency and explicit control, rather than obscuring the underlying logic.

2.  The comparison between the proposed compile-stage activation checkpointing and traditional layer-wise checkpointing is vague. The paper primarily demonstrates advantages over a standard `torch.compile()` baseline but fails to clearly articulate its benefits over conventional layer-wise gradient checkpointing. This is a significant omission, as the latter is a common, effective, and simple-to-use practice in LLM training.

3.  Most critically, the ideas presented are neither groundbreaking nor particularly novel. They appear to be targeted optimizations for implementing a specific function more efficiently within an existing general purpose framework, rather than a substantive innovation. As such, this contribution seems more appropriate for a pull request to the PyTorch repository than a publication at ICLR.

### Questions
1.  Why was DeepSpeed Ulysses selected as the baseline? Does it exclusively support eager-mode, $O(N^2)$ attention? If so, it is an inappropriate baseline. Standard $O(N^2)$ attention is outdated; modern implementations using Flash Attention or PyTorch SDPA can easily train 16K contexts on a single 80GB GPU, especially with DeepSpeed ZeRO-3 enabled. A valid comparison would require using faster context-parallel (CP) implementations, such as Megatron-LM's CP or Ring Flash Attention [1].

2.  Using `torch.compile()` for activation checkpointing is uncommon in practice. Standard LLM implementations (e.g., in Hugging Face) wrap each layer with `torch.utils.checkpoint`. This standard, LLM-specific approach should be discussed. The official PyTorch API is not complex:

    ```python
    torch.utils.checkpoint.checkpoint(layer_function, *inputs, use_reentrant=False)
    ```
    Please clarify the efficiency difference between your activation checkpointing method and this standard `torch.utils.checkpoint` implementation.

[1] https://github.com/zhuzilin/ring-flash-attention

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces two compiler passes to achieve sequence parallel. It inserts all-to-all collectives and transform single GPU model into sequence parallelsized models. It also consider activation checkpointing that selectively rematerializes activations by finding best memory and perf trade offs. Similar to existing work like simpleFSDP and deepcompile, it capture the graph single GPU model and transform the IR with collectives and IR passes. It mains simple UX and single-GPU style authoring.

### Strengths
fair originality: this is indeed the 1st paper I saw to use compiler pass to achieve sequence parallel. DeepSpeed-Ulysses and RingAttention implement SP at the framework level that affects the single gpu authoring. This paper lifts SP into the compiler layer. Joint optimization with activation checkpointing is non-trivial and essential to make good memory perf trade offs. 

enough quality: it touched the reasoning behind choosing which layer of IRs, and how collectives are inserted and optimized for better perf.   Evaluation is done on both NVIDIA and AMD GPUs, including eager and compiler baselines

descent clarify: The main technical sections (Sections 3.1–3.2) may challenge readers unfamiliar with PyTorch internals. Fortunately figures and intros effectively lowed the bar to keep up with the content.

highly significant: it's critical to showcase how to use IR passes to achieve SP. Long-context LLM training is a critical challenge. It's espectially benifitial to minimal developer friction. This could potentially influence future designs of pytorch compiler solution for parallelsims, like simpleFSDP

### Weaknesses
originiality mainly comes from using IR to implement SP. SP itself was previously implemented in open source library like DeepSpeed-Ulysses. There are also similar work that moves FSDP into IR passes. This paper is more like an extension of the idea to more parallelsims. 

technical depth: it remains questionable how non-trivial it is to come up with IR passes to achieve SP, considering we have open source implementation in eager mode. The non-trivial evaluation should be done for people with enough understanding of pytorch 2 compiler stack. But I agree joint optimization with activation checkpoint is non-trivial

### Questions
Explain why it's non-trivial to come up with IR passes according to open source eager SP implementation like DeepSpeed-Ulysses

For maximum sequence length, analyze memory snapshot for each baseline and show more insights into memory usage: when did the peak happen, % of memory on model/opt state, activation, and intermidate tensors

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces AutoSP, a compiler-based system designed to optimize the training of large language models (LLMs) for long-context scenarios. AutoSP automates sequence parallelism and activation checkpointing specifically tailored for long-contexts, aiming to enhance trainability without sacrificing runtime performance. The system is evaluated on both NVIDIA and AMD hardware, showing significant improvements in trainable sequence lengths compared to existing methods.

### Strengths
The paper presents a compiler-based solution for optimizing LLM training in long-context scenarios, which maybe easy to use.

### Weaknesses
1. Lack of novelty. Automatic parallelism has been thoroughly studied and there are lots of work about the automatic or dynamic sequence parallelism including selective activation checkpointing. This article integrates these elements into the compiler, which more like an engineering project.
 2.The paper's baseline comparisons may not fully represent the current state-of-the-art techniques, particularly in terms of hand-optimized implementations. 
3.Lots of errors. Such as line 328 , “Grouped-Query-Attention (GQA) or Full-Attention” GQA and Full-attention are completely compatible.

### Questions
see weakness above.

### Soundness
2

### Presentation
2

### Contribution
1
