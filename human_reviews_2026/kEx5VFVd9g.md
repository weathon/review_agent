# Efficient Fine-tuning via Auxiliary Representation

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
The widespread adoption of large pretrained models has made fine-tuning an essential step for tailoring models to specific tasks. As these models continue to scale larger and as the demand for task-specific and personalized adaptation grows, parameter-efficient fine-tuning (PEFT) has emerged as a practical alternative to full fine-tuning. PEFT enables effective adaptation while updating only a small fraction of the total parameters. While various PEFT techniques have shown strong performance, many still suffer from increased inference latency and inefficiencies in multi-adapter scenarios. Motivated by these limitations, we propose a novel PEFT approach that leverages auxiliary representations to enable fast and flexible inference. In our method, Latent Task Embedding fine-tuning, a small task-specific latent embedding is concatenated to the original embedding. The corresponding weight matrices are extended, and only the additional parameters introduced by this expansion are trained. This design allows for efficient inference using a single matrix multiplication per weight, minimizing latency overhead, and supports task-specific masking to handle multiple adapters within a single model. We evaluate our method on large language models and latent diffusion models, demonstrating competitive accuracy with existing PEFT baselines while providing faster inference and enabling efficient intra-batch multi-task processing.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Latent Task Embedding (LTE), a novel parameter-efficient fine-tuning framework. Specifically, LTE prepends a small task-specific latent vector to the original embeddings and correspondingly extends each weight matrix so that only the newly introduced parameters are trained, enabling single-matrix-multiplication inference and task-wise masking—differentiating it from prior PEFT methods that rely on fully shared or multi-stage computations.

### Strengths
1. Extensive experiments on language and vision tasks show its effectiveness.

2. the paper is well written and easy to follow

### Weaknesses
W1. Lack of novelty. Despite its motivation, the contribution of this work appears limited. Widening the original model’s width is not a new idea; on the contrary, it is quite straightforward and has been discussed in net2net (ICLR 2016) and LLaMA-Pro (2024).

W2. The improvement is marginal. Under the same parameter budget, latTE shows less than a 0.5% improvement over lora across multiple datasets.

W3. Although the authors claim their method has lower latency than unmerged lora, lora achieves the fastest speed after merging. Moreover, the authors do not discuss coupling with common inference acceleration techniques, such as weight quantization.

### Questions
Please see weaknesses W1~3.

Minor:
Training efficiency. Under the same parameter budget as LoRA, are the training-time memory usage and the time per iteration similar?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LatTE, a new PEFT method that introduces a small auxiliary latent embedding concatenated to the original input embeddings. During fine-tuning, only the parameters associated with this expanded embedding are updated, while the base model remains frozen.
Unlike traditional methods such as LoRA or adapters, LatTE maintains single-matrix-multiplication inference, achieving comparable accuracy with minimal latency overhead. The paper demonstrates LatTE’s effectiveness across both LLMs and diffusion models, reporting competitive or superior performance to baselines like LoRA, BOFT, and OFT.
The approach also supports task-specific masking for multi-adapter and multi-task inference, making it scalable for personalized or edge-device deployment scenarios.

### Strengths
1. Instead of updating weights (LoRA) or inserting modules (adapters), LatTE introduces auxiliary representation-level adaptation, expanding the embedding space while keeping inference cost minimal.

2. The design ensures single matrix multiplication per layer without additional latency, which is a well-motivated improvement for deployment and edge use cases.

3. The extension of LatTE to both NLP and vision-generation tasks (text-to-image) demonstrates generality beyond transformer-based text models.

### Weaknesses
1. While the paper reports that LatTE “matches or exceeds” existing PEFT baselines, most observed improvements are marginal (e.g., Qwen2.5-3B in Table 1: 86.63 vs. 86.60 for LoRA). Across many tasks, the best results are still obtained by other methods such as OFT or BOFT. Given the small numerical gaps and lack of confidence intervals or statistical tests, it is difficult to conclude that LatTE offers a consistently superior adaptation performance rather than random variation.

2. All experiments are conducted on relatively small or medium-sized backbones (≤ 8 B parameters). Since PEFT is mainly motivated by the impracticality of full fine-tuning in very large models (tens or hundreds of billions of parameters), the absence of larger-scale experiments leaves open whether LatTE remains effective and efficient when model size grows substantially.

### Questions
1. Why do the model sizes used in Table 1 (up to 8 B) and Table 2 (up to 3 B) differ? This inconsistency makes it difficult to compare results or assess how the method behaves across model scales and task types

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
The paper proposes Latent Task Embedding, a parameter-efficient fine-tuning (PEFT) approach that concatenates a small task-specific latent embedding to model inputs, expands the associated projection matrices, and trains only the introduced parameters. The claim is that LatTE achieves PEFT-level parameter efficiency while preserving single-matrix-multiplication inference (low latency) and enabling efficient multi-adapter composition via task-specific masking. Evaluation is reported on LLMs and latent diffusion models, with theoretical analysis supporting equivalence or bounded error relative to related PEFT methods.

### Strengths
Clear, simple construction: concatenating a learned low-dimensional task embedding and expanding projection weights is easy to implement and reason about. 

Practical motivation: latency in multi-adapter scenarios is an important deployment concern; addressing it directly is valuable. 

Empirical breadth: experiments on both autoregressive LLM tasks and latent diffusion models (reported) show broad applicability.

### Weaknesses
Theoretical arguments rely on informal equivalence / bounded-difference claims but do not fully characterize when LatTE can fail (e.g., when task embeddings need to encode highly non-linear, high-rank corrections). The bounds lack discussion of constants and dependence on embedding dimension. 

Inference-latency claims are asserted as single-matrix-multiplication equivalence, but the paper under-specifies real-system measurements (how expanded matrices affect cache locality, memory bandwidth, precision trade-offs). No detailed ablation on embedding size vs. latency/accuracy trade-off.

Comparison to some very recent PEFT variants (that also target inference efficiency) is limited: authors should benchmark against the most recent low-latency schemes and provide wall-clock latency/memory breakdowns.

### Questions
What are the formal assumptions behind the theoretical bounds? For example, do bounds assume small-norm embeddings or specific activation linearity approximations? Please state exact assumptions and constants. 

How does your method perform when the required task adaptation is not well-approximated by augmenting input-space (i.e., when adaptation needs internal layer reparametrization)? Provide failure cases or diagnostics.

Please provide precise latency/memory benchmarking (hardware, batch sizes, eff. throughput) and show embedding-dimension vs. accuracy/latency curves.

### Soundness
2

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
To improve the inference latency and efficiencies in multi-adapter scenarios of Low rank adaptors methods, the paper propose method, Latent Task Embedding fine-tuning, a small task-specific latent embedding is concatenated to the original embedding. The corresponding weight matrices are extended, and only the additional parameters introduced by this expansion are trained. This design aims to achieve better efficient inference using a single matrix multiplication per weight, minimizing latency overhead, and supports task-specific masking to handle multiple adapters within a single model.

### Strengths
- The paper is clearly presented and easy to follow, with only a few minor typos.
- The proposed LatTE method is conceptually straightforward and practically implementable.
- The experiments are comprehensive, covering LLM fine-tuning across QA, reasoning, and diffusion tasks. LatTE achieves comparable performance to existing PEFT methods such as LoRA on models including LLaMA and Qwen.

### Weaknesses
- The reported improvements in performance and inference latency over baseline LoRA are modest and may not convincingly demonstrate practical advantages.

- The discussion of related work is limited in scope, particularly regarding sparsity-based methods in the parameter-efficient fine-tuning (PEFT) literature.

- In PEFT, existing approaches typically fall into two categories:
(1) Low-rank adaptation methods (e.g., LoRA) and
(2) Subset or sparsity-based fine-tuning methods, which selectively update parts of the model parameters.
The related work section would benefit from a clearer categorization and inclusion of recent sparsity-based works such as:
Separating [4–8] from module- or weight-based methods and creating a distinct subsection on sparsity-based PEFT would strengthen the paper’s positioning and contextual completeness. Adding a discussion of the sparsity methods in the related work section with recent papers[1-3] would strengthen the contextual foundation of this paper since they are emerging trends in PEFT field.



### Typos:

“the third category focus” → “the third category focuses”
“utilize auxiliary latent embedding” → “utilizes auxiliary latent embeddings”
* “Another group in this category modify or edit” → “modifies or edits”
* “combines LoRA with pruning or quantization” → remove “s” → “combine LoRA with pruning or quantization”
* “We now ready to implement it” → “We are now ready to implement it”

* “reasoning tasks additionally requires” → “reasoning tasks additionally require”
* “the idential positions” → “the identical positions”
* “LoRA and LatTE both shows” → “both show”
* “trainable parameters closely matches” → “closely match”



[1] Scaling Sparse Fine-Tuning to Large Language Models

[2] Sparse Matrix in Large Language Model Fine-Tuning

[3] The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

[4] Parameter-Efficient Fine-Tuning without Introducing New Latency

[5] Parameter-Efficient Transfer Learning with Diff Pruning

[6] Diff Pruning: Parameter-Efficient Transfer Learning with Diff Pruning

[7] Training Neural Networks with Fixed Sparse Masks

[8] Composable Sparse Fine-Tuning for Cross-Lingual Transfer

### Questions
- Latency Motivation:
The paper argues that LoRA introduces latency due to additional adapters. Could the authors clarify why these adapters are a latency bottleneck? Each adapter involves small matrix multiplications per layer, which typically contribute marginally to total inference time. Moreover, LatTE also introduces extra embedding transformations (𝑓_in and 𝑓_out), which are applied to every FFN layer. Why, then, is LatTE expected to reduce latency rather than increase it?

- Latency Profiling:
In line 454, the paper reports latency measurements for generating 100 tokens with a 10k context length, averaged over 10 trials. However, the observed improvement seems trivial. Could the authors provide more detailed time profiling (e.g., breakdown by attention, FFN, embedding) to demonstrate the latency behavior of LatTE versus LoRA?

- Architectural Overhead:
Both “more-heads” and “wider-heads” LatTE variants introduce additional projections, which likely increase inference time. Is there quantitative evidence or profiling that measures the actual latency introduced by these modifications?

- Initialization Strategy:
In line 194, the paper states that LatTE initializes the embedding with a constant value. What initialization strategy is used for the additional matrix dimensions (𝐴 and 𝐵)? Initialization plays a crucial role in the performance of low-rank methods, and more discussion on initialization sensitivity would be valuable.

I would like to discuss the questions I raised regarding the weaknesses and concerns with the authors. If my concerns are adequately addressed, I would be willing to reconsider my rating.

### Soundness
2

### Presentation
3

### Contribution
2
