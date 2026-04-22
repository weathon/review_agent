# Explore More, Learn Better: Parallel MLLM Embeddings under Mutual Information Minimization

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Embedding models are a cornerstone of modern AI. Driven by Multimodal Large Language Models (MLLMs), they have made great progress in architecture and data curation, while the holistic paradigm is still limited to SSC, {\em i.e.}, single input, singular embedding, contrastive supervision, which collapses rich, multifaceted inputs into monolithic embeddings and fails to fully exploit MLLM capabilities. In this paper, we tailor one \textbf{P}arallel \textbf{D}ecoupling \textbf{F}ramework (PDF) for multimodal embedding learning, by utilizing the proprietary steerability of MLLMs, {\em i.e.}, their ability to flexibly generate quite differentiated response under explicit instructions. Concretely, PDF conditions a shared MLLM backbone on distinct, learnable prefixes to roll out multiple parallel paths for one input, then relies on these paths to obtain parallel embeddings. To promote full parallel diversity, we employ Mutual Information Minimization (MIM) as an explicit constraint, coupled with per-path contrastive supervision to maintain semantic alignment. Such dual-objectives force PDF to yield robust semantic coverage and a generalizable embedding space. Ultimately, the remarkable embedding space are accessible at inference via one single forward pass, incurring negligible computational overhead. We instantiate PDF on multiple MLLM backbones and prove its effectiveness on MMEB benchmark. Significant gains are consistently achieved across various resolutions and model sizes, {\em e.g.}, boosting the VLM2Vec-LLaVA-1.6-LR model by a remarkable +8.9\% (7B), and the VLM2Vec-Qwen2VL models by +4.2\% (2B) and +3.1\% (7B). In terms of efficiency, our 2B model surpasses its baseline by +2.6\% using only half the computational budget. Code will be available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel Parallel Decoupling Framework (PDF) designed to improve the training of MLLMs for embedding learning. The main idea is to exploit the steerability of MLLMs by generating multiple parallel embeddings for a single input. This approach is realized by conditioning the MLLM backbone on distinct, learnable prefixes, which in turn create multiple parallel paths for embedding generation. Here is the core contributions of the paper:

1) Parallel Embedding Generation: The paper proposes using a shared MLLM backbone with multiple parallel paths, each conditioned by unique prefixes. This allows the model to generate diverse embeddings, fully utilizing the input's semantic richness.

2) Mutual Information Minimization (MIM): To avoid collapse into redundant embeddings, the paper introduces a mutual information minimization objective. This explicitly minimizes statistical dependencies between parallel embeddings, enhancing diversity and ensuring robust semantic coverage.

3) Training Efficiency: The framework introduces minimal computational overhead during inference, as the system can select a single path for inference, maintaining the benefits of parallel training without extra cost.

4) Empirical Validation: The PDF framework is instantiated on several MLLM backbones, such as VLM2Vec with LLaVA and Qwen2VL, achieving significant performance improvements across multiple benchmarks (e.g., MMEB), with gains ranging from +4.2% to +8.9% over baseline models. The model also shows superior training efficiency, surpassing the baseline in fewer iterations and with reduced computational budget.

### Strengths
The paper’s strengths span multiple dimensions. In terms of originality, it presents a fresh reformulation of multimodal embedding learning by replacing the conventional Single input–Singular embedding–Contrastive supervision (SSC) paradigm with the proposed Single input–Parallel paths–Parallel outputs (SPP) paradigm. This idea, realizing multiple, decorrelated embeddings via deep prefix injection and mutual information minimization, creatively leverages the inherent steerability of MLLMs, representing a novel and well-motivated contribution. In terms of quality, the technical design is rigorous: the dual-objective optimization (contrastive loss + variational MIM) is theoretically grounded, experimentally validated across multiple MLLM backbones, and supported by comprehensive ablation, efficiency, and zero-shot evaluations. In addition, the paper is well organized, with clear figures and consistent notations that make complex mechanisms (e.g., prefix-based parallel paths and MI estimation) easy to follow. Regarding significance, the proposed framework achieves notable gains (up to +8.9% on MMEB) while requiring negligible inference overhead, suggesting strong practical impact and potential influence on future MLLM embedding research. Overall, the work is a empirically convincing contribution to representation learning for multimodal foundation models

### Weaknesses
1. Most experiments are conducted on the MMEB benchmark, with limited evaluation across other multimodal embedding datasets (e.g., MSCOCO, LAION-Aesthetics, or ImageNet-Text retrieval). This restricts the evidence for generalization and may introduce benchmark-specific bias. Adding results on diverse domains would better demonstrate robustness and transferability.

2. The paper proposes learnable prefixes for parallel embedding generation but provides little insight into how prefix dimensionality, initialization, or number of paths 𝑁 affect performance. A systematic analysis (e.g., scaling 𝑁=2, 4, 8) could clarify whether improvements stem from architectural diversity or training regularization.

3. The framework is demonstrated for multimodal retrieval and classification, but its applicability to downstream generation or reasoning tasks is not discussed. Since MLLMs are often used in generative settings, evaluating or at least discussing potential extensions to instruction-following or captioning would broaden the paper’s impact.

4. The baselines largely include VLM2Vec and a few classic models (e.g., CLIP, BLIP2). However, more recent multimodal embedding approaches (e.g., MM-VLM, EVA-CLIP, E5-V, MME5) are not included or discussed in depth. A more thorough comparative evaluation would improve contextual positioning and claim of novelty.

5. Every method has limitations. For example, perhaps in data-scarce settings, adding many paths could over­fit; or perhaps for very simple inputs, parallel embeddings may be redundant. The paper would improve by including a short discussion of when the proposed method might not offer gains (and perhaps empirical evidence of such cases).

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

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
The key idea is to replace the standard “single input–single embedding” paradigm with a deep prefix injection mechanism that conditions multiple parallel paths through learnable prefix vectors injected at each transformer layer.

### Strengths
1. The deep prefix injection design is a clever extension of prefix-tuning that operates at every layer, rather than only at the input, offering a plausible mechanism to steer MLLMs toward richer, more diverse embedding spaces.
2. The authors present extensive benchmarks on multiple model scales and backbones

### Weaknesses
1. Lack of evidence that deep prefix injection truly induces semantic diversity. There is no analysis of attention distributions, token contributions, or feature subspace diversity.
2. Table 3 shows nearly identical performance between Single Prefix and Aggregate inference strategies. If the learned embeddings are genuinely diverse, aggregation should bring at least a small improvement. This raises reasonable doubt that the parallel paths have collapsed or that diversity is not reflected in performance.
3. No demonstration that deep injection outperforms prompt-based methods. MetaEoL [1] is relevant baseline. Without a direct comparison, it is unclear whether the reported improvements arise from deep injection itself or from generic multi-prompt effects. Moreover, prompt-based methods usually offer better interpretability—something the current approach lacks.

[1] Meta-Task Prompting Elicits Embeddings from Large Language Models

### Questions
1. The scores in Table 3 are almost identical. Is this expected? Please report the individual performance of each path to clarify whether the embeddings are genuinely diverse. If diversity exists, why does aggregation bring no additional benefit?
2. To substantiate the “different subspace / computation trajectory” claim, please include stronger evidence.
3. Please provide quantitative comparison with MetaEoL or similar prefix/prompting methods under the same setup. This would clarify whether “deep injection” offers real gains beyond prompting and whether the improvements come from deeper modulation or simply from prompt-like effects.

### Soundness
2

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
3

### Summary
This paper introduces the Parallel Decoupling Framework (PDF): a single MLLM backbone conditioned by learnable deep prefixes to produce multiple parallel embeddings for one input. Diversity is enforced via Mutual Information Minimization (vCLUB estimator) while each path also receives a contrastive objective. At inference, a single forward pass yields an aggregated embedding. PDF improves MMEB results on VLM2Vec-LLaVA-1.6 (+8.9 points, 7B) and Qwen2VL (+4.2 at 2B; +3.1 at 7B), with efficiency gains.

### Strengths
- Parallel prefix-conditioned paths + explicit MI minimization within MLLMs.
- Gains across tasks/scales, including compute-reduced settings.
- Clear framing (SSC→SPP) and conceptual diagrams.
- Practical recipe for better MLLM embeddings with negligible inference overhead.

### Weaknesses
- Possible instability/bias of vCLUB MI estimation; lack of analysis on estimator variance and its effect on training.
- Limited ablation on #paths, prefix depth/placement, and aggregation design.
- Generalization beyond MMEB (e.g., long-form retrieval/RAG latency-quality tradeoffs) not explored.

### Questions
- How does performance scale with number of parallel paths and MI weight?
- Does MI minimization ever hurt semantic alignment (failure cases)?
- Can the aggregated embedding be replaced by path-specific retrieval (multi-vector index), and what are latency implications?

### Soundness
3

### Presentation
3

### Contribution
3
