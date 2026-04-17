# PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
Multimodal large language models (MLLMs) have achieved strong performance on vision-language tasks, yet often suffer from inefficiencies due to redundant visual tokens. Existing token merging methods reduce sequence length but frequently disrupt spatial layouts and temporal continuity by disregarding positional relationships. In this work, we propose a novel encoding operator dubbed as **P**ositional **P**reservation **E**mbedding (**PPE**), which has the main hallmark of preservation of spatiotemporal structure during visual token compression. PPE explicitly introduces the disentangled encoding of 3D positions in the token dimension, enabling each compressed token to encapsulate different positions from multiple original tokens. Furthermore, we show that PPE can effectively support cascade clustering --- a progressive token compression strategy that leads to better performance retention. PPE is a parameter-free and generic operator that can be seamlessly integrated into existing token merging methods without any adjustments. Applied to state-of-the-art token merging framework, PPE achieves consistent improvements of 2\%~5\% across multiple vision-language benchmarks, including MMBench (general vision understanding), TextVQA (layout understanding)  and VideoMME (temporal understanding). These results demonstrate that preserving positional cues is critical for efficient and effective MLLM reasoning. Our code is available at https://github.com/MouxiaoHuang/PPE

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Positional Preservation Embedding, a positional encoding scheme for multimodal LLMs that compress visual tokens without destroying spatial or temporal structure. Standard token merging reduces sequence length by clustering similar visual tokens, but it typically drops most positional IDs, so the merged tokens lose fine-grained layout and temporal continuity. PPE instead packs multiple positional IDs (2D for images, 3D for video) into different embedding sub-dimensions of each merged token, so a single compressed token keeps location/ordering information from several original tokens. PPE is parameter-free, attaches to existing token merging, and supports cascade compression across multiple transformer layers. The authors evaluate PPE on Qwen2.5-VL-3B (fine-tuned) for both image and video understanding.

### Strengths
1. Problem formulation is clear and technical idea is simple and clean, reuse RoPE’s per-dimension independence to “multiplex” multiple spatial / temporal position IDs into one merged token. 
2. Plug-and-play, no new trainable parameters, negligible runtime overhead.
3. Ablations are thoughtful, like where to insert compression, compatibility with other methods, compute/memory cost.

### Weaknesses
My main concern is the empirical evaluation. The current experiments are not sufficient. The method is not compared against several widely used token reduction / compression baselines such as ToMe, FastV, SparseVLM, PyramidDrop, VisionZip, etc. The set of baselines is very limited, which makes it hard to judge relative progress. Also, the set of benchmarks is also limited, many common benchmarks are not included. Please address these points in the rebuttal. I am willing to raise my score if the empirical evaluation is strengthened.

### Questions
1. How does PPE compare quantitatively to other popular token reduction approaches such as ToMe, FastV, SparseVLM, PyramidDrop, and VisionZip?
2. Can the authors expand the evaluation to include more common benchmarks?
3. Does PPE maintain its claimed efficiency gains when applied to larger MLLMs or higher-resolution inputs beyond the 3B model setting?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Position Preservation Embeddings (PPE), a novel position encoding mechanism designed to preserve spatio-temporal structure during visual token compression in multimodal large language models (MLLMs). PPE explicitly encodes multiple position identifiers into a single compressed token by partitioning the embedding dimension into groups of independent storage spaces. This enables the merged token to simultaneously represent multiple original positions. This method requires no parameter tuning, is plug-and-play, and is compatible with existing token merging algorithms like Chat-UniVi and  PACT. Across various visual-language benchmarks, PPE achieves 2–5% performance gains over state-of-the-art methods, maintaining advantages even under extreme compression rates.

### Strengths
1. **Novel Concept**: PPE preserves the spatio-temporal integrity of visual markers by encoding multiple position identifiers within a single compressed marker. Even under high compression ratios, it significantly reduces computational and memory overhead while maintaining performance.
2. **Parameter-free Compatibility**: The PPE strategy requires no additional training parameters or architectural modifications and can be seamlessly integrated into existing large-scale language models and token compression pipelines. Furthermore, PPE demonstrates strong adaptability and generalization capabilities across architectures and foundational models, achieving compatibility with various clustering compression frameworks such as Chat-UniVi, PACT, and ToMe.
3. **Cascade Compression**: This paper systematically extends PPE to multi-level cascaded compression within large language models, demonstrating that the progressive merging technique maintains high accuracy even at token compression rates exceeding 90%. Furthermore, experiments reveal that PPE achieves stable quantization gains while preserving overall computational efficiency, offering significant practical value for large-scale multimodal inference.
4. **Comprehensive Evaluation**: Across extensive assessments of image and video tasks, PPE demonstrates remarkable effectiveness in preserving accuracy while maintaining inference efficiency. This paper designs rich ablation experiments including K-value analysis, rate trade-offs, and attention visualization, which robustly validate the design decisions underlying PPE's token compression process.

### Weaknesses
1. **Benchmark Coverage for Layout-Sensitive Tasks**: The current evaluation primarily targets general QA-type tasks (e.g., MMBench, VideoMME). While TextVQA is included, the paper lacks comprehensive validation on benchmarks specifically designed for fine-grained spatial and layout understanding, such as OCR-intensive tasks. It is recommended to incorporate dedicated OCR benchmarks (e.g., OCRBench and DocVQA) to rigorously validate PPE's capability in preserving precise spatial relationships for text-heavy scenarios, which is a core claimed advantage of the method.
2. **Ablation Analysis**: Table 5 presents results for the number of retained positions K = 1, 8, and 24. However, it does not discuss the more granular effects of K on computational complexity or changes in attention distribution. Furthermore, the choice of K lacks universality across different tasks. While the attention visualization in ablation experiments demonstrates qualitative improvements for PPE, a more nuanced interpretive discussion could be developed by integrating spatial contextual relevance.
3. **Related Work Could Be More Comprehensive**: The discussion of visual token merging in Section 2.2 would benefit from incorporating several recent and highly relevant works (e.g., Shang et al. 2025 - Llava-prumerge; Li et al. 2024 – TokenPacker; Yuhang Han et al. 2024 - Filter, Correlate, Compress: Training-Free Token Reduction for MLLM Acceleration; Mark Endo et al. 2024 - Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration; and Chen et al. 2024 - Recoverable compression: A multimodal vision token recovery mechanism guided by text information). Engaging with these methods would help to better position PPE's contributions within the current landscape.
4. **Lack of Detailed Efficiency Analysis**: The paper would be significantly strengthened by a more thorough analysis of computational and memory efficiency. While token reduction inherently implies speedup, quantitative evidence is currently lacking. We recommend providing comprehensive comparisons on key metrics including computational cost (e.g., FLOPs reduction in LLM self-attention modules), memory footprint (peak GPU memory usage during inference), and inference latency (end-to-end inference time or throughput). A comparative analysis against other token reduction methods on these metrics would provide a more complete picture of PPE's practical advantages.

My overall stance on this paper is weak accept. If the authors could provide additional details or point out areas I may have overlooked, I would be willing to further increase my score.

### Questions
Refer to Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the limitations of existing visual token compression methods in MLLMs that disrupt spatial layouts and temporal continuity. The authors propose Positional Preservation Embedding (PPE), a parameter-free positional encoding operator that explicitly retains multiple spatiotemporal positions within each compressed token by splitting RoPE embedding dimensions into K groups, each encoding different position IDs from merged tokens. The method integrates into existing token merging frameworks and supports cascade compression across transformer layers. Experiments on Qwen2.5-VL-3B demonstrate 2-5% improvements across multiple benchmarks (MMBench, TextVQA, VideoMME) while achieving 55-94% token reduction.

### Strengths
1. The paper clearly identifies that existing token compression methods lose fine-grained positional information, with compelling evidence (Figure 1, attention visualizations) showing how this impacts layout-sensitive tasks.

2. The authors provide thorough ablation studies covering K values, reduction ratios, cascade compression strategies, and include attention visualizations and failure case analyses.

3. PPE supports cascade compression, enabling 90% token reduction while maintaining performance through multi-stage processing.

### Weaknesses
1. Critically incomplete baseline comparisons: The paper lacks comparisons with several mainstream token compression methods like FastV, VisionZip, MustDrop, and TokenCarve[1-4].

2. Unvalidated position correspondence assumption: The core assumption that compressed tokens retain a meaningful correspondence to original image positions after vision encoder processing (ViT layers, pooling) is not validated and is questionable for many architectures. This is particularly relevant for models like InternVL-3 with its “Tiles+Thumbnail” approach or cross-attention models, where position IDs might not directly map to original spatial locations, fundamentally questioning the meaningfulness of "preserved" positions for the model.

3. Limited architectural generalization: All primary experiments are confined to Qwen2.5-VL-3B and LLaVA-OV-0.5B. On LLaVA-OV-0.5B, experiments show PPE underperforms the Dense baseline (44.74% vs 44.80%). There is no evaluation on other mainstream MLLMs, or larger models (7B+), making PPE’s general applicability uncertain.

4. Confusing experimental setup: The authors claim PPE to be “plug-and-play”, but conduct SFT to obtain a dense fine-tuned model. The paper lacks the validation of PPE on the official pre-trained Qwen2.5-VL-3B model and may compromise the reproducibility and fairness of the experimental comparisons.

[1] An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models

[2] Visionzip: Longer is better but not necessary in vision language models

[3] Multi-stage vision token dropping: Towards efficient multimodal large language model

[4] Tokencarve: Information-preserving visual token compression in multimodal large language models

### Questions
1. Insufficient theoretical justification: The choice of K based on GCD of M-RoPE sections is presented as a design principle but functions more as a constraint, and there's no formal analysis of information preservation or a clear definition of what "77% IDs retained" means.

2. Unclear Result Presentation and Incorrect Marking: The tables in the paper fail to specify the meaning of bold and underlined markings. Additionally, there is no explanation of whether higher values or lower values of the metrics are better. As a result, the results in many tables are extremely difficult to understand. For instance, in Table 8, the highest result for the "Short" metric is bolded, while the highest results for other metrics are underlined.

### Soundness
2

### Presentation
2

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
This paper proposes PPE to enhance visual token compression in vision-language model. Instead of assigning a single positional ID to a merged token, PPE encodes multiple positional cues by splitting embedding dimensions and preserving RoPE structure. The method enables high-ratio token compression and supports cascade compression across transformer layers.

Experiments show consistent improvements over clustering-based compression (e.g., Chat-UniVi), especially under aggressive reduction settings. PPE is simple, model-agnostic though current results are limited to clustering-based methods and evaluation breadth.

### Strengths
1. Motivation is clear: positional degradation under token compression.

2. Good solution: leveraging RoPE dimension independence.

3. Strong efficiency gains with competitive accuracy at high compression ratios.

### Weaknesses
1. Evaluated mainly with clustering (DPC-KNN); no results with learning-based compression.

2. Unclear applicability to ALiBi or other positional schemes.

3. More evaluation is needed, in the paper, layout-heavy or OCR-centric tasks partially explored.

### Questions
1. PPE relies on RoPE’s dimension-wise rotational independence. How would this method generalize to non-RoPE architectures (e.g., ALiBi, learned absolute embeddings)?

2. Experiments focus on clustering-based token merging. Have you evaluated PPE with learned token selection/aggregation methods?

3. Have you explored other position grouping strategies?

### Soundness
3

### Presentation
3

### Contribution
3
