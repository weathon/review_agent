# EventFlash: Towards Efficient MLLMs for Event-Based Vision

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Event-based multimodal large language models (MLLMs) enable robust perception in high-speed and low-light scenarios, addressing key limitations of frame-based MLLMs. However, current event-based MLLMs often rely on dense image-like processing paradigms, overlooking the spatiotemporal sparsity of event streams and resulting in high computational cost. In this paper, we propose EventFlash, the first efficient MLLM to explore spatiotemporal token sparsification for reducing data redundancy and accelerating inference. Technically, we first build EventMind, a large-scale and scene-diverse dataset with over 500k instruction sets, providing both short and long event stream sequences to support our curriculum training strategy. Then, we present the adaptive temporal window aggregation module for efficient temporal sampling, which adaptively compresses temporal tokens while retaining key temporal cues. Finally, the sparse density-guided attention module is designed to improve spatial token efficiency by selecting informative regions and suppressing empty or sparse areas. Experimental results show that EventFlash achieves a 12.4x throughput improvement over the baseline (EventFlash-Zero) while maintaining comparable performance. It supports long-range event stream processing with up to 1,000 bins, significantly outperforming EventGPT’s 5-bin limit. We believe EventFlash serves as an efficient foundation model for event-based vision.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an event-based multimodal large model named EventFlash. The method explores reducing data redundancy and accelerating inference for event cameras through spatiotemporal token sparsification. It introduces an adaptive temporal window aggregation module and a sparse density-guided attention module to perform efficient sampling and selection of spatiotemporal information. In addition, the authors construct a large-scale, scene-diverse dataset called EventMind. Experimental results demonstrate that the proposed approach significantly improves throughput while maintaining competitive performance.

### Strengths
1. The paper presents a clear motivation. It effectively addresses the high temporal resolution and sparsity characteristics of event camera data, significantly reducing data redundancy and improving the efficiency of large language model (LLM) inference.

2. It provides a scene-driven large-scale event dataset, EventMind, which makes a valuable contribution to the advancement of this field.

3. The experimental results are comprehensive, including images, tables, videos, and code. The writing is fluent, logically structured, and easy to follow.

### Weaknesses
1. The authors emphasize the advantages of their method in terms of efficiency and throughput. However, in Table 1, EventFlash-7B does not show a throughput advantage compared with EventGPT-7B. The authors should provide a discussion to clarify this discrepancy.

2. EventGPT-7B was not evaluated on the EventMind dataset. According to Figure 6, EventGPT-7B appears to work on the EventMind dataset, so why are there no quantitative comparison results presented in Table 1? The authors should explain this omission.

3. The application scope seems limited. When constructing the EventMind dataset, the authors used GPT-4o and Qwen-VL-Max to automatically generate annotations. This suggests that similar tasks could potentially be handled using standard RGB cameras combined with LLMs. Therefore, the authors should clarify why event cameras and EventFlash are necessary in this context, and how the advantages of event cameras are effectively leveraged.

### Questions
1. When processing the event data, do the authors first stack events into event counts at fixed time intervals, and then aggregate them into W₁, W₂, …, Wₙ using the Adaptive Temporal Window Aggregation module?

2. In Table 1, are the video-based methods also using event data as input? Have these video-based methods been fine-tuned on event data before comparison?

3. Can the proposed Adaptive Temporal Window Aggregation module and Sparse Density-Guided Attention module be applied to other event-based tasks, such as semantic segmentation or optical flow estimation?

### Soundness
3

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
5

### Summary
This paper presents an efficient MLLM for event-based vision tasks. The authors claim that this is the first efficient MLLM to explore spatio-temporal token sparsification, aiming to reduce redundancy, accelerate inference, and enable long-range event stream understanding. To address temporal inefficiency, the authors propose a two-stage density-guided Adaptive Temporal Window Aggregation (ATWA) module; to address spatial redundancy, they introduce a Sparse Density-Guided Attention (SDGA) module. To demonstrate performance on long-range event tasks, the authors also generate a large dataset containing 500k events. Experiments are conducted to evaluate the proposed framework on the self-generated dataset across seven tasks: motion captioning, event question answering (Event QA), human action QA, multiple-choice QA (MCQA), simple captioning, fine-grained QA (FGQA), and scene captioning. The authors report throughput performance compared to state-of-the-art (SOTA) methods using an A100 GPU. Some qualitative results are also provided.

### Strengths
1. A novel self-generated dataset that could be useful for the event-based MLLM research community.
2. Two interesting modules appear to help improve throughput while achieving comparable accuracy.
3. Some qualitative analyses are provided.

### Weaknesses
1. The experimental comparison to actual SOTA MLLMs (e.g., Qwen2.5-VL, InternVL, LLaVA-v1.6) is critically undermined by the paper's failure to state whether these baselines were finetuned on event data. This strongly suggests an unfair comparison against zero-shot models, rendering the performance results unreliable.
2. The reliance on "RGB-style" (image-like) representations is a significant limitation. The method's effectiveness is never validated against true native event representations (e.g., point clouds, graphs), which would be necessary to prove its value for event-based vision.
3. The framework appears to be an incremental application of common MLLM techniques (e.g., standard token sparsification, three-stage training) to a pre-processed, image-like data format, rather than a fundamental innovation for "raw event streams."
4. The paper's second major contribution, the EventMind dataset, is undermined by critical concerns. Significant doubts exist regarding the quality of annotations generated by GPT-4o, especially given its visual token limits and the challenge of processing long event sequences. The "human checking" and validation process lacks sufficient detail to be verifiable.
5. The paper's headline claim of a "12.4× throughput improvement" is highly misleading. This gain is measured against "EventFlash-Zero," an internal baseline, not against established SOTA models.
6. Benchmarking is incomplete, omitting comparisons against key models like EventGPT-13B and failing to report results on established metrics from prior work (e.g., EventGPT's VQA setting).
7. A fundamental contradiction exists between the paper's premise and its implementation. The paper claims to explore a "sparsification strategy for raw event streams," but the methodology reveals it operates on "image-based tokens" from a standard ViT encoder.

### Questions
See weakness. I am frustrated that the author did not make corresponding revisions to the submitted paper based on the NeurIPS review comments. This work has so many limitation, although its packaging is good. I think the event community need the solid and usefull MLLM-based works, not limited in the Line MLLM+some existing techinuqes+datasets''.

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
2

### Summary
This paper addresses the inefficiency of applying dense, frame-like processing to sparse event-based MLLMs. It makes two main contributions: (1) EventMind, a new 500k-sample instruction dataset for event vision, enabling a short-to-long curriculum learning strategy; and (2) EventFlash, an efficient MLLM using adaptive temporal (ATWA) and sparse spatial (SDGA) token sparsification(2). Experiments show EventFlash achieves a 12.4x throughput gain over its non-sparse baseline and can process much longer event sequences (1,000 bins).

### Strengths
1. The creation of the 500k-sample EventMind dataset is a major contribution that addresses a critical resource gap for training and benchmarking event-based MLLMs.
2. The paper targets the correct bottleneck: the inefficiency of applying dense methods to sparse data. The proposed spatiotemporal sparsification modules (ATWA and SDGA) are an intuitive and direct solution to this problem.

### Weaknesses
1. Insufficient comparison to SOTA event-based models: The paper fails to benchmark EventFlash against its direct competitors. On its new EventMind dataset, it only compares against frame-based models (Table 1). On the existing EventChat-Sub dataset, it only compares against EventGPT, omitting other SOTA event models like EventVL mentioned in the related work. This makes the SOTA performance claims unsubstantiated.
2. Missing Key Methodological Ablations: The core Adaptive Temporal Window Aggregation (ATWA) module is a complex two-stage process (a spike-based merge followed by a semantic-based merge. However, the ablation study (Table 2) only validates the entire "+T" (Temporal) block at once. It never justifies the necessity of this complex two-stage design over a simpler, single-stage alternative.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
