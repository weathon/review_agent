# Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Unified Multimodal Models (UMMs) built on shared autoregressive (AR) transformers are attractive for their architectural simplicity. However, we identify a critical limitation: when trained on multimodal inputs, modality-shared transformers suffer from severe gradient conflicts between vision and text, particularly in shallow and deep layers. We trace this issue to the fundamentally different low-level statistical properties of images and text, while noting that conflicts diminish in middle layers where representations become more abstract and semantically aligned. 
To overcome this challenge, we propose Uni-X, a two-end-separated, middle-shared architecture. Uni-X dedicates its initial and final layers to modality-specific processing, while maintaining shared parameters in the middle layers for high-level semantic fusion. This X-shaped design not only eliminates gradient conflicts at both ends but also further alleviates residual conflicts in the shared layers.
Extensive experiments validate the effectiveness of Uni-X. Under identical training conditions, Uni-X achieves superior training efficiency compared to strong baselines. When scaled to 3B parameters with larger training data, Uni-X matches or surpasses 7B AR-based UMMs, achieving a GenEval score of 82 for image generation alongside strong performance in text and vision understanding tasks.
These results establish Uni-X as a parameter-efficient and scalable foundation for future unified multimodal modeling.
Our code is available at https://github.com/CURRENTF/Uni-X.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Uni-X, a two-ended, middle-shared architecture: modality-specific blocks at the input and output are paired with a shared intermediate trunk for middle-level semantic fusion. This X-shaped layout mitigates gradient conflicts at both ends and further reduces residual interference within the shared layers. The design offers a clear architectural principle for unified generation models and merits wider exploration. Extensive experiments validate its effectiveness. Under identical training settings, Uni-X trains more efficiently than strong baselines. Moreover, the architecture scales to larger model sizes while delivering performance competitive with established large models.

### Strengths
Uni-X’s strengths lie in a clear diagnosis of why shared autoregressive multimodal transformers struggle and a simple remedy that matches the problem: the paper pinpoints where vision–text conflicts arise across depth, then proposes a two-end-separated, middle-shared 
``X-shape'' layout that keeps early and late processing modality-specific while fusing semantics in the middle, yielding a lightweight design that reduces conflict without adding heavy subsystems. Empirically, the model shows strong results across understanding and generation, improves training throughput under matched settings, and scales efficiently, with a 3B variant achieving performance competitive with or better than several 7B baselines, which supports the claim that the approach is both practical and parameter-efficient for unified modeling.

### Weaknesses
The paper's weaknesses center on evidentiary clarity and completeness: the gradient-conflict metric $c_g$ is ambiguously defined (the roles of $g_{\text{any}}^{1}$ and $g_{\text{any}}^{2}$, indexing, aggregation, and normalization are unclear); inference efficiency is not evaluated, even though the X-shaped split-and-share design may impede parallelism relative to fully shared Transformers; Table 4 omits the exact number of shared layers and split points needed for replication; and the in-context learning claim is supported only by qualitative figures without quantitative benchmarks. However, I do not view these as major issues. The paper offers a systematic analysis of gradient conflict in unified generation models and provides useful insights for the community, so I believe it is a valuable contribution that deserves encouragement.

### Questions
1. I am not convinced that the conflict factor $c_g$ is defined in a way that faithfully reflects gradient conflicts between image and text inputs. In particular, the meanings of $g_{\text{any}}^{1}$ and $g_{\text{any}}^{2}$ are unclear. Please provide precise definitions on what ``any'' indexes. Since $c_g$ underpins the paper’s claims, a clear mathematical specification would be very helpful.

2. I noticed that Table 3 reports training efficiency. What about inference efficiency? Compared with a standard fully shared Transformer, the X-shaped split-and-share design may be less friendly to parallelization. I recommend evaluating Uni-X at inference time and discussing any trade-offs in latency and accuracy.

3. What is the number of shard layers in Table 4?

4. Uni-X’s in-context learning capability is demonstrated only qualitatively in Figure 5. Could you provide quantitative or statistical results to substantiate this claim?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose Uni-X, a two-end-separated, middle-shared architecture, whose X-shaped design not only eliminates gradient conflicts at both ends but also further alleviates residual conflicts in the shared layers. When scaled to 3B parameters with larger training data, Uni-X matches or surpasses 7B AR-based UMMs, achieving a GenEval score of 82 for image generation alongside strong performance in text and vision understanding tasks. These results establish Uni-X as a parameter-efficient and scalable foundation for future unified multimodal modeling.

### Strengths
1. It proposes the Uni-X architecture, which processes vision and text inputs separately, fuses them for computation only in the intermediate layers, and then separates them again for modality-specific outputs.
2. It defines the concept of "Gradient Conflict" to represent the conflict between the vision and text components during model training. This concept is highly extensible to other multimodal models for analyzing the interplay between different modalities, making it valuable for future research in the field.

### Weaknesses
1. The paper defines *Gradient Conflict*, but the later sections lack further analysis. What I would like to understand is how Gradient Conflict reduces training efficiency or how it leads to model failures during inference. Is Gradient Conflict severe in mainstream models such as Qwen? Specifically, I hope to see a quantitative comparison of Gradient Conflict across different models and an analysis of its impacts.

2. The experimental results in the paper show that Uni-X is comparable to other large models, but we notice that Uni-X performs worse than its base model Qwen2.5-3B on multiple benchmarks such as ARC-C and MMLU. I am concerned about whether such a Uni-X architecture is reasonable, and I hope to see further analysis of this phenomenon.

### Questions
1. The paper defines  $S_{base} = \cos(g^1_{any}, g^2_{any})$ . How exactly are ( $g^1_{any}$ ) and ( $g^2_{any}$ ) computed? Are they averaged over multiple batches? For different types of multimodal data such as understanding, captioning, and generation, is ( $S_{base}$ ) stable? In other words, is this an inherent internal characteristic of the model?

2. The paper compares the influence of different ratios between shallow separated layers ($x$) and deep separated layers ($y$). Another point of interest is the parameter ratio between t-layers and v-layers within the separated layers. I would like to see an ablation study regarding this aspect.

In summary, my primary concerns lie in **Weaknesses 1 and Weaknesses 2**. If the responses are reasonable and acceptable, I will raise the score.

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
3

### Summary
The paper introduces a novel unified multimodal model architecture named Uni-X, which aims to address the gradient conflict issue in current unified multimodal models (UMMs) based on shared autoregressive (AR) Transformers. Uni-X employs an "X-shaped" structure design with "separation at both ends and sharing in the middle," effectively alleviating conflicts between the visual and textual modalities while maintaining model simplicity. This design enhances training efficiency and performance in generation and understanding.

### Strengths
The paper measures the degree of cross-modal gradient conflict by comparing the similarity of backpropagated gradients. It is found that visual and textual modalities cause severe gradient conflicts in both shallow and deep layers due to their vastly different underlying statistical properties, making it difficult for the model to optimize both simultaneously. In contrast, middle layers exhibit less conflict as they represent more abstract and semantically aligned features. Therefore, the authors propose designing model architectures based on modality characteristics rather than enforcing shared parameters across all layers. A series of experiments demonstrate the effectiveness of the proposed architecture.

The paper presents innovative insights, with solid experimental quality and clear expression, and holds significant importance.

### Weaknesses
1. The authors' explanation of using gradient similarity as a measure of gradient conflict is not particularly clear. Additional explanation is needed to clarify why lower similarity indicates stronger gradient conflict.

2. Uni-X modifies the fully shared network structure by introducing separate branches for vision and text. This seems to revert to the classical dual-tower structure, with separated shallow and deep layers seemingly taking on some of the functions of a multimodal encoder-decoder. The authors do not elaborate on the similarities and differences between the Uni-X structure and classical structure.

3. The paper lacks experimental comparisons with some recent models, such as blipo3-Next.

### Questions
1. Please explain why lower gradient similarity indicates stronger gradient conflict.

2. Please elaborate on the similarities and differences between the Uni-X structure and the dual-tower architecture. If gradient conflicts imply the separation of the model, does it mean that a dual-tower architecture rather than a fully shared architecture is more suitable for generating and understanding integrated tasks?

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
3

### Summary
This paper introduces Uni-X, a new unified mllm architecture that addresses the "gradient conflict" phenomenon during unified training. By separating first and final layers while fusing different modalities in the middle layers, conflited gradients are alleviated compared to standard architectures. By scaling on data and compute, Uni-X achieves good performance in generation and understanding compared to its counterparts.

### Strengths
1. The presentation is clear and the paper is easy to follow.
2. The code is already open-sourced, which is a plus for the community.

### Weaknesses
1. More evaluations on image generation benchmarks, e.g., MSCOCO, T2I-CompBench, should be considered to further validate the effectiveness of Uni-X.
2. Some baselines like Blip3o and Bagel are also evaluated on image editing benchmarks like ImgEdit or OmniContext, why is Uni-X not evaluated on such benchmarks? I believe if Uni-X is more modality unified as claimed, it should be competent in tasks like image editing that requires both image and text understanding.
3. The overall performance of Uni-X is mediocore compared with other baselines.

### Questions
1. Why does the authors choose cosine similarity as metrics for gradient-conflict? I believe more insights on this would be beneficial.

### Soundness
3

### Presentation
2

### Contribution
2
